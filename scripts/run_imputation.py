import copy
import datetime
import os
import pathlib
from argparse import ArgumentParser, Namespace
import swanlab
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import sys

# 获取当前脚本的父目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(script_dir)
# 将项目根目录添加到Python路径中
sys.path.append(project_root)
from lib import fillers, datasets, config
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import ImputationDataset, GraphImputationDataset
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool

import warnings
warnings.filterwarnings("ignore")


def has_graph_support(model_cls):
    return model_cls in [models.GRINet, models.MPGRUNet, models.BiMPGRUNet]


def get_model_classes(model_str):
    if model_str == 'brits':
        model, filler = models.BRITSNet, fillers.BRITSFiller
    elif model_str == 'grin':
        model, filler = models.GRINet, fillers.GraphFiller
    elif model_str == 'mpgru':
        model, filler = models.MPGRUNet, fillers.GraphFiller
    elif model_str == 'bimpgru':
        model, filler = models.BiMPGRUNet, fillers.GraphFiller
    elif model_str == 'var':
        model, filler = models.VARImputer, fillers.Filler
    elif model_str == 'gain':
        model, filler = models.RGAINNet, fillers.RGAINFiller
    elif model_str == 'birnn':
        model, filler = models.BiRNNImputer, fillers.MultiImputationFiller
    elif model_str == 'rnn':
        model, filler = models.RNNImputer, fillers.Filler
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name):
    if dataset_name[:3] == 'air':
        dataset = datasets.AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    elif dataset_name == 'bay_block':
        dataset = datasets.MissingValuesPemsBay()
    elif dataset_name == 'la_block':
        dataset = datasets.MissingValuesMetrLA()
    elif dataset_name == 'la_point':
        dataset = datasets.MissingValuesMetrLA(p_fault=0., p_noise=0.25)
    elif dataset_name == 'bay_point':
        dataset = datasets.MissingValuesPemsBay(p_fault=0., p_noise=0.25)
    elif dataset_name == 'guangzhou':
        dataset = datasets.MyCity(impute_nans=True, data_dir='guangzhou')
    elif dataset_name == 'beijing':
        dataset = datasets.MyCity(impute_nans=True, data_dir='beijing')
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='grin')
    parser.add_argument("--dataset-name", type=str, default='guangzhou')
    parser.add_argument("--config", type=str, default=None)
    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--consistency-loss', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    # gain hparams
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--hint-rate', type=float, default=0.7)
    parser.add_argument('--g-train-freq', type=int, default=1)
    parser.add_argument('--d-train-freq', type=int, default=5)

    known_args, _ = parser.parse_known_args()
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])
            if arg == 'samples_per_epoch' and config_args[arg] == 'None':
                setattr(args, arg, None)

    return args


def seed_everything(seed=None):
    """
    替换 pl.seed_everything
    """
    import random
    import numpy as np
    import torch
    
    if seed is None or seed < 0:
        seed = random.randint(0, 10000)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def run_experiment(args, exp_dir=None):
    # Set configuration
    torch.set_num_threads(1)
    
    # 替换 pl.seed_everything
    seed = seed_everything(args.seed)

    model_cls, filler_cls = get_model_classes(args.model_name)
    dataset = get_dataset(args.dataset_name)

    ########################################
    # create logdir and save configuration #
    ########################################
    if exp_dir is None:
        exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    else:
        exp_name = exp_dir
    # save config for logging
    logdir = os.path.join(config['logs'], args.dataset_name, args.model_name, exp_name)
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)
    ########################################
    # data module                          #
    ########################################

    # instantiate dataset
    dataset_cls = GraphImputationDataset if has_graph_support(model_cls) else ImputationDataset
    torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                mask=dataset.training_mask,
                                eval_mask=dataset.eval_mask,
                                window=args.window,
                                stride=args.stride)

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(args, dataset.splitter, return_dict=True)
    train_idxs, val_idxs, test_idxs = dataset.splitter(torch_dataset, **split_conf)

    # configure datamodule
    data_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                  **data_conf)
    dm.setup()

    # if out of sample in air, add values removed for evaluation in train set
    if not args.in_sample and args.dataset_name[:3] == 'air':
        dm.torch_dataset.mask[dm.train_slice] |= dm.torch_dataset.eval_mask[dm.train_slice]

    # get adjacency matrix
    adj = dataset.get_similarity(thr=args.adj_threshold)
    # force adj with no self loop
    np.fill_diagonal(adj, 0.)
    # ===== city semantic embedding =====
    city_emb = dataset.city_emb

    ########################################
    # predictor                            #
    ########################################

    # model's inputs
    additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes, city_emb=city_emb, dist_mat=dataset.dist, use_proportion_aware=args.use_proportion_aware)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(),
               'mape': MaskedMAPE(),
               'mse': MaskedMSE(),
               'mre': MaskedMRE()}

    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr,
                                                   'weight_decay': args.l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': args.epochs
                                     },
                                     alpha=args.alpha,
                                     hint_rate=args.hint_rate,
                                     g_train_freq=args.g_train_freq,
                                     d_train_freq=args.d_train_freq)
    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)
    filler = filler_cls(**filler_kwargs)

    ########################################
    # training                             #
    ########################################

    # 获取数据加载器
    train_loader = dm.get_train_dataloader()
    val_loader = dm.get_val_dataloader()
    test_loader = dm.get_test_dataloader()

    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filler.device = device
    filler.to(device)

    # 训练参数
    best_val_mae = float('inf')
    patience_counter = 0
    best_model_path = None
    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建一个SwanLab项目
    swanlab.init(
        # 设置项目名
        project=f"{args.dataset_name}-{args.model_name}",
        experiment_name=exp_name,
        # 设置超参数
        config=vars(args),
        description=f"Training {args.model_name} on {args.dataset_name} with seed {seed}",
        logdir=logdir
    )

    # 使用 tqdm 显示训练进度
    print(f"\n开始训练 {args.model_name} 在 {args.dataset_name}")
    print(f"设备: {device} | 随机种子: {seed} | 批次大小: {args.batch_size}")
    
    # 创建进度条
    epoch_pbar = tqdm(range(args.epochs), desc="训练进度", unit="epoch")
    
    for epoch in epoch_pbar:
        # 训练
        filler.train()
        
        train_loss = 0.0
        train_batches = 0
        
        # 使用 tqdm 显示 batch 进度
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", 
                         leave=False)
        
        for batch in batch_pbar:
            batch = filler._move_to_device(batch, device)
            filler.optimizer.zero_grad()
            loss = filler.training_step(batch, 0)
            loss.backward()
            
            if args.grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(filler.model.parameters(), args.grad_clip_val)
            
            filler.optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # 更新 batch 进度条
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # # 记录 batch 损失
            # if train_batches % 10 == 0:
            #     swanlab.log({'train/batch_loss': loss.item()})
        
        batch_pbar.close()
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        train_metrics = filler.compute_metrics('train')
        
        # 验证
        filler.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = filler._move_to_device(batch, device)
                val_loss_batch = filler.validation_step(batch, 0)
                val_loss += val_loss_batch.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        val_metrics = filler.compute_metrics('val')
        
        # 记录到 SWANLab
        epoch_logs = {
            'epoch': epoch + 1,
            'train/loss': avg_train_loss,
            'val/loss': avg_val_loss,
            'train/lr': filler.get_lr()
        }
        
        for metric_name, metric_value in train_metrics.items():
            simple_name = metric_name[6:] if metric_name.startswith('train_') else metric_name
            if hasattr(metric_value, 'item'):
                metric_value = metric_value.item()
            epoch_logs[f'train/{simple_name}'] = metric_value
        for metric_name, metric_value in val_metrics.items():
            simple_name = metric_name[4:] if metric_name.startswith('val_') else metric_name
            if hasattr(metric_value, 'item'):
                metric_value = metric_value.item()
            epoch_logs[f'val/{simple_name}'] = metric_value
        
        swanlab.log(epoch_logs, step=epoch)
        
        # 更新 epoch 进度条
        val_mae = val_metrics.get('val_mae', 0)
        val_mse = val_metrics.get('val_mse', 0)
        val_mre = val_metrics.get('val_mre', 0)
        val_mape = val_metrics.get('val_mape', 0)
        # 转换为标量
        if hasattr(val_mae, 'item'):
            val_mae = val_mae.item()
        if hasattr(val_mse, 'item'):
            val_mse = val_mse.item()
        if hasattr(val_mre, 'item'):
            val_mre = val_mre.item()
        if hasattr(val_mape, 'item'):
            val_mape = val_mape.item()
        # 更新进度条，只显示训练损失和验证的四个指标
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_mae': f'{val_mae:.4f}',
            'val_mse': f'{val_mse:.4f}',
            'val_mre': f'{val_mre:.4f}',
            'val_mape': f'{val_mape:.4f}'
        })
        
        # 保存检查点
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth')
        filler.save_checkpoint(checkpoint_path, epoch)
        
        # 保存最佳模型
        val_mae = val_metrics.get('val_mae', float('inf'))
        if hasattr(val_mae, 'item'):
            val_mae = val_mae.item()
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            filler.save_checkpoint(best_model_path, epoch, is_best=True)
            patience_counter = 0
            epoch_pbar.write(f"Epoch {epoch+1}: 保存最佳模型 (val_mae: {val_mae:.4f})")
        else:
            patience_counter += 1
            epoch_pbar.write(f"Epoch {epoch+1}: 耐心计数 {patience_counter}/{args.patience}")
        
        # 学习率调度
        if filler.scheduler is not None:
            filler.scheduler.step()
        
        # 早停检查
        if patience_counter >= args.patience:
            epoch_pbar.write(f"早停在 epoch {epoch+1}")
            swanlab.log({'info/early_stopped': True, 'info/stopped_at_epoch': epoch + 1})
            break
    
    epoch_pbar.close()
    
    ########################################
    # testing                              #
    ########################################

    print("\n开始测试...")

    if best_model_path and os.path.exists(best_model_path):
        filler.load_checkpoint(best_model_path, device)
        print(f"已加载最佳模型: {best_model_path}")
    else:
        print("警告: 未找到最佳模型，使用最后保存的模型")

    filler.eval()
    filler.freeze = lambda: filler.eval()

    if torch.cuda.is_available():
        filler.cuda()

    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(test_loader, device=device, return_mask=True)
    y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[:3])  # reshape to (eventually) squeeze node channels

    # Test imputations in whole series
    eval_mask = dataset.eval_mask[dm.test_slice]
    df_true = dataset.df.iloc[dm.test_slice]
    metrics = {
        'mae': numpy_metrics.masked_mae,
        'mse': numpy_metrics.masked_mse,
        'mre': numpy_metrics.masked_mre,
        'mape': numpy_metrics.masked_mape
    }
    # Aggregate predictions in dataframes
    index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
    aggr_methods = ensure_list(args.aggregate_by)
    df_hats = prediction_dataframe(y_hat, index, dataset.df.columns, aggregate_by=aggr_methods)
    df_hats = dict(zip(aggr_methods, df_hats))

    results = {}
    test_logs = {}
    
    # 打印测试结果
    print("\n测试结果:")
    for aggr_by, df_hat in df_hats.items():
        results[aggr_by] = {}
        for metric_name, metric_fn in metrics.items():
            error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
            results[aggr_by][metric_name] = error
            test_logs[f'test/{aggr_by}/{metric_name}'] = error
            print(f"  {aggr_by.upper()} {metric_name.upper()}: {error:.4f}")
    
    # 记录测试结果
    swanlab.log(test_logs)
    swanlab.log({'info/best_val_mae': best_val_mae})
    
    # 完成日志记录
    swanlab.finish()
    
    # 打印最佳结果
    print(f"\n训练完成! 最佳验证 MAE: {best_val_mae:.4f}")
    print(f"实验结果已保存到: {logdir}")
    
    return results


if __name__ == '__main__':
    args = parse_args()
    
    # 打印配置
    print("实验配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # 设置种子列表
    if args.seed < 0:
        # 如果指定了负的种子，使用5个随机种子
        np.random.seed(42)  # 固定随机数生成器，确保可重复性
        seed_list = [np.random.randint(1e9) for _ in range(5)]
    else:
        # 如果指定了特定种子，只使用这一个种子
        seed_list = [args.seed]
    
    # 创建主实验文件夹
    main_logdir = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    all_results = []
    for seed in seed_list:
        print(f"\n========== Running seed {seed} ==========")
        args_i = copy.deepcopy(args)
        args_i.seed = seed
        # 为当前种子创建子目录
        seed_logdir = os.path.join(main_logdir, f"seed_{seed}")
        res = run_experiment(args_i, exp_dir=seed_logdir)
        all_results.append(res)
    # ===== 汇总结果 =====
    print("\n========== FINAL RESULTS (mean ± std) ==========")
    aggr = 'mean'
    metric_names = all_results[0][aggr].keys()

    for metric in metric_names:
        values = [r[aggr][metric] for r in all_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric}: {mean:.4f} ± {std:.4f}")