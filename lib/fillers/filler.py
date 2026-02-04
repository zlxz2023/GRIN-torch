import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Union
import inspect
import numpy as np

from .. import epsilon
from ..nn.utils.metric_base import MaskedMetric
from ..utils.utils import ensure_list


class Filler(nn.Module):
    def __init__(self,
                 model_class,
                 model_kwargs: Dict,
                 optim_class,
                 optim_kwargs: Dict,
                 loss_fn,
                 scaled_target: bool = False,
                 whiten_prob: float = 0.05,
                 metrics: Optional[Dict] = None,
                 scheduler_class=None,
                 scheduler_kwargs: Optional[Dict] = None):
        """
        PyTorch Module 实现缺失值填充器

        :param model_class: torch.nn.Module 的实现类
        :param model_kwargs: 模型的参数
        :param optim_class: 优化器类
        :param optim_kwargs: 优化器参数
        :param loss_fn: 训练使用的损失函数
        :param scaled_target: 是否在计算损失前缩放目标
        :param whiten_prob: 在训练中将值置为缺失并用作真值的概率
        :param metrics: 指标字典
        :param scheduler_class: 学习率调度器类
        :param scheduler_kwargs: 调度器参数
        """
        super(Filler, self).__init__()
        
        # 保存参数
        self.model_cls = model_class
        self.model_kwargs = model_kwargs
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}
        
        # 损失函数
        if loss_fn is not None:
            self.loss_fn = self._check_metric(loss_fn)
        else:
            self.loss_fn = None
        
        self.scaled_target = scaled_target
        
        # 训练时保留真值的概率
        assert 0. <= whiten_prob <= 1.
        self.keep_prob = 1. - whiten_prob
        
        # 指标
        if metrics is None:
            metrics = {}
        self._set_metrics(metrics)
        
        # 实例化模型
        self.model = self.model_cls(**self.model_kwargs)
        
        # 优化器和调度器
        self.optimizer = None
        self.scheduler = None
        self._configure_optimizers()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        
    def reset_model(self):
        """重置模型"""
        self.model = self.model_cls(**self.model_kwargs)
        self._configure_optimizers()
    
    @property
    def trainable_parameters(self):
        """可训练参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.model(*args, **kwargs)
    
    @staticmethod
    def _check_metric(metric, compute_on_step: bool = False):
        """
        检查指标，如果是函数则包装为 MaskedMetric
        """
        if not isinstance(metric, MaskedMetric):
            if 'reduction' in inspect.getfullargspec(metric).args:
                metric_kwargs = {'reduction': 'none'}
            else:
                metric_kwargs = {}
            return MaskedMetric(metric, compute_on_step=compute_on_step, metric_kwargs=metric_kwargs)
        return deepcopy(metric)
    
    def _set_metrics(self, metrics: Dict):
        """设置训练、验证、测试指标"""
        self.train_metrics = {f'train_{k}': self._check_metric(m, compute_on_step=True) 
                            for k, m in metrics.items()}
        self.val_metrics = {f'val_{k}': self._check_metric(m) for k, m in metrics.items()}
        self.test_metrics = {f'test_{k}': self._check_metric(m) for k, m in metrics.items()}
    
    def _preprocess(self, data, batch_preprocessing: Dict):
        """
        预处理数据
        
        :param data: torch.Tensor [batch, steps, nodes, features]
        :param batch_preprocessing: 预处理字典
        :return: 预处理后的数据
        """
        if isinstance(data, (list, tuple)):
            return [self._preprocess(d, batch_preprocessing) for d in data]
        
        trend = batch_preprocessing.get('trend', 0.)
        bias = batch_preprocessing.get('bias', 0.)
        scale = batch_preprocessing.get('scale', 1.)
        
        return (data - trend - bias) / (scale + epsilon)
    
    def _postprocess(self, data, batch_preprocessing: Dict):
        """
        后处理（逆变换）数据
        
        :param data: torch.Tensor [batch, steps, nodes, features]
        :param batch_preprocessing: 预处理字典
        :return: 逆变换后的数据
        """
        if isinstance(data, (list, tuple)):
            return [self._postprocess(d, batch_preprocessing) for d in data]
        
        trend = batch_preprocessing.get('trend', 0.)
        bias = batch_preprocessing.get('bias', 0.)
        scale = batch_preprocessing.get('scale', 1.)
        
        return data * (scale + epsilon) + bias + trend
    
    def predict_batch(self, batch, preprocess: bool = False, postprocess: bool = True, 
                     return_target: bool = False):
        """
        预测一个批次的数据
        
        :param batch: 数据批次
        :param preprocess: 是否预处理
        :param postprocess: 是否后处理
        :param return_target: 是否返回目标
        :return: 预测结果
        """
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        
        if preprocess:
            x = batch_data.pop('x')
            x = self._preprocess(x, batch_preprocessing)
            y_hat = self.forward(x, **batch_data)
        else:
            y_hat = self.forward(**batch_data)
        
        # 后处理
        if postprocess:
            y_hat = self._postprocess(y_hat, batch_preprocessing)
        
        if return_target:
            y = batch_data.get('y')
            mask = batch_data.get('mask', None)
            return y, y_hat, mask
        
        return y_hat
    
    def predict_loader(self, loader, preprocess: bool = False, postprocess: bool = True, 
                      return_mask: bool = True, device: str = 'cuda'):
        """
        预测整个数据加载器的数据
        
        :param loader: 数据加载器
        :param preprocess: 是否预处理
        :param postprocess: 是否后处理
        :param return_mask: 是否返回掩码
        :param device: 设备
        :return: 预测结果
        """
        targets, imputations, masks = [], [], []
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                # 移动数据到设备
                batch = self._move_to_device(batch, device)
                batch_data, batch_preprocessing = self._unpack_batch(batch)
                
                # 提取掩码和目标
                eval_mask = batch_data.pop('eval_mask', None)
                y = batch_data.pop('y')
                
                y_hat = self.predict_batch(batch, preprocess=preprocess, postprocess=postprocess)
                
                if isinstance(y_hat, (list, tuple)):
                    y_hat = y_hat[0]
                
                targets.append(y)
                imputations.append(y_hat)
                masks.append(eval_mask)
        
        y = torch.cat(targets, 0)
        y_hat = torch.cat(imputations, 0)
        
        if return_mask:
            mask = torch.cat(masks, 0) if masks[0] is not None else None
            return y, y_hat, mask
        
        return y, y_hat
    
    def _unpack_batch(self, batch):
        """
        解包批次数据
        
        :param batch: 批次数据
        :return: 数据字典和预处理字典
        """
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            batch_data, batch_preprocessing = batch
        else:
            batch_data = batch
            batch_preprocessing = {}
        return batch_data, batch_preprocessing
    
    def _move_to_device(self, batch, device: str):
        """移动数据到设备"""
        if isinstance(batch, (tuple, list)):
            return [self._move_to_device(b, device) for b in batch]
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch
    
    def training_step(self, batch, batch_idx: int = 0):
        """
        训练步骤
        
        :param batch: 批次数据
        :param batch_idx: 批次索引
        :return: 损失
        """
        # 解包批次
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        
        # 提取掩码和目标
        mask = batch_data['mask'].clone().detach()
        batch_data['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_prob).bool()
        eval_mask = batch_data.pop('eval_mask')
        eval_mask = (mask | eval_mask) - batch_data['mask']
        
        y = batch_data.pop('y')
        
        # 计算预测和损失
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)
        
        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
        
        loss = self.loss_fn(imputation, target, mask)
        
        # 计算指标
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        
        # 更新训练指标
        for metric_name, metric in self.train_metrics.items():
            metric.update(imputation.detach(), y, eval_mask)
        
        self.global_step += 1
        
        return {
            'loss': loss,
            'imputation': imputation.detach(),
            'y': y,
            'eval_mask': eval_mask
        }
    
    def validation_step(self, batch, batch_idx: int = 0):
        """
        验证步骤
        
        :param batch: 批次数据
        :param batch_idx: 批次索引
        :return: 验证损失
        """
        # 解包批次
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        
        # 提取掩码和目标
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')
        
        # 计算预测和损失
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)
        
        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
        
        val_loss = self.loss_fn(imputation, target, eval_mask)
        
        # 计算指标
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        
        # 更新验证指标
        for metric_name, metric in self.val_metrics.items():
            metric.update(imputation.detach(), y, eval_mask)
        
        return {
            'val_loss': val_loss,
            'imputation': imputation.detach(),
            'y': y,
            'eval_mask': eval_mask
        }
    
    def test_step(self, batch, batch_idx: int = 0):
        """
        测试步骤
        
        :param batch: 批次数据
        :param batch_idx: 批次索引
        :return: 测试损失
        """
        # 解包批次
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        
        # 提取掩码和目标
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')
        
        # 计算输出和重缩放
        imputation = self.predict_batch(batch, preprocess=False, postprocess=True)
        test_loss = self.loss_fn(imputation, y, eval_mask)
        
        # 更新测试指标
        for metric_name, metric in self.test_metrics.items():
            metric.update(imputation.detach(), y, eval_mask)
        
        return {
            'test_loss': test_loss,
            'imputation': imputation.detach(),
            'y': y,
            'eval_mask': eval_mask
        }
    
    def compute_metrics(self, stage: str = 'train'):
        """
        计算指标
        
        :param stage: 阶段（train, val, test）
        :return: 指标字典
        """
        if stage == 'train':
            metrics = self.train_metrics
        elif stage == 'val':
            metrics = self.val_metrics
        elif stage == 'test':
            metrics = self.test_metrics
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        results = {}
        for metric_name, metric in metrics.items():
            results[metric_name] = metric.compute()
            metric.reset()  # 重置指标
        
        return results
    
    def _configure_optimizers(self):
        """配置优化器和调度器"""
        self.optimizer = self.optim_class(self.model.parameters(), **self.optim_kwargs)
        
        if self.scheduler_class is not None:
            self.scheduler = self.scheduler_class(self.optimizer, **self.scheduler_kwargs)
        else:
            self.scheduler = None
    
    def get_lr(self):
        """获取当前学习率"""
        if self.optimizer is None:
            return 0.0
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0
    
    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_metrics': {k: v.state_dict() for k, v in self.train_metrics.items()},
            'val_metrics': {k: v.state_dict() for k, v in self.val_metrics.items()},
            'test_metrics': {k: v.state_dict() for k, v in self.test_metrics.items()},
        }
        
        torch.save(checkpoint, path)
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str, device: str = 'cpu'):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        # 加载指标状态
        if 'train_metrics' in checkpoint:
            for k, v in self.train_metrics.items():
                v.load_state_dict(checkpoint['train_metrics'][k])
        
        if 'val_metrics' in checkpoint:
            for k, v in self.val_metrics.items():
                v.load_state_dict(checkpoint['val_metrics'][k])
        
        if 'test_metrics' in checkpoint:
            for k, v in self.test_metrics.items():
                v.load_state_dict(checkpoint['test_metrics'][k])
        
        return checkpoint['epoch']
    
    def train_epoch(self, train_loader, device: str = 'cuda', epoch: int = 0):
        """
        训练一个epoch
        
        :param train_loader: 训练数据加载器
        :param device: 设备
        :param epoch: 当前epoch
        :return: 平均损失
        """
        self.train()
        self.to(device)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动数据到设备
            batch = self._move_to_device(batch, device)
            
            # 训练步骤
            self.optimizer.zero_grad()
            result = self.training_step(batch, batch_idx)
            loss = result['loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（如果需要）
            if hasattr(self, 'grad_clip_val') and self.grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
            
            # 优化器步骤
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 每100个batch打印一次
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        self.current_epoch = epoch
        
        # 学习率调度
        if self.scheduler:
            self.scheduler.step()
        
        return avg_loss
    
    def validate(self, val_loader, device: str = 'cuda'):
        """
        验证
        
        :param val_loader: 验证数据加载器
        :param device: 设备
        :return: 平均验证损失
        """
        self.eval()
        self.to(device)
        
        total_val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = self._move_to_device(batch, device)
                result = self.validation_step(batch, batch_idx)
                total_val_loss += result['val_loss'].item()
                num_batches += 1
        
        avg_val_loss = total_val_loss / num_batches
        return avg_val_loss
    
    def test(self, test_loader, device: str = 'cuda'):
        """
        测试
        
        :param test_loader: 测试数据加载器
        :param device: 设备
        :return: 平均测试损失
        """
        self.eval()
        self.to(device)
        
        total_test_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = self._move_to_device(batch, device)
                result = self.test_step(batch, batch_idx)
                total_test_loss += result['test_loss'].item()
                num_batches += 1
        
        avg_test_loss = total_test_loss / num_batches
        return avg_test_loss