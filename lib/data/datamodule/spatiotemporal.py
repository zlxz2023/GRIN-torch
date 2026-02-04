from torch.utils.data import DataLoader, Subset, RandomSampler
import torch

from .. import TemporalDataset, SpatioTemporalDataset
from ..preprocessing import StandardScaler, MinMaxScaler
from ...utils import ensure_list
from ...utils.parser_utils import str_to_bool


class SpatioTemporalDataModule:
    def __init__(self, 
                 dataset: TemporalDataset,
                 scale=True,
                 scaling_axis='samples',
                 scaling_type='std',
                 scale_exogenous=None,
                 train_idxs=None,
                 val_idxs=None,
                 test_idxs=None,
                 batch_size=32,
                 workers=1,
                 samples_per_epoch=None):
        """
        初始化数据模块
        
        :param dataset: 时间序列数据集
        :param scale: 是否进行标准化
        :param scaling_axis: 标准化轴
        :param scaling_type: 标准化类型 (std/minmax)
        :param scale_exogenous: 需要标准化的外生变量
        :param train_idxs: 训练集索引
        :param val_idxs: 验证集索引
        :param test_idxs: 测试集索引
        :param batch_size: 批次大小
        :param workers: 数据加载工作线程数
        :param samples_per_epoch: 每个epoch的样本数（用于重采样）
        """
        self.torch_dataset = dataset
        
        # 数据划分
        self.trainset = Subset(self.torch_dataset, train_idxs if train_idxs is not None else [])
        self.valset = Subset(self.torch_dataset, val_idxs if val_idxs is not None else [])
        self.testset = Subset(self.torch_dataset, test_idxs if test_idxs is not None else [])
        
        # 预处理配置
        self.scale = scale
        self.scaling_type = scaling_type
        self.scaling_axis = scaling_axis
        self.scale_exogenous = ensure_list(scale_exogenous) if scale_exogenous is not None else None
        
        # 数据加载器配置
        self.batch_size = batch_size
        self.workers = workers
        self.samples_per_epoch = samples_per_epoch
        
        # 是否已进行标准化设置
        self._is_setup = False
    
    @property
    def is_spatial(self):
        """是否为空间时间序列数据"""
        return isinstance(self.torch_dataset, SpatioTemporalDataset)
    
    @property
    def n_nodes(self):
        """节点数（仅空间数据有效）"""
        if not self._is_setup:
            raise ValueError('You should initialize the datamodule first(function:setup()).')
        return self.torch_dataset.n_nodes if self.is_spatial else None
    
    @property
    def d_in(self):
        """输入维度"""
        if not self._is_setup:
            raise ValueError('You should initialize the datamodule first(function:setup()).')
        return self.torch_dataset.n_channels
    
    @property
    def d_out(self):
        """输出维度（预测步长）"""
        if not self._is_setup:
            raise ValueError('You should initialize the datamodule first(function:setup()).')
        return self.torch_dataset.horizon
    
    @property
    def train_slice(self):
        """训练集切片"""
        return self.torch_dataset.expand_indices(self.trainset.indices, merge=True)
    
    @property
    def val_slice(self):
        """验证集切片"""
        return self.torch_dataset.expand_indices(self.valset.indices, merge=True)
    
    @property
    def test_slice(self):
        """测试集切片"""
        return self.torch_dataset.expand_indices(self.testset.indices, merge=True)
    
    def get_scaling_axes(self, dim='global'):
        """获取标准化轴"""
        scaling_axis = tuple()
        if dim == 'global':
            scaling_axis = (0, 1, 2)
        elif dim == 'channels':
            scaling_axis = (0, 1)
        elif dim == 'nodes':
            scaling_axis = (0,)
        
        # 对于非空间时间序列数据，移除最后一个维度
        if not self.is_spatial:
            scaling_axis = scaling_axis[:-1]
        
        if not len(scaling_axis):
            raise ValueError(f'标准化轴 "{dim}" 无效')
        
        return scaling_axis
    
    def get_scaler(self):
        """获取标准化器"""
        if self.scaling_type == 'std':
            return StandardScaler
        elif self.scaling_type == 'minmax':
            return MinMaxScaler
        else:
            raise NotImplementedError(f'不支持的标准化类型: {self.scaling_type}')
    
    def setup(self, stage=None):
        """
        设置数据模块，进行标准化等预处理
        
        :param stage: 阶段
        """
        if self.scale and not self._is_setup:
            scaling_axis = self.get_scaling_axes(self.scaling_axis)
            train = self.torch_dataset.data.numpy()[self.train_slice]
            
            # 如果有掩码，使用掩码进行标准化
            train_mask = self.torch_dataset.mask.numpy()[self.train_slice] if 'mask' in self.torch_dataset else None
            
            # 拟合标准化器
            scaler = self.get_scaler()(scaling_axis).fit(train, mask=train_mask, keepdims=True).to_torch()
            self.torch_dataset.scaler = scaler
            
            # 标准化外生变量（如果指定）
            if self.scale_exogenous is not None:
                for label in self.scale_exogenous:
                    if hasattr(self.torch_dataset, label):
                        exo = getattr(self.torch_dataset, label)
                        scaler = self.get_scaler()(scaling_axis)
                        scaler.fit(exo[self.train_slice], keepdims=True).to_torch()
                        setattr(self.torch_dataset, label, scaler.transform(exo))
        
        self._is_setup = True
    
    def _data_loader(self, dataset, shuffle=False, batch_size=None, **kwargs):
        """创建数据加载器"""
        batch_size = self.batch_size if batch_size is None else batch_size
        
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=self.workers,
            **kwargs
        )
    
    def get_train_dataloader(self, shuffle=True, batch_size=None):
        """获取训练数据加载器"""
        if not self._is_setup:
            self.setup()
        
        if self.samples_per_epoch is not None:
            # 如果指定了每个epoch的样本数，使用重采样
            sampler = RandomSampler(self.trainset, replacement=True, num_samples=self.samples_per_epoch)
            return self._data_loader(
                self.trainset, 
                shuffle=False,  # 使用sampler时shuffle应为False
                batch_size=batch_size, 
                sampler=sampler, 
                drop_last=True
            )
        
        return self._data_loader(self.trainset, shuffle, batch_size, drop_last=True)
    
    def get_val_dataloader(self, shuffle=False, batch_size=None):
        """获取验证数据加载器"""
        if not self._is_setup:
            self.setup()
        
        return self._data_loader(self.valset, shuffle, batch_size)
    
    def get_test_dataloader(self, shuffle=False, batch_size=None):
        """获取测试数据加载器"""
        if not self._is_setup:
            self.setup()
        
        return self._data_loader(self.testset, shuffle, batch_size)
    
    def get_all_dataloaders(self, train_shuffle=True, val_shuffle=False, test_shuffle=False):
        """一次性获取所有数据加载器"""
        if not self._is_setup:
            self.setup()
        
        return {
            'train': self.get_train_dataloader(shuffle=train_shuffle),
            'val': self.get_val_dataloader(shuffle=val_shuffle),
            'test': self.get_test_dataloader(shuffle=test_shuffle)
        }
    
    def get_dataset_info(self):
        """获取数据集信息"""
        if not self._is_setup:
            self.setup()
        
        info = {
            'n_nodes': self.n_nodes,
            'd_in': self.d_in,
            'd_out': self.d_out,
            'train_samples': len(self.trainset),
            'val_samples': len(self.valset),
            'test_samples': len(self.testset),
            'is_spatial': self.is_spatial,
            'is_scaled': self.scale,
            'scaling_type': self.scaling_type
        }
        return info
    
    def print_info(self):
        """打印数据集信息"""
        info = self.get_dataset_info()
        print("=" * 50)
        print("数据集信息:")
        print(f"  是否为空间数据: {info['is_spatial']}")
        if info['is_spatial']:
            print(f"  节点数: {info['n_nodes']}")
        print(f"  输入维度: {info['d_in']}")
        print(f"  输出步长: {info['d_out']}")
        print(f"  训练集样本数: {info['train_samples']}")
        print(f"  验证集样本数: {info['val_samples']}")
        print(f"  测试集样本数: {info['test_samples']}")
        print(f"  是否标准化: {info['is_scaled']}")
        if info['is_scaled']:
            print(f"  标准化类型: {info['scaling_type']}")
        print("=" * 50)
    
    @staticmethod
    def add_argparse_args(parser, **kwargs):
        """
        添加命令行参数
        
        注意：在纯PyTorch中，这个函数主要用于解析参数，
        实际使用时应从args中提取参数创建实例
        """
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--scaling-axis', type=str, default="channels")
        parser.add_argument('--scaling-type', type=str, default="std")
        parser.add_argument('--scale', type=str_to_bool, nargs='?', const=True, default=True)
        parser.add_argument('--workers', type=int, default=0)
        parser.add_argument('--samples-per-epoch', type=int, default=None)
        return parser