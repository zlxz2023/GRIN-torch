from functools import partial
import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any, Callable


class MaskedMetric(nn.Module):
    def __init__(self,
                 metric_fn: Callable,
                 mask_nans: bool = False,
                 mask_inf: bool = False,
                 metric_kwargs: Optional[Dict] = None,
                 at: Optional[int] = None):
        """
        初始化掩码指标
        
        :param metric_fn: 基础指标函数
        :param mask_nans: 是否屏蔽 NaN 值
        :param mask_inf: 是否屏蔽无穷大值
        :param metric_kwargs: 指标函数的关键字参数
        :param at: 时间步索引None 表示所有步
        """
        super(MaskedMetric, self).__init__()
        
        if metric_kwargs is None:
            metric_kwargs = {}
        
        # 使用 functools.partial 预设参数
        self.metric_fn = partial(metric_fn, **metric_kwargs)
        self.mask_nans = mask_nans
        self.mask_inf = mask_inf
        # 时间步切片
        if at is None:
            self.at = slice(None)
        else:
            self.at = slice(at, at + 1)
        
        # 初始化状态
        self.register_buffer('value', torch.tensor(0.).float(), persistent=True)
        self.register_buffer('numel', torch.tensor(0), persistent=True)
        
        # 重置状态
        self.reset()
    
    def reset(self) -> None:
        """重置指标状态"""
        self.value.fill_(0.)
        self.numel.fill_(0)
    
    def _check_same_shape(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
        """检查两个张量形状是否相同"""
        if tensor1.shape != tensor2.shape:
            raise ValueError(
                f"张量形状不匹配: {tensor1.shape} 和 {tensor2.shape}"
            )
    
    def _check_mask(self, mask: Optional[torch.Tensor], val: torch.Tensor) -> torch.Tensor:
        """
        检查和生成掩码
        
        :param mask: 输入掩码
        :param val: 值张量
        :return: 处理后的掩码
        """
        if mask is None:
            mask = torch.ones_like(val, dtype=torch.bool)
        else:
            self._check_same_shape(mask, val)
        
        # 转换为布尔类型
        mask = mask.bool()
        
        # 屏蔽 NaN
        if self.mask_nans:
            mask = mask & ~torch.isnan(val)
        
        # 屏蔽无穷大
        if self.mask_inf:
            mask = mask & ~torch.isinf(val)
        
        return mask
    
    def _compute_masked(self, 
                       y_hat: torch.Tensor, 
                       y: torch.Tensor, 
                       mask: Optional[torch.Tensor] = None) -> tuple:
        """
        计算带掩码的指标值
        
        :param y_hat: 预测值
        :param y: 真实值
        :param mask: 掩码
        :return: (指标值之和, 有效元素数)
        """
        self._check_same_shape(y_hat, y)
        
        # 计算基础指标
        val = self.metric_fn(y_hat, y)
        
        # 处理掩码
        mask = self._check_mask(mask, val)
        
        # 应用掩码
        val = torch.where(mask, val, torch.tensor(0., device=val.device).float())
        
        return val.sum(), mask.sum()
    
    def _compute_std(self, y_hat: torch.Tensor, y: torch.Tensor) -> tuple:
        """
        计算标准指标值（无掩码）
        
        :param y_hat: 预测值
        :param y: 真实值
        :return: (指标值之和, 总元素数)
        """
        self._check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        return val.sum(), val.numel()
    
    def is_masked(self, mask: Optional[torch.Tensor]) -> bool:
        """检查是否需要使用掩码"""
        return self.mask_inf or self.mask_nans or (mask is not None)
    
    def update(self, 
               y_hat: torch.Tensor, 
               y: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> None:
        """
        更新指标状态
        
        :param y_hat: 预测值
        :param y: 真实值
        :param mask: 掩码
        """
        # 切片处理
        y_hat = y_hat[:, self.at]
        y = y[:, self.at]
        
        if mask is not None:
            mask = mask[:, self.at]
        
        # 计算指标
        if self.is_masked(mask):
            val, numel = self._compute_masked(y_hat, y, mask)
        else:
            val, numel = self._compute_std(y_hat, y)
        
        # 更新状态
        self.value += val
        self.numel += numel
    
    def compute(self) -> torch.Tensor:
        """
        计算最终指标值
        
        :return: 平均指标值
        """
        if self.numel > 0:
            return self.value / self.numel
        return self.value
    
    def forward(self, 
                y_hat: torch.Tensor, 
                y: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播（一次性计算，不更新内部状态）
        
        :param y_hat: 预测值
        :param y: 真实值
        :param mask: 掩码
        :return: 指标值
        """
        # 切片处理
        y_hat = y_hat[:, self.at]
        y = y[:, self.at]
        
        if mask is not None:
            mask = mask[:, self.at]
        
        # 计算指标
        if self.is_masked(mask):
            val, numel = self._compute_masked(y_hat, y, mask)
        else:
            val, numel = self._compute_std(y_hat, y)
        
        if numel > 0:
            return val / numel
        return torch.tensor(0., device=val.device)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """获取状态字典"""
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """加载状态字典"""
        return super().load_state_dict(state_dict, strict)
    
    def extra_repr(self) -> str:
        """额外表示信息"""
        info = []
        info.append(f"metric_fn={self.metric_fn.func.__name__}")
        info.append(f"mask_nans={self.mask_nans}")
        info.append(f"mask_inf={self.mask_inf}")
        info.append(f"at={self.at}")
        return ', '.join(info)
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"MaskedMetric({self.extra_repr()})"
    
    def __repr__(self) -> str:
        """详细表示"""
        return (f"MaskedMetric(metric_fn={self.metric_fn}, "
                f"mask_nans={self.mask_nans}, mask_inf={self.mask_inf}, "
                f"at={self.at}, value={self.value.item():.4f}, "
                f"numel={self.numel.item()})")