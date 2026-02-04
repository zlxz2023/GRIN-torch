from .metric_base import MaskedMetric
from .ops import mape
from torch.nn import functional as F
import torch

from ... import epsilon


class MaskedMAE(MaskedMetric):
    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 at=None):
        super(MaskedMAE, self).__init__(metric_fn=F.l1_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        metric_kwargs={'reduction': 'none'},
                                        at=at)


class MaskedMAPE(MaskedMetric):
    def __init__(self,
                 mask_nans=False,
                 at=None):
        super(MaskedMAPE, self).__init__(metric_fn=mape,
                                         mask_nans=mask_nans,
                                         mask_inf=True,
                                         at=at)


class MaskedMSE(MaskedMetric):
    def __init__(self,
                 mask_nans=False,
                 at=None):
        super(MaskedMSE, self).__init__(metric_fn=F.mse_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=True,
                                        metric_kwargs={'reduction': 'none'},
                                        at=at)


class MaskedMRE(MaskedMetric):
    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 at=None):
        super(MaskedMRE, self).__init__(metric_fn=F.l1_loss,
                                        mask_nans=mask_nans,
                                        mask_inf=mask_inf,
                                        metric_kwargs={'reduction': 'none'},
                                        at=at)
        self.register_buffer('tot', torch.tensor(0., dtype=torch.float))

    def reset(self):
        """重置所有状态"""
        super().reset()  # 重置基类的 value 和 numel
        self.tot.zero_()

    def _compute_masked(self, y_hat, y, mask):
        self._check_same_shape(y_hat, y)
        # 计算绝对误差
        val = self.metric_fn(y_hat, y)
        # 获取掩码
        mask = self._check_mask(mask, val)
        # 应用掩码到误差
        val = torch.where(mask, val, torch.tensor(0., device=y.device, dtype=torch.float))
        # 应用掩码到真实值
        y_masked = torch.where(mask, y, torch.tensor(0., device=y.device, dtype=torch.float))
        return val.sum(), mask.sum(), y_masked.sum()

    def _compute_std(self, y_hat, y):
        self._check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        return val.sum(), val.numel(), y.sum()

    def compute(self):
        if self.tot > epsilon:
            return self.value / self.tot
        return self.value

    def update(self, y_hat, y, mask=None):
        y_hat = y_hat[:, self.at]
        y = y[:, self.at]
        if mask is not None:
            mask = mask[:, self.at]
        if self.is_masked(mask):
            val, numel, tot = self._compute_masked(y_hat, y, mask)
        else:
            val, numel, tot = self._compute_std(y_hat, y)
        self.value += val
        self.numel += numel
        self.tot += tot

    def forward(self, y_hat, y, mask=None):
        """一次性计算 MRE，不更新内部状态"""
        y_hat = y_hat[:, self.at]
        y = y[:, self.at]
        
        if mask is not None:
            mask = mask[:, self.at]
        
        if self.is_masked(mask):
            val, numel, tot = self._compute_masked(y_hat, y, mask)
        else:
            val, numel, tot = self._compute_std(y_hat, y)
        
        if tot > epsilon:
            return val / tot
        return torch.tensor(0., device=val.device)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """获取状态字典"""
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """加载状态字典"""
        return super().load_state_dict(state_dict, strict)
