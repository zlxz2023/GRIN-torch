import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union

from .filler import Filler
from ..nn.models import MPGRUNet, GRINet, BiMPGRUNet


class GraphFiller(Filler):
    def __init__(self,
                 model_class,
                 model_kwargs: Dict,
                 optim_class,
                 optim_kwargs: Dict,
                 loss_fn,
                 scaled_target: bool = False,
                 whiten_prob: float = 0.05,
                 pred_loss_weight: float = 1.,
                 warm_up: int = 0,
                 metrics: Optional[Dict] = None,
                 scheduler_class=None,
                 scheduler_kwargs: Optional[Dict] = None):
        """
        初始化 GraphFiller
        
        :param model_class: 模型类
        :param model_kwargs: 模型参数
        :param optim_class: 优化器类
        :param optim_kwargs: 优化器参数
        :param loss_fn: 损失函数
        :param scaled_target: 是否缩放目标
        :param whiten_prob: 空白概率
        :param pred_loss_weight: 预测损失权重
        :param warm_up: 预热步数
        :param metrics: 指标字典
        :param scheduler_class: 调度器类
        :param scheduler_kwargs: 调度器参数
        """
        super(GraphFiller, self).__init__(
            model_class=model_class,
            model_kwargs=model_kwargs,
            optim_class=optim_class,
            optim_kwargs=optim_kwargs,
            loss_fn=loss_fn,
            scaled_target=scaled_target,
            whiten_prob=whiten_prob,
            metrics=metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs
        )
        
        self.tradeoff = pred_loss_weight
        
        # 根据模型类型设置 trimming
        if model_class is MPGRUNet:
            self.trimming = (warm_up, 0)
        elif model_class in [GRINet, BiMPGRUNet]:
            self.trimming = (warm_up, warm_up)
        else:
            self.trimming = (0, 0)
    
    def trim_seq(self, *seq):
        """
        修剪序列，去除 warm-up 部分
        
        :param seq: 序列列表
        :return: 修剪后的序列
        """
        trimmed_seq = []
        for s in seq:
            if s is None:
                trimmed_seq.append(None)
            else:
                start = self.trimming[0]
                end = s.size(1) - self.trimming[1]
                trimmed_seq.append(s[:, start:end])
        
        if len(trimmed_seq) == 1:
            return trimmed_seq[0]
        return trimmed_seq
    
    def training_step(self, batch, batch_idx: int = 0):
        """
        训练步骤
        
        :param batch: 批次数据
        :param batch_idx: 批次索引
        :return: 训练结果字典
        """
        # 解包批次
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        
        # 计算掩码
        mask = batch_data['mask'].clone().detach()
        batch_data['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_prob).bool()
        eval_mask = batch_data.pop('eval_mask', None)
        eval_mask = (mask | eval_mask) - batch_data['mask']  # 所有未观测数据
        
        y = batch_data.pop('y')
        
        # 计算预测和损失
        res = self.predict_batch(batch, preprocess=False, postprocess=False)
        
        # 处理多个输出
        if isinstance(res, (list, tuple)):
            imputation, predictions = res[0], res[1:]
        else:
            imputation, predictions = res, []
        
        # 修剪序列
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)
        predictions = self.trim_seq(*predictions)
        
        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            for i in range(len(predictions)):
                predictions[i] = self._postprocess(predictions[i], batch_preprocessing)
        
        # 计算损失
        loss = self.loss_fn(imputation, target, mask)
        for pred in predictions:
            loss += self.tradeoff * self.loss_fn(pred, target, mask)
        
        # 更新训练指标
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        
        for metric_name, metric in self.train_metrics.items():
            metric.update(imputation.detach(), y, eval_mask)
        
        self.global_step += 1
        
        return {
            'loss': loss,
            'imputation': imputation.detach(),
            'y': y,
            'eval_mask': eval_mask,
            'predictions': predictions
        }
    
    def validation_step(self, batch, batch_idx: int = 0):
        """
        验证步骤
        
        :param batch: 批次数据
        :param batch_idx: 批次索引
        :return: 验证结果字典
        """
        # 解包批次
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        
        # 提取掩码和目标
        mask = batch_data.get('mask')
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')
        
        # 计算预测和损失
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)
        
        # 修剪序列
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)
        
        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
        
        val_loss = self.loss_fn(imputation, target, eval_mask)
        
        # 更新验证指标
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        
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
        :return: 测试结果字典
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
    
    def predict_batch(self, batch, preprocess: bool = False, postprocess: bool = True, 
                     return_target: bool = False):
        """
        重写预测方法以处理多输出模型
        
        :param batch: 批次数据
        :param preprocess: 是否预处理
        :param postprocess: 是否后处理
        :param return_target: 是否返回目标
        :return: 预测结果
        """
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        
        if preprocess:
            x = batch_data.pop('x')
            x = self._preprocess(x, batch_preprocessing)
            output = self.forward(x, **batch_data)
        else:
            output = self.forward(**batch_data)
        
        # 处理多个输出
        if isinstance(output, (list, tuple)):
            if postprocess:
                output = [self._postprocess(o, batch_preprocessing) for o in output]
        else:
            if postprocess:
                output = self._postprocess(output, batch_preprocessing)
        
        if return_target:
            y = batch_data.get('y')
            mask = batch_data.get('mask', None)
            return y, output, mask
        
        return output