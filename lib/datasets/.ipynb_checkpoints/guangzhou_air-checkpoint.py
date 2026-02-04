import os
import numpy as np
import pandas as pd
import h5py

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils.utils import infer_mask, compute_mean, geographical_distance, thresholded_gaussian_kernel

class GuangzhouAirQuality(PandasDataset):
    SEED = 3210

    def __init__(self, impute_nans=False, freq='60T', excluded_sensors=None, val_ratio=0.1, test_ratio=0.2):
        self.random = np.random.default_rng(self.SEED)
        # 验证集掩码
        self.eval_mask = None
        # 测试集掩码
        self.test_mask = None
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.test_station_indices = None
        self.train_station_indices = None
        df, dist, mask = self.load(impute_nans=impute_nans, excluded_sensors=excluded_sensors)
        self.dist = dist
        super().__init__(dataframe=df, u=None, mask=mask, name='guangzhou', freq=freq, aggr='nearest')

    def load_raw(self, small=False):
        path = os.path.join(datasets_path['guangzhou'], 'pm25_gz.h5')
        eval_mask = None
        df = pd.read_hdf(path, 'pm25')
        stations = pd.read_hdf(path, 'stations')
        return df, stations, eval_mask
    
    def load(self, impute_nans=True, excluded_sensors=None):
        # 加载数据和站点元数据
        df, stations, eval_mask = self.load_raw()
        
        # 保存原始mask用于eval_mask设置
        mask = (~np.isnan(df.values)).astype('uint8')
        
        # 划分训练、验证、测试
        if excluded_sensors is None:
            excluded_sensors = []

        stations_num = df.shape[1]  # 站点个数
        times_num = df.shape[0]  # 时间步
        stations_list = list(range(stations_num))  # 站点索引
        candidate_for_test = [idx for idx in stations_list if idx not in excluded_sensors]  # 测试站点备选
        self.random.shuffle(candidate_for_test)  # 随机打乱
        self.test_station_indices = sorted(candidate_for_test[:int(stations_num * self.test_ratio)])  # 取测试站点
        self.train_station_indices = sorted([idx for idx in stations_list if idx not in self.test_station_indices])  # 其他站点均为训练站点
        mask[:, self.test_station_indices] = 0  # 设置测试站点数据不可见
        print("*" * 30)
        print("数据集划分:")
        print(f"总站点数: {stations_num}")
        print(f"训练站点 ({len(self.train_station_indices)}): {self.train_station_indices}")
        print(f"测试站点 ({len(self.test_station_indices)}): {self.test_station_indices}")
        print("*" * 30)
        train_end = int(times_num * (1 - self.val_ratio))
        # 设置验证集掩码
        eval_mask = np.zeros_like(mask)
        eval_mask[train_end:, self.train_station_indices] = 1
        self.eval_mask = eval_mask
        # 设置测试集掩码
        test_mask = np.zeros_like(mask)
        test_mask[:, self.test_station_indices] = 1
        self.test_mask = test_mask

        # 用每周同时段的平均值替换NaN值
        if impute_nans:
            df = df.fillna(compute_mean(df))

        # 计算基于经纬度的距离
        st_coord = stations.loc[:, ['latitude', 'longitude']]
        dist = geographical_distance(st_coord, to_rad=True).values
        
        return df, dist, mask

    def splitter(self, dataset, in_sample=False, window=0):
        total_len = len(dataset)
        if in_sample == False: 
            val_len = int(total_len * self.val_ratio)
            train_len = total_len - val_len
            # 计算起始和结束位置，考虑window作为缓冲区 
            val_start = total_len - val_len
            train_end = max(0, val_start - window)
            train_idxs = np.arange(train_end)
            val_idxs = np.arange(val_start, total_len)
            test_idxs = np.arange(total_len)
            # 如果window > 0，仍需要检查重叠 
            if window > 0:
                # 获取重叠的索引位置 
                overlapping_mask, _ = dataset.overlapping_indices(train_idxs, val_idxs, synch_mode='horizon', as_mask=True) 
                # 只对train_idxs应用掩码，确保维度匹配
                if len(overlapping_mask) == len(train_idxs): 
                    train_idxs = train_idxs[~overlapping_mask] 
                else: 
                    # 如果掩码维度不匹配，跳过重叠处理
                    print(f"Warning: overlapping_mask dimension {len(overlapping_mask)} doesn't match train_idxs {len(train_idxs)}") 
        else:
            raise NotImplementedError("目前仅支持样本外预测, 请设置in_sample=False")
        return [train_idxs, val_idxs, test_idxs]

    def get_similarity(self, thr=0.1, include_self=False, force_symmetric=False, sparse=False, **kwargs):
        # 对于广州数据集，使用距离的标准差作为theta
        theta = np.std(self.dist)
        adj = thresholded_gaussian_kernel(self.dist, theta=theta, threshold=thr)
        
        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
            
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
            
        return adj

    @property
    def mask(self):
        return self._mask

    @property
    def training_mask(self):
        return self._mask if self.eval_mask is None else (self._mask & (1 - self.eval_mask))