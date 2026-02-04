import os

import numpy as np
import pandas as pd

from lib import datasets_path, masked_sensors_gz, masked_sensors_bj
from .pd_dataset import PandasDataset
from ..utils.utils import compute_mean, geographical_distance, thresholded_gaussian_kernel


class MyCity(PandasDataset):
    SEED = 3210

    def __init__(self, impute_nans=False, freq='60T', data_dir='guangzhou'):
        self.random = np.random.default_rng(self.SEED)
        self.eval_mask = None
        self.data_dir = data_dir
        self.city_emb = None
        df, dist, mask = self.load(impute_nans=impute_nans)
        self.dist = dist
        super().__init__(dataframe=df, u=None, mask=mask, name=data_dir, freq=freq, aggr='nearest')

    def load_raw(self):
        path = os.path.join(os.path.join(datasets_path['air_city'], self.data_dir), 'pm25.h5')
        eval_mask = None
        df = pd.read_hdf(path, 'pm25')
        stations = pd.read_hdf(path, 'stations')
        return df, stations, eval_mask

    def load(self, impute_nans=True):
        # 加载元数据
        df, stations, eval_mask = self.load_raw()
        # 计算数据真实存在掩码
        mask = (~np.isnan(df.values)).astype('uint8')  # 1 if value is not nan else 0
        masked_sensors = masked_sensors_gz if self.data_dir == 'guangzhou' else masked_sensors_bj
        # 评估掩码
        if eval_mask is not None:
            eval_mask = eval_mask.values.astype('uint8')
        else:
            eval_mask = np.zeros_like(mask, dtype='uint8')
            eval_mask[:, masked_sensors] = np.where(mask[:, masked_sensors], 1, 0)
        # 保存评估掩码
        self.eval_mask = eval_mask
        # 填补Nan值
        if impute_nans:
            df = df.fillna(compute_mean(df))
        # compute distances from latitude and longitude degrees
        st_coord = stations.loc[:, ['latitude', 'longitude']]
        dist = geographical_distance(st_coord, to_rad=True).values
        # 站点表征
        city_emb_path = os.path.join(
            datasets_path['air_city'],
            self.data_dir,
            'node_semantic.csv'
        )
        city_emb_df = pd.read_csv(city_emb_path)
        # 明确排除非语义列
        non_semantic_cols = ['station_name', 'cid']
        city_emb_df = city_emb_df.drop(columns=non_semantic_cols, errors='ignore')
        city_emb = city_emb_df.values.astype('float32')

        self.city_emb = city_emb  # (N, D_city)
        return df, dist, mask

    def splitter(self, dataset, val_len=0.1, window=24, **kwargs):
        n = len(dataset)

        n_test = int(n * 0.2)
        n_val = int(n * val_len)

        test_start = n - n_test
        val_start = n - n_test - n_val

        # 向前收缩 window，防止上下文泄露
        train_end = val_start - window
        val_end = test_start - window

        train_idxs = np.arange(0, train_end)
        val_idxs = np.arange(val_start, val_end)
        test_idxs = np.arange(test_start, n)

        return train_idxs, val_idxs, test_idxs


    def get_similarity(self, thr=0.1, include_self=False, **kwargs):
        # 计算theta
        theta = np.std(self.dist)
        # 得到邻接矩阵
        adj = thresholded_gaussian_kernel(self.dist, theta=theta, threshold=thr)
        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        return adj

    @property
    def mask(self):
        return self._mask

    @property
    def training_mask(self):
        return self._mask & (1 - self.eval_mask)