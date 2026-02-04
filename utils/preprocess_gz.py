import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta


def create_h5_for_grin(data_dir='datasets/air_city', output_path='datasets/air_city/guangzhou/pm25.h5'):
    """
        创建GRIN兼容的HDF5文件
    """
    # 1. 加载站点信息
    locations_file = os.path.join(os.path.join(data_dir, 'guangzhou'), 'direction.csv')
    locations_df = pd.read_csv(locations_file, encoding='gb2312')
    locations_df = locations_df.iloc[:, 1:4]
    sensor_count = locations_df.shape[0]
    
    # 2. 加载PM2.5数据
    pm25_file = os.path.join(os.path.join(data_dir, 'guangzhou'), 'pm25_gz.csv')
    pm25_df = pd.read_csv(pm25_file, encoding='gb2312', header=None)
    
    # 3. 定位PM2.5数据列
    pm25_start_idx = -1
    for i, item in enumerate(pm25_df.iloc[0]):
        if "PM2.5" in str(item):
            pm25_start_idx = i
            break
    
    pm25_end_idx = pm25_start_idx + sensor_count
    pm25_data = pm25_df.iloc[2:, pm25_start_idx:pm25_end_idx]
    
    # 4. 数据清洗：转为数字，过滤无效值
    pm25_data = pm25_data.apply(pd.to_numeric, errors='coerce')
    pm25_data = pm25_data.where(pm25_data > 0, np.nan)
    
    # 5. 创建时间索引
    n_timesteps = pm25_data.shape[0]
    start_time = datetime(2024, 10, 1, 1, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_timesteps)]
    
    # 6. 提取站点名称
    site_names = []
    if '市发布名称' in locations_df.columns:
        site_names = locations_df['市发布名称'].tolist()
    
    # 7. 创建PM2.5 DataFrame
    pm25_df = pd.DataFrame(
        pm25_data.values,
        index=timestamps,
        columns=site_names
    )
    
    # 8. 准备站点信息（GRIN需要latitude和longitude列）
    if '纬度' in locations_df.columns and '经度' in locations_df.columns and '市发布名称' in locations_df.columns:
        # 重命名列以匹配GRIN要求
        locations_df = locations_df.rename(columns={'市发布名称': 'station_name', '纬度': 'latitude', '经度': 'longitude'})
    
    # 9. 保存为H5文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.HDFStore(output_path, 'w') as store:
        # 只保存GRIN必需的数据集
        store.put('pm25', pm25_df)
        store.put('stations', locations_df)
    
    print(f"H5文件已创建: {output_path}")
    print(f"数据维度: {pm25_df.shape}")


if __name__ == '__main__':
    create_h5_for_grin()
    pm25 = pd.read_hdf('datasets/air_city/guangzhou/pm25_gz.h5', key='pm25')
    stations = pd.read_hdf('datasets/air_city/guangzhou/pm25_gz.h5', key='stations')
    print(pm25)
    print(stations)
    # 统计每个站点的NaN值数量
    nan_counts = pm25.isnull().sum()
    # 查看NaN值比例
    nan_ratio = pm25.isnull().mean()
    # 添加序号信息
    station_order = {station: idx + 1 for idx, station in enumerate(pm25.columns)}
    # 创建一个包含站点、缺失数量和比例的DataFrame，并按缺失比例降序排序
    nan_stats = pd.DataFrame({
        'station': pm25.columns,
        'nan_count': nan_counts.values,
        'nan_ratio': nan_ratio.values
    }).sort_values('nan_ratio', ascending=False)

    excluded_sensors = []
    print("\n站点序号对照表(按缺失比例由大到小排序):")
    for i, (_, row) in enumerate(nan_stats.iterrows(), 1):
        station = row['station']
        nan_count = row['nan_count']
        nan_pct = row['nan_ratio'] * 100
        original_order = station_order[station]
        print(f"序号 {original_order:2d}: {station} (NaN数量: {nan_count}, 缺失比例: {nan_pct:.1f}%)")
        if nan_pct <= 0.2:
            excluded_sensors.append(original_order-1)

    print(excluded_sensors)