import pandas as pd
import numpy as np
from pygeohash import decode

# ----------------------
# 1. 数据加载与预处理
# ----------------------
# 使用你指定的本地文件路径和输出文件名
file_path = r"D:\THU\fourthfall\DataMotivatedDecisions\dataset\dataset_training_20240408_20240524.xlsx"

try:
    df = pd.read_excel(file_path)
    print("文件读取成功！")
except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到，请检查文件路径是否正确。")
    exit()

# 删除包含任何缺失值的整行
df.dropna(inplace=True)

# 将时间戳列转换为 datetime 类型
print("正在转换时间戳...")
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
print("时间戳转换完成。")

# ----------------------
# 2. 矢量化特征工程 (关键优化)
# ----------------------
print("正在进行地理信息转换和距离计算 (矢量化操作)...")

# 矢量化：使用列表推导式进行 geohash 转换
start_coords = [decode(g) for g in df['start_geohash']]
end_coords = [decode(g) for g in df['end_geohash']]

# 创建新的列
df['start_lat'] = [c[0] for c in start_coords]
df['start_lon'] = [c[1] for c in start_coords]
df['end_lat'] = [c[0] for c in end_coords]
df['end_lon'] = [c[1] for c in end_coords]

# 矢量化：使用 Haversine 公式计算骑行距离
R = 6371  # 地球半径（公里）

# 将经纬度从度转换为弧度
lat1_rad = np.radians(df['start_lat'])
lon1_rad = np.radians(df['start_lon'])
lat2_rad = np.radians(df['end_lat'])
lon2_rad = np.radians(df['end_lon'])

# Haversine 公式
dlon = lon2_rad - lon1_rad
dlat = lat2_rad - lat1_rad
a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
df['trip_distance_km'] = R * c
print("地理信息转换和距离计算完成。")

# ----------------------
# 3. 异常值处理与特征创建
# ----------------------
# 过滤掉明显的地理位置异常点
lat_min, lat_max = 39.4, 41.6
lon_min, lon_max = 115.7, 117.4

df = df[(df['start_lat'] > lat_min) & (df['start_lat'] < lat_max) &
        (df['start_lon'] > lon_min) & (df['start_lon'] < lon_max) &
        (df['end_lat'] > lat_min) & (df['end_lat'] < lat_max) &
        (df['end_lon'] > lon_min) & (df['end_lon'] < lon_max)]

# 计算骑行时长（以分钟为单位），并过滤异常值
df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
df = df[(df['duration_minutes'] > 1) & (df['duration_minutes'] < 120)]

# 计算速度（公里/小时），并过滤异常值
df['speed_kmh'] = df.apply(
    lambda row: (row['trip_distance_km'] / row['duration_minutes']) * 60 if row['duration_minutes'] > 0 else 0,
    axis=1
)
df = df[df['speed_kmh'] <= 30]

# ----------------------
# 4. 数据保存到新文件
# ----------------------
# 定义新的输出文件路径和需要导出的列名列表
selected_output_file_path = "processed_bike_data.xlsx"
columns_to_export = [
    'bike_id',
    'start_time',
    'end_time',
    'weekday',
    'trip_distance_km',
    'duration_minutes',
    'start_lat',
    'start_lon',
    'end_lat',
    'end_lon'
]

# 从原始df中选择需要的列，创建一个新的DataFrame
df_to_export = df[columns_to_export]

# 将选择的列保存为Excel文件
print(f"正在将选择的列保存到 '{selected_output_file_path}'...")
df_to_export.to_excel(selected_output_file_path, index=False)
print(f"数据已成功保存到 '{selected_output_file_path}'。")

print(f"\n新数据表共 {len(df_to_export)} 行。")

print("\n分析完成。")