import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
# ==============================================================================
try:
    df = pd.read_excel('processed_bike_data.xlsx')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'processed_bike_data.xlsx' not found. Please ensure the file is in the same directory.")
    exit()

df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])


# 2. 供需平衡网格图可视化函数 (已修改为非线性颜色分配)
# ==============================================================================
def visualize_supply_demand_balance_grid_all_days(data, lat_range, lon_range, hour_range, filename_suffix):
    """
    在均匀网格上可视化所有天中指定时间段内的供需平衡，使用非线性颜色分配。

    Args:
        data (pd.DataFrame): 包含单车订单数据的DataFrame。
        lat_range (tuple): 纬度范围 (min_lat, max_lat)。
        lon_range (tuple): 经度范围 (min_lon, max_lon)。
        hour_range (tuple): 一个元组，(开始小时, 结束小时)，用于筛选数据。
        filename_suffix (str): 输出的 HTML 文件名后缀。
    """
    print(f"\n正在分析所有天 {filename_suffix} 的供需平衡...")

    # 根据用户提供的精确时间段进行筛选
    if filename_suffix == 'morning':
        time_filter = ((data['start_time'].dt.hour == 7) & (data['start_time'].dt.minute >= 30)) | \
                      (data['start_time'].dt.hour == 8) | \
                      ((data['start_time'].dt.hour == 9) & (data['start_time'].dt.minute <= 30))
    elif filename_suffix == 'lunch':
        time_filter = ((data['start_time'].dt.hour == 11) & (data['start_time'].dt.minute >= 30)) | \
                      (data['start_time'].dt.hour == 12) | \
                      ((data['start_time'].dt.hour == 13) & (data['start_time'].dt.minute <= 0))
    elif filename_suffix == 'evening':
        time_filter = ((data['start_time'].dt.hour == 17) & (data['start_time'].dt.minute >= 0)) | \
                      ((data['start_time'].dt.hour == 18) & (data['start_time'].dt.minute <= 30))
    else:
        print("Invalid filename_suffix. Please use 'morning', 'lunch', or 'evening'.")
        return

    filtered_data = data[time_filter].copy()

    print(f"筛选后的数据量: {len(filtered_data)} 条记录")
    if filtered_data.empty:
        print(f"警告：在指定的日期和时间段内没有找到订单数据。")
        return

    # 定义网格参数
    num_lat_cells = 30
    num_lon_cells = 20
    lat_bins = np.linspace(lat_range[0], lat_range[1], num_lat_cells + 1)
    lon_bins = np.linspace(lon_range[0], lon_range[1], num_lon_cells + 1)

    # 分配到网格单元
    filtered_data['start_lat_bin'] = pd.cut(filtered_data['start_lat'], bins=lat_bins, labels=False,
                                            include_lowest=True)
    filtered_data['start_lon_bin'] = pd.cut(filtered_data['start_lon'], bins=lon_bins, labels=False,
                                            include_lowest=True)
    filtered_data['end_lat_bin'] = pd.cut(filtered_data['end_lat'], bins=lat_bins, labels=False, include_lowest=True)
    filtered_data['end_lon_bin'] = pd.cut(filtered_data['end_lon'], bins=lon_bins, labels=False, include_lowest=True)

    filtered_data['start_cell'] = filtered_data['start_lat_bin'].astype(str) + '_' + filtered_data[
        'start_lon_bin'].astype(str)
    filtered_data['end_cell'] = filtered_data['end_lat_bin'].astype(str) + '_' + filtered_data['end_lon_bin'].astype(
        str)

    # 计算供需平衡
    demand = filtered_data.groupby('start_cell').size().rename('demand')
    supply = filtered_data.groupby('end_cell').size().rename('supply')
    all_cells = pd.Index(demand.index.union(supply.index))
    balance_df = pd.DataFrame(index=all_cells)
    balance_df['demand'] = demand
    balance_df['supply'] = supply
    balance_df = balance_df.fillna(0)
    balance_df['balance'] = balance_df['supply'] - balance_df['demand']

    # --- 核心修改部分：对供需平衡值进行非线性变换 ---
    # 使用np.log1p(abs(x))对数值进行压缩，并保留其正负号
    balance_df['transformed_balance'] = np.sign(balance_df['balance']) * np.log1p(balance_df['balance'].abs())

    # 诊断性输出
    print("\n原始供需平衡值（部分）：\n", balance_df['balance'].head())
    print("\n变换后的供需平衡值（部分）：\n", balance_df['transformed_balance'].head())

    # 创建地图
    map_center = [(lat_range[0] + lat_range[1]) / 2, (lon_range[0] + lon_range[1]) / 2]
    m = folium.Map(location=map_center, zoom_start=14,
                   tiles='https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
                   attr='高德地图')

    # 获取变换后的最大绝对平衡值，用于颜色映射
    max_abs_transformed_balance = balance_df['transformed_balance'].abs().max()

    # --- 核心修改部分：使用更深的颜色和变换后的值进行映射 ---
    if max_abs_transformed_balance > 0:
        colormap = folium.LinearColormap(['#B22222', 'white', '#1E90FF'],
                                         vmin=-max_abs_transformed_balance, vmax=max_abs_transformed_balance)
        colormap.caption = 'Supply-Demand Balance (Supply - Demand, Log-transformed)'
        m.add_child(colormap)
    else:
        colormap = None

    geojson_feature_group = folium.FeatureGroup(name='Supply-Demand Grid').add_to(m)

    # 在地图上绘制每个网格单元
    for i in range(num_lat_cells):
        for j in range(num_lon_cells):
            cell_id = f'{i}.0_{j}.0'
            cell_balance = balance_df.loc[cell_id, 'balance'] if cell_id in balance_df.index else 0
            # 使用变换后的值进行颜色映射
            cell_transformed_balance = balance_df.loc[
                cell_id, 'transformed_balance'] if cell_id in balance_df.index else 0

            bounds = [(lat_bins[i], lon_bins[j]), (lat_bins[i + 1], lon_bins[j + 1])]
            geojson = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [bounds[0][1], bounds[0][0]],
                        [bounds[1][1], bounds[0][0]],
                        [bounds[1][1], bounds[1][0]],
                        [bounds[0][1], bounds[1][0]],
                        [bounds[0][1], bounds[0][0]]
                    ]]
                },
                "properties": {
                    "balance": cell_transformed_balance,  # 将变换后的值放入properties
                    "tooltip_text": f"Original Balance: {cell_balance:.0f}<br>Transformed Balance: {cell_transformed_balance:.2f}"
                }
            }

            folium.GeoJson(
                geojson,
                style_function=lambda feature: {
                    'fillColor': colormap(feature['properties']['balance']) if colormap and abs(
                        feature['properties']['balance']) > 0 else 'white',
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.7 if abs(feature['properties']['balance']) > 0 else 0.1
                },
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['tooltip_text'],
                    aliases=[''],
                    sticky=False,
                    localize=True
                )
            ).add_to(geojson_feature_group)

    filename = f'pictures/supply_demand_balance_log_scaled_{filename_suffix}_peak.html'
    m.save(filename)
    print(f"供需平衡图已保存为 '{filename}'.")


# 3. 主程序执行
# ==============================================================================
# 定义分析区域的经纬度范围
lat_range = (39.995, 40.0135)
lon_range = (116.318, 116.338)

# 分别为三个高峰时段调用新函数
visualize_supply_demand_balance_grid_all_days(df, lat_range, lon_range, (7, 10), 'morning')
visualize_supply_demand_balance_grid_all_days(df, lat_range, lon_range, (11, 14), 'lunch')
visualize_supply_demand_balance_grid_all_days(df, lat_range, lon_range, (17, 19), 'evening')