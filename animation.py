import pandas as pd
import numpy as np
import folium
from folium.plugins import TimestampedGeoJson, HeatMapWithTime, HeatMap
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

# ----------------------
# 1. 数据加载与预处理
# ----------------------
file_path = "processed_bike_data.xlsx"

try:
    df = pd.read_excel(file_path)
    logging.info(f"已成功载入数据，记录数：{len(df)}")
except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到。请先运行特征工程部分以生成该文件。")
    exit()
except Exception as e:
    print(f"错误: 加载数据失败: {e}")
    exit()

# 验证必需列
required_columns = ['bike_id', 'start_time', 'end_time', 'start_lat', 'start_lon', 'end_lat', 'end_lon', 'weekday']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"错误: 缺少以下必需列: {missing_columns}")
    exit()

# 验证数据完整性
if df[required_columns].isnull().any().any():
    print("警告: 数据中存在缺失值，已删除缺失值记录")
    df = df.dropna(subset=required_columns)

# 验证时间格式
if not pd.api.types.is_datetime64_any_dtype(df['start_time']) or not pd.api.types.is_datetime64_any_dtype(
        df['end_time']):
    print("错误: start_time 或 end_time 列不是 datetime 类型")
    exit()

# 验证坐标范围
invalid_coords = df[
    (df['start_lat'].abs() > 90) | (df['start_lon'].abs() > 180) |
    (df['end_lat'].abs() > 90) | (df['end_lon'].abs() > 180)
    ]
if not invalid_coords.empty:
    print("警告: 检测到无效坐标，已删除")
    df = df[~df.index.isin(invalid_coords.index)]

# 打印数据样本以检查格式
logging.info(f"数据样本（前5行）:\n{df[['start_time', 'start_lat', 'start_lon', 'weekday']].head().to_string()}")

# ----------------------
# 2. 生成3辆单车轨迹时间滑块图
# ----------------------
print("\n正在生成3辆单车轨迹时间滑块图...")

# 筛选特定日期的所有订单
date_to_analyze = '2024-04-09'
df_day = df[df['start_time'].dt.date == pd.to_datetime(date_to_analyze).date()]

if df_day.empty:
    print(f"在 {date_to_analyze} 这一天没有找到订单数据。")
else:
    # 找出这一天使用次数最多的3辆单车
    top_3_bikes = df_day['bike_id'].value_counts().head(3).index.tolist()
    if len(top_3_bikes) < 3:
        print(f"警告: 仅找到 {len(top_3_bikes)} 辆单车的数据")
    print(f"在 {date_to_analyze} 这一天使用次数最多的3辆单车ID是：{top_3_bikes}")

    # 预定义颜色列表
    colors = ['#FF0000', '#00FF00', '#0000FF']
    features = []

    # 为每辆热门单车创建轨迹动画数据
    for i, bike_id in enumerate(top_3_bikes):
        bike_trips = df_day[df_day['bike_id'] == bike_id].sort_values('start_time')
        color = colors[i % len(colors)]

        for _, trip in bike_trips.iterrows():
            start_point = [trip['start_lat'], trip['start_lon']]
            end_point = [trip['end_lat'], trip['end_lon']]

            # 添加骑行轨迹
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [
                        [trip['start_lon'], trip['start_lat']],
                        [trip['end_lon'], trip['end_lat']]
                    ]
                },
                'properties': {
                    'times': [
                        trip['start_time'].strftime('%Y-%m-%dT%H:%M:%S'),
                        trip['end_time'].strftime('%Y-%m-%dT%H:%M:%S')
                    ],
                    'style': {'color': color, 'weight': 4},
                    'popup': f"Bike ID: {bike_id}<br>Start: {trip['start_time'].strftime('%H:%M')}<br>End: {trip['end_time'].strftime('%H:%M')}"
                }
            })

            # 添加起点
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [trip['start_lon'], trip['start_lat']]
                },
                'properties': {
                    'time': trip['start_time'].strftime('%Y-%m-%dT%H:%M:%S'),
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': color,
                        'fillOpacity': 1,
                        'stroke': False,
                        'radius': 6
                    },
                    'popup': f"Bike ID: {bike_id}<br>Start: {trip['start_time'].strftime('%H:%M')}"
                }
            })

            # 添加终点
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [trip['end_lon'], trip['end_lat']]
                },
                'properties': {
                    'time': trip['end_time'].strftime('%Y-%m-%dT%H:%M:%S'),
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': color,
                        'fillOpacity': 1,
                        'stroke': False,
                        'radius': 6
                    },
                    'popup': f"Bike ID: {bike_id}<br>End: {trip['end_time'].strftime('%H:%M')}"
                }
            })

    # 创建地图
    map_center = [df_day['start_lat'].mean(), df_day['start_lon'].mean()]
    m_bike_animation = folium.Map(location=map_center, zoom_start=14, tiles='CartoDB positron')

    # 添加时间动画插件
    try:
        TimestampedGeoJson(
            {'type': 'FeatureCollection', 'features': features},
            period='PT1H',
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=10,
            transition_time=500
        ).add_to(m_bike_animation)
    except Exception as e:
        print(f"错误: 生成轨迹动画失败: {e}")
        exit()

    # 添加图例
    legend_html = '<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background: white; padding: 10px; border: 2px solid black;">'
    legend_html += f'<h4>Bike Trajectories on {date_to_analyze}</h4>'
    for i, bike_id in enumerate(top_3_bikes):
        legend_html += f'<p><span style="color:{colors[i % len(colors)]}">■</span> Bike ID: {bike_id}</p>'
    legend_html += '</div>'
    m_bike_animation.get_root().html.add_child(folium.Element(legend_html))

    # 添加标题和比例尺
    m_bike_animation.get_root().html.add_child(folium.Element(
        f'<h3 style="position: fixed; top: 10px; left: 50px; z-index: 1000;">Bike Trajectories on {date_to_analyze}</h3>'
    ))
    folium.plugins.MiniMap().add_to(m_bike_animation)

    # 保存为 HTML 文件
    animation_filename = f'bike_trajectory_animation_{date_to_analyze}.html'
    try:
        m_bike_animation.save(animation_filename)
        print(f"3辆单车轨迹动画地图已成功保存到文件：{animation_filename}")
    except Exception as e:
        print(f"错误: 保存轨迹动画地图失败: {e}")
        exit()

# ----------------------
# 3. 生成某一天（周一）订单起始时间滑块热力图
# ----------------------
print("\n正在生成某一天（周一）订单起始时间滑块热力图...")

# 筛选某一天（2024-04-08，确认是周一）的订单
monday_date = '2024-04-08'
df_monday = df[(df['weekday'] == 1) & (df['start_time'].dt.date == pd.to_datetime(monday_date).date())].copy()

if df_monday.empty:
    print(f"在 {monday_date} 这一天没有找到周一的订单数据，无法生成热力图动画。")
    # 尝试选择另一个周一日期
    alternative_monday = df[df['weekday'] == 1]['start_time'].dt.date.min()
    if alternative_monday:
        print(f"尝试使用最早的周一日期: {alternative_monday}")
        df_monday = df[(df['weekday'] == 1) & (df['start_time'].dt.date == alternative_monday)].copy()
        monday_date = alternative_monday.strftime('%Y-%m-%d')
    else:
        print("错误: 数据中没有周一数据，无法生成热力图")
        exit()

# 打印筛选后的数据样本（保留原始坐标精度）
logging.info(f"{monday_date} 数据包含 {len(df_monday)} 条记录")
logging.info(f"数据样本（前5行）:\n{df_monday[['start_time', 'start_lat', 'start_lon']].head().to_string()}")
logging.info(f"纬度范围: {df_monday['start_lat'].min()} - {df_monday['start_lat'].max()}")
logging.info(f"经度范围: {df_monday['start_lon'].min()} - {df_monday['start_lon'].max()}")

# 按1小时聚合数据（恢复24小时分割）
df_monday['hour'] = df_monday['start_time'].dt.floor('1h')
hours = sorted(df_monday['hour'].unique())
logging.info(f"{monday_date} 数据覆盖 {len(hours)} 个小时")

# 准备热力图数据，过滤低点数时段并采样
heat_data = []
time_index = []
max_points_per_hour = 500  # 每小时最多保留500个点
min_points_per_hour = 30  # 最小点数阈值
for hour in hours:
    df_hour = df_monday[df_monday['hour'] == hour]
    if len(df_hour) > max_points_per_hour:
        df_hour = df_hour.sample(n=max_points_per_hour, random_state=42)
        logging.info(f"小时 {hour.strftime('%H:%M:%S')} 采样到 {max_points_per_hour} 个点")
    else:
        logging.info(f"小时 {hour.strftime('%H:%M:%S')} 包含 {len(df_hour)} 个点")

    if len(df_hour) >= min_points_per_hour:
        points = [[row['start_lat'], row['start_lon'], 2] for _, row in df_hour.iterrows()]  # 权重设为2
        heat_data.append(points)
        time_index.append(hour.strftime('%Y-%m-%dT%H:%M:%S'))
        logging.info(f"小时 {hour.strftime('%H:%M:%S')} 坐标样本: {points[:2]}")
    else:
        logging.warning(f"小时 {hour.strftime('%H:%M:%S')} 点数 ({len(df_hour)}) 少于 {min_points_per_hour}，已跳过")

# 创建地图（动态热力图）
map_center = [df_monday['start_lat'].mean(), df_monday['start_lon'].mean()]
m_monday_animation = folium.Map(location=map_center, zoom_start=14, tiles='CartoDB positron')

if heat_data:
    # 添加动态热力图
    try:
        HeatMapWithTime(
            heat_data,
            index=time_index,
            radius=10,  # 减小半径
            blur=20,  # 保持模糊效果
            min_opacity=0.6,
            max_opacity=0.8,
            scale_radius=False,
            gradient={0.1: 'blue', 0.3: 'lime', 0.5: 'yellow', 0.7: 'orange', 1.0: 'red'},
            auto_play=False,
            max_speed=10
        ).add_to(m_monday_animation)
    except Exception as e:
        print(f"错误: 生成动态热力图动画失败: {e}")
    else:
        # 添加标题和比例尺
        m_monday_animation.get_root().html.add_child(folium.Element(
            f'<h3 style="position: fixed; top: 10px; left: 50px; z-index: 1000; background: white; padding: 5px;">Bike Start Heatmap on {monday_date}</h3>'
        ))
        folium.plugins.MiniMap().add_to(m_monday_animation)
        # 添加JavaScript日志
        m_monday_animation.get_root().html.add_child(folium.Element(
            f'<script>console.log("HeatMapWithTime initialized for {monday_date} with {len(heat_data)} time steps");</script>'
        ))
        # 保存动态热力图
        animation_filename = f'monday_start_heatmap_animation_{monday_date}.html'
        try:
            m_monday_animation.save(animation_filename)
            print(f"{monday_date} 订单起始热力图动画已成功保存到文件：{animation_filename}")
        except Exception as e:
            print(f"错误: 保存热力图动画失败: {e}")

# 始终生成静态热力图用于调试
m_monday_static = folium.Map(location=map_center, zoom_start=14, tiles='CartoDB positron')
points = [[row['start_lat'], row['start_lon'], 2] for _, row in df_monday.iterrows()]  # 权重设为2
if points:
    HeatMap(
        points,
        radius=10,
        blur=20,
        min_opacity=0.6,
        max_opacity=0.8,
        gradient={0.1: 'blue', 0.3: 'lime', 0.5: 'yellow', 0.7: 'orange', 1.0: 'red'}
    ).add_to(m_monday_static)
    m_monday_static.get_root().html.add_child(folium.Element(
        f'<h3 style="position: fixed; top: 10px; left: 50px; z-index: 1000; background: white; padding: 5px;">Static Bike Start Heatmap on {monday_date}</h3>'
    ))
    folium.plugins.MiniMap().add_to(m_monday_static)
    m_monday_static.get_root().html.add_child(folium.Element(
        f'<script>console.log("Static HeatMap initialized for {monday_date}");</script>'
    ))
    static_filename = f'static_bike_heatmap_{monday_date}.html'
    try:
        m_monday_static.save(static_filename)
        print(f"静态热力图已保存到文件：{static_filename}")
    except Exception as e:
        print(f"错误: 保存静态热力图失败: {e}")
else:
    print(f"错误: {monday_date} 静态热力图数据为空")

print("\n分析完成。")