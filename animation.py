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
    animation_filename = f'pictures/bike_trajectory_animation_{date_to_analyze}.html'
    try:
        m_bike_animation.save(animation_filename)
        print(f"3辆单车轨迹动画地图已成功保存到文件：{animation_filename}")
    except Exception as e:
        print(f"错误: 保存轨迹动画地图失败: {e}")
        exit()

