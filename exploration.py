import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')

# ----------------------
# 5. 探索性分析与可视化
# ----------------------
# 从处理后的 Excel 文件中加载数据
file_path = "processed_bike_data.xlsx"

try:
    df = pd.read_excel(file_path)
    print("已成功载入处理后的数据。")
except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到。请先运行特徵工程部分以生成该文件。")
    exit()

# 确保时间列是 datetime 类型
df['start_time'] = pd.to_datetime(df['start_time'])

print("\n开始探索性分析与可视化...")

# 5.1 骑行高峰分析
hourly_orders = df.groupby(df['start_time'].dt.hour)['order_id'].count()
plt.figure(figsize=(10, 6))
hourly_orders.plot(kind='bar', color='skyblue')
plt.title('Daily Hourly Bike Trip Demand')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Trips')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('daily_hourly_bike_trip_demand.png', bbox_inches='tight')  # 保存图片
plt.show()

# 5.2 相关性分析 (关键优化: 对大型数据集进行采样)
print("正在对数据集进行采样，以进行相关性分析...")
# 随机抽取1000条数据进行分析，避免性能问题
sampled_df = df.sample(n=10000, random_state=42)

numerical_vars = [
    'start_time',
    'weekday',
    'duration_minutes',
    'trip_distance_km',
    'start_lat',
    'start_lon',
]

# 绘制相关性热力图
corr_matrix = sampled_df[numerical_vars].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=0.5, linecolor='black')
plt.title('Correlation Matrix of Numerical Variables (Sampled Data)')
plt.savefig('correlation_matrix.png', bbox_inches='tight')  # 保存图片
plt.show()

# 绘制 pairwise scatter plots
sns.pairplot(sampled_df[numerical_vars])
plt.suptitle('Pairwise Scatter Plots of Numerical Variables (Sampled Data)', y=1.02)
plt.savefig('pairwise_scatter_plots.png', bbox_inches='tight')  # 保存图片
plt.show()

# 5.3 地图可视化 (Folium)
print("正在生成热力图...")
start_locations = df[['start_lat', 'start_lon']].dropna().values.tolist()
if start_locations:
    m = folium.Map(
        location=[df['start_lat'].mean(), df['start_lon'].mean()],
        zoom_start=14,
        tiles='https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德地图'
    )
    HeatMap(start_locations).add_to(m)
    map_filename = 'bike_trip_heatmap_start.html'
    m.save(map_filename)
    print(f"热门起始地点热力图已保存到文件：{map_filename}")

end_locations = df[['end_lat', 'end_lon']].dropna().values.tolist()
if end_locations:
    m_end = folium.Map(
        location=[df['end_lat'].mean(), df['end_lon'].mean()],
        zoom_start=14,
        tiles='https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德地图'
    )
    HeatMap(end_locations).add_to(m_end)
    map_filename_end = 'bike_trip_heatmap_end.html'
    m_end.save(map_filename_end)
    print(f"热门结束地点热力图已保存到文件：{map_filename_end}")

print("\n分析完成。")