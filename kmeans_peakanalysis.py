#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @ Time:2025/11/3 12:38
# @ Author:86155
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @ Time:2025/11/3 10:21
# @ Author:86155
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import folium
import warnings

warnings.filterwarnings('ignore')


class EnhancedWeightedKMeans:
    def __init__(self, n_clusters=10, max_iters=300, random_state=42, alpha=1.5):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.alpha = alpha  # 权重放大指数

    def enhanced_manhattan_distance(self, X, centers):
        """Calculate enhanced weighted Manhattan distance with alpha exponent"""
        distances = np.zeros((X.shape[0], centers.shape[0]))
        for i in range(centers.shape[0]):
            # Manhattan distance: |lat1-lat2| + |lon1-lon2|
            lat_diff = np.abs(X[:, 0] - centers[i, 0])
            lon_diff = np.abs(X[:, 1] - centers[i, 1])
            # Apply enhanced weights: distance × (order_count^alpha)
            weighted_dist = (lat_diff + lon_diff) * (X[:, 2] ** self.alpha)
            distances[:, i] = weighted_dist
        return distances

    def fit(self, X):
        """Train enhanced weighted K-means model"""
        np.random.seed(self.random_state)

        # Initialize cluster centers
        n_samples = X.shape[0]
        centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False), :2]

        for iteration in range(self.max_iters):
            # Calculate enhanced weighted distances and assign clusters
            distances = self.enhanced_manhattan_distance(X, centroids)
            labels = np.argmin(distances, axis=1)

            # Update cluster centers using weighted average
            new_centroids = np.zeros((self.n_clusters, 2))
            for i in range(self.n_clusters):
                cluster_mask = (labels == i)
                if np.sum(cluster_mask) > 0:
                    cluster_points = X[cluster_mask]
                    # Use weighted average based on order counts
                    weights = cluster_points[:, 2]
                    total_weight = np.sum(weights)
                    if total_weight > 0:
                        new_centroids[i, 0] = np.sum(cluster_points[:, 0] * weights) / total_weight
                        new_centroids[i, 1] = np.sum(cluster_points[:, 1] * weights) / total_weight
                    else:
                        new_centroids[i] = np.mean(cluster_points[:, :2], axis=0)
                else:
                    # If cluster is empty, reinitialize with a random point
                    new_centroids[i] = X[np.random.randint(0, n_samples), :2]

            # Check for convergence
            if np.allclose(centroids, new_centroids, rtol=1e-6):
                print(f"Converged at iteration {iteration}")
                break

            centroids = new_centroids

        self.cluster_centers_ = centroids
        self.labels_ = labels
        return self


def load_and_preprocess_data(file_path):
    """Load and preprocess data"""
    # Read data
    df = pd.read_excel(file_path)

    print("Basic data information:")
    print(f"Data shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print("\nFirst 5 rows of data:")
    print(df.head())

    # Check for missing values
    print(f"\nMissing value statistics:")
    print(df.isnull().sum())

    # 确保时间是datetime格式
    df['start_time'] = pd.to_datetime(df['start_time'])

    # 提取小时信息用于高峰时段分析
    df['hour'] = df['start_time'].dt.hour

    return df


def create_grid_system_start_only(df, grid_size=0.001):
    """Create grid system and calculate order count for each grid - ONLY START POINTS"""
    # Use only start locations (departure points)
    start_points = df[['start_lat', 'start_lon']].rename(columns={'start_lat': 'lat', 'start_lon': 'lon'})

    print(f"Using only start points for clustering")
    print(f"Number of start points: {len(start_points)}")

    # Create grid
    min_lat, max_lat = start_points['lat'].min(), start_points['lat'].max()
    min_lon, max_lon = start_points['lon'].min(), start_points['lon'].max()

    print(f"\nCoordinate range (start points only):")
    print(f"Latitude: {min_lat:.6f} - {max_lat:.6f}")
    print(f"Longitude: {min_lon:.6f} - {max_lon:.6f}")

    # Assign grid ID to each start point
    start_points['grid_lat'] = ((start_points['lat'] - min_lat) / grid_size).astype(int)
    start_points['grid_lon'] = ((start_points['lon'] - min_lon) / grid_size).astype(int)

    # Calculate order count for each grid (only start points)
    grid_counts = start_points.groupby(['grid_lat', 'grid_lon']).size().reset_index(name='order_count')

    # Calculate grid center coordinates
    grid_counts['center_lat'] = min_lat + (grid_counts['grid_lat'] + 0.5) * grid_size
    grid_counts['center_lon'] = min_lon + (grid_counts['grid_lon'] + 0.5) * grid_size

    print(f"\nGrid system (start points only):")
    print(f"Grid size: {grid_size}")
    print(f"Number of grids: {len(grid_counts)}")
    print(f"Total start points: {len(start_points)}")

    return grid_counts, min_lat, max_lat, min_lon, max_lon


def create_grid_system_peak_hours(df, start_hour, end_hour, grid_size=0.001):
    """Create grid system for specific peak hours"""
    # Filter data for peak hours
    peak_data = df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)].copy()

    print(f"\nPeak hour analysis: {start_hour:02d}:00 - {end_hour:02d}:00")
    print(f"Number of orders in this period: {len(peak_data)}")

    # Use only start locations for peak hours
    start_points = peak_data[['start_lat', 'start_lon']].rename(columns={'start_lat': 'lat', 'start_lon': 'lon'})

    # Create grid
    min_lat, max_lat = start_points['lat'].min(), start_points['lat'].max()
    min_lon, max_lon = start_points['lon'].min(), start_points['lon'].max()

    # Assign grid ID to each start point
    start_points['grid_lat'] = ((start_points['lat'] - min_lat) / grid_size).astype(int)
    start_points['grid_lon'] = ((start_points['lon'] - min_lon) / grid_size).astype(int)

    # Calculate order count for each grid
    grid_counts = start_points.groupby(['grid_lat', 'grid_lon']).size().reset_index(name='order_count')

    # Calculate grid center coordinates
    grid_counts['center_lat'] = min_lat + (grid_counts['grid_lat'] + 0.5) * grid_size
    grid_counts['center_lon'] = min_lon + (grid_counts['grid_lon'] + 0.5) * grid_size

    print(f"Number of grids with orders: {len(grid_counts)}")
    print(f"Total orders in grids: {grid_counts['order_count'].sum()}")

    return grid_counts, min_lat, max_lat, min_lon, max_lon


def perform_enhanced_clustering(grid_counts, k=10, alpha=1.5):
    """Perform enhanced weighted K-means clustering with alpha exponent"""
    # Prepare clustering data [latitude, longitude, order count weight]
    X = grid_counts[['center_lat', 'center_lon', 'order_count']].values

    print(f"\nStarting enhanced weighted K-means clustering, k={k}, alpha={alpha}")

    # Use enhanced weighted K-means
    kmeans = EnhancedWeightedKMeans(n_clusters=k, alpha=alpha, random_state=42)
    kmeans.fit(X)

    # Assign cluster labels to each grid - 确保是整数类型
    grid_counts['cluster'] = kmeans.labels_.astype(int)

    # Calculate statistics for each cluster
    cluster_stats = grid_counts.groupby('cluster').agg({
        'order_count': ['count', 'sum', 'mean'],
        'center_lat': 'mean',
        'center_lon': 'mean'
    }).round(6)

    cluster_stats.columns = ['Grid Count', 'Total Orders', 'Average Orders', 'Center Latitude', 'Center Longitude']

    print("\nEnhanced clustering results statistics (start points only):")
    print(cluster_stats)

    # Calculate weighted center for comparison
    weighted_centers = []
    for cluster_id in range(k):
        cluster_data = grid_counts[grid_counts['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            weights = cluster_data['order_count'].values
            total_weight = np.sum(weights)
            weighted_lat = np.sum(cluster_data['center_lat'] * weights) / total_weight
            weighted_lon = np.sum(cluster_data['center_lon'] * weights) / total_weight
            weighted_centers.append((weighted_lat, weighted_lon))
        else:
            weighted_centers.append((np.nan, np.nan))

    print("\nWeighted cluster centers (for reference):")
    for i, (lat, lon) in enumerate(weighted_centers):
        print(f"Cluster {i}: ({lat:.6f}, {lon:.6f})")

    return kmeans, grid_counts, cluster_stats


def visualize_clusters_folium(grid_counts, kmeans, min_lat, max_lat, min_lon, max_lon,
                              grid_size=0.001, alpha=1.5, title_suffix=""):
    """使用Folium创建交互式聚类可视化地图 - 增强加权版本"""

    # 创建地图中心点
    map_center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]

    # 创建Folium地图，使用高德地图作为底图
    m = folium.Map(
        location=map_center,
        zoom_start=12,
        tiles='https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德地图'
    )

    # 定义聚类颜色方案
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]

    # 为每个聚类创建FeatureGroup
    feature_groups = {}
    for cluster_id in range(kmeans.n_clusters):
        feature_groups[cluster_id] = folium.FeatureGroup(name=f'Cluster {cluster_id}')
        m.add_child(feature_groups[cluster_id])

    # 添加聚类中心点
    centers_group = folium.FeatureGroup(name='Cluster Centers')
    for i, center in enumerate(kmeans.cluster_centers_):
        folium.Marker(
            location=[center[0], center[1]],
            popup=f'Cluster {i} Center<br>Lat: {center[0]:.6f}<br>Lon: {center[1]:.6f}<br>Alpha: {alpha}',
            icon=folium.Icon(color='red', icon='star', prefix='fa')
        ).add_to(centers_group)
    m.add_child(centers_group)

    # 绘制每个网格
    for _, row in grid_counts.iterrows():
        # 确保cluster_id是整数
        cluster_id = int(row['cluster'])
        color = colors[cluster_id % len(colors)]

        # 确保网格索引是整数
        grid_lat = int(row['grid_lat'])
        grid_lon = int(row['grid_lon'])

        # 计算网格边界
        lat_min = min_lat + grid_lat * grid_size
        lat_max = lat_min + grid_size
        lon_min = min_lon + grid_lon * grid_size
        lon_max = lon_min + grid_size

        # 创建网格多边形
        bounds = [
            [lat_min, lon_min],
            [lat_max, lon_min],
            [lat_max, lon_max],
            [lat_min, lon_max]
        ]

        # 创建GeoJSON特征
        geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min]
                ]]
            },
            "properties": {
                "cluster": cluster_id,
                "order_count": int(row['order_count']),  # 确保是整数
                "tooltip_text": f"Cluster: {cluster_id}<br>Start Orders: {int(row['order_count'])}<br>Grid: ({grid_lat}, {grid_lon})<br>Alpha: {alpha}"
            }
        }

        # 添加到地图
        folium.GeoJson(
            geojson,
            style_function=lambda feature, color=color: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.6
            },
            tooltip=folium.features.GeoJsonTooltip(
                fields=['tooltip_text'],
                aliases=[''],
                sticky=False,
                localize=True
            )
        ).add_to(feature_groups[cluster_id])

    # 添加图层控制
    folium.LayerControl().add_to(m)

    # 添加标题
    title_html = f'''
                 <h3 align="center" style="font-size:20px"><b>{title_suffix} - 加权K-means聚类分析 (α={alpha})</b></h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))

    # 保存地图
    filename = f'enhanced_kmeans_clustering_{title_suffix.replace(" ", "_")}_alpha_{alpha}.html'
    m.save(filename)
    print(f"\n交互式聚类地图已保存为: {filename} (alpha={alpha})")

    return m


def visualize_cluster_characteristics_folium(grid_counts, kmeans, alpha=1.5, title_suffix=""):
    """使用Folium创建聚类特征分析地图 - 增强加权版本"""

    # 计算每个聚类的统计数据
    cluster_stats = grid_counts.groupby('cluster').agg({
        'order_count': ['sum', 'mean', 'count']
    }).round(2)
    cluster_stats.columns = ['Total Start Orders', 'Avg Start Orders per Grid', 'Grid Count']

    # 创建地图
    map_center = [grid_counts['center_lat'].mean(), grid_counts['center_lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    # 为每个聚类创建圆形标记，大小表示订单数量
    for cluster_id in range(kmeans.n_clusters):
        # 确保cluster_id是整数
        cluster_id_int = int(cluster_id)

        # 检查聚类是否存在统计数据中
        if cluster_id_int not in cluster_stats.index:
            print(f"Warning: Cluster {cluster_id_int} not found in cluster_stats")
            continue

        cluster_data = grid_counts[grid_counts['cluster'] == cluster_id_int]
        center_lat = kmeans.cluster_centers_[cluster_id_int, 0]
        center_lon = kmeans.cluster_centers_[cluster_id_int, 1]

        total_orders = cluster_stats.loc[cluster_id_int, 'Total Start Orders']
        grid_count = cluster_stats.loc[cluster_id_int, 'Grid Count']
        avg_orders = cluster_stats.loc[cluster_id_int, 'Avg Start Orders per Grid']

        # 圆形半径基于订单数量的对数（避免过大差异）
        radius = np.log(total_orders + 1) * 100

        folium.CircleMarker(
            location=[center_lat, center_lon],
            radius=float(radius),  # 确保是浮点数
            popup=(
                f"<b>Cluster {cluster_id_int}</b><br>"
                f"Total Start Orders: {int(total_orders):,}<br>"
                f"Grid Count: {int(grid_count)}<br>"
                f"Avg Start Orders/Grid: {avg_orders:.1f}<br>"
                f"Center: ({center_lat:.6f}, {center_lon:.6f})<br>"
                f"Alpha: {alpha}"
            ),
            color='blue',
            fill=True,
            fillOpacity=0.6
        ).add_to(m)

    # 添加标题
    title_html = f'''
                 <h3 align="center" style="font-size:20px"><b>{title_suffix} - 聚类特征分析 (α={alpha})</b></h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))

    # 保存特征分析地图
    filename = f'enhanced_cluster_characteristics_{title_suffix.replace(" ", "_")}_alpha_{alpha}.html'
    m.save(filename)
    print(f"聚类特征分析地图已保存为: {filename} (alpha={alpha})")

    return m


def visualize_peak_hour_comparison(morning_clusters, noon_clusters, alpha=1.5):
    """创建早高峰和午高峰对比地图"""

    # 创建对比地图
    map_center = [morning_clusters['center_lat'].mean(), morning_clusters['center_lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    # 早高峰聚类 - 用蓝色表示
    morning_group = folium.FeatureGroup(name='早高峰 (8:00-10:00)')
    for _, row in morning_clusters.iterrows():
        folium.CircleMarker(
            location=[row['center_lat'], row['center_lon']],
            radius=np.log(row['order_count'] + 1) * 50,
            popup=f"早高峰<br>订单数: {row['order_count']}<br>聚类: {row['cluster']}",
            color='blue',
            fill=True,
            fillOpacity=0.6
        ).add_to(morning_group)

    # 午高峰聚类 - 用红色表示
    noon_group = folium.FeatureGroup(name='午高峰 (11:00-13:00)')
    for _, row in noon_clusters.iterrows():
        folium.CircleMarker(
            location=[row['center_lat'], row['center_lon']],
            radius=np.log(row['order_count'] + 1) * 50,
            popup=f"午高峰<br>订单数: {row['order_count']}<br>聚类: {row['cluster']}",
            color='red',
            fill=True,
            fillOpacity=0.6
        ).add_to(noon_group)

    m.add_child(morning_group)
    m.add_child(noon_group)

    # 添加图层控制
    folium.LayerControl().add_to(m)

    # 添加标题
    title_html = '''
                 <h3 align="center" style="font-size:20px"><b>早晚高峰聚类对比分析</b></h3>
                 <p align="center">蓝色: 早高峰 (8:00-10:00) | 红色: 午高峰 (11:00-13:00)</p>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))

    # 保存对比地图
    filename = f'peak_hours_comparison_alpha_{alpha}.html'
    m.save(filename)
    print(f"高峰时段对比地图已保存为: {filename}")

    return m


def analyze_cluster_characteristics(grid_counts, kmeans, alpha=1.5, period_name=""):
    """Analyze characteristics of each cluster - enhanced version"""
    print("\n" + "=" * 60)
    print(f"Enhanced Cluster Analysis - {period_name} - Alpha={alpha}")
    print("=" * 60)

    # 计算加权中心点
    weighted_centers = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_data = grid_counts[grid_counts['cluster'] == cluster_id]
        if len(cluster_data) > 0:
            weights = cluster_data['order_count'].values
            total_weight = np.sum(weights)
            weighted_lat = np.sum(cluster_data['center_lat'] * weights) / total_weight
            weighted_lon = np.sum(cluster_data['center_lon'] * weights) / total_weight
            weighted_centers.append((weighted_lat, weighted_lon))
        else:
            weighted_centers.append((np.nan, np.nan))

    for cluster_id in range(kmeans.n_clusters):
        cluster_data = grid_counts[grid_counts['cluster'] == cluster_id]

        print(f"\nCluster {cluster_id}:")
        print(f"  Number of grids: {len(cluster_data)}")
        print(f"  Total start orders: {cluster_data['order_count'].sum()}")
        print(f"  Average start orders per grid: {cluster_data['order_count'].mean():.2f}")
        print(f"  Algorithm center: ({kmeans.cluster_centers_[cluster_id, 0]:.6f}, "
              f"{kmeans.cluster_centers_[cluster_id, 1]:.6f})")
        print(f"  Weighted center: ({weighted_centers[cluster_id][0]:.6f}, "
              f"{weighted_centers[cluster_id][1]:.6f})")

        # Calculate cluster boundaries
        lat_min = cluster_data['center_lat'].min()
        lat_max = cluster_data['center_lat'].max()
        lon_min = cluster_data['center_lon'].min()
        lon_max = cluster_data['center_lon'].max()

        print(f"  Latitude range: {lat_min:.6f} - {lat_max:.6f}")
        print(f"  Longitude range: {lon_min:.6f} - {lon_max:.6f}")


def analyze_peak_hour_patterns(morning_results, noon_results):
    """分析早晚高峰模式差异"""
    print("\n" + "=" * 60)
    print("PEAK HOUR PATTERN COMPARISON")
    print("=" * 60)

    morning_grids, morning_kmeans, morning_stats = morning_results
    noon_grids, noon_kmeans, noon_stats = noon_results

    # 计算总体统计
    morning_total = morning_grids['order_count'].sum()
    noon_total = noon_grids['order_count'].sum()

    print(f"早高峰总订单数: {morning_total}")
    print(f"午高峰总订单数: {noon_total}")
    print(f"订单比例 (早/午): {morning_total / noon_total:.2f}:1")

    # 分析热点区域变化
    morning_top_clusters = morning_stats.nlargest(3, 'Total Orders')
    noon_top_clusters = noon_stats.nlargest(3, 'Total Orders')

    print(f"\n早高峰Top 3聚类:")
    for idx, row in morning_top_clusters.iterrows():
        print(f"  聚类 {idx}: {row['Total Orders']} 订单")

    print(f"\n午高峰Top 3聚类:")
    for idx, row in noon_top_clusters.iterrows():
        print(f"  聚类 {idx}: {row['Total Orders']} 订单")


def main():
    # Parameter settings
    FILE_PATH = "data_shaode.xlsx"  # Please replace with your file path
    K_CLUSTERS = 25
    GRID_SIZE = 0.001  # Grid size, can be adjusted based on campus size
    ALPHA = 2.5  # Weight exponent for enhanced clustering

    try:
        # 1. Load data
        print("Step 1: Loading data...")
        df = load_and_preprocess_data(FILE_PATH)

        # 2. Create grid system - ONLY START POINTS (全天数据)
        print("\nStep 2: Creating grid system (start points only)...")
        grid_counts, min_lat, max_lat, min_lon, max_lon = create_grid_system_start_only(df, GRID_SIZE)

        # 3. Perform enhanced clustering with alpha=2.5 (全天数据)
        print(f"\nStep 3: Performing enhanced weighted K-means clustering (alpha={ALPHA})...")
        kmeans, grid_counts, cluster_stats = perform_enhanced_clustering(grid_counts, K_CLUSTERS, ALPHA)

        # 4. 使用Folium创建交互式可视化 (全天数据)
        print("\nStep 4: Generating interactive visualizations with Folium...")
        folium_map = visualize_clusters_folium(
            grid_counts, kmeans, min_lat, max_lat, min_lon, max_lon,
            GRID_SIZE, ALPHA, "全天数据"
        )

        # 5. 创建聚类特征分析地图 (全天数据)
        characteristics_map = visualize_cluster_characteristics_folium(
            grid_counts, kmeans, ALPHA, "全天数据"
        )

        # 6. Detailed analysis (全天数据)
        analyze_cluster_characteristics(grid_counts, kmeans, ALPHA, "全天数据")

        # 7. 早高峰分析 (8:00-10:00)
        print("\n" + "=" * 50)
        print("MORNING PEAK ANALYSIS (8:00-10:00)")
        print("=" * 50)

        morning_grids, m_min_lat, m_max_lat, m_min_lon, m_max_lon = create_grid_system_peak_hours(
            df, 8, 10, GRID_SIZE
        )

        if len(morning_grids) > 0:
            morning_kmeans, morning_grids, morning_stats = perform_enhanced_clustering(
                morning_grids, K_CLUSTERS, ALPHA
            )

            # 早高峰可视化
            morning_map = visualize_clusters_folium(
                morning_grids, morning_kmeans, m_min_lat, m_max_lat, m_min_lon, m_max_lon,
                GRID_SIZE, ALPHA, "早高峰_8-10点"
            )

            morning_chars_map = visualize_cluster_characteristics_folium(
                morning_grids, morning_kmeans, ALPHA, "早高峰_8-10点"
            )

            analyze_cluster_characteristics(morning_grids, morning_kmeans, ALPHA, "早高峰")
        else:
            print("No data available for morning peak hours")
            morning_results = None

        # 8. 午高峰分析 (11:00-13:00)
        print("\n" + "=" * 50)
        print("NOON PEAK ANALYSIS (11:00-13:00)")
        print("=" * 50)

        noon_grids, n_min_lat, n_max_lat, n_min_lon, n_max_lon = create_grid_system_peak_hours(
            df, 11, 13, GRID_SIZE
        )

        if len(noon_grids) > 0:
            noon_kmeans, noon_grids, noon_stats = perform_enhanced_clustering(
                noon_grids, K_CLUSTERS, ALPHA
            )

            # 午高峰可视化
            noon_map = visualize_clusters_folium(
                noon_grids, noon_kmeans, n_min_lat, n_max_lat, n_min_lon, n_max_lon,
                GRID_SIZE, ALPHA, "午高峰_11-13点"
            )

            noon_chars_map = visualize_cluster_characteristics_folium(
                noon_grids, noon_kmeans, ALPHA, "午高峰_11-13点"
            )

            analyze_cluster_characteristics(noon_grids, noon_kmeans, ALPHA, "午高峰")
        else:
            print("No data available for noon peak hours")
            noon_results = None

        # 9. 创建高峰时段对比地图
        if morning_results is not None and noon_results is not None:
            comparison_map = visualize_peak_hour_comparison(morning_grids, noon_grids, ALPHA)
            analyze_peak_hour_patterns(
                (morning_grids, morning_kmeans, morning_stats),
                (noon_grids, noon_kmeans, noon_stats)
            )

        # 10. Save results
        output_file = f"enhanced_campus_clustering_alpha_{ALPHA}.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            grid_counts.to_excel(writer, sheet_name='全天聚类结果', index=False)
            cluster_stats.to_excel(writer, sheet_name='全天聚类统计')

            # Save cluster centers
            centers_df = pd.DataFrame({
                'cluster': range(K_CLUSTERS),
                'center_lat': kmeans.cluster_centers_[:, 0],
                'center_lon': kmeans.cluster_centers_[:, 1]
            })
            centers_df.to_excel(writer, sheet_name='聚类中心', index=False)

            # 保存高峰时段结果
            if morning_results is not None:
                morning_grids.to_excel(writer, sheet_name='早高峰聚类', index=False)
                morning_stats.to_excel(writer, sheet_name='早高峰统计', index=False)

            if noon_results is not None:
                noon_grids.to_excel(writer, sheet_name='午高峰聚类', index=False)
                noon_stats.to_excel(writer, sheet_name='午高峰统计', index=False)

        print(f"\nEnhanced results saved to: {output_file}")

        return kmeans, grid_counts, cluster_stats, folium_map

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# If running this script directly
if __name__ == "__main__":
    kmeans_model, grid_data, stats, folium_map = main()

    # Usage example
    if kmeans_model is not None:
        print("\nUsage example:")
        print("To predict cluster label for a new location, use the following code:")
        print("""
        # Assume a new location (latitude, longitude)
        new_point = np.array([[new_lat, new_lon]])
        # First need to find the grid and order count weight for this location
        # Then use enhanced weighted distance to calculate the nearest cluster
        """)