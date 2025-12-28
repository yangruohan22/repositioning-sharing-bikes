import math
import pandas as pd
import matplotlib.pyplot as plt
import math
import json
import os
import csv
from datetime import datetime, time, timedelta

# --- Matplotlib 配置 (保持不变) ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# --- 全局时间常量 ---
START_TIME_HOUR = 11
END_TIME_HOUR = 15.5
TIME_FREQ_MINUTES = 5  # 聚合时间间隔（分钟）


# --- Grid 类 (保持不变) ---
class Grid:
    """表示单个网格的类"""

    def __init__(self, min_lat, max_lat, min_lon, max_lon, row, col, is_outside=False):
        self.min_lat = min_lat  # 网格最小纬度
        self.max_lat = max_lat  # 网格最大纬度
        self.min_lon = min_lon  # 网格最小经度
        self.max_lon = max_lon  # 网格最大经度
        self.row = row  # 网格行索引
        self.col = col  # 网格列索引
        self.is_outside = is_outside  # 是否为校外网格

    def get_center(self):
        """获取网格中心点的经纬度"""
        if self.is_outside:
            return (0.0, 0.0)  # 校外网格返回默认中心点
        center_lat = (self.min_lat + self.max_lat) / 2
        center_lon = (self.min_lon + self.max_lon) / 2
        return (center_lat, center_lon)

    def contains_point(self, lat, lon):
        """检查一个点是否在网格内"""
        return (self.min_lat <= lat <= self.max_lat and
                self.min_lon <= lon <= self.max_lon)

    def get_id(self):
        """获取网格ID"""
        return f"{self.row}_{self.col}"

    def __str__(self):
        """网格的字符串表示"""
        return (f"网格({self.row}, {self.col}): "
                f"纬度[{self.min_lat:.8f}, {self.max_lat:.8f}], "
                f"经度[{self.min_lon:.8f}, {self.max_lon:.8f}]")


# --- GridQHPark 类 (主要修改集中于 process_trips_by_cluster 和 visualize_cluster_flow) ---
class GridQHPark:
    def __init__(self, lat_csv_path, lon_csv_path):
        # 读取经纬度边界文件
        self.lat_boundary = pd.read_csv(lat_csv_path)
        self.lon_boundary = pd.read_csv(lon_csv_path)

        # 计算清华园的整体边界范围
        self.min_lat = self.lat_boundary['Latitude'].min()
        self.max_lat = self.lat_boundary['Latitude'].max()
        self.min_lon = self.lon_boundary['Longitude'].min()
        self.max_lon = self.lon_boundary['Longitude'].max()

        # 初始化网格参数
        self.grid_size = None  # 网格大小(米)
        self.lat_per_meter = None  # 每米对应的纬度变化
        self.lon_per_meter = None  # 每米对应的经度变化
        self.grid_rows = None  # 网格行数
        self.grid_cols = None  # 网格列数
        self.grids = None  # 网格数据结构
        # 用于存储网格到聚集点ID的映射
        self.grid_to_cluster = {}
        # 用于存储原始聚集点ID到新顺序ID的映射
        self.cluster_id_map = {}

        # 创建校外特殊网格
        self.outside_grid = Grid(
            min_lat=39.994,
            max_lat=39.9945,
            min_lon=116.322,  # 全球最小经度
            max_lon=116.3225,  # 全球最大经度
            row=-1,  # 特殊行索引
            col=-1,  # 特殊列索引
            is_outside=True  # 标记为校外网格
        )

        # 定义时间范围用于筛选
        self.filter_start_time = time(START_TIME_HOUR, 0, 0)  # 11:00:00
        hour = int(END_TIME_HOUR)  # 取整数部分：15
        minute = int((END_TIME_HOUR - hour) * 60)  # 取小数部分乘以60：0.5 * 60 = 30

        self.filter_end_time = time(hour, minute, 0)

    def _calculate_conversion_factors(self):
        # 计算经纬度与米之间的转换因子
        # 在北纬40度附近，1度纬度约等于111.194公里
        self.lat_per_meter = 1 / (111194.0)  # 每米对应的纬度变化

        # 在北纬40度附近，1度经度约等于85.228公里
        self.lon_per_meter = 1 / (85228.0)  # 每米对应的经度变化

    def _is_grid_inside_qhpark(self, grid_min_lat, grid_max_lat, grid_min_lon, grid_max_lon):
        """根据cell_boundary_lat.csv判断网格是否在清华园校内"""
        # 检查网格中心点是否在清华园内
        grid_center_lat = (grid_min_lat + grid_max_lat) / 2

        # 找到与网格中心点纬度最接近的记录
        closest_lat_idx = (self.lat_boundary['Latitude'] - grid_center_lat).abs().idxmin()
        closest_lat_row = self.lat_boundary.iloc[closest_lat_idx]

        # 检查网格中心点的经度是否在对应的经度范围内
        if grid_max_lon < closest_lat_row['Min_longitude'] or grid_min_lon > closest_lat_row['Max_longitude']:
            return False

        return True

    def create_grid(self, grid_size):
        """创建n*n米的方格覆盖清华园街道"""
        self.grid_size = grid_size
        self._calculate_conversion_factors()

        # 计算网格数量
        lat_diff = self.max_lat - self.min_lat
        lon_diff = self.max_lon - self.min_lon

        self.grid_rows = math.ceil(lat_diff / (self.lat_per_meter * grid_size))
        self.grid_cols = math.ceil(lon_diff / (self.lon_per_meter * grid_size))

        # 初始化网格数据结构
        # 使用字典存储每个网格的信息
        self.grids = {}

        # 统计校内和校外网格数量
        inside_count = 0
        outside_count = 0

        # 创建网格并存储边界信息
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # 计算网格的经纬度边界
                grid_min_lat = self.min_lat + row * self.lat_per_meter * grid_size
                grid_max_lat = min(grid_min_lat + self.lat_per_meter * grid_size, self.max_lat)
                grid_min_lon = self.min_lon + col * self.lon_per_meter * grid_size
                grid_max_lon = min(grid_min_lon + self.lon_per_meter * grid_size, self.max_lon)

                # 判断网格是否在清华园内
                is_outside = not self._is_grid_inside_qhpark(grid_min_lat, grid_max_lat, grid_min_lon, grid_max_lon)

                if is_outside:
                    outside_count += 1
                else:
                    inside_count += 1

                # 创建Grid对象并存储
                grid = Grid(
                    min_lat=grid_min_lat,
                    max_lat=grid_max_lat,
                    min_lon=grid_min_lon,
                    max_lon=grid_max_lon,
                    row=row,
                    col=col,
                    is_outside=is_outside
                )
                self.grids[grid.get_id()] = grid

        # 将校外特殊网格添加到grids字典中
        self.grids[self.outside_grid.get_id()] = self.outside_grid
        outside_count += 1

        print(f"创建了{self.grid_rows}行{self.grid_cols}列的网格，共{len(self.grids)}个网格")
        print(f"其中校内网格数量：{inside_count}，校外网格数量：{outside_count}")
        print(f"网格覆盖范围：纬度{self.min_lat:.6f}-{self.max_lat:.6f}，经度{self.min_lon:.6f}-{self.max_lon:.6f}")

    def get_grid_by_latlon(self, lat, lon):
        """根据经纬度确定在哪个方格里"""
        if self.grids is None:
            return self.outside_grid

        # 检查经纬度是否在清华园大范围或边界上
        if not (self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon):
            return self.outside_grid

        # 计算所在网格的行和列
        row = math.floor((lat - self.min_lat) / (self.lat_per_meter * self.grid_size))
        col = math.floor((lon - self.min_lon) / (self.lon_per_meter * self.grid_size))

        # 确保行和列在有效范围内
        row = max(0, min(row, self.grid_rows - 1))
        col = max(0, min(col, self.grid_cols - 1))

        grid_id = f"{row}_{col}"
        return self.grids.get(grid_id) if grid_id in self.grids else self.outside_grid

    # ... (get_grid_neighbors 和 get_total_grids 方法保持不变)

    def load_cluster_data(self, cluster_json_path):
        """加载聚类数据并设置聚类ID映射"""
        try:
            with open(cluster_json_path, 'r', encoding='utf-8') as f:
                cluster_data = json.load(f)
            self.grid_to_cluster = cluster_data.get('grid_to_cluster', {})

            # 1. 排除 cluster_29
            # 修改后的代码
            raw_ids = [
                cid for cid in set(self.grid_to_cluster.values()) if cid != 'cluster_29'
            ]

            # 使用 lambda 函数提取 '_' 后面的数字并转为 int 进行排序
            valid_cluster_ids = sorted(raw_ids, key=lambda x: int(x.split('_')[1]))

            # 2. 重新命名聚类ID（1到15）
            self.cluster_id_map = {
                old_id: f"Cluster {i + 1}" for i, old_id in enumerate(valid_cluster_ids)
            }
            print(f"已加载 {len(valid_cluster_ids)} 个有效聚集点并重新命名。")

            return True
        except FileNotFoundError:
            print(f"错误：聚类文件未找到：{cluster_json_path}")
            return False
        except json.JSONDecodeError:
            print(f"错误：聚类文件格式错误：{cluster_json_path}")
            return False

    def process_trips_by_cluster(self, trip_data_path, output_dir="clustered_data_11_16"):
        """
        根据聚类结果和11:00-16:00时间段筛选订单，并输出为CSV文件。
        假定 load_cluster_data 已调用。
        """
        if not self.cluster_id_map:
            print("错误：未加载或初始化聚类数据。")
            return

        # 1. 读取订单数据
        try:
            trip_df = pd.read_excel(trip_data_path)

            required_cols = ['start_lat', 'start_lon', 'end_lat', 'end_lon', 'start_time', 'end_time']
            if not all(col in trip_df.columns for col in required_cols):
                print("错误：订单数据中缺少必要的经纬度或时间列。")
                return

            # 确保时间列是 datetime 对象
            trip_df['start_time'] = pd.to_datetime(trip_df['start_time'])
            trip_df['end_time'] = pd.to_datetime(trip_df['end_time'])

            # 提取时间部分用于筛选
            trip_df['start_time_obj'] = trip_df['start_time'].dt.time
            trip_df['end_time_obj'] = trip_df['end_time'].dt.time

            csv_header = list(trip_df.drop(columns=['start_time_obj', 'end_time_obj']).columns)

        except FileNotFoundError:
            print(f"错误：订单文件未找到：{trip_data_path}")
            return
        except Exception as e:
            print(f"错误：读取订单文件失败: {e}")
            return

        # 初始化订单收集字典，使用新的 Cluster ID
        new_cluster_ids = list(self.cluster_id_map.values())
        cluster_start_trips = {cid: [] for cid in new_cluster_ids}
        cluster_end_trips = {cid: [] for cid in new_cluster_ids}

        # 2. 遍历订单并分配到相应的聚集点 (增加时间筛选)
        print(f"开始处理 {len(trip_df)} 条订单数据，并筛选 11:00-16:00 时间段...")

        for index, row in trip_df.iterrows():
            try:
                start_lat, start_lon = float(row['start_lat']), float(row['start_lon'])
                end_lat, end_lon = float(row['end_lat']), float(row['end_lon'])

                start_time_obj = row['start_time_obj']
                end_time_obj = row['end_time_obj']

            except (ValueError, TypeError):
                continue

            start_grid_id = self.get_grid_by_latlon(start_lat, start_lon).get_id()
            end_grid_id = self.get_grid_by_latlon(end_lat, end_lon).get_id()

            # 将网格ID映射到新的Cluster ID
            old_start_cid = self.grid_to_cluster.get(start_grid_id)
            new_start_cid = self.cluster_id_map.get(old_start_cid)

            old_end_cid = self.grid_to_cluster.get(end_grid_id)
            new_end_cid = self.cluster_id_map.get(old_end_cid)

            # 原始订单数据（不包含临时的时间对象列）
            trip_data = row.drop(['start_time_obj', 'end_time_obj']).to_list()

            # --- 筛选逻辑 ---

            # i. 筛选起点订单
            if new_start_cid:
                # 检查订单的开始时间是否在 [11:00:00, 16:00:00) 范围内
                if self.filter_start_time <= start_time_obj < self.filter_end_time:
                    cluster_start_trips[new_start_cid].append(trip_data)

            # ii. 筛选终点订单
            if new_end_cid:
                # 检查订单的结束时间是否在 [11:00:00, 16:00:00) 范围内
                if self.filter_start_time <= end_time_obj < self.filter_end_time:
                    cluster_end_trips[new_end_cid].append(trip_data)

        print("订单处理和时间筛选完成。")

        # 3. 输出结果到指定文件夹

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_count = 0

        for new_cid in new_cluster_ids:
            # i. 输出起点订单
            start_trips = cluster_start_trips[new_cid]
            if start_trips:
                start_filename = os.path.join(output_dir, f"{new_cid.replace(' ', '_')}_start_trips.csv")
                with open(start_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_header)
                    writer.writerows(start_trips)
                output_count += 1

            # ii. 输出终点订单
            end_trips = cluster_end_trips[new_cid]
            if end_trips:
                end_filename = os.path.join(output_dir, f"{new_cid.replace(' ', '_')}_end_trips.csv")
                with open(end_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_header)
                    writer.writerows(end_trips)
                output_count += 1

        print(f"总共输出了 {output_count} 个 CSV 文件到 {output_dir} 文件夹。")

    def visualize_cluster_flow(self, trip_data_path, output_dir="cluster_picture_11_16"):
        """
        可视化每个聚集点到达和离开的流量折线图 (11:00-16:00, 跨日期合并)。
        假定 load_cluster_data 已调用。
        """
        if not self.cluster_id_map:
            print("错误：未加载或初始化聚类数据。")
            return

        print("\n---------- 开始生成流量折线图 (11:00-16:00 合并) ----------")

        # 1. 读取订单数据
        try:
            trip_df = pd.read_excel(trip_data_path)
            trip_df['start_time'] = pd.to_datetime(trip_df['start_time'])
            trip_df['end_time'] = pd.to_datetime(trip_df['end_time'])

        except Exception as e:
            print(f"错误：读取或处理订单时间数据失败: {e}")
            return

        # 2. 创建输出文件夹
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出文件夹: {output_dir}")

        # 3. 预处理数据 (网格映射和时间筛选)

        # 增加网格和Cluster ID
        trip_df['start_grid_id'] = trip_df.apply(
            lambda row: self.get_grid_by_latlon(row['start_lat'], row['start_lon']).get_id(), axis=1
        )
        trip_df['end_grid_id'] = trip_df.apply(
            lambda row: self.get_grid_by_latlon(row['end_lat'], row['end_lon']).get_id(), axis=1
        )

        trip_df['start_cluster'] = trip_df['start_grid_id'].map(self.grid_to_cluster).map(self.cluster_id_map)
        trip_df['end_cluster'] = trip_df['end_grid_id'].map(self.grid_to_cluster).map(self.cluster_id_map)

        # 移除不在有效聚集点内的订单
        valid_df = trip_df.dropna(subset=['start_cluster', 'end_cluster'], how='all').copy()

        # 4. 定义时间筛选和分组函数

        # 将时间转换为从 11:00 开始的分钟数，并按 5 分钟间隔取整
        def time_to_bin(dt, start_hour=START_TIME_HOUR, freq_min=TIME_FREQ_MINUTES):
            # 11:00:00 转换为 time 对象
            base_time = time(start_hour, 0)

            # 检查时间是否在 [11:00:00, 16:00:00) 范围内
            if not (self.filter_start_time <= dt.time() < self.filter_end_time):
                return -1  # 标记为无效时间

            # 计算时间点相对于 11:00 的总分钟数
            total_minutes = (dt.hour - start_hour) * 60 + dt.minute

            # 按 5 分钟间隔取整
            time_bin = (total_minutes // freq_min) * freq_min
            return time_bin

        # 5. 跨日期统计流量

        # 应用时间分箱函数
        valid_df['start_time_bin'] = valid_df['start_time'].apply(time_to_bin)
        valid_df['end_time_bin'] = valid_df['end_time'].apply(time_to_bin)

        # 移除无效时间段的订单
        start_flow_df = valid_df[valid_df['start_time_bin'] != -1]
        end_flow_df = valid_df[valid_df['end_time_bin'] != -1]

        # 创建时间轴 (0, 5, 10, ..., 295 分钟)
        # 11:00 到 16:00 共有 5 小时 = 300 分钟。最后一个间隔是 15:55-16:00
        time_bins = list(range(0, (END_TIME_HOUR - START_TIME_HOUR) * 60, TIME_FREQ_MINUTES))
        time_index = pd.Index(time_bins, name='time_bin')

        # 6. 按聚集点生成图表
        for new_cid in self.cluster_id_map.values():
            # 统计离开流量 (Start Flow) - 按 time_bin 和 cluster 分组
            start_counts = start_flow_df[start_flow_df['start_cluster'] == new_cid].groupby(
                'start_time_bin'
            )['bike_id'].count().reindex(time_index, fill_value=0).rename('离开 (Start)')

            # 统计到达流量 (End Flow) - 按 time_bin 和 cluster 分组
            end_counts = end_flow_df[end_flow_df['end_cluster'] == new_cid].groupby(
                'end_time_bin'
            )['bike_id'].count().reindex(time_index, fill_value=0).rename('到达 (End)')

            # 将分钟数转换为时间标签用于绘图
            def format_min_to_time(minute_val):
                dt_obj = datetime(1, 1, 1, START_TIME_HOUR, 0) + timedelta(minutes=minute_val)
                return dt_obj.strftime('%H:%M')

            # 绘图
            plt.figure(figsize=(12, 6))

            # x轴是分钟数
            x_labels = [format_min_to_time(m) for m in time_bins]
            x_ticks = time_bins

            plt.plot(x_ticks, start_counts.values, label='Departures (Start)', color='red', marker='o',
                     linewidth=2)
            plt.plot(x_ticks, end_counts.values, label='Arrivals (End)', color='blue', marker='x',
                     linewidth=2)

            plt.ylim(0, 525)
            plt.title(f'Gathering Point {new_cid} Orders', fontsize=16)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Total Orders', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)

            # 设置X轴刻度和标签，只显示小时整点
            hourly_ticks = [m for m in x_ticks if m % 60 == 0]
            hourly_labels = [format_min_to_time(m) for m in hourly_ticks]
            plt.xticks(hourly_ticks, hourly_labels, rotation=45)

            plt.tight_layout()

            # 保存图表
            filename = os.path.join(output_dir, f"{new_cid.replace(' ', '_')}_flow_chart_11_16_merged.png")
            plt.savefig(filename)
            plt.close()

            print(f"已生成 {new_cid} 的流量图。")

        print("所有流量图生成完成。")


# 使用示例
if __name__ == "__main__":
    # 实例化网格类
    # 请根据您的实际文件路径修改以下路径
    grid_qhpark = GridQHPark(
        "data/cell_boundary_lat.csv",
        "data/cell_boundary_lon.csv"
    )

    # 网格边长为 40 米
    grid_size = 40
    grid_qhpark.create_grid(grid_size)

    # -----------------------------------------------------------
    # 文件路径设置
    TRIP_DATA_PATH = "data/processed_bike_data.xlsx"
    CLUSTER_JSON_PATH = "data/clustering_results_tensorflow_merged.json"

    # 1. 加载和重命名聚集点ID
    if grid_qhpark.load_cluster_data(CLUSTER_JSON_PATH):
        # 2. 执行订单筛选和输出 (11:00-16:00)
        print("\n---------- 开始订单聚类筛选和输出 (11:00-16:00) ----------")
        grid_qhpark.process_trips_by_cluster(
            trip_data_path=TRIP_DATA_PATH,
            output_dir="clustered_data"  # CSV输出文件夹
        )
        print("---------------------------------------------")

        # 3. 可视化流量 (11:00-16:00 跨日期合并)
        grid_qhpark.visualize_cluster_flow(
            trip_data_path=TRIP_DATA_PATH,
            output_dir="clustered_pictures"  # 图片输出文件夹
        )
        print("---------------------------------------------")