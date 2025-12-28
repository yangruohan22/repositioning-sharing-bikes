import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

class Grid:
    """表示单个网格的类"""
    def __init__(self, min_lat, max_lat, min_lon, max_lon, row, col, is_outside=False):
        self.min_lat = min_lat  # 网格最小纬度
        self.max_lat = max_lat  # 网格最大纬度
        self.min_lon = min_lon  # 网格最小经度
        self.max_lon = max_lon  # 网格最大经度
        self.row = row          # 网格行索引
        self.col = col          # 网格列索引
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
        
        # 创建校外特殊网格
        self.outside_grid = Grid(
            min_lat=39.994,   
            max_lat=39.9945,    
            min_lon=116.322,  # 全球最小经度
            max_lon=116.3225,   # 全球最大经度
            row=-1,          # 特殊行索引
            col=-1,          # 特殊列索引
            is_outside=True  # 标记为校外网格
        )
    
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
            raise ValueError("请先调用create_grid方法创建网格")
        
        # 检查经纬度是否在清华园范围内
        if not (self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon):
            # 返回校外特殊网格
            return self.outside_grid
        
        # 计算所在网格的行和列
        row = math.floor((lat - self.min_lat) / (self.lat_per_meter * self.grid_size))
        col = math.floor((lon - self.min_lon) / (self.lon_per_meter * self.grid_size))
        
        # 确保行和列在有效范围内
        row = max(0, min(row, self.grid_rows - 1))
        col = max(0, min(col, self.grid_cols - 1))
        
        grid_id = f"{row}_{col}"
        return self.grids.get(grid_id) if grid_id in self.grids else self.outside_grid
    
    
    def get_grid_neighbors(self, grid):
        """获取网格的相邻网格"""
        if self.grids is None or grid is None:
            return []
        
        # 如果是校外网格，没有相邻网格
        if grid.is_outside:
            return []
        
        row, col = grid.row, grid.col
        neighbors = []
        
        # 检查8个方向的相邻网格
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        
        for dr, dc in directions:
            neighbor_row, neighbor_col = row + dr, col + dc
            if 0 <= neighbor_row < self.grid_rows and 0 <= neighbor_col < self.grid_cols:
                neighbor_id = f"{neighbor_row}_{neighbor_col}"
                if neighbor_id in self.grids:
                    neighbors.append(self.grids[neighbor_id])
        
        return neighbors
    
    def get_total_grids(self):
        """获取网格总数"""
        if self.grids is None:
            return 0
        return len(self.grids)
    
    def visualize_grids(self, figsize=(12, 10), dpi=100):
        """可视化网格，用不同颜色显示校内和校外网格"""
        if self.grids is None:
            raise ValueError("请先调用create_grid方法创建网格")
        
        # 设置matplotlib中文字体支持
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
        
        # 创建图形
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        
        # 设置图形标题和标签
        plt.title(f'清华园网格可视化 ({self.grid_size}米 × {self.grid_size}米)', fontsize=15)
        plt.xlabel('经度', fontsize=12)
        plt.ylabel('纬度', fontsize=12)
        
        # 绘制所有网格
        for grid_id, grid in self.grids.items():
            # 跳过特殊的校外网格(-1_-1)
            if grid.row == -1 and grid.col == -1:
                continue
            
            # 根据是否为校外网格设置颜色
            if grid.is_outside:
                # 校外网格用浅蓝色表示
                color = 'lightblue'
                alpha = 0.3
            else:
                # 校内网格用绿色表示
                color = 'lightgreen'
                alpha = 0.7
            
            # 创建矩形补丁表示网格
            rect = patches.Rectangle(
                (grid.min_lon, grid.min_lat),  # 左下角坐标
                grid.max_lon - grid.min_lon,    # 宽度
                grid.max_lat - grid.min_lat,    # 高度
                edgecolor='gray',               # 边框颜色
                facecolor=color,                # 填充颜色
                alpha=alpha                     # 透明度
            )
            
            # 添加矩形到图形
            ax.add_patch(rect)
        
        # 绘制清华园实际边界（用于参考）
        # 从lat_boundary数据绘制边界线
        for _, row in self.lat_boundary.iterrows():
            plt.plot([row['Min_longitude'], row['Max_longitude']], 
                     [row['Latitude'], row['Latitude']], 'k-', linewidth=1)
        
        # 设置坐标轴范围
        plt.xlim(self.min_lon - 0.001, self.max_lon + 0.001)
        plt.ylim(self.min_lat - 0.001, self.max_lat + 0.001)
        
        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lightgreen', lw=4, label='校内网格'),
            Line2D([0], [0], color='lightblue', lw=4, label='校外网格')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 启用网格线以便查看
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 保持纵横比
        plt.axis('equal')
        
        # 显示图形
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 实例化网格类
    grid_qhpark = GridQHPark(
        "data/cell_boundary_lat.csv",
        "data/cell_boundary_lon.csv"
    )
    
    # 创建10*10米的网格
    grid_size = 40  # 40米
    grid_qhpark.create_grid(grid_size)
    
    # 测试根据经纬度查找网格 - 校内位置
    test_lat, test_lon = 40.0, 116.33  # 清华园附近的经纬度
    grid = grid_qhpark.get_grid_by_latlon(test_lat, test_lon)
    
    if grid:
        if grid.is_outside:
            print(f"\n经纬度({test_lat}, {test_lon})在校外网格")
        else:
            print(f"\n经纬度({test_lat}, {test_lon})所在的网格：")
            print(f"网格ID: {grid.get_id()}")
            print(f"网格边界：")
            print(f"  纬度范围: {grid.min_lat:.6f} - {grid.max_lat:.6f}")
            print(f"  经度范围: {grid.min_lon:.6f} - {grid.max_lon:.6f}")
            
            # 获取网格中心点
            center = grid.get_center()
            if center:
                print(f"网格中心点：({center[0]:.6f}, {center[1]:.6f})")
            
            # 直接调用Grid类的contains_point方法
            is_contained = grid.contains_point(test_lat, test_lon)
            print(f"网格包含该点: {is_contained}")
            
            # 获取相邻网格
            neighbors = grid_qhpark.get_grid_neighbors(grid)
            print(f"相邻网格数量: {len(neighbors)}")
    
    # 测试根据经纬度查找网格 - 校外位置
    test_lat_outside, test_lon_outside = 40.009,116.322  # 清华园外的经纬度
    grid_outside = grid_qhpark.get_grid_by_latlon(test_lat_outside, test_lon_outside)
    
    if grid_outside:
        if grid_outside.is_outside:
            print(f"\n经纬度({test_lat_outside}, {test_lon_outside})在校外网格")
            print(f"校外网格标识: row={grid_outside.row}, col={grid_outside.col}")
            print(f"校外网格边界范围: 全球范围")
        else:
            print(f"\n经纬度({test_lat_outside}, {test_lon_outside})所在的网格：")
            print(f"网格ID: {grid_outside.get_id()}")
    
    # 可视化网格
    print("\n正在生成网格可视化...")
    print("绿色表示校内网格，蓝色表示校外网格")
    grid_qhpark.visualize_grids(figsize=(12, 10), dpi=100)