#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @ Time:2025/12/22 6:37
# @ Author:86155
import pandas as pd
import numpy as np
import heapq
import json
import re
import os
import random
from datetime import timedelta, datetime, time, date
from typing import Dict, List, Tuple, Optional, Any
import warnings
from collections import defaultdict
# matplotlib核心绘图
import matplotlib.pyplot as plt
# 日期格式化（x轴时间显示）
from matplotlib.dates import DateFormatter, MinuteLocator
# （可选）更美观的样式
import seaborn as sns

warnings.filterwarnings('ignore')

# ======================== 基础配置（新增率数据文件路径）========================
FIXED_NODE_CAPACITIES = {
    1: 54, 2: 60, 3: 65, 4: 50, 5: 59, 6: 158, 7: 71, 8: 135,
    9: 57, 10: 139, 11: 31, 12: 71, 13: 65, 14: 108, 15: 112
}

PFA_PARAMS = {
    'THETA_SUPPLY_L': 0.228,  # 供应节点下阈值
    'THETA_SUPPLY_U': 1.503,  # 供应节点上阈值
    'THETA_DEMAND_L': 0.501,  # 需求节点下阈值
    'THETA_DEMAND_U': 1.038,  # 需求节点上阈值
    'TAU_L': 12,  # 库存决策前瞻时间(分钟)
    'TAU_D': 59,  # 路由决策前瞻时间(分钟)
    'DELTA': 0.721,  # 容量阈值参数
    'BETA': 0.604,  # 集合大小参数
    'PHI': 1.231,  # 途中时间限制参数
    'ENROUTE_UNLOAD': False  # 强制关闭弧上卸载
}

SYSTEM_PARAMS = {
    # 车辆参数
    'TRUCK_CAPACITY': 25,  # 车辆容量
    'TRUCK_SPEED_KMH': 15.0,  # 车辆速度(km/h)
    # 工作时间（仅该时间段内节点出发订单算作需求）
    'WORK_START_TIME': time(11, 0),  # 11:00
    'WORK_END_TIME': time(15, 30),  # 15:30
    # 操作时间(分钟/辆)
    'NODE_OPERATION_TIME_PER_BIKE': 0.25,  # 节点装车/卸车统一时间
    'ARC_COLLECTION_TIME_PER_BIKE': 0.5,  # 弧上收集时间
    # 其他参数
    'NODE_CAPACITY_DEFAULT': 50,  # 未指定节点的默认容量
    'ARC_CAPACITY_INFINITE': True,  # 弧容量无限
    'TIME_SLOT_MINUTES': 5,  # 时间槽长度（5分钟粒度）
    'DECISION_INTERVAL': 5,  # 决策冷却时间（避免死循环）
    'DECISION_COOLDOWN': 1.0,  # 最小决策间隔（分钟）
    'FIXED_SETUP_TIME': 0,
    'INVENTORY_UPDATE_INTERVAL': 5,  # 率驱动库存更新间隔（分钟）
}
DATA_PATHS = {
    'arc_init': 'arcinitial.xlsx',
    'node_init': 'nodeinitial.xlsx',
    'orders': 'tagged_bike_data_semicolon.xlsx',
    'network': 'arc_gdf.geojson',
    'rates': 'arrival_departure_rates.xlsx'  # 新增：到达率/离开率文件
}
SIMULATION_CONFIG = {
    'base_date': date(2024, 5, 27),  # 基准日期
    'initial_truck_node': 2,  # 卡车初始节点
    'debug': True,  # 开启调试
    'print_progress': True,  # 打印进度
    'time_range_only': True,  # 仅仿真指定时间范围
    'track_arc_inventory': True,  # 追踪弧库存变动
    'max_decision_events': 1000,  # 最大决策事件数（防死循环）
}


# ======================== 工具函数（新增：时间槽匹配+率驱动库存计算）========================
def get_time_slot_column(current_time: datetime) -> int:
    """
    匹配当前时间对应的5分钟粒度列索引（11:00=0列，11:05=1列...15:25=最后一列）
    返回：列索引（从0开始），超出范围返回-1
    """
    work_start = time(11, 0)
    work_end = time(15, 25)  # 最后一个时间槽是15:25

    # 提取时间部分
    current_time_only = current_time.time()
    if current_time_only < work_start or current_time_only > work_end:
        return -1

    # 计算与11:00的分钟差
    minutes_since_start = (current_time_only.hour - 11) * 60 + current_time_only.minute
    slot_idx = minutes_since_start // SYSTEM_PARAMS['TIME_SLOT_MINUTES']

    return int(slot_idx)


def calculate_inventory_change_by_rate(node_id: int, arc_id: str,
                                       current_time: datetime, time_delta_minutes: float,
                                       data_loader) -> Tuple[float, float]:
    """
    基于率数据计算时间差内的库存变动
    :param node_id: 节点ID（None则不计算节点）
    :param arc_id: 弧ID（None则不计算弧）
    :param current_time: 当前时间
    :param time_delta_minutes: 时间差（分钟）
    :param data_loader: 数据加载器
    :return: (节点库存变动量, 弧库存变动量) 正数=增加，负数=减少
    """
    node_change = 0.0
    arc_change = 0.0

    # 计算节点库存变动：到达率×时间 - 离开率×时间
    if node_id is not None and time_delta_minutes > 0:
        arrival_rate = data_loader.get_node_rate(node_id, current_time, 'arrival')
        departure_rate = data_loader.get_node_rate(node_id, current_time, 'departure')
        node_change = (arrival_rate - departure_rate) * time_delta_minutes

    # 计算弧库存变动（仅使用弧的率数据）
    if arc_id is not None and time_delta_minutes > 0 and arc_id in data_loader.arc_rates:
        slot_idx = get_time_slot_column(current_time)
        if slot_idx >= 0 and slot_idx < len(data_loader.arc_rates[arc_id]):
            arc_rate = data_loader.arc_rates[arc_id][slot_idx]
            arc_change = arc_rate * time_delta_minutes

    return node_change, arc_change


# ======================== 数据加载类（核心修改：加载率数据）========================
class DataLoader:
    """加载和管理所有输入数据（新增率数据加载）"""

    def __init__(self):
        self.network = NetworkManager(DATA_PATHS['network'])
        self.initial_inventory = {}
        self.node_capacities = {}  # 固定节点容量
        self.arc_capacities = {}  # 弧容量（无限，仅占位）
        self.orders_df = None

        # 新增：率数据存储
        self.node_departure_rates = {}  # 节点离开率 {node_id: [0.1, 0.2, ...]} 5分钟粒度
        self.node_arrival_rates = {}  # 节点到达率 {node_id: [0.1, 0.2, ...]} 5分钟粒度
        self.arc_rates = {}  # 弧段率 {arc_id: [0.1, 0.2, ...]} 5分钟粒度

        self._load_all_data()

    def _load_all_data(self):
        """加载所有必要数据（新增率数据加载）"""
        # 1. 加载初始库存
        self._load_initial_inventory()

        # 2. 加载订单数据（仅用于统计满足率，不参与库存变更）
        self._load_orders()

        # 3. 设置固定节点容量
        self._set_fixed_node_capacities()

        # 4. 新增：加载到达率/离开率数据
        self._load_arrival_departure_rates()

    def _load_initial_inventory(self):
        """加载初始库存分布"""
        # 加载节点库存
        try:
            print("正在加载节点初始库存...")
            df_nodes = pd.read_excel(DATA_PATHS['node_init'], engine='openpyxl')
            for _, row in df_nodes.iterrows():
                node_id = int(float(row.iloc[0]))
                inventory = int(row.iloc[2])
                self.initial_inventory[f"node_{node_id}"] = inventory
            print(f"成功加载 {len([k for k in self.initial_inventory if k.startswith('node_')])} 个节点的库存")
        except Exception as e:
            print(f"加载节点库存失败: {e}")

        # 加载弧库存
        try:
            print("正在加载弧初始库存...")
            df_arcs = pd.read_excel(DATA_PATHS['arc_init'], engine='openpyxl')
            for _, row in df_arcs.iterrows():
                raw_id = row.iloc[0]
                std_id = self.network.parse_arc_id(raw_id)
                if std_id:
                    arc_key = f"arc_{std_id}"
                    inventory = int(row.iloc[2])
                    self.initial_inventory[arc_key] = inventory
            print(f"成功加载 {len([k for k in self.initial_inventory if k.startswith('arc_')])} 条弧的库存")
        except Exception as e:
            print(f"加载弧库存失败: {e}")

    def _set_fixed_node_capacities(self):
        """设置固定节点容量"""
        print("正在设置固定节点容量...")
        # 转换为代码中使用的key格式（node_xxx）
        for node_id, capacity in FIXED_NODE_CAPACITIES.items():
            self.node_capacities[f"node_{node_id}"] = capacity
            print(f"  节点{node_id}: 容量={capacity}")

        # 为未指定的节点设置默认容量
        for key in self.initial_inventory:
            if key.startswith('node_') and key not in self.node_capacities:
                node_id = int(key.split('_')[1])
                self.node_capacities[key] = SYSTEM_PARAMS['NODE_CAPACITY_DEFAULT']
                print(f"  节点{node_id}: 使用默认容量={SYSTEM_PARAMS['NODE_CAPACITY_DEFAULT']}")

        print(f"共设置 {len(self.node_capacities)} 个节点的容量")

    def _load_orders(self):
        """加载订单数据（仅用于统计满足率，不参与库存变更）"""
        try:
            print(f"正在加载订单数据...")
            self.orders_df = pd.read_excel(DATA_PATHS['orders'])

            # 显示数据基本信息
            print(f"数据形状: {self.orders_df.shape}")
            print(f"列名: {list(self.orders_df.columns)}")

            # 转换时间列（兼容多格式）
            time_cols = ['start_time', 'end_time']
            for col in time_cols:
                if col in self.orders_df.columns:
                    try:
                        self.orders_df[col] = pd.to_datetime(
                            self.orders_df[col],
                            errors='coerce',
                            format='mixed'  # 自动识别多种时间格式
                        )
                    except Exception as e:
                        print(f"转换列 {col} 时出错: {e}")

            # 确保必要列存在
            required_cols = ['start_node_ids', 'end_node_ids', 'start_arc_ids', 'end_arc_ids']
            for col in required_cols:
                if col not in self.orders_df.columns:
                    print(f"警告: 列 '{col}' 不存在，使用默认值")
                    self.orders_df[col] = np.nan

            # 按开始时间排序
            if 'start_time' in self.orders_df.columns:
                self.orders_df = self.orders_df.dropna(subset=['start_time'])
                self.orders_df.sort_values('start_time', inplace=True)

            # 分析日期范围
            if 'start_time' in self.orders_df.columns and len(self.orders_df) > 0:
                min_date = self.orders_df['start_time'].min()
                max_date = self.orders_df['start_time'].max()
                print(f"订单日期范围: {min_date} 到 {max_date}")

                # 提取2024年5月27日到6月14日的订单
                may_june_orders = self.get_orders_in_date_range('2024-05-27', '2024-06-14')
                print(f"2024年5月27日到6月14日的订单数: {len(may_june_orders)}")

            print(f"订单数据加载完成，共 {len(self.orders_df)} 条记录（仅用于统计满足率）")

        except Exception as e:
            print(f"加载订单数据失败: {e}")
            print("错误详情:", e)
            self.orders_df = pd.DataFrame()

    def _load_arrival_departure_rates(self):
        """新增：加载到达率/离开率数据（修复NaN转换错误）"""
        try:
            print("\n正在加载到达率/离开率数据...")
            # 1. 加载节点离开率（sheet1）
            df_depart = pd.read_excel(DATA_PATHS['rates'], sheet_name="节点_离开率", engine='openpyxl')
            # 数据清洗：过滤ID列空值，重置索引
            df_depart = df_depart.dropna(subset=[df_depart.columns[0]]).reset_index(drop=True)
            print(f"节点离开率数据形状: {df_depart.shape} (已过滤空ID行)")

            # 第一列为node_id，后续列为5分钟粒度的率（辆/分钟）
            for _, row in df_depart.iterrows():
                try:
                    # 处理ID：先转字符串去空，再转数字
                    id_raw = str(row.iloc[0]).strip()
                    if not id_raw or id_raw.lower() == 'nan':
                        continue
                    node_id = int(float(id_raw))  # 兼容浮点型ID（如10.0→10）
                    # 率数据：填充NaN为0，转浮点数
                    rates = row.iloc[1:].fillna(0.0).astype(float).tolist()
                    self.node_departure_rates[node_id] = rates
                except (ValueError, TypeError) as e:
                    print(f"  跳过无效节点行: {row.iloc[0]} (错误: {e})")
                    continue
            print(f"成功加载 {len(self.node_departure_rates)} 个节点的离开率")

            # 2. 加载节点到达率（sheet2）
            df_arrival = pd.read_excel(DATA_PATHS['rates'], sheet_name="节点_到达率", engine='openpyxl')
            df_arrival = df_arrival.dropna(subset=[df_arrival.columns[0]]).reset_index(drop=True)
            print(f"节点到达率数据形状: {df_arrival.shape} (已过滤空ID行)")

            for _, row in df_arrival.iterrows():
                try:
                    id_raw = str(row.iloc[0]).strip()
                    if not id_raw or id_raw.lower() == 'nan':
                        continue
                    node_id = int(float(id_raw))
                    rates = row.iloc[1:].fillna(0.0).astype(float).tolist()
                    self.node_arrival_rates[node_id] = rates
                except (ValueError, TypeError) as e:
                    print(f"  跳过无效节点行: {row.iloc[0]} (错误: {e})")
                    continue
            print(f"成功加载 {len(self.node_arrival_rates)} 个节点的到达率")

            # 3. 加载弧段率（sheet3）
            df_arc = pd.read_excel(DATA_PATHS['rates'], sheet_name="弧段", engine='openpyxl')
            df_arc = df_arc.dropna(subset=[df_arc.columns[0]]).reset_index(drop=True)
            print(f"弧段率数据形状: {df_arc.shape} (已过滤空ID行)")

            for _, row in df_arc.iterrows():
                try:
                    id_raw = str(row.iloc[0]).strip()
                    if not id_raw or id_raw.lower() == 'nan':
                        continue
                    std_arc_id = self.network.parse_arc_id(id_raw)
                    if std_arc_id:
                        rates = row.iloc[1:].fillna(0.0).astype(float).tolist()
                        self.arc_rates[std_arc_id] = rates
                except (ValueError, TypeError) as e:
                    print(f"  跳过无效弧段行: {row.iloc[0]} (错误: {e})")
                    continue
            print(f"成功加载 {len(self.arc_rates)} 条弧段的率数据")

        except Exception as e:
            print(f"加载到达率/离开率数据失败: {e}")
            # 打印详细错误栈，方便定位
            import traceback
            traceback.print_exc()
            raise e

    def get_node_rate(self, node_id: int, current_time: datetime, rate_type: str) -> float:
        """
        获取节点在当前时间的到达率/离开率（辆/分钟）
        :param node_id: 节点ID
        :param current_time: 当前时间
        :param rate_type: 'departure'（离开率）/ 'arrival'（到达率）
        :return: 率（辆/分钟），无数据返回0
        """
        # 1. 匹配时间槽列索引
        slot_idx = get_time_slot_column(current_time)
        if slot_idx < 0:
            return 0.0

        # 2. 获取对应率数据
        rate_dict = self.node_departure_rates if rate_type == 'departure' else self.node_arrival_rates
        if node_id not in rate_dict:
            if SIMULATION_CONFIG['debug']:
                print(f"警告：节点{node_id}无{rate_type}率数据，返回0")
            return 0.0

        rates = rate_dict[node_id]
        # 容错：时间槽索引超出率列表长度时
        if slot_idx >= len(rates):
            # 取最后一个有效值，无有效值则返回0
            return rates[-1] if len(rates) > 0 else 0.0

        # 确保返回值是浮点数，避免None/NaN
        rate_value = rates[slot_idx]
        return float(rate_value) if not pd.isna(rate_value) else 0.0

    def get_orders_in_date_range(self, start_date_str, end_date_str):
        """获取指定日期范围内的订单（仅用于统计满足率）"""
        if self.orders_df is None or 'start_time' not in self.orders_df.columns:
            return pd.DataFrame()

        start_date = pd.to_datetime(start_date_str).date()
        end_date = pd.to_datetime(end_date_str).date()

        mask = (self.orders_df['start_time'].dt.date >= start_date) & \
               (self.orders_df['start_time'].dt.date <= end_date)

        return self.orders_df[mask]


# ======================== 网络管理类（无修改）========================
class NetworkManager:
    """管理网络拓扑和距离"""

    def __init__(self, network_path):
        self.distances = {}  # (u,v) -> 距离(km)
        if os.path.exists(network_path):
            self._load_network(network_path)
        else:
            print(f"网络文件不存在: {network_path}")

    def _load_network(self, path):
        """从GeoJSON加载网络"""
        try:
            print(f"正在加载网络数据: {path}")
            # 处理BOM头和编码问题
            with open(path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)

            arc_count = 0
            for feature in data['features']:
                props = feature['properties']
                arc_id = str(props.get('arc_id', ''))

                # 解析节点
                nodes = self._extract_nodes_from_id(arc_id)
                if nodes and len(nodes) >= 2:
                    u, v = nodes[0], nodes[1]
                    length_km = props.get('length', 0) / 1000.0
                    key = tuple(sorted((u, v)))
                    self.distances[key] = length_km
                    arc_count += 1

            print(f"成功加载 {arc_count} 条弧的距离数据")

        except Exception as e:
            print(f"加载网络失败: {e}")

    def _extract_nodes_from_id(self, arc_id):
        """从弧ID中提取节点"""
        try:
            arc_id_str = str(arc_id)

            # 尝试匹配 "u_v" 格式
            if '_' in arc_id_str:
                parts = arc_id_str.split('_')
                if len(parts) >= 2:
                    return [int(parts[0]), int(parts[1])]

            # 尝试匹配括号格式 (u,v)
            if '(' in arc_id_str and ')' in arc_id_str:
                match = re.search(r'\((\d+),\s*(\d+)\)', arc_id_str)
                if match:
                    return [int(match.group(1)), int(match.group(2))]

            # 尝试匹配纯数字
            numbers = re.findall(r'\d+', arc_id_str)
            if len(numbers) >= 2:
                return [int(numbers[0]), int(numbers[1])]

        except Exception as e:
            print(f"提取节点ID失败: {arc_id}, 错误: {e}")
        return None

    def parse_arc_id(self, raw_id):
        """解析弧ID为标准格式（u_v）"""
        nodes = self._extract_nodes_from_id(raw_id)
        if nodes and len(nodes) >= 2:
            u, v = nodes[0], nodes[1]
            return f"{min(u, v)}_{max(u, v)}"
        return None

    def get_distance(self, u, v):
        """获取两点距离(km)"""
        if u == v:
            return 0.0
        key = tuple(sorted((u, v)))
        return self.distances.get(key, 1.0)  # 默认1km

    def get_travel_time(self, u, v):
        """获取行驶时间(分钟)"""
        distance = self.get_distance(u, v)
        speed = SYSTEM_PARAMS['TRUCK_SPEED_KMH']
        return (distance / max(speed, 1.0)) * 60.0


# ======================== PFA策略类（核心修改：率驱动的供需计算）========================
class PFAStrategy:
    """PFA策略实现（禁用弧上卸载，基于到达率/离开率计算）"""

    def __init__(self, params, network, data_loader):
        self.params = params
        self.network = network
        self.data_loader = data_loader
        self.current_day_orders = []  # 当前天的订单数据（仅用于统计满足率）
        self.current_day = None

    def set_current_day(self, current_day):
        """设置当前仿真日期"""
        self.current_day = current_day
        if self.data_loader.orders_df is not None and 'start_time' in self.data_loader.orders_df.columns:
            day_mask = (self.data_loader.orders_df['start_time'].dt.date == current_day)
            self.current_day_orders = self.data_loader.orders_df[day_mask].sort_values('start_time').to_dict('records')
        else:
            self.current_day_orders = []

    def compute_inventory_decision(self, node_id, current_inv, truck_load, current_time):
        """库存决策（仅节点）- 基于到达率/离开率计算（对齐论文逻辑）"""
        tau_L = self.params['TAU_L']  # 库存决策前瞻时间（分钟）

        # 1. 获取当前节点的到达率/离开率（辆/分钟）
        departure_rate = self.data_loader.get_node_rate(node_id, current_time, 'departure')  # 离开率（λ^-）
        arrival_rate = self.data_loader.get_node_rate(node_id, current_time, 'arrival')  # 到达率（λ^+）

        # 2. 计算前瞻时间内的预计数量（论文公式：数量=率×时间）
        expected_departures = departure_rate * tau_L  # 前瞻时间内预计离开数量
        expected_arrivals = arrival_rate * tau_L  # 前瞻时间内预计到达数量
        net_demand = expected_departures - expected_arrivals  # 净需求=离开-到达

        # 3. 应用论文阈值参数（约束目标库存）
        capacity = self.data_loader.node_capacities.get(f"node_{node_id}", SYSTEM_PARAMS['NODE_CAPACITY_DEFAULT'])
        theta_supply_l = self.params['THETA_SUPPLY_L']
        theta_supply_u = self.params['THETA_SUPPLY_U']
        theta_demand_l = self.params['THETA_DEMAND_L']
        theta_demand_u = self.params['THETA_DEMAND_U']

        # 4. 计算目标库存（对齐论文的上下限逻辑）
        if net_demand > 0:  # 需求节点：需要补车
            # 需求节点下阈值约束：目标库存 ≥ θ_demand_l × 容量
            min_target = theta_demand_l * capacity
            # 需求节点上阈值约束：目标库存 ≤ θ_demand_u × 容量
            max_target = theta_demand_u * capacity
            # 目标库存 = 当前库存 + 净需求（约束在阈值范围内）
            target_inv = np.clip(current_inv + net_demand, min_target, max_target)
            needed = max(0, target_inv - current_inv)  # 需要补充的数量
            can_unload = truck_load
            return -min(needed, can_unload)  # 负号表示卸载

        elif net_demand < 0:  # 供应节点：需要运走
            # 供应节点下阈值约束：目标库存 ≥ θ_supply_l × 容量
            min_target = theta_supply_l * capacity
            # 供应节点上阈值约束：目标库存 ≤ θ_supply_u × 容量
            max_target = theta_supply_u * capacity
            # 目标库存 = 当前库存 + 净需求（约束在阈值范围内）
            target_inv = np.clip(current_inv + net_demand, min_target, max_target)
            excess = max(0, current_inv - target_inv)  # 多余的数量
            can_load = SYSTEM_PARAMS['TRUCK_CAPACITY'] - truck_load
            return min(excess, can_load)  # 正号表示装载

        else:  # 平衡节点：无需操作
            return 0

    def _calculate_net_demand(self, node_id, current_time, lookahead_minutes):
        """兼容原有接口的净需求计算（基于率数据）"""
        # 获取到达率/离开率
        departure_rate = self.data_loader.get_node_rate(node_id, current_time, 'departure')
        arrival_rate = self.data_loader.get_node_rate(node_id, current_time, 'arrival')

        # 预计数量=率×前瞻时间
        expected_departures = departure_rate * lookahead_minutes
        expected_arrivals = arrival_rate * lookahead_minutes

        return expected_departures - expected_arrivals

    def _extract_node_ids(self, node_str):
        """解析节点ID"""
        if pd.isna(node_str):
            return []
        try:
            node_str = str(node_str).replace('[', '').replace(']', '').replace(' ', '')
            nodes = node_str.split(';')
            return [int(float(node)) for node in nodes if node]
        except:
            return []

    def compute_routing_decision(self, current_node, truck_load, inventory_decision,
                                 system_inv, current_time):
        """路由决策（基于率数据的需求预测）"""
        try:
            post_load = truck_load + inventory_decision
            capacity = SYSTEM_PARAMS['TRUCK_CAPACITY']
            load_ratio = (capacity - post_load) / capacity if capacity > 0 else 1.0

            # 获取候选节点
            candidate_nodes = []
            for key in system_inv:
                if key.startswith('node_'):
                    try:
                        node_id = int(key.split('_')[1])
                        if node_id != current_node:
                            candidate_nodes.append(node_id)
                    except Exception as e:
                        if SIMULATION_CONFIG['debug']:
                            print(f"解析节点ID失败: {key}, 错误: {e}")
                        continue

            if not candidate_nodes:
                return current_node

            # 根据负载率选择模式
            if load_ratio <= self.params['DELTA']:
                return self._select_node_for_unloading(current_node, candidate_nodes,
                                                       current_time, system_inv)
            else:
                return self._select_node_for_loading(current_node, candidate_nodes, system_inv)
        except Exception as e:
            if SIMULATION_CONFIG['debug']:
                print(f"路由决策失败: {e} (错误类型: {type(e).__name__})")
            return current_node  # 兜底：返回当前节点

    def _select_node_for_unloading(self, current_node, candidates, current_time, system_inv):
        """选择卸载节点（基于率数据的未来需求）"""
        best_node = current_node
        best_score = -1

        for node_id in candidates:
            # 基础得分：基于率的未来需求与库存的差距
            tau_D = self.params['TAU_D']
            departure_rate = self.data_loader.get_node_rate(node_id, current_time, 'departure')
            expected_future_demand = departure_rate * tau_D  # 前瞻时间内预计需求
            current_inv = system_inv.get(f"node_{node_id}", 0)
            base_score = max(0, expected_future_demand - current_inv)

            # 加上相邻弧的库存补充得分
            arc_bonus = 0
            arc_id = self.network.parse_arc_id(f"{current_node}_{node_id}")
            if arc_id:
                arc_key = f"arc_{arc_id}"
                arc_inv = system_inv.get(arc_key, 0)
                arc_bonus = min(arc_inv / 5, 5)

            total_score = base_score + arc_bonus

            if total_score > best_score:
                best_score = total_score
                best_node = node_id
            elif total_score == best_score:
                dist1 = self.network.get_distance(current_node, best_node)
                dist2 = self.network.get_distance(current_node, node_id)
                if dist2 < dist1:
                    best_node = node_id
                elif dist2 == dist1:
                    arc1_id = self.network.parse_arc_id(f"{current_node}_{best_node}")
                    arc2_id = self.network.parse_arc_id(f"{current_node}_{node_id}")
                    arc1_inv = system_inv.get(f"arc_{arc1_id}", 0) if arc1_id else 0
                    arc2_inv = system_inv.get(f"arc_{arc2_id}", 0) if arc2_id else 0
                    if arc2_inv > arc1_inv:
                        best_node = node_id

        return best_node

    def _select_node_for_loading(self, current_node, candidates, system_inv):
        """选择装载节点（基于率数据的供应能力）"""
        best_node = current_node
        best_inventory = -1

        for node_id in candidates:
            # 基础库存：节点当前库存
            node_inv = system_inv.get(f"node_{node_id}", 0)
            # 加上相邻弧的库存
            arc_id = self.network.parse_arc_id(f"{current_node}_{node_id}")
            arc_inv = system_inv.get(f"arc_{arc_id}", 0) if arc_id else 0
            total_inv = node_inv + arc_inv

            if total_inv > best_inventory:
                best_inventory = total_inv
                best_node = node_id
            elif total_inv == best_inventory:
                dist1 = self.network.get_distance(current_node, best_node)
                dist2 = self.network.get_distance(current_node, node_id)
                if dist2 > dist1:
                    best_node = node_id
                elif dist2 == dist1:
                    arc1_id = self.network.parse_arc_id(f"{current_node}_{best_node}")
                    arc2_id = self.network.parse_arc_id(f"{current_node}_{node_id}")
                    arc1_inv = system_inv.get(f"arc_{arc1_id}", 0) if arc1_id else 0
                    arc2_inv = system_inv.get(f"arc_{arc2_id}", 0) if arc2_id else 0
                    if arc2_inv < arc1_inv:
                        best_node = node_id

        return best_node

    def _estimate_future_demand(self, node_id, current_time, tau_D):
        """估算节点未来需求（基于率数据）"""
        departure_rate = self.data_loader.get_node_rate(node_id, current_time, 'departure')
        return departure_rate * tau_D  # 预计未来需求=离开率×前瞻时间

    def compute_enroute_decision(self, current_node, next_node, truck_load, system_inv):
        """途中决策（仅返回可收集量，禁用卸载）"""
        travel_time = self.network.get_travel_time(current_node, next_node)
        phi = self.params['PHI']
        time_limit = phi * travel_time
        collection_time = SYSTEM_PARAMS['ARC_COLLECTION_TIME_PER_BIKE']

        # 仅计算可收集量（禁用卸载）
        max_by_time_collect = int(time_limit / collection_time)
        remaining_capacity = SYSTEM_PARAMS['TRUCK_CAPACITY'] - truck_load
        max_collect = min(max_by_time_collect, remaining_capacity)

        return {
            'collect': max_collect,
            'unload': 0,  # 强制返回0，禁用卸载
            'arc_id': self.network.parse_arc_id(f"{current_node}_{next_node}")
        }


# ======================== 不作为策略类（无修改）========================
class DoNothingStrategy:
    """不作为策略（基准对比）"""

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def set_current_day(self, current_day):
        pass

    def compute_inventory_decision(self, node_id, current_inv, truck_load, current_time):
        return 0

    def compute_routing_decision(self, current_node, truck_load, inventory_decision,
                                 system_inv, current_time):
        return current_node

    def compute_enroute_decision(self, current_node, next_node, truck_load, system_inv):
        # 不作为策略：不收集/卸载弧上自行车
        return {'collect': 0, 'unload': 0, 'arc_id': None}


# ======================== 仿真引擎类（核心修改：率驱动库存+移除还车）========================
class BikeSharingSimulator:
    """共享单车调度仿真器（率驱动库存，订单仅统计满足率）"""

    def __init__(self, data_loader, strategy, strategy_name="Strategy"):
        self.data = data_loader
        self.strategy = strategy
        self.strategy_name = strategy_name

        # 仿真状态
        self.current_time = None
        self.inventory = None
        self.truck_node = None
        self.truck_load = None
        self.last_inventory_update_time = None  # 上次库存更新时间（率驱动）

        # 防死循环控制
        self.decision_event_count = 0  # 决策事件计数器
        self.last_decision_time = None  # 上一次决策时间
        self.last_decision_node = None  # 上一次决策节点

        # 统计信息（仅11:00-15:30节点出发订单为需求）
        self.satisfied_demand = 0
        self.total_demand = 0
        self.truck_operations = 0
        self.daily_stats = []
        self.arc_inventory_history = defaultdict(list)  # 弧库存变动历史
        self.node_inventory_history = defaultdict(list)  # 节点库存变动历史

        self.node_inventory_history = []  # 每条记录：{'time': datetime, 'node_id': int, 'inventory': int, 'change': int, 'change_type': str}
        self.arc_inventory_history = []  # 每条记录：{'time': datetime, 'arc_id': str, 'inventory': int, 'change': int, 'change_type': str}
        self.truck_movement_history = []  # 卡车移动历史：{'time': datetime, 'from_node': int, 'to_node': int, 'load': int, 'operation_type': str}

    def run_for_date_range(self, params=None, date_range_start=None, date_range_end=None):
        """运行指定时间范围的策略验证"""
        print(f"\n开始运行策略: {self.strategy_name}")
        print("核心规则：库存纯靠率数据驱动，订单仅用于统计满足率，无还车逻辑")

        if date_range_start and date_range_end:
            if isinstance(date_range_start, str):
                date_range_start = datetime.strptime(date_range_start, '%Y-%m-%d').date()
            if isinstance(date_range_end, str):
                date_range_end = datetime.strptime(date_range_end, '%Y-%m-%d').date()

            print(f"仿真日期范围: {date_range_start} 到 {date_range_end}")
            simulation_dates = []
            current_date = date_range_start
            while current_date <= date_range_end:
                simulation_dates.append(current_date)
                current_date += timedelta(days=1)
        else:
            if self.data.orders_df is not None and 'start_time' in self.data.orders_df.columns:
                simulation_dates = sorted(self.data.orders_df['start_time'].dt.date.unique())
                print(f"使用订单数据中的所有日期: {len(simulation_dates)} 天")
            else:
                simulation_dates = [SIMULATION_CONFIG['base_date'] + timedelta(days=i) for i in range(19)]
                print("没有订单数据，使用默认日期范围")

        print(f"仿真天数: {len(simulation_dates)}")

        for day_idx, sim_date in enumerate(simulation_dates, 1):
            if SIMULATION_CONFIG['print_progress']:
                print(f"  正在仿真第 {day_idx}/{len(simulation_dates)} 天 ({sim_date})...")
            # 重置防死循环计数器
            self.decision_event_count = 0
            self.last_decision_time = None
            self.last_decision_node = None
            # 清空当日库存历史
            self.arc_inventory_history = defaultdict(list)
            self.node_inventory_history = defaultdict(list)
            day_result = self._run_single_day(sim_date)
            self.daily_stats.append(day_result)

            # 打印当日库存变动
            if SIMULATION_CONFIG['debug']:
                self._print_inventory_history(sim_date)

        # 计算总体结果（仅11:00-15:30节点出发订单）
        if len(self.daily_stats) > 0:
            total_satisfied = sum(s['satisfied'] for s in self.daily_stats)
            total_demand = sum(s['total'] for s in self.daily_stats)
            total_operations = sum(s['operations'] for s in self.daily_stats)
            satisfaction_rate = total_satisfied / total_demand if total_demand > 0 else 0
        else:
            total_satisfied = 0
            total_demand = 0
            total_operations = 0
            satisfaction_rate = 0

        print(f"\n策略 '{self.strategy_name}' 完成:")
        print(f"  总天数: {len(simulation_dates)}")
        print(f"  有效需求: {total_demand} (仅11:00-15:30节点出发订单)")
        print(f"  满足需求: {total_satisfied} (仅有效需求)")
        print(f"  需求满足率: {satisfaction_rate:.2%} (仅有效需求)")
        print(f"  总调度操作: {total_operations} (含弧收集)")

        return {
            'satisfaction_rate': satisfaction_rate,
            'total_satisfied': total_satisfied,
            'total_demand': total_demand,
            'total_operations': total_operations,
            'daily_stats': self.daily_stats,
            'arc_inventory_history': dict(self.arc_inventory_history),
            'node_inventory_history': dict(self.node_inventory_history)
        }

    def _run_single_day(self, sim_date):
        """运行单日仿真（核心：率驱动库存，订单仅统计满足率，无还车）"""
        if hasattr(self.strategy, 'set_current_day'):
            self.strategy.set_current_day(sim_date)

        # 初始化状态
        self.inventory = self.data.initial_inventory.copy()
        self.truck_node = SIMULATION_CONFIG['initial_truck_node']
        self.truck_load = 0
        self.last_inventory_update_time = None  # 初始化率驱动库存更新时间
        day_satisfied = 0  # 仅11:00-15:30节点出发且满足的订单
        day_demand = 0  # 仅11:00-15:30节点出发的订单
        day_operations = 0

        # 设置工作时间（11:00-15:30）
        work_start = datetime.combine(sim_date, SYSTEM_PARAMS['WORK_START_TIME'])
        work_end = datetime.combine(sim_date, SYSTEM_PARAMS['WORK_END_TIME'])
        self.last_inventory_update_time = work_start  # 初始库存更新时间

        # 获取当日所有订单（仅用于统计满足率）
        day_orders = []
        if self.data.orders_df is not None and 'start_time' in self.data.orders_df.columns:
            day_mask = (self.data.orders_df['start_time'].dt.date == sim_date)
            day_orders = self.data.orders_df[day_mask].sort_values('start_time').to_dict('records')

        if len(day_orders) == 0:
            return {
                'date': sim_date,
                'satisfied': 0,
                'total': 0,
                'satisfaction_rate': 0,
                'operations': 0
            }

        # 创建事件队列（仅包含订单出发和决策事件，无还车事件）
        events = []
        event_counter = 0

        # 添加订单事件（仅用于统计满足率，不修改库存）
        for order in day_orders:
            if 'start_time' in order and pd.notna(order['start_time']):
                event_counter += 1
                heapq.heappush(events, (order['start_time'], event_counter, 'order_departure', order))

        # 添加初始决策事件（工作开始时间）
        event_counter += 1
        heapq.heappush(events, (work_start, event_counter, 'decision', None))

        # 事件循环
        self.current_time = work_start

        while events and self.current_time <= work_end:
            try:
                event_time, event_id, event_type, event_data = heapq.heappop(events)
            except (ValueError, IndexError):
                break
            self.current_time = event_time

            # 防死循环：决策事件超过最大值则退出
            if event_type == 'decision':
                self.decision_event_count += 1
                if self.decision_event_count > SIMULATION_CONFIG['max_decision_events']:
                    if SIMULATION_CONFIG['debug']:
                        print(f"    ⚠️  决策事件数超过最大值 {SIMULATION_CONFIG['max_decision_events']}，终止事件循环")
                    break

            # 执行率驱动的库存更新（核心修改）
            self._update_inventory_by_rates()

            if event_type == 'order_departure':
                # 处理订单出发：仅统计满足率，不修改库存
                result = self._process_order_departure(event_data, work_start, work_end)
                # 仅当是有效需求时，才累加总需求和满足数
                if result['is_valid_demand']:
                    day_demand += 1
                    day_satisfied += result['satisfied']

            elif event_type == 'decision':
                # 决策并执行调度（传入事件队列和计数器）
                operations, event_counter = self._make_decision(work_end, events, event_counter)
                day_operations += operations

        # 计算满足率（仅有效需求）
        satisfaction_rate = day_satisfied / day_demand if day_demand > 0 else 0

        return {
            'date': sim_date,
            'satisfied': day_satisfied,
            'total': day_demand,
            'satisfaction_rate': satisfaction_rate,
            'operations': day_operations
        }

    def _update_inventory_by_rates(self):
        """核心修改：基于率数据更新所有节点/弧的库存（无订单参与）"""
        if self.last_inventory_update_time is None or self.current_time <= self.last_inventory_update_time:
            return

        # 计算时间差（分钟）
        time_delta = (self.current_time - self.last_inventory_update_time).total_seconds() / 60
        if time_delta < SYSTEM_PARAMS['INVENTORY_UPDATE_INTERVAL']:
            return

        if SIMULATION_CONFIG['debug']:
            print(f"    [率驱动库存更新] 时间差: {time_delta:.1f}分钟, 当前时间: {self.current_time.strftime('%H:%M:%S')}")

        # 更新所有节点库存
        for node_key in list(self.inventory.keys()):
            if node_key.startswith('node_'):
                try:
                    node_id = int(node_key.split('_')[1])
                    # 计算节点库存变动
                    node_change, _ = calculate_inventory_change_by_rate(
                        node_id=node_id,
                        arc_id=None,
                        current_time=self.current_time,
                        time_delta_minutes=time_delta,
                        data_loader=self.data
                    )

                    if node_change != 0:
                        # 应用库存变动（取整，确保非负）
                        new_inv = max(0, self.inventory[node_key] + int(round(node_change)))
                        # 检查容量限制
                        capacity = self.data.node_capacities.get(node_key, SYSTEM_PARAMS['NODE_CAPACITY_DEFAULT'])
                        new_inv = min(new_inv, capacity)

                        change_amount = new_inv - self.inventory[node_key]
                        if change_amount != 0:
                            self.inventory[node_key] = new_inv
                            self._record_node_inventory_change(node_key, change_amount, "rate_driven")
                            if SIMULATION_CONFIG['debug']:
                                print(f"      节点{node_key}: 率驱动变动 {change_amount}辆 → 库存{self.inventory[node_key]} (容量{capacity})")
                except Exception as e:
                    if SIMULATION_CONFIG['debug']:
                        print(f"      节点{node_key}库存更新失败: {e}")
                    continue

        # 更新所有弧库存
        for arc_key in list(self.inventory.keys()):
            if arc_key.startswith('arc_'):
                try:
                    arc_id = arc_key.split('_')[1]
                    # 计算弧库存变动
                    _, arc_change = calculate_inventory_change_by_rate(
                        node_id=None,
                        arc_id=arc_id,
                        current_time=self.current_time,
                        time_delta_minutes=time_delta,
                        data_loader=self.data
                    )

                    if arc_change != 0:
                        # 应用库存变动（取整，确保非负，弧容量无限）
                        new_inv = max(0, self.inventory[arc_key] + int(round(arc_change)))
                        change_amount = new_inv - self.inventory[arc_key]
                        if change_amount != 0:
                            self.inventory[arc_key] = new_inv
                            self._record_arc_inventory_change(arc_key, change_amount, "rate_driven")
                            if SIMULATION_CONFIG['debug']:
                                print(f"      弧{arc_key}: 率驱动变动 {change_amount}辆 → 库存{self.inventory[arc_key]} (容量无限)")
                except Exception as e:
                    if SIMULATION_CONFIG['debug']:
                        print(f"      弧{arc_key}库存更新失败: {e}")
                    continue

        # 更新最后库存更新时间
        self.last_inventory_update_time = self.current_time

    def _is_valid_demand(self, order, work_start, work_end):
        """判断订单是否为有效需求：11:00-15:30 + 从节点出发"""
        # 1. 检查订单时间是否在工作时间内
        start_time = order.get('start_time')
        if not start_time or pd.isna(start_time):
            return False
        if start_time < work_start or start_time > work_end:
            return False

        # 2. 检查是否有有效出发节点
        start_node_ids = self._extract_node_ids(order.get('start_node_ids', ''))
        if not start_node_ids:
            return False

        return True

    def _extract_arc_ids(self, arc_str):
        """解析多弧段ID（支持分号分隔，如(3,8);(5,8);(8,14)）"""
        if pd.isna(arc_str):
            return []

        try:
            # 清理字符串
            arc_str = str(arc_str).replace('[', '').replace(']', '').replace(' ', '')
            # 按分号分割多个弧段
            arc_parts = arc_str.split(';')
            parsed_arcs = []

            for arc in arc_parts:
                if arc and arc.strip():
                    std_arc_id = self.data.network.parse_arc_id(arc.strip())
                    if std_arc_id:
                        parsed_arcs.append(std_arc_id)

            if SIMULATION_CONFIG['debug'] and parsed_arcs:
                print(f"    解析多弧段成功: {arc_str} → {parsed_arcs}")
            return parsed_arcs

        except Exception as e:
            if SIMULATION_CONFIG['debug']:
                print(f"解析多弧段ID失败: {arc_str}, 错误: {e}")
            return []

    def _process_order_departure(self, order, work_start, work_end):
        """核心修改：订单仅统计满足率，不修改库存"""
        # 第一步：判断是否为有效需求
        is_valid_demand = self._is_valid_demand(order, work_start, work_end)
        satisfied = 0

        # 第二步：仅检查库存是否满足（不扣减）
        if is_valid_demand:
            start_node_ids = self._extract_node_ids(order.get('start_node_ids', ''))
            if start_node_ids:
                start_node_loc = f"node_{start_node_ids[0]}"
                current_inv = self.inventory.get(start_node_loc, 0)
                # 仅判断是否满足，不修改库存
                satisfied = 1 if current_inv > 0 else 0

                if SIMULATION_CONFIG['debug']:
                    status = "满足" if satisfied else "不满足"
                    print(f"    [有效需求] 订单{status}：节点{start_node_loc}，当前库存{current_inv} (容量={self.data.node_capacities.get(start_node_loc, '默认')})")

        return {
            'satisfied': satisfied,  # 是否满足（仅判断，不扣减）
            'is_valid_demand': is_valid_demand  # 是否为有效需求
        }

    def _extract_node_ids(self, node_str):
        """解析节点ID"""
        if pd.isna(node_str):
            return []
        try:
            node_str = str(node_str).replace('[', '').replace(']', '').replace(' ', '')
            nodes = node_str.split(';')
            return [int(float(node)) for node in nodes if node]
        except Exception as e:
            if SIMULATION_CONFIG['debug']:
                print(f"解析节点ID失败: {node_str}, 错误: {e}")
            return []

    def _make_decision(self, work_end, events, event_counter):
        """做出调度决策（防死循环+到达节点决策）"""
        if self.current_time >= work_end:
            return 0, event_counter

        # 防死循环：同一节点短时间内不重复决策
        if (self.last_decision_node == self.truck_node and
                self.last_decision_time is not None and
                (self.current_time - self.last_decision_time).total_seconds() / 60 < SYSTEM_PARAMS[
                    'DECISION_COOLDOWN']):
            if SIMULATION_CONFIG['debug']:
                print(f"    ⚠️  冷却时间内，跳过节点{self.truck_node}的重复决策")
            # 延迟添加下一次决策事件
            next_decision_time = self.current_time + timedelta(minutes=SYSTEM_PARAMS['DECISION_COOLDOWN'])
            if next_decision_time <= work_end:
                event_counter += 1
                heapq.heappush(events, (next_decision_time, event_counter, 'decision', None))
            return 0, event_counter

        # 记录本次决策信息
        self.last_decision_time = self.current_time
        self.last_decision_node = self.truck_node

        current_node = self.truck_node
        truck_load = self.truck_load
        node_key = f"node_{current_node}"
        current_inv = self.inventory.get(node_key, 0)
        operations = 0
        total_time = 0.0  # 所有操作总耗时（分钟）
        node_operation_time = SYSTEM_PARAMS['NODE_OPERATION_TIME_PER_BIKE']

        # 1. 库存决策（仅节点装卸，基于率数据）
        try:
            inv_decision = self.strategy.compute_inventory_decision(
                current_node, current_inv, truck_load, self.current_time
            )
        except Exception as e:
            if SIMULATION_CONFIG['debug']:
                print(f"库存决策失败: {e}")
            inv_decision = 0

        # 节点操作（装车/卸车）
        if inv_decision > 0:  # 节点装车
            actual = min(inv_decision, current_inv)
            if actual > 0:
                if self.inventory[node_key] >= actual:
                    self.inventory[node_key] -= actual
                    self.truck_load += actual
                    operations += actual
                    # 计算操作时间并累加
                    operation_time = actual * node_operation_time + SYSTEM_PARAMS['FIXED_SETUP_TIME']
                    total_time += operation_time
                    # 记录节点库存变动
                    self._record_node_inventory_change(node_key, -actual, "truck_load")
                    if SIMULATION_CONFIG['debug']:
                        print(
                            f"    在节点{current_node}装车{actual}辆，耗时{operation_time:.1f}分钟，车辆负载: {self.truck_load} (节点容量={self.data.node_capacities.get(node_key, '默认')})")

        elif inv_decision < 0:  # 节点卸车
            actual = min(abs(inv_decision), truck_load)
            if actual > 0:
                node_capacity = self.data.node_capacities.get(node_key, SYSTEM_PARAMS['NODE_CAPACITY_DEFAULT'])
                if self.inventory[node_key] + actual <= node_capacity:
                    self.inventory[node_key] += actual
                    self.truck_load -= actual
                    operations += actual
                    # 计算操作时间并累加
                    operation_time = actual * node_operation_time + SYSTEM_PARAMS['FIXED_SETUP_TIME']
                    total_time += operation_time
                    # 记录节点库存变动
                    self._record_node_inventory_change(node_key, +actual, "truck_unload")
                    if SIMULATION_CONFIG['debug']:
                        print(
                            f"    在节点{current_node}卸车{actual}辆，耗时{operation_time:.1f}分钟，车辆负载: {self.truck_load} (节点容量={node_capacity})")
                elif SIMULATION_CONFIG['debug']:
                    print(
                        f"    节点{current_node}容量不足，无法卸车{actual}辆 (当前库存={self.inventory[node_key]}, 容量={node_capacity})")

        # 2. 路由决策（基于率数据）
        next_node = self.strategy.compute_routing_decision(
            current_node, self.truck_load, inv_decision,
            self.inventory, self.current_time
        )

        if next_node == current_node:
            if SIMULATION_CONFIG['debug']:
                print(f"    车辆停留在节点{current_node}")
            # 更新当前时间（即使停留也需要计算节点操作时间）
            self.current_time += timedelta(minutes=total_time)

            # 防死循环：停留时延长决策间隔
            cooldown = max(SYSTEM_PARAMS['DECISION_COOLDOWN'], SYSTEM_PARAMS['DECISION_INTERVAL'])
            next_decision_time = self.current_time + timedelta(minutes=cooldown)

            if next_decision_time <= work_end:
                # 检查是否已有相同时间的决策事件
                has_duplicate = any(
                    evt[0] == next_decision_time and evt[2] == 'decision'
                    for evt in events
                )
                if not has_duplicate:
                    event_counter += 1
                    heapq.heappush(events, (next_decision_time, event_counter, 'decision', None))
                    if SIMULATION_CONFIG['debug']:
                        print(
                            f"    停留节点{current_node}，冷却{cooldown}分钟，下次决策时间: {next_decision_time.strftime('%H:%M:%S')}")
            return operations, event_counter

        # 3. 途中决策（仅收集弧上车辆，基于率数据）
        try:
            enroute_decision = self.strategy.compute_enroute_decision(
                current_node, next_node, self.truck_load, self.inventory
            )
        except Exception as e:
            if SIMULATION_CONFIG['debug']:
                print(f"途中决策失败: {e}")
            enroute_decision = {'collect': 0, 'unload': 0, 'arc_id': None}

        arc_id = enroute_decision['arc_id']
        arc_key = f"arc_{arc_id}" if arc_id else None

        # 弧收集操作
        if arc_key and arc_key in self.inventory:
            # 仅收集弧上车辆
            max_collect = enroute_decision['collect']
            if max_collect > 0:
                available = self.inventory.get(arc_key, 0)
                actual_collect = min(max_collect, available, SYSTEM_PARAMS['TRUCK_CAPACITY'] - self.truck_load)
                if actual_collect > 0:
                    # 扣减弧库存
                    self.inventory[arc_key] -= actual_collect
                    self.truck_load += actual_collect
                    operations += actual_collect
                    # 计算收集时间并累加
                    collection_time = actual_collect * SYSTEM_PARAMS['ARC_COLLECTION_TIME_PER_BIKE']
                    total_time += collection_time
                    # 记录弧库存变动
                    self._record_arc_inventory_change(arc_key, -actual_collect, "truck_collect")
                    if SIMULATION_CONFIG['debug']:
                        print(
                            f"    在弧段{arc_key}收集{actual_collect}辆，耗时{collection_time:.1f}分钟，弧库存: {self.inventory.get(arc_key, 0)}, 车辆负载: {self.truck_load} (弧容量无限)")

        # 计算行驶时间并累加
        travel_time = self.data.network.get_travel_time(current_node, next_node)
        total_time += travel_time
        if SIMULATION_CONFIG['debug']:
            print(f"    车辆从节点{current_node}行驶到节点{next_node}，预计耗时{travel_time:.1f}分钟")

        # 更新当前时间（所有操作总耗时）
        self.current_time += timedelta(minutes=total_time)
        if SIMULATION_CONFIG['debug']:
            print(f"    所有操作总耗时{total_time:.1f}分钟，当前时间更新为{self.current_time.strftime('%H:%M:%S')}")

        # 更新卡车位置
        self.truck_node = next_node
        if SIMULATION_CONFIG['debug']:
            print(f"    车辆到达节点{next_node}")

        # 到达新节点后，添加新的决策事件（防死循环+重复检查）
        if self.current_time <= work_end:
            # 检查是否已有相同时间的决策事件
            has_duplicate = any(
                evt[0] == self.current_time and evt[2] == 'decision'
                for evt in events
            )
            if not has_duplicate:
                event_counter += 1
                heapq.heappush(events, (self.current_time, event_counter, 'decision', None))
                if SIMULATION_CONFIG['debug']:
                    print(f"    到达节点{next_node}，添加新的决策事件（时间: {self.current_time.strftime('%H:%M:%S')}）")

        return operations, event_counter

    def _record_node_inventory_change(self, node_key, change, operation_type):
        """记录节点库存变动（适配可视化）"""
        try:
            node_id = int(node_key.split('_')[1])
            current_inv = self.inventory.get(node_key, 0)
            self.node_inventory_history.append({
                'time': self.current_time,
                'node_id': node_id,
                'inventory': current_inv,
                'change': change,
                'change_type': operation_type
            })
        except Exception as e:
            if SIMULATION_CONFIG['debug']:
                print(f"记录节点库存变动失败: {e}")

    def _record_arc_inventory_change(self, arc_key, change, operation_type):
        """记录弧库存变动（适配可视化）"""
        try:
            arc_id = arc_key.split('_')[1]
            current_inv = self.inventory.get(arc_key, 0)
            self.arc_inventory_history.append({
                'time': self.current_time,
                'arc_id': arc_id,
                'inventory': current_inv,
                'change': change,
                'change_type': operation_type
            })
        except Exception as e:
            if SIMULATION_CONFIG['debug']:
                print(f"记录弧库存变动失败: {e}")

    def _record_truck_movement(self, from_node, to_node, load, operation_type):
        """记录卡车移动（新增）"""
        self.truck_movement_history.append({
            'time': self.current_time,
            'from_node': from_node,
            'to_node': to_node,
            'load': load,
            'operation_type': operation_type
        })

    def _print_inventory_history(self, sim_date):
        """打印当日库存变动"""
        print(f"\n  当日({sim_date})节点库存变动记录 (率驱动为主):")
        for node_key, changes in self.node_inventory_history.items():
            if changes:
                print(f"    {node_key} (容量={self.data.node_capacities.get(node_key, '默认')}):")
                for change in changes[:3]:  # 仅打印前3条
                    print(
                        f"      时间: {change['time'].strftime('%H:%M:%S')}, 操作: {change['operation']}, 变动: {change['change']}, 当前库存: {change['current_inventory']}")
                if len(changes) > 3:
                    print(f"      ... 共{len(changes)}条变动记录")

        print(f"\n  当日({sim_date})弧段库存变动记录 (率驱动为主):")
        for arc_key, changes in self.arc_inventory_history.items():
            if changes:
                print(f"    {arc_key} (容量无限):")
                for change in changes[:3]:  # 仅打印前3条
                    print(
                        f"      时间: {change['time'].strftime('%H:%M:%S')}, 操作: {change['operation']}, 变动: {change['change']}, 当前库存: {change['current_inventory']}")
                if len(changes) > 3:
                    print(f"      ... 共{len(changes)}条变动记录")


# ======================== 策略验证函数（无核心修改）========================
def validate_pfa_for_date_range(pfa_params=None, compare_with_do_nothing=True,
                                date_range_start='2024-05-27', date_range_end='2024-06-14'):
    """验证PFA策略主函数"""
    print("=" * 60)
    print("共享单车调度策略验证 (率驱动库存，订单仅统计满足率)")
    print("需求定义：仅11:00-15:30从节点出发的订单算作需求")
    print("库存规则：纯基于arrival_departure_rates.xlsx的5分钟粒度率数据更新")
    print("核心变更：无还车逻辑，Mover动作仅基于率数据")
    print("=" * 60)

    print("\n检查数据文件...")
    for file_key, file_path in DATA_PATHS.items():
        if os.path.exists(file_path):
            print(f"  ✓ {file_key}: {file_path}")
        else:
            print(f"  ✗ {file_key}: {file_path} (文件不存在)")

    print("\n1. 加载数据...")
    try:
        data_loader = DataLoader()
        print("数据加载成功")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return {}

    results = {}

    print("\n2. 验证PFA策略...")
    if pfa_params is None:
        pfa_params = PFA_PARAMS

    pfa_strategy = PFAStrategy(
        params=pfa_params,
        network=data_loader.network,
        data_loader=data_loader
    )

    pfa_simulator = BikeSharingSimulator(
        data_loader=data_loader,
        strategy=pfa_strategy,
        strategy_name="PFA Strategy (率驱动库存)"
    )

    pfa_results = pfa_simulator.run_for_date_range(
        params=pfa_params,
        date_range_start=date_range_start,
        date_range_end=date_range_end
    )
    results['pfa'] = pfa_results

    if compare_with_do_nothing:
        print("\n3. 验证不作为策略（基准）...")
        do_nothing_strategy = DoNothingStrategy(data_loader=data_loader)
        dn_simulator = BikeSharingSimulator(
            data_loader=data_loader,
            strategy=do_nothing_strategy,
            strategy_name="Do-Nothing Strategy"
        )

        dn_results = dn_simulator.run_for_date_range(
            date_range_start=date_range_start,
            date_range_end=date_range_end
        )
        results['do_nothing'] = dn_results

        if len(dn_results['daily_stats']) > 0:
            print("\n4. 策略比较 (仅11:00-15:30节点出发订单):")
            print("-" * 40)
            pfa_rate = pfa_results['satisfaction_rate']
            dn_rate = dn_results['satisfaction_rate']

            print(f"  不作为策略满足率: {dn_rate:.2%}")
            print(f"  PFA策略满足率: {pfa_rate:.2%}")

            # 计算满足率差值并标注提升/下降
            rate_diff = pfa_rate - dn_rate
            if rate_diff > 0:
                print(f"  PFA策略相对不作为策略提升: {rate_diff:.2%}")
            elif rate_diff < 0:
                print(f"  PFA策略相对不作为策略下降: {abs(rate_diff):.2%}")
            else:
                print(f"  PFA策略与不作为策略满足率持平")

            # 统计两种策略覆盖的订单量（假设daily_stats包含order_count字段，可根据实际字段名调整）
            # 遍历daily_stats列表，累加所有日期的order_count
            dn_order_count = sum(day.get('order_count', 0) for day in dn_results['daily_stats'])
            # 先获取daily_stats列表（为空则返回空列表），再遍历求和
            daily_stats = pfa_results.get('daily_stats', [])
            pfa_order_count = sum(day.get('order_count', 0) for day in daily_stats)

            print(f"\n  订单量统计:")
            print(f"    不作为策略覆盖订单数: {dn_order_count:,}")
            print(f"    PFA策略覆盖订单数: {pfa_order_count:,}")

            # 计算满足订单数的绝对差异（更直观的业务效果）
            if dn_order_count > 0:
                dn_satisfied = int(dn_rate * dn_order_count)
                pfa_satisfied = int(pfa_rate * pfa_order_count)
                satisfied_diff = pfa_satisfied - dn_satisfied

                print(f"  满足订单数统计:")
                print(f"    不作为策略满足订单数: {dn_satisfied:,}")
                print(f"    PFA策略满足订单数: {pfa_satisfied:,}")

                if satisfied_diff > 0:
                    print(f"    PFA策略多满足订单数: {satisfied_diff:,}")
                elif satisfied_diff < 0:
                    print(f"    PFA策略少满足订单数: {abs(satisfied_diff):,}")
                else:
                    print(f"    两种策略满足订单数相同")

            print("-" * 40)


# ======================== 新增：可视化核心函数 ========================
def plot_node_inventory_dynamics(simulator, target_date=date(2024, 5, 27), selected_nodes=None):
    """
    绘制指定日期各节点库存动态变化曲线
    :param simulator: 仿真器实例
    :param target_date: 目标日期
    :param selected_nodes: 可选，指定要展示的节点列表，None则展示所有节点
    """
    # 筛选目标日期的节点库存数据
    date_filtered = [
        rec for rec in simulator.node_inventory_history
        if rec['time'].date() == target_date
    ]

    if not date_filtered:
        print(f"⚠️  无{target_date}的节点库存数据")
        return

    # 转换为DataFrame便于处理
    df = pd.DataFrame(date_filtered)

    # 筛选指定节点
    if selected_nodes:
        df = df[df['node_id'].isin(selected_nodes)]

    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 8))

    # 按节点分组绘制曲线
    for node_id, group in df.groupby('node_id'):
        # 按时间排序
        group = group.sort_values('time')
        # 获取节点容量
        capacity = simulator.data.node_capacities.get(f"node_{node_id}", SYSTEM_PARAMS['NODE_CAPACITY_DEFAULT'])

        # 绘制库存曲线
        ax.plot(group['time'], group['inventory'],
                label=f'节点{node_id} (容量:{capacity})',
                linewidth=2, marker='o', markersize=3)

        # 绘制容量上限线
        ax.axhline(y=capacity, color=ax.lines[-1].get_color(),
                   linestyle='--', alpha=0.5, linewidth=1)

    # 设置时间格式
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(MinuteLocator(interval=30))
    plt.xticks(rotation=45)

    # 设置标签和标题
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('库存数量（辆）', fontsize=12)
    ax.set_title(f'{target_date} 节点库存动态变化', fontsize=14, fontweight='bold')

    # 添加网格和图例
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f'node_inventory_dynamics_{target_date}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_truck_movement(simulator, target_date=date(2024, 5, 27)):
    """
    绘制卡车移动轨迹和负载变化
    :param simulator: 仿真器实例
    :param target_date: 目标日期
    """
    # 筛选目标日期的卡车数据
    date_filtered = [
        rec for rec in simulator.truck_movement_history
        if rec['time'].date() == target_date
    ]

    if not date_filtered:
        print(f"⚠️  无{target_date}的卡车移动数据")
        return

    df = pd.DataFrame(date_filtered)
    df = df.sort_values('time')

    # 创建双轴图
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 绘制卡车负载变化（左轴）
    ax1.plot(df['time'], df['load'], color='red', linewidth=3, marker='s',
             label='卡车负载', markersize=5)
    ax1.axhline(y=SYSTEM_PARAMS['TRUCK_CAPACITY'], color='red',
                linestyle='--', alpha=0.7, label=f'卡车容量({SYSTEM_PARAMS["TRUCK_CAPACITY"]})')
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('卡车负载（辆）', fontsize=12, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, alpha=0.3)

    # 绘制节点转移（右轴）
    ax2 = ax1.twinx()
    node_sequence = [rec['from_node'] for rec in df] + [df.iloc[-1]['to_node']]
    time_sequence = list(df['time']) + [df.iloc[-1]['time'] + timedelta(minutes=10)]

    ax2.step(time_sequence, node_sequence, color='blue', linewidth=2,
             where='post', label='当前节点', marker='o')
    ax2.set_ylabel('卡车所在节点', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # 设置时间格式
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(MinuteLocator(interval=30))
    plt.xticks(rotation=45)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc='upper left')

    # 设置标题
    plt.title(f'{target_date} 卡车移动轨迹与负载变化', fontsize=14, fontweight='bold')

    # 保存并显示
    plt.tight_layout()
    plt.savefig(f'truck_movement_{target_date}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_inventory_change_type(simulator, target_date=date(2024, 5, 27)):
    """
    绘制库存变动类型统计（率驱动/卡车操作）
    :param simulator: 仿真器实例
    :param target_date: 目标日期
    """
    # 筛选目标日期的节点库存变动
    date_filtered = [
        rec for rec in simulator.node_inventory_history
        if rec['time'].date() == target_date and rec['change'] != 0
    ]

    if not date_filtered:
        print(f"⚠️  无{target_date}的库存变动数据")
        return

    df = pd.DataFrame(date_filtered)

    # 统计变动类型
    change_stats = df.groupby(['change_type', 'node_id'])['change'].agg(['sum', 'count']).reset_index()
    change_stats['abs_change'] = change_stats['sum'].abs()

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：各类型总变动量
    type_totals = change_stats.groupby('change_type')['abs_change'].sum()
    colors = {'rate_driven': '#2E86AB', 'truck_load': '#A23B72', 'truck_unload': '#F18F01'}
    ax1.pie(type_totals.values, labels=type_totals.index, autopct='%1.1f%%',
            colors=[colors.get(t, '#8B8B8B') for t in type_totals.index],
            startangle=90)
    ax1.set_title('库存变动类型占比（总变动量）', fontsize=12, fontweight='bold')

    # 右图：各节点变动次数
    node_counts = change_stats.groupby('node_id')['count'].sum().sort_values(ascending=False)[:10]
    ax2.bar(node_counts.index.astype(str), node_counts.values, color='#4CAF50', alpha=0.8)
    ax2.set_xlabel('节点ID', fontsize=12)
    ax2.set_ylabel('变动次数', fontsize=12)
    ax2.set_title('Top10 变动最频繁节点', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 整体标题
    fig.suptitle(f'{target_date} 库存变动类型分析', fontsize=14, fontweight='bold')

    # 保存并显示
    plt.tight_layout()
    plt.savefig(f'inventory_change_type_{target_date}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_satisfaction_rate_comparison(results):
    """
    绘制PFA策略与不作为策略的满足率对比
    :param results: 仿真结果字典
    """
    if 'pfa' not in results or 'do_nothing' not in results:
        print("⚠️  缺少策略对比数据")
        return

    # 提取每日满足率
    pfa_daily = results['pfa']['daily_stats']
    dn_daily = results['do_nothing']['daily_stats']

    # 对齐日期
    dates = [d['date'] for d in pfa_daily]
    pfa_rates = [d['satisfaction_rate'] for d in pfa_daily]
    dn_rates = [d['satisfaction_rate'] for d in dn_daily[:len(pfa_rates)]]

    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 7))

    # 绘制对比曲线
    x = range(len(dates))
    ax.plot(x, pfa_rates, 'o-', linewidth=2, markersize=6,
            color='#2196F3', label='PFA策略')
    ax.plot(x, dn_rates, 's-', linewidth=2, markersize=6,
            color='#FF5722', label='不作为策略')

    # 设置标签
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('需求满足率', fontsize=12)
    ax.set_title('PFA策略 vs 不作为策略 每日满足率对比', fontsize=14, fontweight='bold')

    # 设置x轴刻度
    ax.set_xticks(x[::2])  # 每2天显示一个刻度，避免拥挤
    ax.set_xticklabels([str(dates[i]) for i in x[::2]], rotation=45)

    # 添加数值标签（关键节点）
    for i in range(0, len(x), 5):  # 每5天显示一次数值
        ax.text(i, pfa_rates[i], f'{pfa_rates[i]:.1%}',
                ha='center', va='bottom', fontsize=9, color='#2196F3')
        ax.text(i, dn_rates[i], f'{dn_rates[i]:.1%}',
                ha='center', va='top', fontsize=9, color='#FF5722')

    # 添加网格和图例
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # 保存并显示
    plt.tight_layout()
    plt.savefig('satisfaction_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def generate_visualization_report(simulator_pfa, simulator_dn=None, results=None, target_date=date(2024, 5, 27)):
    """
    生成完整的可视化报告
    :param simulator_pfa: PFA策略仿真器
    :param simulator_dn: 不作为策略仿真器（可选）
    :param results: 仿真结果（可选）
    :param target_date: 重点分析日期
    """
    print(f"\n📊 生成{target_date}可视化报告...")

    # 1. 节点库存动态变化
    print("  - 绘制节点库存动态变化图")
    plot_node_inventory_dynamics(simulator_pfa, target_date=target_date,
                                 selected_nodes=[2, 6, 8, 10])  # 重点节点

    # 2. 卡车移动轨迹
    print("  - 绘制卡车移动轨迹图")
    plot_truck_movement(simulator_pfa, target_date=target_date)

    # 3. 库存变动类型分析
    print("  - 绘制库存变动类型统计图")
    plot_inventory_change_type(simulator_pfa, target_date=target_date)

    # 4. 策略满足率对比（如果有数据）
    if results and 'do_nothing' in results:
        print("  - 绘制策略满足率对比图")
        plot_satisfaction_rate_comparison(results)

    print("✅ 可视化报告生成完成！")

# ======================== 主函数（程序入口）========================
def main():
    """
    程序主入口：
    1. 配置仿真参数
    2. 执行策略验证
    3. 保存仿真结果
    4. 生成可视化报告
    5. 输出汇总报告
    """
    # -------------------------- 1. 基础配置 --------------------------
    # 可自定义的仿真参数
    CONFIG = {
        'DATE_RANGE_START': '2024-05-27',  # 仿真开始日期
        'DATE_RANGE_END': '2024-06-14',    # 仿真结束日期
        'COMPARE_WITH_DO_NOTHING': True,   # 是否对比不作为策略
        'SAVE_RESULTS': True,              # 是否保存结果到JSON文件
        'RESULT_SAVE_PATH': 'simulation_results.json',  # 结果保存路径
        'SHOW_DETAILED_REPORT': True,      # 是否展示详细日报表
        'GENERATE_VISUALIZATION': True,    # 是否生成可视化报告
        'VISUALIZATION_TARGET_DATE': date(2024, 5, 27)  # 可视化重点分析日期
    }

    print("=" * 80)
    print("🚲 共享单车调度仿真系统 (率驱动库存版 + 可视化)")
    print(f"📅 仿真时间范围: {CONFIG['DATE_RANGE_START']} ~ {CONFIG['DATE_RANGE_END']}")
    print(f"📊 可视化分析日期: {CONFIG['VISUALIZATION_TARGET_DATE']}")
    print("=" * 80)

    # -------------------------- 2. 执行策略验证 --------------------------
    try:
        # 执行PFA策略验证（含可选的不作为策略对比）
        simulation_results = validate_pfa_for_date_range(
            pfa_params=PFA_PARAMS,
            compare_with_do_nothing=CONFIG['COMPARE_WITH_DO_NOTHING'],
            date_range_start=CONFIG['DATE_RANGE_START'],
            date_range_end=CONFIG['DATE_RANGE_END']
        )

        # 无有效结果时退出
        if not simulation_results:
            print("\n❌ 仿真执行失败，未生成有效结果")
            return

    except Exception as e:
        print(f"\n❌ 仿真过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # -------------------------- 3. 保存仿真结果 --------------------------
    if CONFIG['SAVE_RESULTS']:
        try:
            # 自定义JSON序列化器（处理datetime/date对象）
            def serialize(obj):
                if isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"无法序列化 {type(obj)} 类型")

            # 构造保存的数据结构
            save_data = {
                'simulation_config': CONFIG,
                'pfa_params': PFA_PARAMS,
                'system_params': SYSTEM_PARAMS,
                'fixed_node_capacities': FIXED_NODE_CAPACITIES,
                'results': simulation_results,
                'run_time': datetime.now().isoformat()
            }

            # 保存到JSON文件
            with open(CONFIG['RESULT_SAVE_PATH'], 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=serialize)

            print(f"\n✅ 仿真结果已保存至: {os.path.abspath(CONFIG['RESULT_SAVE_PATH'])}")

        except Exception as e:
            print(f"\n⚠️  结果保存失败: {e}")

    # -------------------------- 4. 生成可视化报告 --------------------------
    if CONFIG['GENERATE_VISUALIZATION'] and 'pfa' in simulation_results:
        # 获取PFA仿真器实例（需要修改validate_pfa_for_date_range函数返回仿真器）
        # 注：需确保validate_pfa_for_date_range函数返回simulator实例
        generate_visualization_report(
            simulator_pfa=simulation_results['pfa_simulator'],
            simulator_dn=simulation_results.get('dn_simulator'),
            results=simulation_results,
            target_date=CONFIG['VISUALIZATION_TARGET_DATE']
        )

    # -------------------------- 5. 输出详细日报表（可选）--------------------------
    if CONFIG['SHOW_DETAILED_REPORT'] and simulation_results:
        print("\n" + "-" * 80)
        print("📊 每日仿真结果详情 (仅11:00-15:30节点出发订单)")
        print("-" * 80)

        # 整理每日数据（对齐PFA和不作为策略）
        pfa_daily = simulation_results.get('pfa', {}).get('daily_stats', [])
        dn_daily = simulation_results.get('do_nothing', {}).get('daily_stats', []) if CONFIG['COMPARE_WITH_DO_NOTHING'] else []

        # 打印表头
        headers = ["日期", "PFA满足率", "PFA满足数", "PFA总需求", "不作为满足率", "不作为满足数", "不作为总需求"] if CONFIG['COMPARE_WITH_DO_NOTHING'] else ["日期", "PFA满足率", "PFA满足数", "PFA总需求"]
        print(f"{'| '.join([h.ljust(12) for h in headers])}")
        print("-" * 80)

        # 打印每日数据
        max_days = max(len(pfa_daily), len(dn_daily))
        for i in range(max_days):
            # PFA数据
            pfa_day = pfa_daily[i] if i < len(pfa_daily) else {'date': '-', 'satisfaction_rate': 0, 'satisfied': 0, 'total': 0}
            pfa_date = pfa_day['date'].isoformat() if isinstance(pfa_day['date'], date) else pfa_day['date']
            pfa_rate = f"{pfa_day['satisfaction_rate']:.2%}".ljust(12)
            pfa_sat = f"{pfa_day['satisfied']:,}".ljust(12)
            pfa_total = f"{pfa_day['total']:,}".ljust(12)

            # 不作为策略数据
            if CONFIG['COMPARE_WITH_DO_NOTHING']:
                dn_day = dn_daily[i] if i < len(dn_daily) else {'satisfaction_rate': 0, 'satisfied': 0, 'total': 0}
                dn_rate = f"{dn_day['satisfaction_rate']:.2%}".ljust(12)
                dn_sat = f"{dn_day['satisfied']:,}".ljust(12)
                dn_total = f"{dn_day['total']:,}".ljust(12)
                row = f"{pfa_date.ljust(12)}| {pfa_rate}| {pfa_sat}| {pfa_total}| {dn_rate}| {dn_sat}| {dn_total}"
            else:
                row = f"{pfa_date.ljust(12)}| {pfa_rate}| {pfa_sat}| {pfa_total}"

            print(row)

        print("-" * 80)

    # -------------------------- 6. 最终汇总 --------------------------
    print("\n" + "=" * 80)
    print("🏁 仿真执行完成 - 核心指标汇总")
    print("=" * 80)

    # PFA策略汇总
    pfa_res = simulation_results.get('pfa', {})
    print(f"📈 PFA策略:")
    print(f"   总满足率: {pfa_res.get('satisfaction_rate', 0):.2%}")
    print(f"   总满足订单: {pfa_res.get('total_satisfied', 0):,}")
    print(f"   总有效需求: {pfa_res.get('total_demand', 0):,}")
    print(f"   总调度操作: {pfa_res.get('total_operations', 0):,}")

    # 不作为策略汇总（可选）
    if CONFIG['COMPARE_WITH_DO_NOTHING']:
        dn_res = simulation_results.get('do_nothing', {})
        print(f"\n📉 不作为策略:")
        print(f"   总满足率: {dn_res.get('satisfaction_rate', 0):.2%}")
        print(f"   总满足订单: {dn_res.get('total_satisfied', 0):,}")
        print(f"   总有效需求: {dn_res.get('total_demand', 0):,}")
        print(f"   总调度操作: {dn_res.get('total_operations', 0):,}")

        # 计算相对提升
        rate_diff = pfa_res.get('satisfaction_rate', 0) - dn_res.get('satisfaction_rate', 0)
        sat_diff = pfa_res.get('total_satisfied', 0) - dn_res.get('total_satisfied', 0)
        print(f"\n📊 策略对比:")
        print(f"   满足率相对提升: {rate_diff:.2%}")
        print(f"   满足订单数绝对提升: {sat_diff:,} 单")

    print("\n✅ 程序执行完毕！")


# ======================== 程序入口 ========================
if __name__ == "__main__":
    # 设置随机种子（保证仿真可复现）
    random.seed(2025)
    np.random.seed(2025)

    # 执行主函数
    main()
