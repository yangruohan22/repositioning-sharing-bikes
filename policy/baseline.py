#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @ Time:2025/12/1 5:26
# @ Author:86155
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
import math
import random
from datetime import datetime, timedelta
import warnings

# 添加 Optuna 优化库
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')


def load_and_process_data():
    """加载和处理所有数据"""
    print("正在加载数据...")

    # 1. 节点容量
    node_capacities = {
        1: 54, 2: 60, 3: 65, 4: 50, 5: 59, 6: 158, 7: 71, 8: 135,
        9: 57, 10: 139, 11: 31, 12: 71, 13: 65, 14: 108, 15: 112
    }

    # 2. 读取到达/离开率数据
    try:
        node_departure_df = pd.read_excel('arrival_departure_rates.xlsx', sheet_name='节点_离开率')
        node_arrival_df = pd.read_excel('arrival_departure_rates.xlsx', sheet_name='节点_到达率')
        arc_rates_df = pd.read_excel('arrival_departure_rates.xlsx', sheet_name='弧段')

        print(f"节点离开率数据形状: {node_departure_df.shape}")
        print(f"节点到达率数据形状: {node_arrival_df.shape}")
        print(f"弧段数据形状: {arc_rates_df.shape}")

    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return None

    # 3. 读取弧段长度数据
    try:
        arc_lengths_df = pd.read_csv('arc_lengths(1).csv')
        print(f"弧段长度数据形状: {arc_lengths_df.shape}")

        # 创建弧段长度字典
        arc_lengths = {}
        for _, row in arc_lengths_df.iterrows():
            try:
                arc_id_str = row['arc_id']
                if isinstance(arc_id_str, str) and arc_id_str.startswith('(') and arc_id_str.endswith(')'):
                    arc_id = eval(arc_id_str)
                    length = float(row['length'])
                    arc_lengths[arc_id] = length
            except (ValueError, KeyError, SyntaxError) as e:
                print(f"处理弧段长度数据时出错，行数据: {row}, 错误: {e}")
                continue

        print(f"成功加载 {len(arc_lengths)} 个弧段的长度数据")

    except Exception as e:
        print(f"读取弧段长度文件时出错: {e}")
        return None

    # 数据清洗：处理NaN值
    node_departure_df = node_departure_df.dropna(subset=['Node_ID'])
    node_arrival_df = node_arrival_df.dropna(subset=['Node_ID'])
    arc_rates_df = arc_rates_df.dropna(subset=['Arc_ID'])

    # 处理节点需求率数据
    node_demand_rates = {}
    node_supply_rates = {}

    print("处理节点需求率数据...")
    for _, row in node_departure_df.iterrows():
        try:
            node_id = int(float(row['Node_ID']))  # 先转float再转int，处理可能的浮点数
            node_demand_rates[node_id] = {}

            for col in node_departure_df.columns[1:]:  # 时间列
                if pd.notna(row[col]):
                    node_demand_rates[node_id][col] = float(row[col])
                else:
                    node_demand_rates[node_id][col] = 0.0  # 用0填充NaN

        except (ValueError, KeyError) as e:
            print(f"处理节点离开率数据时出错，行数据: {row}, 错误: {e}")
            continue

    print("处理节点供应率数据...")
    for _, row in node_arrival_df.iterrows():
        try:
            node_id = int(float(row['Node_ID']))
            node_supply_rates[node_id] = {}

            for col in node_arrival_df.columns[1:]:  # 时间列
                if pd.notna(row[col]):
                    node_supply_rates[node_id][col] = float(row[col])
                else:
                    node_supply_rates[node_id][col] = 0.0

        except (ValueError, KeyError) as e:
            print(f"处理节点到达率数据时出错，行数据: {row}, 错误: {e}")
            continue

    # 处理弧段数据
    arc_demand_rates = {}
    arc_supply_rates = {}
    travel_times = {}

    print("处理弧段数据...")
    for _, row in arc_rates_df.iterrows():
        try:
            arc_id_str = row['Arc_ID']
            # 处理弧段ID格式
            if isinstance(arc_id_str, str) and arc_id_str.startswith('(') and arc_id_str.endswith(')'):
                arc_id = eval(arc_id_str)
            else:
                print(f"跳过无效的弧段ID: {arc_id_str}")
                continue

            demand_rate = float(row['离开率（单位：辆/分钟）']) if pd.notna(row['离开率（单位：辆/分钟）']) else 0.0
            supply_rate = float(row['到达率（单位：辆/分钟）']) if pd.notna(row['到达率（单位：辆/分钟）']) else 0.0

            arc_demand_rates[arc_id] = demand_rate
            arc_supply_rates[arc_id] = supply_rate

            # 使用弧段长度和速度计算旅行时间
            if arc_id in arc_lengths:
                length_km = arc_lengths[arc_id]
                # 速度25km/h = 25/60 km/min
                speed_km_per_min = 25 / 60
                travel_time = length_km / speed_km_per_min
                travel_times[arc_id] = travel_time
            else:
                # 如果没有长度数据，使用默认值10分钟
                travel_times[arc_id] = 10
                print(f"警告: 弧段 {arc_id} 没有长度数据，使用默认旅行时间10分钟")

        except (ValueError, KeyError, SyntaxError) as e:
            print(f"处理弧段数据时出错，行数据: {row}, 错误: {e}")
            continue

    # 3. 读取初始分布数据
    try:
        node_initial_df = pd.read_excel('node_combined_11am_bike_distribution_15x35.xlsx')
        arc_initial_df = pd.read_excel('arc_combined_11am_bike_distribution_105x35.xlsx')

        print(f"节点初始分布数据形状: {node_initial_df.shape}")
        print(f"弧段初始分布数据形状: {arc_initial_df.shape}")

    except Exception as e:
        print(f"读取初始分布数据时出错: {e}")
        return None

    # 数据清洗
    node_initial_df = node_initial_df.dropna(subset=['node_id'])
    arc_initial_df = arc_initial_df.dropna(subset=['arc_id'])

    # 处理初始分布
    initial_node_bikes = {}
    initial_arc_bikes = {}

    print("处理初始节点分布...")
    # 使用第一天的数据作为初始状态
    if len(node_initial_df.columns) > 1:
        first_date_col = node_initial_df.columns[1]
        for _, row in node_initial_df.iterrows():
            try:
                node_id = int(float(row['node_id']))
                bike_count = int(float(row[first_date_col])) if pd.notna(row[first_date_col]) else 0
                initial_node_bikes[node_id] = bike_count
            except (ValueError, KeyError) as e:
                print(f"处理节点初始分布时出错，行数据: {row}, 错误: {e}")
                continue
    else:
        print("节点初始分布数据列数不足")

    print("处理初始弧段分布...")
    if len(arc_initial_df.columns) > 1:
        first_date_col = arc_initial_df.columns[1]
        for _, row in arc_initial_df.iterrows():
            try:
                arc_id_str = row['arc_id']
                if isinstance(arc_id_str, str) and arc_id_str.startswith('(') and arc_id_str.endswith(')'):
                    arc_id = eval(arc_id_str)
                else:
                    continue

                bike_count = int(float(row[first_date_col])) if pd.notna(row[first_date_col]) else 0
                initial_arc_bikes[arc_id] = bike_count
            except (ValueError, KeyError, SyntaxError) as e:
                print(f"处理弧段初始分布时出错，行数据: {row}, 错误: {e}")
                continue
    else:
        print("弧段初始分布数据列数不足")

    # 补充缺失的弧段旅行时间
    print("补充缺失的弧段旅行时间...")
    for i in range(1, 16):
        for j in range(1, 16):
            if i != j:
                arc_id = (i, j)
                if arc_id not in travel_times:
                    if arc_id in arc_lengths:
                        # 使用实际长度计算旅行时间
                        length_km = arc_lengths[arc_id]
                        speed_km_per_min = 25 / 60
                        travel_time = length_km / speed_km_per_min
                        travel_times[arc_id] = travel_time
                    else:
                        # 使用默认值10分钟
                        travel_times[arc_id] = 10
                        print(f"警告: 弧段 {arc_id} 没有长度数据，使用默认旅行时间10分钟")

    # 验证数据完整性
    print("\n数据验证:")
    print(f"节点容量: {len(node_capacities)} 个节点")
    print(f"节点需求率: {len(node_demand_rates)} 个节点")
    print(f"节点供应率: {len(node_supply_rates)} 个节点")
    print(f"弧段需求率: {len(arc_demand_rates)} 个弧段")
    print(f"弧段供应率: {len(arc_supply_rates)} 个弧段")
    print(f"弧段长度: {len(arc_lengths)} 个弧段")
    print(f"初始节点分布: {len(initial_node_bikes)} 个节点")
    print(f"初始弧段分布: {len(initial_arc_bikes)} 个弧段")
    print(f"旅行时间: {len(travel_times)} 个弧段")

    # 打印旅行时间统计信息
    travel_time_values = list(travel_times.values())
    print(
        f"旅行时间统计 - 最小值: {min(travel_time_values):.2f}分钟, 最大值: {max(travel_time_values):.2f}分钟, 平均值: {np.mean(travel_time_values):.2f}分钟")

    return {
        'node_capacities': node_capacities,
        'travel_times': travel_times,
        'node_demand_rates': node_demand_rates,
        'node_supply_rates': node_supply_rates,
        'arc_demand_rates': arc_demand_rates,
        'arc_supply_rates': arc_supply_rates,
        'initial_node_bikes': initial_node_bikes,
        'initial_arc_bikes': initial_arc_bikes,
        'arc_lengths': arc_lengths  # 新增弧段长度数据
    }


class BaseStrategy:
    """策略基类"""

    def __init__(self, operational_params: Dict[str, Any], policy_params: Dict[str, Any], network_data: Dict[str, Any]):
        self.operational_params = operational_params
        self.policy_params = policy_params
        self.network_data = network_data
        self.node_capacities = network_data['node_capacities']

    def get_time_index(self, current_time: float) -> str:
        """将分钟时间转换为时间字符串格式"""
        start_hour = 11  # 11:00开始
        total_minutes = start_hour * 60 + current_time
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        # 确保分钟是5的倍数（匹配5分钟间隔数据）
        rounded_minutes = (minutes // 5) * 5
        return f"{hours:02d}:{rounded_minutes:02d}"

    def get_supply_rate(self, node_id: int, current_time: float) -> float:
        """获取节点供应率"""
        time_key = self.get_time_index(current_time)
        if node_id in self.network_data['node_supply_rates'] and time_key in self.network_data['node_supply_rates'][
            node_id]:
            return self.network_data['node_supply_rates'][node_id][time_key]
        return 0.0

    def get_demand_rate(self, node_id: int, current_time: float) -> float:
        """获取节点需求率"""
        time_key = self.get_time_index(current_time)
        if node_id in self.network_data['node_demand_rates'] and time_key in self.network_data['node_demand_rates'][
            node_id]:
            return self.network_data['node_demand_rates'][node_id][time_key]
        return 0.0

    def get_arc_supply_rate(self, arc_id: Tuple[int, int]) -> float:
        """获取弧段供应率"""
        return self.network_data['arc_supply_rates'].get(arc_id, 0.0)

    def get_arc_demand_rate(self, arc_id: Tuple[int, int]) -> float:
        """获取弧段需求率"""
        return self.network_data['arc_demand_rates'].get(arc_id, 0.0)

    def inventory_decision(self, current_time: float, current_node: int,
                           node_bikes: Dict[int, int], vehicle_capacity: int) -> int:
        """库存决策 - 子类需要实现"""
        raise NotImplementedError

    def routing_decision(self, current_time: float, current_node: int,
                         vehicle_capacity: int, inventory_decision: int,
                         node_bikes: Dict[int, int], arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """路径决策 - 子类需要实现"""
        raise NotImplementedError

    def en_route_decision(self, current_node: int, next_node: int,
                          post_capacity: int, arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """途中决策 - 子类需要实现"""
        raise NotImplementedError


class DoNothingStrategy(BaseStrategy):
    """Do Nothing 策略 - 车辆不进行任何调度操作"""

    def inventory_decision(self, current_time: float, current_node: int,
                           node_bikes: Dict[int, int], vehicle_capacity: int) -> int:
        """不进行库存操作"""
        return 0

    def routing_decision(self, current_time: float, current_node: int,
                         vehicle_capacity: int, inventory_decision: int,
                         node_bikes: Dict[int, int], arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """随机选择下一个节点，但不主动调度"""
        gathering_points = [i for i in range(1, 16)]
        if current_node == 0:  # 从仓库出发
            return random.choice(gathering_points)
        else:
            # 随机选择一个不同于当前节点的节点
            available_nodes = [n for n in gathering_points if n != current_node]
            return random.choice(available_nodes) if available_nodes else current_node

    def en_route_decision(self, current_node: int, next_node: int,
                          post_capacity: int, arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """不进行途中装载"""
        return 0


class RandomStrategy(BaseStrategy):
    """Random 策略 - 随机调度"""

    def inventory_decision(self, current_time: float, current_node: int,
                           node_bikes: Dict[int, int], vehicle_capacity: int) -> int:
        """随机库存决策"""
        if current_node == 0:  # 仓库不进行操作
            return 0

        current_bikes = node_bikes.get(current_node, 0)
        node_capacity = self.node_capacities.get(current_node, 0)

        # 随机决定是装车还是卸车
        decision_type = random.choice(['load', 'unload', 'none'])

        if decision_type == 'load' and current_bikes > 0:
            # 随机装载数量
            max_load = min(current_bikes, vehicle_capacity)
            return random.randint(1, max(1, max_load))
        elif decision_type == 'unload' and vehicle_capacity > 0:
            # 随机卸载数量
            max_unload = min(vehicle_capacity, node_capacity - current_bikes)
            return -random.randint(1, max(1, max_unload))
        else:
            return 0

    def routing_decision(self, current_time: float, current_node: int,
                         vehicle_capacity: int, inventory_decision: int,
                         node_bikes: Dict[int, int], arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """随机路径决策"""
        gathering_points = [i for i in range(1, 16)]
        if current_node == 0:  # 从仓库出发
            return random.choice(gathering_points)
        else:
            # 随机选择一个不同于当前节点的节点
            available_nodes = [n for n in gathering_points if n != current_node]
            return random.choice(available_nodes) if available_nodes else current_node

    def en_route_decision(self, current_node: int, next_node: int,
                          post_capacity: int, arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """随机途中决策"""
        if post_capacity > 0 and random.random() < 0.3:  # 30%概率进行途中装载
            arc_bike_count = arc_bikes.get((current_node, next_node), 0)
            if arc_bike_count > 0:
                return random.randint(1, min(arc_bike_count, post_capacity))
        return 0


class GreedyStrategy(BaseStrategy):
    """Greedy 策略 - 基于当前状态的贪婪调度"""

    def inventory_decision(self, current_time: float, current_node: int,
                           node_bikes: Dict[int, int], vehicle_capacity: int) -> int:
        """贪婪库存决策"""
        if current_node == 0:  # 仓库不进行操作
            return 0

        current_bikes = node_bikes.get(current_node, 0)
        node_capacity = self.node_capacities.get(current_node, 0)

        # 获取当前供需率
        supply_rate = self.get_supply_rate(current_node, current_time)
        demand_rate = self.get_demand_rate(current_node, current_time)

        # 简单贪婪策略
        if supply_rate > demand_rate * 1.5 and current_bikes > node_capacity * 0.7:
            # 供应远大于需求且库存较高，需要装车
            load_amount = min(current_bikes - node_capacity * 0.5, vehicle_capacity)
            return max(0, load_amount)
        elif demand_rate > supply_rate * 1.5 and current_bikes < node_capacity * 0.3:
            # 需求远大于供应且库存较低，需要卸车
            unload_amount = min(node_capacity * 0.5 - current_bikes, self.operational_params['Q'] - vehicle_capacity)
            return -max(0, unload_amount)
        else:
            return 0

    def routing_decision(self, current_time: float, current_node: int,
                         vehicle_capacity: int, inventory_decision: int,
                         node_bikes: Dict[int, int], arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """贪婪路径决策"""
        gathering_points = [i for i in range(1, 16)]

        if current_node == 0:  # 从仓库出发
            # 选择当前库存最低的节点
            low_inventory_nodes = []
            for node_id in gathering_points:
                bikes = node_bikes.get(node_id, 0)
                capacity = self.node_capacities.get(node_id, 100)
                if bikes < capacity * 0.3:  # 库存低于30%
                    low_inventory_nodes.append(node_id)

            if low_inventory_nodes:
                return random.choice(low_inventory_nodes)
            else:
                return random.choice(gathering_points)
        else:
            # 基于车辆容量决定去向
            capacity_ratio = vehicle_capacity / self.operational_params['Q']

            if capacity_ratio > 0.7:  # 车辆较满，去需求高的节点
                high_demand_nodes = []
                for node_id in gathering_points:
                    if node_id == current_node:
                        continue
                    demand_rate = self.get_demand_rate(node_id, current_time)
                    if demand_rate > 0.5:  # 需求率较高
                        high_demand_nodes.append(node_id)

                if high_demand_nodes:
                    # 选择旅行时间最短的高需求节点
                    return min(high_demand_nodes,
                               key=lambda n: self.network_data['travel_times'].get((current_node, n), 10))
                else:
                    available_nodes = [n for n in gathering_points if n != current_node]
                    return random.choice(available_nodes) if available_nodes else current_node
            else:  # 车辆较空，去供应高的节点
                high_supply_nodes = []
                for node_id in gathering_points:
                    if node_id == current_node:
                        continue
                    supply_rate = self.get_supply_rate(node_id, current_time)
                    if supply_rate > 0.5:  # 供应率较高
                        high_supply_nodes.append(node_id)

                if high_supply_nodes:
                    # 选择旅行时间最短的高供应节点
                    return min(high_supply_nodes,
                               key=lambda n: self.network_data['travel_times'].get((current_node, n), 10))
                else:
                    available_nodes = [n for n in gathering_points if n != current_node]
                    return random.choice(available_nodes) if available_nodes else current_node

    def en_route_decision(self, current_node: int, next_node: int,
                          post_capacity: int, arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """贪婪途中决策"""
        arc_bike_count = arc_bikes.get((current_node, next_node), 0)

        if post_capacity > 0 and arc_bike_count > 0:
            # 如果车辆有空余容量且弧段上有自行车，尽量装载
            return min(arc_bike_count, post_capacity)
        else:
            return 0


class PFAStrategy(BaseStrategy):
    def __init__(self, operational_params: Dict[str, Any], policy_params: Dict[str, Any], network_data: Dict[str, Any]):
        super().__init__(operational_params, policy_params, network_data)

    def inventory_decision(self, current_time: float, current_node: int,
                           node_bikes: Dict[int, int], vehicle_capacity: int) -> int:
        """4.1 库存决策 - 双阈值策略"""
        if current_node == 0:  # 仓库不进行操作
            return 0

        # 获取当前节点的即时供需率
        supply_rate = self.get_supply_rate(current_node, current_time)
        demand_rate = self.get_demand_rate(current_node, current_time)

        # 判断节点类型
        if supply_rate > demand_rate:
            theta_low = self.policy_params['θ⁺_low']
            theta_high = theta_low + self.policy_params['θ⁺_high_gap']
            node_type = "supply"
        elif supply_rate < demand_rate:
            theta_low = self.policy_params['θ⁻_low']
            theta_high = theta_low + self.policy_params['θ⁻_high_gap']
            node_type = "demand"
        else:
            theta_low = self.policy_params['θ⁺_low']
            theta_high = theta_low + self.policy_params['θ⁺_high_gap']
            node_type = "balanced"

        # 计算库存阈值
        L_it, U_it = self.calculate_inventory_thresholds(
            current_node, current_time, theta_low, theta_high, node_type
        )

        current_bikes = node_bikes.get(current_node, 0)
        node_capacity = self.node_capacities.get(current_node, 0)

        # 应用双阈值策略
        if current_bikes < L_it:
            # 需要卸车
            unload_amount = min(L_it - current_bikes, self.operational_params['Q'] - vehicle_capacity)
            return -unload_amount
        elif current_bikes > U_it:
            # 需要装车
            load_amount = min(current_bikes - U_it, node_capacity - current_bikes,
                              vehicle_capacity)
            return load_amount
        else:
            return 0

    def calculate_inventory_thresholds(self, node_id: int, current_time: float,
                                       theta_low: float, theta_high: float, node_type: str) -> Tuple[float, float]:
        """计算库存阈值 L_it 和 U_it"""
        tau_L = self.policy_params['τ_L']  # 前瞻时间

        # 计算 alpha 因子
        alpha_sum = 0
        time_points = np.arange(current_time, min(current_time + tau_L, self.operational_params['T']), 5)  # 5分钟间隔

        for t in time_points:
            supply_rate = self.get_supply_rate(node_id, t)
            demand_rate = self.get_demand_rate(node_id, t)

            if node_type in ["supply", "balanced"]:
                alpha = 1 / (abs(supply_rate - demand_rate) + 1)
            else:  # demand node
                alpha = abs(supply_rate - demand_rate) + 1
            alpha_sum += alpha

        if len(time_points) > 0:
            alpha_avg = alpha_sum / len(time_points)
        else:
            alpha_avg = 1

        # 计算阈值
        node_capacity = self.node_capacities.get(node_id, 100)
        L_it = min(node_capacity, theta_low * alpha_avg * node_capacity)
        U_it = min(node_capacity, theta_high * alpha_avg * node_capacity)

        return L_it, U_it

    def routing_decision(self, current_time: float, current_node: int,
                         vehicle_capacity: int, inventory_decision: int,
                         node_bikes: Dict[int, int], arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """4.2 路径决策"""
        post_capacity = vehicle_capacity - inventory_decision
        capacity_ratio = post_capacity / self.operational_params['Q']

        gathering_points = [i for i in range(1, 16)]  # 节点1-15

        if capacity_ratio <= self.policy_params['δ']:
            # 车辆中自行车较多，倾向于前往需要自行车的节点
            return self._select_demand_node(current_time, current_node, post_capacity,
                                            node_bikes, arc_bikes, gathering_points)
        else:
            # 车辆容量较多，倾向于前往可以装载自行车的节点
            return self._select_supply_node(current_time, current_node, post_capacity,
                                            node_bikes, arc_bikes, gathering_points)

    def _select_demand_node(self, current_time: float, current_node: int,
                            post_capacity: int, node_bikes: Dict[int, int],
                            arc_bikes: Dict[Tuple[int, int], int], gathering_points: List[int]) -> int:
        """选择需求节点"""
        unmet_demands = {}

        for node_id in gathering_points:
            if node_id == current_node:
                continue

            # 计算预计到达时间
            travel_time = self.network_data['travel_times'].get((current_node, node_id), 10)

            # 计算预期未满足需求
            unmet_demand = self.calculate_anticipated_unmet_demand(
                node_id, current_time + travel_time, self.policy_params['τ_D']
            )
            unmet_demands[node_id] = unmet_demand

        if not unmet_demands:
            return random.choice(gathering_points)

        # 选择未满足需求较大的节点
        max_unmet = max(unmet_demands.values())
        candidate_nodes = [n for n, u in unmet_demands.items()
                           if u >= self.policy_params['β'] * max_unmet]

        # 选择旅行时间最短的节点
        if candidate_nodes:
            return min(candidate_nodes, key=lambda n: self.network_data['travel_times'].get((current_node, n), 10))
        else:
            return random.choice(gathering_points)

    def _select_supply_node(self, current_time: float, current_node: int,
                            post_capacity: int, node_bikes: Dict[int, int],
                            arc_bikes: Dict[Tuple[int, int], int], gathering_points: List[int]) -> int:
        """选择供应节点"""
        loading_amounts = {}

        for node_id in gathering_points:
            if node_id == current_node:
                continue

            travel_time = self.network_data['travel_times'].get((current_node, node_id), 10)

            # 计算途中装载量
            en_route_load = self.en_route_decision(current_node, node_id, post_capacity, arc_bikes)

            # 计算在节点的装载量
            node_arrival_time = current_time + travel_time
            node_inventory_decision = self.inventory_decision(
                node_arrival_time, node_id, node_bikes, post_capacity - en_route_load
            )
            node_load = max(node_inventory_decision, 0)

            loading_amounts[node_id] = en_route_load + node_load

        if not loading_amounts:
            return random.choice(gathering_points)

        # 选择装载量较大的节点
        max_load = max(loading_amounts.values())
        candidate_nodes = [n for n, l in loading_amounts.items()
                           if l >= self.policy_params['β'] * max_load]

        # 选择旅行时间最短的节点
        if candidate_nodes:
            return min(candidate_nodes, key=lambda n: self.network_data['travel_times'].get((current_node, n), 10))
        else:
            return random.choice(gathering_points)

    def en_route_decision(self, current_node: int, next_node: int,
                          post_capacity: int, arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """4.3 途中决策"""
        travel_time = self.network_data['travel_times'].get((current_node, next_node), 10)

        # 时间限制下的最大装载量
        time_limit_load = math.floor(self.policy_params['φ'] * travel_time / self.operational_params['τ_A'])

        # 弧段上的自行车数量限制
        arc_bike_count = arc_bikes.get((current_node, next_node), 0)

        # 容量限制
        capacity_limit = post_capacity

        return min(time_limit_load, arc_bike_count, capacity_limit)

    def calculate_anticipated_unmet_demand(self, node_id: int, start_time: float,
                                           lookahead_time: float) -> float:
        """计算预期未满足需求"""
        total_unmet = 0
        current_inventory = 0  # 简化处理

        time_slots = np.arange(start_time, start_time + lookahead_time, 5)  # 5分钟间隔

        for t in time_slots:
            if t >= self.operational_params['T']:
                break

            supply = self.get_supply_rate(node_id, t) * 5  # 5分钟内的供应
            demand = self.get_demand_rate(node_id, t) * 5  # 5分钟内的需求

            # 计算未满足需求
            unmet = max(0, demand - max(0, current_inventory + supply))
            total_unmet += unmet

            # 更新库存
            net_change = supply - demand
            current_inventory = max(0, min(self.node_capacities.get(node_id, 100),
                                           current_inventory + net_change))

        return total_unmet


class BikeSharingSimulator:
    def __init__(self, network_data: Dict[str, Any], initial_state: Dict[str, Any],
                 params: Dict[str, Any]):
        self.network_data = network_data
        self.initial_state = initial_state
        self.params = params
        # 添加累计需求跟踪
        self.total_demand_generated = 0
        self.total_supply_generated = 0

    def simulate_demand_supply_events(self, state: Dict[str, Any], start_time: float,
                                      end_time: float) -> float:
        """修正版本：按5分钟间隔离散模拟供需事件"""
        satisfied_demand = 0
        current_t = start_time

        # 按5分钟间隔逐步模拟
        while current_t < end_time:
            time_slot_end = min(current_t + 5, end_time)
            slot_duration = time_slot_end - current_t

            # 模拟这个5分钟时间槽内的供需
            slot_satisfied = self._simulate_time_slot(state, current_t, slot_duration)
            satisfied_demand += slot_satisfied

            current_t = time_slot_end

        return satisfied_demand

    def _simulate_time_slot(self, state: Dict[str, Any], start_time: float,
                            duration: float) -> float:
        """模拟单个时间槽内的供需事件"""
        satisfied_demand = 0

        for node_id, current_bikes in state['node_bikes'].items():
            if node_id == 0:  # 仓库没有需求
                continue

            node_capacity = self.network_data['node_capacities'].get(node_id, 100)

            # 获取当前时间点的需求率和供应率
            demand_rate = self.get_demand_rate(node_id, start_time)
            supply_rate = self.get_supply_rate(node_id, start_time)

            # 计算这个时间槽内的期望需求
            expected_demand = demand_rate * duration  # 辆/分钟 × 分钟 = 辆
            expected_supply = supply_rate * duration

            # 使用泊松分布生成实际事件
            actual_demand = np.random.poisson(expected_demand)
            actual_supply = np.random.poisson(expected_supply)

            # 跟踪总生成量
            self.total_demand_generated += actual_demand
            self.total_supply_generated += actual_supply

            # 计算实际满足的需求（受库存限制）
            actual_satisfied = min(actual_demand, current_bikes)
            satisfied_demand += actual_satisfied

            # 更新节点库存
            net_change = actual_supply - actual_satisfied
            new_bikes = max(0, min(node_capacity, current_bikes + net_change))
            state['node_bikes'][node_id] = new_bikes

        return satisfied_demand

    def get_demand_rate(self, node_id: int, current_time: float) -> float:
        """获取节点需求率 - 修正时间索引"""
        time_key = self.get_time_index(current_time)
        if (node_id in self.network_data['node_demand_rates'] and
                time_key in self.network_data['node_demand_rates'][node_id]):
            return self.network_data['node_demand_rates'][node_id][time_key]
        return 0.0

    def get_supply_rate(self, node_id: int, current_time: float) -> float:
        """获取节点供应率 - 修正时间索引"""
        time_key = self.get_time_index(current_time)
        if (node_id in self.network_data['node_supply_rates'] and
                time_key in self.network_data['node_supply_rates'][node_id]):
            return self.network_data['node_supply_rates'][node_id][time_key]
        return 0.0

    def get_time_index(self, current_time: float) -> str:
        """将分钟时间转换为时间字符串格式（匹配Excel中的时间格式）"""
        start_hour = 11  # 11:00开始
        total_minutes = start_hour * 60 + current_time
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)

        # 确保分钟是5的倍数（匹配5分钟间隔数据）
        rounded_minutes = (minutes // 5) * 5
        return f"{hours:02d}:{rounded_minutes:02d}"

    def validate_data_rates(self):
        """验证需求率数据的合理性"""
        print("\n=== 需求率数据验证 ===")

        total_expected_demand = 0
        total_expected_supply = 0

        # 计算4.5小时内的总期望需求
        for node_id in range(1, 16):
            node_demand = 0
            node_supply = 0

            # 遍历所有5分钟时间槽
            for t in range(0, 271, 5):  # 11:00-15:30，共270分钟
                demand_rate = self.get_demand_rate(node_id, t)
                supply_rate = self.get_supply_rate(node_id, t)
                node_demand += demand_rate * 5  # 5分钟的需求量
                node_supply += supply_rate * 5  # 5分钟的供应量

            total_expected_demand += node_demand
            total_expected_supply += node_supply

            print(f"节点 {node_id}: 期望需求 = {node_demand:.1f}辆, 期望供应 = {node_supply:.1f}辆")

        print(f"\n系统总期望需求: {total_expected_demand:.1f}辆")
        print(f"系统总期望供应: {total_expected_supply:.1f}辆")

        # 合理性判断
        if total_expected_demand < 100:
            print("⚠️ 警告: 总期望需求过低，请检查数据单位！")
        elif total_expected_demand > 10000:
            print("⚠️ 警告: 总期望需求过高，请检查数据单位！")
        else:
            print("✅ 数据范围合理")

    def simulate_with_strategy(self, strategy: BaseStrategy, random_seed: int = None) -> Dict[str, Any]:
        """使用指定策略执行一次仿真运行，返回详细统计"""
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # 重置累计统计
        self.total_demand_generated = 0
        self.total_supply_generated = 0

        # 初始化状态
        state = self.initialize_state()
        total_satisfied_demand = 0
        current_time = 0  # 11:00开始
        current_node = 0  # 从仓库开始
        decision_count = 0

        # 主仿真循环
        while current_time < self.params['T']:  # 11:00-15:30
            decision_count += 1

            # 获取当前决策
            if current_node == 0:  # 在仓库
                # 从仓库出发，选择第一个节点
                next_node = random.choice(list(range(1, 16)))
                inventory_decision = 0
                en_route_decision = 0
            else:
                # 使用指定策略做决策
                inventory_decision = strategy.inventory_decision(
                    current_time, current_node,
                    state['node_bikes'], state['vehicle']['current_load']
                )

                next_node = strategy.routing_decision(
                    current_time, current_node,
                    state['vehicle']['current_load'], inventory_decision,
                    state['node_bikes'], state['arc_bikes']
                )

                en_route_decision = strategy.en_route_decision(
                    current_node, next_node,
                    state['vehicle']['current_load'] - inventory_decision,
                    state['arc_bikes']
                )

            # 应用决策并更新状态
            new_state, satisfied_demand = self.transition_state(
                state, current_node, next_node, inventory_decision, en_route_decision, current_time
            )

            total_satisfied_demand += satisfied_demand
            state = new_state

            # 更新时间和位置
            travel_time = self.network_data['travel_times'].get((current_node, next_node), 10)
            inventory_time = abs(inventory_decision) * self.params['τ_N']
            en_route_time = en_route_decision * self.params['τ_A']
            current_time += travel_time + inventory_time + en_route_time
            current_node = next_node

        # 返回详细统计
        return {
            'total_satisfied_demand': total_satisfied_demand,
            'total_demand_generated': self.total_demand_generated,
            'total_supply_generated': self.total_supply_generated,
            'demand_satisfaction_ratio': total_satisfied_demand / self.total_demand_generated if self.total_demand_generated > 0 else 0,
            'decision_count': decision_count,
            'final_time': current_time
        }

    def simulate(self, policy_params: Dict[str, Any], random_seed: int = None) -> Dict[str, Any]:
        """向后兼容的simulate方法，使用PFA策略"""
        pfa_strategy = PFAStrategy(self.params, policy_params, self.network_data)
        return self.simulate_with_strategy(pfa_strategy, random_seed)

    def initialize_state(self) -> Dict[str, Any]:
        """初始化仿真状态"""
        return {
            'node_bikes': self.initial_state['node_bikes'].copy(),
            'arc_bikes': self.initial_state['arc_bikes'].copy(),
            'vehicle': {
                'location': 0,  # 仓库
                'current_load': 0
            }
        }

    def transition_state(self, state: Dict[str, Any], current_node: int, next_node: int,
                         inventory_decision: int, en_route_decision: int, current_time: float) -> Tuple[
        Dict[str, Any], float]:
        """状态转移函数 - 修正版本"""
        new_state = {
            'node_bikes': state['node_bikes'].copy(),
            'arc_bikes': state['arc_bikes'].copy(),
            'vehicle': state['vehicle'].copy()
        }

        # 1. 首先处理节点库存决策（瞬时完成）
        if current_node != 0:  # 仓库不处理库存
            current_bikes = new_state['node_bikes'].get(current_node, 0)
            node_capacity = self.network_data['node_capacities'].get(current_node, 100)

            if inventory_decision > 0:  # 装车
                actual_load = min(inventory_decision, current_bikes,
                                  self.params['Q'] - new_state['vehicle']['current_load'])
                new_state['node_bikes'][current_node] = current_bikes - actual_load
                new_state['vehicle']['current_load'] += actual_load
            elif inventory_decision < 0:  # 卸车
                actual_unload = min(-inventory_decision, new_state['vehicle']['current_load'],
                                    node_capacity - current_bikes)
                new_state['node_bikes'][current_node] = current_bikes + actual_unload
                new_state['vehicle']['current_load'] -= actual_unload

        # 2. 计算整个决策区间的时间
        travel_time = self.network_data['travel_times'].get((current_node, next_node), 10)
        inventory_time = abs(inventory_decision) * self.params['τ_N']
        en_route_time = en_route_decision * self.params['τ_A']
        total_decision_time = travel_time + inventory_time + en_route_time

        end_time = current_time + total_decision_time

        # 3. 处理途中决策（在旅行期间发生）
        if en_route_decision > 0:
            arc_key = (current_node, next_node)
            arc_bikes = new_state['arc_bikes'].get(arc_key, 0)
            actual_en_route_load = min(en_route_decision, arc_bikes,
                                       self.params['Q'] - new_state['vehicle']['current_load'])
            new_state['arc_bikes'][arc_key] = arc_bikes - actual_en_route_load
            new_state['vehicle']['current_load'] += actual_en_route_load

        # 4. 模拟整个决策区间内的供需事件
        satisfied_demand = self.simulate_demand_supply_events(
            new_state, current_time, end_time
        )

        return new_state, satisfied_demand

    def validate_demand_calculation(self):
        """验证需求计算是否合理"""
        print("=== 需求计算验证 ===")

        # 使用论文中的最佳参数进行测试
        best_params_from_paper = {
            'θ⁺_low': 0,
            'θ⁺_high_gap': 0,
            'θ⁻_low': 0.5,
            'θ⁻_high_gap': 0.5,
            'τ_L': 180,
            'τ_D': 180,
            'δ': 0.75,
            'β': 0.75,
            'φ': 1
        }

        # 测试一次完整的仿真
        results = self.simulate(best_params_from_paper)

        print(f"总生成需求: {results['total_demand_generated']:.2f}")
        print(f"总满足需求: {results['total_satisfied_demand']:.2f}")
        print(f"需求满足率: {results['demand_satisfaction_ratio']:.1%}")
        print(f"总生成供应: {results['total_supply_generated']:.2f}")
        print(f"决策次数: {results['decision_count']}")
        print(f"最终时间: {results['final_time']:.1f}分钟")

        # 检查合理性
        if results['demand_satisfaction_ratio'] < 0.5:
            print("警告: 需求满足率过低!")
            print("可能原因:")
            print("1. 需求率数据单位错误")
            print("2. 初始自行车数量不足")
            print("3. 调度策略效率低下")
        elif results['demand_satisfaction_ratio'] > 1.0:
            print("错误: 需求满足率超过100%!")
            print("可能原因:")
            print("1. 需求计算重复")
            print("2. 供应计算错误")

    def run_benchmark_comparison(self, pfa_params: Dict[str, Any], n_runs: int = 5) -> Dict[str, Any]:
        """运行基准策略比较"""
        print("\n" + "=" * 60)
        print("开始基准策略比较")
        print("=" * 60)

        strategies = {
            'PFA': PFAStrategy(self.params, pfa_params, self.network_data),
            'DoNothing': DoNothingStrategy(self.params, {}, self.network_data),
            'Random': RandomStrategy(self.params, {}, self.network_data),
            'Greedy': GreedyStrategy(self.params, {}, self.network_data)
        }

        results = {}

        for strategy_name, strategy in strategies.items():
            print(f"\n正在测试 {strategy_name} 策略...")

            strategy_results = []
            for run in range(n_runs):
                result = self.simulate_with_strategy(strategy, random_seed=run)
                strategy_results.append(result)

                print(f"  运行 {run + 1}: 满足需求={result['total_satisfied_demand']:.2f}, "
                      f"满足率={result['demand_satisfaction_ratio']:.1%}")

            # 计算统计信息
            satisfied_demands = [r['total_satisfied_demand'] for r in strategy_results]
            satisfaction_ratios = [r['demand_satisfaction_ratio'] for r in strategy_results]
            decision_counts = [r['decision_count'] for r in strategy_results]

            results[strategy_name] = {
                'avg_satisfied_demand': np.mean(satisfied_demands),
                'std_satisfied_demand': np.std(satisfied_demands),
                'avg_satisfaction_ratio': np.mean(satisfaction_ratios),
                'std_satisfaction_ratio': np.std(satisfaction_ratios),
                'avg_decision_count': np.mean(decision_counts),
                'all_runs': strategy_results
            }

            print(f"  {strategy_name} 平均结果:")
            print(f"    平均满足需求: {results[strategy_name]['avg_satisfied_demand']:.2f} ± "
                  f"{results[strategy_name]['std_satisfied_demand']:.2f}")
            print(f"    平均满足率: {results[strategy_name]['avg_satisfaction_ratio']:.1%} ± "
                  f"{results[strategy_name]['std_satisfaction_ratio']:.1%}")
            print(f"    平均决策次数: {results[strategy_name]['avg_decision_count']:.1f}")

        return results


class OptunaOptimizer:
    def __init__(self, parameter_space: Dict[str, List[Any]], optuna_params: Dict[str, Any]):
        self.parameter_space = parameter_space
        self.optuna_params = optuna_params
        self.study = None

    def optimize(self, simulator: BikeSharingSimulator) -> Dict[str, Any]:
        """执行Optuna优化"""
        n_trials = self.optuna_params.get('n_trials', 100)
        n_startup_trials = self.optuna_params.get('n_startup_trials', 20)
        random_state = self.optuna_params.get('random_state', 42)

        print(f"开始Optuna优化，总试验次数: {n_trials}")

        # 创建Optuna研究
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(n_startup_trials=n_startup_trials, seed=random_state)
        )

        # 定义目标函数
        def objective(trial):
            # 建议参数值
            theta_plus_low = trial.suggest_float('θ⁺_low', 0.0, 2.0)
            theta_plus_high_gap = trial.suggest_float('θ⁺_high_gap', 0.0, 2.0)
            theta_minus_low = trial.suggest_float('θ⁻_low', 0.0, 2.0)
            theta_minus_high_gap = trial.suggest_float('θ⁻_high_gap', 0.0, 2.0)
            tau_L = trial.suggest_int('τ_L', 30, 180)
            tau_D = trial.suggest_int('τ_D', 30, 180)
            delta = trial.suggest_float('δ', 0.1, 0.9)
            beta = trial.suggest_float('β', 0.1, 0.9)
            phi = trial.suggest_float('φ', 0.1, 1.5)

            # 构建策略参数
            policy_params = {
                'θ⁺_low': theta_plus_low,
                'θ⁺_high_gap': theta_plus_high_gap,
                'θ⁻_low': theta_minus_low,
                'θ⁻_high_gap': theta_minus_high_gap,
                'τ_L': tau_L,
                'τ_D': tau_D,
                'δ': delta,
                'β': beta,
                'φ': phi
            }

            # 运行多次仿真取平均（减少随机性）
            n_replications = 3  # 每次试验运行3次
            performances = []

            for rep in range(n_replications):
                result = simulator.simulate(policy_params, random_seed=rep)
                performances.append(result['total_satisfied_demand'])

            avg_performance = np.mean(performances)

            # 打印当前试验信息
            print(f"试验 {trial.number}: 参数 = {policy_params}, 平均性能 = {avg_performance:.2f}")

            return avg_performance

        # 执行优化
        self.study.optimize(objective, n_trials=n_trials)

        # 提取最佳参数
        best_params = self.study.best_params

        print(f"\nOptuna优化完成！")
        print(f"最佳试验: {self.study.best_trial.number}")
        print(f"最佳性能: {self.study.best_value:.2f}")
        print(f"最佳参数: {best_params}")

        return best_params

    def get_optimization_history(self):
        """获取优化历史"""
        if self.study is None:
            return None

        trials = self.study.trials
        history = {
            'values': [trial.value for trial in trials if trial.value is not None],
            'params': [trial.params for trial in trials if trial.value is not None]
        }

        return history


def main():
    """主函数"""
    try:
        # 加载和处理数据
        processed_data = load_and_process_data()

        if processed_data is None:
            print("数据加载失败，程序退出")
            return

        # 运营参数
        operational_params = {
            'Q': 25,  # 车辆容量
            'τ_N': 0.25,  # 节点装卸时间（分钟/辆）
            'τ_A': 0.5,  # 弧段装载时间（分钟/辆）
            'τ_W': 5,  # 等待时间（分钟）
            'T': 270,  # 总工作时间（分钟，11:00-15:30）
            'Δt': 5,  # 时间离散化间隔（分钟）
        }

        # PFA参数搜索空间（现在用于Optuna优化）
        pfa_parameter_space = {
            'θ⁺_low': [0, 0.5, 1],
            'θ⁺_high_gap': [0, 0.5, 1],
            'θ⁻_low': [0, 0.5, 1, 1.5, 2],
            'θ⁻_high_gap': [0, 0.5, 1, 1.5, 2],
            'τ_L': [60, 120, 180],
            'τ_D': [60, 120, 180],
            'δ': [0.25, 0.5, 0.75],
            'β': [0.25, 0.5, 0.75],
            'φ': [0.25, 0.5, 0.75, 1]
        }

        # Optuna优化参数
        optuna_params = {
            'n_trials': 100,  # 总试验次数（为了演示减少次数）
            'n_startup_trials': 20,  # 初始随机试验次数
            'random_state': 42  # 随机种子
        }

        print("初始化仿真器...")
        # 创建仿真器
        simulator = BikeSharingSimulator(
            network_data=processed_data,
            initial_state={
                'node_bikes': processed_data['initial_node_bikes'],
                'arc_bikes': processed_data['initial_arc_bikes']
            },
            params=operational_params
        )

        # 首先验证数据率
        print("验证数据率合理性...")
        simulator.validate_data_rates()

        # 然后验证需求计算
        print("验证需求计算...")
        simulator.validate_demand_calculation()

        # 运行基准策略比较（使用默认参数）
        print("\n运行基准策略比较...")
        default_pfa_params = {
            'θ⁺_low': 0.5,
            'θ⁺_high_gap': 0.5,
            'θ⁻_low': 0.5,
            'θ⁻_high_gap': 0.5,
            'τ_L': 120,
            'τ_D': 120,
            'δ': 0.5,
            'β': 0.5,
            'φ': 0.5
        }

        benchmark_results = simulator.run_benchmark_comparison(default_pfa_params, n_runs=3)

        # 保存基准比较结果
        benchmark_df = pd.DataFrame({
            'Strategy': list(benchmark_results.keys()),
            'Avg_Satisfied_Demand': [benchmark_results[s]['avg_satisfied_demand'] for s in benchmark_results],
            'Std_Satisfied_Demand': [benchmark_results[s]['std_satisfied_demand'] for s in benchmark_results],
            'Avg_Satisfaction_Ratio': [benchmark_results[s]['avg_satisfaction_ratio'] for s in benchmark_results],
            'Std_Satisfaction_Ratio': [benchmark_results[s]['std_satisfaction_ratio'] for s in benchmark_results],
            'Avg_Decision_Count': [benchmark_results[s]['avg_decision_count'] for s in benchmark_results]
        })
        benchmark_df.to_excel('benchmark_comparison_results.xlsx', index=False)
        print("基准比较结果已保存到 'benchmark_comparison_results.xlsx'")

        print("\n开始Optuna优化...")
        optimizer = OptunaOptimizer(pfa_parameter_space, optuna_params)
        best_params = optimizer.optimize(simulator)

        print("\n优化完成！最佳参数：")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        # 使用优化后的参数再次进行基准比较
        print("\n使用优化参数进行最终基准比较...")
        final_benchmark_results = simulator.run_benchmark_comparison(best_params, n_runs=5)

        # 保存最终比较结果
        final_benchmark_df = pd.DataFrame({
            'Strategy': list(final_benchmark_results.keys()),
            'Avg_Satisfied_Demand': [final_benchmark_results[s]['avg_satisfied_demand'] for s in
                                     final_benchmark_results],
            'Std_Satisfied_Demand': [final_benchmark_results[s]['std_satisfied_demand'] for s in
                                     final_benchmark_results],
            'Avg_Satisfaction_Ratio': [final_benchmark_results[s]['avg_satisfaction_ratio'] for s in
                                       final_benchmark_results],
            'Std_Satisfaction_Ratio': [final_benchmark_results[s]['std_satisfaction_ratio'] for s in
                                       final_benchmark_results],
            'Avg_Decision_Count': [final_benchmark_results[s]['avg_decision_count'] for s in final_benchmark_results]
        })
        final_benchmark_df.to_excel('final_benchmark_comparison_results.xlsx', index=False)
        print("最终基准比较结果已保存到 'final_benchmark_comparison_results.xlsx'")

        # 验证最佳参数
        print("\n验证最佳参数性能...")
        final_performances = []
        final_satisfaction_ratios = []

        for i in range(5):  # 5次验证运行
            result = simulator.simulate(best_params, random_seed=1000 + i)
            performance = result['total_satisfied_demand']
            satisfaction_ratio = result['demand_satisfaction_ratio']
            final_performances.append(performance)
            final_satisfaction_ratios.append(satisfaction_ratio)
            print(f"验证运行 {i + 1}: 满足需求={performance:.2f}, 满足率={satisfaction_ratio:.1%}")

        avg_performance = np.mean(final_performances)
        std_performance = np.std(final_performances)
        avg_satisfaction_ratio = np.mean(final_satisfaction_ratios)

        print(f"最佳参数下的平均需求满足量: {avg_performance:.2f} ± {std_performance:.2f}")
        print(f"最佳参数下的平均需求满足率: {avg_satisfaction_ratio:.1%}")

        # 保存最佳参数
        best_params_df = pd.DataFrame([best_params])
        best_params_df.to_excel('best_pfa_parameters_optuna.xlsx', index=False)
        print("最佳参数已保存到 'best_pfa_parameters_optuna.xlsx'")

        # 保存优化历史
        history = optimizer.get_optimization_history()
        if history:
            history_df = pd.DataFrame({
                'trial': range(len(history['values'])),
                'performance': history['values'],
                **{key: [params.get(key, None) for params in history['params']] for key in best_params.keys()}
            })
            history_df.to_excel('optimization_history_optuna.xlsx', index=False)
            print("优化历史已保存到 'optimization_history_optuna.xlsx'")

        # 生成比较报告
        print("\n" + "=" * 60)
        print("策略性能比较报告")
        print("=" * 60)

        pfa_performance = final_benchmark_results['PFA']['avg_satisfied_demand']

        for strategy_name, result in final_benchmark_results.items():
            if strategy_name != 'PFA':
                improvement = ((pfa_performance - result['avg_satisfied_demand']) / result[
                    'avg_satisfied_demand']) * 100
                print(f"PFA相比{strategy_name}策略改进: {improvement:+.1f}%")

    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()