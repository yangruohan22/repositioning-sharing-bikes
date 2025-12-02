#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @ Time:2025/12/1 3:52
# @ Author:86155
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
import math
import random
from datetime import datetime, timedelta
import warnings

# 添加 PSO 优化库
import pyswarms as ps

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


class PFAStrategy:
    def __init__(self, operational_params: Dict[str, Any], policy_params: Dict[str, Any], network_data: Dict[str, Any]):
        self.operational_params = operational_params  # 运营参数
        self.policy_params = policy_params  # 策略参数
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

    def simulate(self, policy_params: Dict[str, Any], random_seed: int = None) -> Dict[str, Any]:
        """执行一次仿真运行，返回详细统计"""
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # 重置累计统计
        self.total_demand_generated = 0
        self.total_supply_generated = 0

        # 初始化PFA策略
        pfa = PFAStrategy(self.params, policy_params, self.network_data)

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
                # 使用PFA策略做决策
                inventory_decision = pfa.inventory_decision(
                    current_time, current_node,
                    state['node_bikes'], state['vehicle']['current_load']
                )

                next_node = pfa.routing_decision(
                    current_time, current_node,
                    state['vehicle']['current_load'], inventory_decision,
                    state['node_bikes'], state['arc_bikes']
                )

                en_route_decision = pfa.en_route_decision(
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


class PSOOptimizer:
    def __init__(self, parameter_space: Dict[str, List[Any]], pso_params: Dict[str, Any]):
        self.parameter_space = parameter_space
        self.pso_params = pso_params
        self.best_solution = None
        self.best_value = -np.inf
        self.history = []

    def optimize(self, simulator: BikeSharingSimulator) -> Dict[str, Any]:
        """执行PSO优化"""
        n_particles = self.pso_params.get('n_particles', 20)
        n_iterations = self.pso_params.get('n_iterations', 50)
        random_state = self.pso_params.get('random_state', 42)

        print(f"开始PSO优化，粒子数: {n_particles}, 迭代次数: {n_iterations}")

        # 定义参数边界
        bounds = self._get_parameter_bounds()

        # 定义选项
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        # 初始化优化器
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=len(bounds[0]),
            options=options,
            bounds=bounds
        )

        # 定义目标函数
        def objective_function(particles):
            performances = []

            for i, particle in enumerate(particles):
                # 将粒子位置转换为策略参数
                policy_params = self._particle_to_params(particle)

                # 运行多次仿真取平均（减少随机性）
                n_replications = 2  # 每次试验运行2次
                particle_performances = []

                for rep in range(n_replications):
                    result = simulator.simulate(policy_params, random_seed=rep)
                    particle_performances.append(result['total_satisfied_demand'])

                avg_performance = np.mean(particle_performances)
                performances.append(-avg_performance)  # 负号因为PSO最小化，我们想要最大化

                # 更新最佳解
                if avg_performance > self.best_value:
                    self.best_value = avg_performance
                    self.best_solution = particle.copy()

                # 记录历史
                self.history.append({
                    'iteration': len(self.history) // n_particles,
                    'particle': i,
                    'performance': avg_performance,
                    'params': policy_params
                })

                print(f"迭代 {len(self.history) // n_particles}, 粒子 {i}: 性能 = {avg_performance:.2f}")

            return np.array(performances)

        # 执行优化
        cost, pos = optimizer.optimize(
            objective_function,
            iters=n_iterations,
            verbose=True
        )

        # 提取最佳参数
        best_params = self._particle_to_params(self.best_solution)

        print(f"\nPSO优化完成！")
        print(f"最佳性能: {self.best_value:.2f}")
        print(f"最佳参数: {best_params}")

        return best_params

    def _get_parameter_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取参数边界"""
        # 参数顺序: θ⁺_low, θ⁺_high_gap, θ⁻_low, θ⁻_high_gap, τ_L, τ_D, δ, β, φ
        lower_bounds = np.array([0.0, 0.0, 0.0, 0.0, 30, 30, 0.1, 0.1, 0.1])
        upper_bounds = np.array([2.0, 2.0, 2.0, 2.0, 180, 180, 0.9, 0.9, 1.5])

        return (lower_bounds, upper_bounds)

    def _particle_to_params(self, particle: np.ndarray) -> Dict[str, Any]:
        """将粒子位置转换为策略参数"""
        return {
            'θ⁺_low': float(particle[0]),
            'θ⁺_high_gap': float(particle[1]),
            'θ⁻_low': float(particle[2]),
            'θ⁻_high_gap': float(particle[3]),
            'τ_L': int(particle[4]),  # 转换为整数
            'τ_D': int(particle[5]),  # 转换为整数
            'δ': float(particle[6]),
            'β': float(particle[7]),
            'φ': float(particle[8])
        }

    def get_optimization_history(self):
        """获取优化历史"""
        return self.history


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

        # PFA参数搜索空间
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

        # PSO优化参数
        pso_params = {
            'n_particles': 20,  # 粒子数量
            'n_iterations': 2000,  # 迭代次数
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

        print("开始PSO优化...")
        optimizer = PSOOptimizer(pfa_parameter_space, pso_params)
        best_params = optimizer.optimize(simulator)

        print("\n优化完成！最佳参数：")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

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
        best_params_df.to_excel('best_pfa_parameters_pso.xlsx', index=False)
        print("最佳参数已保存到 'best_pfa_parameters_pso.xlsx'")

        # 保存优化历史
        history = optimizer.get_optimization_history()
        if history:
            # 转换为DataFrame
            history_data = []
            for record in history:
                row = {
                    'iteration': record['iteration'],
                    'particle': record['particle'],
                    'performance': record['performance']
                }
                row.update(record['params'])
                history_data.append(row)

            history_df = pd.DataFrame(history_data)
            history_df.to_excel('optimization_history_pso.xlsx', index=False)
            print("优化历史已保存到 'optimization_history_pso.xlsx'")

    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()