#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @ Time:2025/12/28 22:47
# @ Author:86155
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
import math
import random
from datetime import datetime, timedelta
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

# 核心参数配置
RANDOM_SEEDS = [42, 100, 200, 300, 400]  # 多个随机种子保证稳定性
PERFORMANCE_WEIGHT = 0.7  # 平均性能权重
STABILITY_WEIGHT = 0.3  # 稳定性权重
N_TRIALS = 10000  # 优化试验次数
N_STARTUP_TRIALS = 20  # 初始随机搜索次数


class BikeSharingSimulator:
    def __init__(self, network_data: Dict[str, Any], initial_state: Dict[str, Any], params: Dict[str, Any]):
        self.network_data = network_data
        self.initial_state = initial_state
        self.params = params
        self.total_demand_generated = 0
        self.total_supply_generated = 0

    def simulate_demand_supply_events(self, state: Dict[str, Any], start_time: float, end_time: float) -> float:
        """按5分钟间隔离散模拟供需事件"""
        satisfied_demand = 0
        current_t = start_time

        while current_t < end_time:
            time_slot_end = min(current_t + 5, end_time)
            slot_duration = time_slot_end - current_t

            for node_id, current_bikes in state['node_bikes'].items():
                if node_id == 0: continue

                node_capacity = self.network_data['node_capacities'].get(node_id, 100)
                demand_rate = self.get_demand_rate(node_id, current_t)
                supply_rate = self.get_supply_rate(node_id, current_t)

                expected_demand = demand_rate * slot_duration
                expected_supply = supply_rate * slot_duration

                actual_demand = np.random.poisson(expected_demand)
                actual_supply = np.random.poisson(expected_supply)

                self.total_demand_generated += actual_demand
                self.total_supply_generated += actual_supply

                actual_satisfied = min(actual_demand, current_bikes)
                satisfied_demand += actual_satisfied

                net_change = actual_supply - actual_satisfied
                new_bikes = max(0, min(node_capacity, current_bikes + net_change))
                state['node_bikes'][node_id] = new_bikes

            current_t = time_slot_end

        return satisfied_demand

    def get_demand_rate(self, node_id: int, current_time: float) -> float:
        """获取节点需求率"""
        time_key = self.get_time_index(current_time)
        if node_id in self.network_data['node_demand_rates'] and time_key in self.network_data['node_demand_rates'][
            node_id]:
            return self.network_data['node_demand_rates'][node_id][time_key]
        return 0.0

    def get_supply_rate(self, node_id: int, current_time: float) -> float:
        """获取节点供应率"""
        time_key = self.get_time_index(current_time)
        if node_id in self.network_data['node_supply_rates'] and time_key in self.network_data['node_supply_rates'][
            node_id]:
            return self.network_data['node_supply_rates'][node_id][time_key]
        return 0.0

    def get_time_index(self, current_time: float) -> str:
        """将分钟时间转换为时间字符串格式"""
        start_hour = 11
        total_minutes = start_hour * 60 + current_time
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        rounded_minutes = (minutes // 5) * 5
        return f"{hours:02d}:{rounded_minutes:02d}"

    def simulate(self, policy_params: Dict[str, Any], random_seed: int = None) -> Dict[str, Any]:
        """核心仿真函数"""
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.total_demand_generated = 0
        self.total_supply_generated = 0

        # 初始化PFA策略
        pfa = PFAStrategy(self.params, policy_params, self.network_data)

        # 初始化状态
        state = self.initialize_state()
        total_satisfied_demand = 0
        current_time = 0
        current_node = 0

        # 主仿真循环
        while current_time < self.params['T']:
            if current_node == 0:
                next_node = random.choice(list(range(1, 16)))
                inventory_decision = 0
                en_route_decision = 0
            else:
                inventory_decision = pfa.inventory_decision(
                    current_time, current_node, state['node_bikes'], state['vehicle']['current_load']
                )

                next_node = pfa.routing_decision(
                    current_time, current_node, state['vehicle']['current_load'], inventory_decision,
                    state['node_bikes'], state['arc_bikes']
                )

                en_route_decision = pfa.en_route_decision(
                    current_node, next_node, state['vehicle']['current_load'] - inventory_decision,
                    state['arc_bikes']
                )

            # 状态转移
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

        return {
            'total_satisfied_demand': total_satisfied_demand,
            'total_demand_generated': self.total_demand_generated,
            'demand_satisfaction_ratio': total_satisfied_demand / self.total_demand_generated if self.total_demand_generated > 0 else 0
        }

    def initialize_state(self) -> Dict[str, Any]:
        """初始化仿真状态"""
        return {
            'node_bikes': self.initial_state['node_bikes'].copy(),
            'arc_bikes': self.initial_state['arc_bikes'].copy(),
            'vehicle': {'location': 0, 'current_load': 0}
        }

    def transition_state(self, state: Dict[str, Any], current_node: int, next_node: int,
                         inventory_decision: int, en_route_decision: int, current_time: float) -> Tuple[
        Dict[str, Any], float]:
        """状态转移函数"""
        new_state = {
            'node_bikes': state['node_bikes'].copy(),
            'arc_bikes': state['arc_bikes'].copy(),
            'vehicle': state['vehicle'].copy()
        }

        # 处理节点库存决策
        if current_node != 0:
            current_bikes = new_state['node_bikes'].get(current_node, 0)
            node_capacity = self.network_data['node_capacities'].get(current_node, 100)

            if inventory_decision > 0:
                actual_load = min(inventory_decision, current_bikes,
                                  self.params['Q'] - new_state['vehicle']['current_load'])
                new_state['node_bikes'][current_node] = current_bikes - actual_load
                new_state['vehicle']['current_load'] += actual_load
            elif inventory_decision < 0:
                actual_unload = min(-inventory_decision, new_state['vehicle']['current_load'],
                                    node_capacity - current_bikes)
                new_state['node_bikes'][current_node] = current_bikes + actual_unload
                new_state['vehicle']['current_load'] -= actual_unload

        # 计算时间
        travel_time = self.network_data['travel_times'].get((current_node, next_node), 10)
        inventory_time = abs(inventory_decision) * self.params['τ_N']
        en_route_time = en_route_decision * self.params['τ_A']
        total_decision_time = travel_time + inventory_time + en_route_time
        end_time = current_time + total_decision_time

        # 处理途中决策
        if en_route_decision > 0:
            arc_key = (current_node, next_node)
            arc_bikes = new_state['arc_bikes'].get(arc_key, 0)
            actual_en_route_load = min(en_route_decision, arc_bikes,
                                       self.params['Q'] - new_state['vehicle']['current_load'])
            new_state['arc_bikes'][arc_key] = arc_bikes - actual_en_route_load
            new_state['vehicle']['current_load'] += actual_en_route_load

        # 模拟供需事件
        satisfied_demand = self.simulate_demand_supply_events(new_state, current_time, end_time)
        return new_state, satisfied_demand


class PFAStrategy:
    """PFA核心策略类"""

    def __init__(self, operational_params: Dict[str, Any], policy_params: Dict[str, Any], network_data: Dict[str, Any]):
        self.operational_params = operational_params
        self.policy_params = policy_params
        self.network_data = network_data
        self.node_capacities = network_data['node_capacities']

    def inventory_decision(self, current_time: float, current_node: int,
                           node_bikes: Dict[int, int], vehicle_capacity: int) -> int:
        """库存决策"""
        if current_node == 0:
            return 0

        # 获取供需率
        supply_rate = self.get_supply_rate(current_node, current_time)
        demand_rate = self.get_demand_rate(current_node, current_time)

        # 判断节点类型
        if supply_rate > demand_rate:
            theta_low = self.policy_params['θ⁺_low']
            theta_high = theta_low + self.policy_params['θ⁺_high_gap']
        elif supply_rate < demand_rate:
            theta_low = self.policy_params['θ⁻_low']
            theta_high = theta_low + self.policy_params['θ⁻_high_gap']
        else:
            theta_low = self.policy_params['θ⁺_low']
            theta_high = theta_low + self.policy_params['θ⁺_high_gap']

        # 计算阈值
        L_it, U_it = self.calculate_inventory_thresholds(
            current_node, current_time, theta_low, theta_high
        )

        current_bikes = node_bikes.get(current_node, 0)
        node_capacity = self.node_capacities.get(current_node, 0)

        # 双阈值决策
        if current_bikes < L_it:
            unload_amount = min(L_it - current_bikes, self.operational_params['Q'] - vehicle_capacity)
            return -unload_amount
        elif current_bikes > U_it:
            load_amount = min(current_bikes - U_it, node_capacity - current_bikes, vehicle_capacity)
            return load_amount
        else:
            return 0

    def calculate_inventory_thresholds(self, node_id: int, current_time: float,
                                       theta_low: float, theta_high: float) -> Tuple[float, float]:
        """计算库存阈值"""
        tau_L = self.policy_params['τ_L']
        alpha_sum = 0
        time_points = np.arange(current_time, min(current_time + tau_L, self.operational_params['T']), 5)

        for t in time_points:
            supply_rate = self.get_supply_rate(node_id, t)
            demand_rate = self.get_demand_rate(node_id, t)
            alpha = abs(supply_rate - demand_rate) + 1
            alpha_sum += alpha

        alpha_avg = alpha_sum / len(time_points) if len(time_points) > 0 else 1
        node_capacity = self.node_capacities.get(node_id, 100)

        L_it = min(node_capacity, theta_low * alpha_avg * node_capacity)
        U_it = min(node_capacity, theta_high * alpha_avg * node_capacity)
        return L_it, U_it

    def routing_decision(self, current_time: float, current_node: int,
                         vehicle_capacity: int, inventory_decision: int,
                         node_bikes: Dict[int, int], arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """路径决策"""
        post_capacity = vehicle_capacity - inventory_decision
        capacity_ratio = post_capacity / self.operational_params['Q']
        gathering_points = [i for i in range(1, 16)]

        if capacity_ratio <= self.policy_params['δ']:
            return self._select_demand_node(current_time, current_node, post_capacity, node_bikes, gathering_points)
        else:
            return self._select_supply_node(current_time, current_node, post_capacity, node_bikes, gathering_points)

    def _select_demand_node(self, current_time: float, current_node: int,
                            post_capacity: int, node_bikes: Dict[int, int], gathering_points: List[int]) -> int:
        """选择需求节点"""
        unmet_demands = {}
        for node_id in gathering_points:
            if node_id == current_node:
                continue
            travel_time = self.network_data['travel_times'].get((current_node, node_id), 10)
            unmet_demand = self.calculate_anticipated_unmet_demand(node_id, current_time + travel_time,
                                                                   self.policy_params['τ_D'])
            unmet_demands[node_id] = unmet_demand

        if not unmet_demands:
            return random.choice(gathering_points)

        max_unmet = max(unmet_demands.values())
        candidate_nodes = [n for n, u in unmet_demands.items() if u >= self.policy_params['β'] * max_unmet]
        return min(candidate_nodes, key=lambda n: self.network_data['travel_times'].get((current_node, n), 10))

    def _select_supply_node(self, current_time: float, current_node: int,
                            post_capacity: int, node_bikes: Dict[int, int], gathering_points: List[int]) -> int:
        """选择供应节点"""
        loading_amounts = {}
        for node_id in gathering_points:
            if node_id == current_node:
                continue
            travel_time = self.network_data['travel_times'].get((current_node, node_id), 10)
            node_arrival_time = current_time + travel_time
            node_inventory_decision = self.inventory_decision(node_arrival_time, node_id, node_bikes, post_capacity)
            node_load = max(node_inventory_decision, 0)
            loading_amounts[node_id] = node_load

        if not loading_amounts:
            return random.choice(gathering_points)

        max_load = max(loading_amounts.values())
        candidate_nodes = [n for n, l in loading_amounts.items() if l >= self.policy_params['β'] * max_load]
        return min(candidate_nodes, key=lambda n: self.network_data['travel_times'].get((current_node, n), 10))

    def en_route_decision(self, current_node: int, next_node: int,
                          post_capacity: int, arc_bikes: Dict[Tuple[int, int], int]) -> int:
        """途中决策"""
        travel_time = self.network_data['travel_times'].get((current_node, next_node), 10)
        time_limit_load = math.floor(self.policy_params['φ'] * travel_time / self.operational_params['τ_A'])
        arc_bike_count = arc_bikes.get((current_node, next_node), 0)
        return min(time_limit_load, arc_bike_count, post_capacity)

    def calculate_anticipated_unmet_demand(self, node_id: int, start_time: float, lookahead_time: float) -> float:
        """计算预期未满足需求"""
        total_unmet = 0
        current_inventory = 0
        time_slots = np.arange(start_time, start_time + lookahead_time, 5)

        for t in time_slots:
            if t >= self.operational_params['T']:
                break
            supply = self.get_supply_rate(node_id, t) * 5
            demand = self.get_demand_rate(node_id, t) * 5
            unmet = max(0, demand - max(0, current_inventory + supply))
            total_unmet += unmet
            net_change = supply - demand
            current_inventory = max(0, min(self.node_capacities.get(node_id, 100), current_inventory + net_change))

        return total_unmet

    def get_supply_rate(self, node_id: int, current_time: float) -> float:
        """获取节点供应率"""
        time_key = self._get_time_index(current_time)
        return self.network_data['node_supply_rates'].get(node_id, {}).get(time_key, 0.0)

    def get_demand_rate(self, node_id: int, current_time: float) -> float:
        """获取节点需求率"""
        time_key = self._get_time_index(current_time)
        return self.network_data['node_demand_rates'].get(node_id, {}).get(time_key, 0.0)

    def _get_time_index(self, current_time: float) -> str:
        """时间格式转换"""
        start_hour = 11
        total_minutes = start_hour * 60 + current_time
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        rounded_minutes = (minutes // 5) * 5
        return f"{hours:02d}:{rounded_minutes:02d}"


def load_and_process_data():
    """精简版数据加载函数"""
    print("加载数据...")

    # 节点容量
    node_capacities = {
        1: 54, 2: 60, 3: 65, 4: 50, 5: 59, 6: 158, 7: 71, 8: 135,
        9: 57, 10: 139, 11: 31, 12: 71, 13: 65, 14: 108, 15: 112
    }

    # 读取核心数据
    try:
        node_departure_df = pd.read_excel('arrival_departure_rates.xlsx', sheet_name='节点_离开率')
        node_arrival_df = pd.read_excel('arrival_departure_rates.xlsx', sheet_name='节点_到达率')
        arc_rates_df = pd.read_excel('arrival_departure_rates.xlsx', sheet_name='弧段')
        arc_lengths_df = pd.read_csv('arc_lengths(1).csv')
        node_initial_df = pd.read_excel('nodeinitial.xlsx')
        arc_initial_df = pd.read_excel('arcinitial.xlsx')
    except Exception as e:
        print(f"数据读取错误: {e}")
        return None

    # 处理弧段长度
    arc_lengths = {}
    for _, row in arc_lengths_df.iterrows():
        try:
            arc_id_str = row['arc_id']
            if isinstance(arc_id_str, str) and arc_id_str.startswith('(') and arc_id_str.endswith(')'):
                arc_id = eval(arc_id_str)
                arc_lengths[arc_id] = float(row['length'])
        except:
            continue

    # 处理节点供需率
    node_demand_rates = {}
    node_supply_rates = {}
    for _, row in node_departure_df.dropna(subset=['Node_ID']).iterrows():
        try:
            node_id = int(float(row['Node_ID']))
            node_demand_rates[node_id] = {col: float(row[col]) if pd.notna(row[col]) else 0.0 for col in
                                          node_departure_df.columns[1:]}
        except:
            continue

    for _, row in node_arrival_df.dropna(subset=['Node_ID']).iterrows():
        try:
            node_id = int(float(row['Node_ID']))
            node_supply_rates[node_id] = {col: float(row[col]) if pd.notna(row[col]) else 0.0 for col in
                                          node_arrival_df.columns[1:]}
        except:
            continue

    # 处理旅行时间
    travel_times = {}
    for _, row in arc_rates_df.dropna(subset=['Arc_ID']).iterrows():
        try:
            arc_id_str = row['Arc_ID']
            if isinstance(arc_id_str, str) and arc_id_str.startswith('(') and arc_id_str.endswith(')'):
                arc_id = eval(arc_id_str)
                if arc_id in arc_lengths:
                    travel_times[arc_id] = arc_lengths[arc_id] / (15 / 60)
                else:
                    travel_times[arc_id] = 10
        except:
            continue

    # 补充缺失的旅行时间
    for i in range(1, 16):
        for j in range(1, 16):
            if i != j and (i, j) not in travel_times:
                travel_times[(i, j)] = 10

    # 处理初始分布
    initial_node_bikes = {}
    initial_arc_bikes = {}
    if len(node_initial_df.columns) > 1:
        first_col = node_initial_df.columns[2]
        for _, row in node_initial_df.dropna(subset=['node_id']).iterrows():
            try:
                node_id = int(float(row['node_id']))
                initial_node_bikes[node_id] = int(float(row[first_col])) if pd.notna(row[first_col]) else 0
            except:
                continue

    if len(arc_initial_df.columns) > 1:
        first_col = arc_initial_df.columns[2]
        for _, row in arc_initial_df.dropna(subset=['arc_id']).iterrows():
            try:
                arc_id_str = row['arc_id']
                if isinstance(arc_id_str, str) and arc_id_str.startswith('(') and arc_id_str.endswith(')'):
                    arc_id = eval(arc_id_str)
                    initial_arc_bikes[arc_id] = int(float(row[first_col])) if pd.notna(row[first_col]) else 0
            except:
                continue

    return {
        'node_capacities': node_capacities,
        'travel_times': travel_times,
        'node_demand_rates': node_demand_rates,
        'node_supply_rates': node_supply_rates,
        'initial_node_bikes': initial_node_bikes,
        'initial_arc_bikes': initial_arc_bikes
    }


def calculate_stability_score(satisfaction_ratios: List[float]) -> float:
    """计算稳定性分数（基于变异系数）"""
    if len(satisfaction_ratios) < 2 or np.mean(satisfaction_ratios) == 0:
        return 0.0
    cv = np.std(satisfaction_ratios) / np.mean(satisfaction_ratios)
    return 1 / (1 + cv)  # 转换为0-1之间的分数，越高越稳定


def optimize_pfa_params(simulator: BikeSharingSimulator):
    """核心优化函数（多随机种子稳定性）"""
    print(f"开始优化，使用{len(RANDOM_SEEDS)}个随机种子验证稳定性")

    # 创建Optuna研究
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=42)
    )

    def objective(trial):
        """目标函数：平均性能(0.7) + 稳定性(0.3)"""
        # 参数搜索空间
        policy_params = {
            'θ⁺_low': trial.suggest_float('θ⁺_low', 0.0, 2.0),
            'θ⁺_high_gap': trial.suggest_float('θ⁺_high_gap', 0.0, 2.0),
            'θ⁻_low': trial.suggest_float('θ⁻_low', 0.0, 2.0),
            'θ⁻_high_gap': trial.suggest_float('θ⁻_high_gap', 0.0, 2.0),
            'τ_L': trial.suggest_int('τ_L', 0, 120),
            'τ_D': trial.suggest_int('τ_D', 0, 120),
            'δ': trial.suggest_float('δ', 0.1, 1),
            'β': trial.suggest_float('β', 0.1, 1),
            'φ': trial.suggest_float('φ', 0.1, 2)
        }

        # 多随机种子验证
        satisfaction_ratios = []
        for seed in RANDOM_SEEDS:
            result = simulator.simulate(policy_params, random_seed=seed)
            satisfaction_ratios.append(result['demand_satisfaction_ratio'])

        # 计算综合分数
        mean_perf = np.mean(satisfaction_ratios)
        stability = calculate_stability_score(satisfaction_ratios)
        total_score = mean_perf * PERFORMANCE_WEIGHT + stability * STABILITY_WEIGHT

        # 仅打印关键进度
        if trial.number % 10 == 0:
            print(f"试验 {trial.number}: 平均性能={mean_perf:.2%}, 稳定性={stability:.3f}, 综合分数={total_score:.3f}")

        return total_score

    # 执行优化
    study.optimize(objective, n_trials=N_TRIALS)

    # 验证最优参数
    best_params = study.best_params
    print("\n最优参数验证：")
    final_ratios = []
    for seed in RANDOM_SEEDS:
        res = simulator.simulate(best_params, random_seed=seed)
        final_ratios.append(res['demand_satisfaction_ratio'])

    final_mean = np.mean(final_ratios)
    final_stability = calculate_stability_score(final_ratios)
    final_score = final_mean * PERFORMANCE_WEIGHT + final_stability * STABILITY_WEIGHT

    # 保存结果
    result = {
        'best_params': best_params,
        'mean_performance': final_mean,
        'stability_score': final_stability,
        'total_score': final_score,
        'detailed_results': {f'seed_{seed}': ratio for seed, ratio in zip(RANDOM_SEEDS, final_ratios)},
        'optimization_completed_at': datetime.now().isoformat()
    }

    # 保存JSON结果
    with open('pfa_optimization_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 打印最终结果
    print("\n优化完成！")
    print(f"最优参数: {best_params}")
    print(f"平均需求满足率: {final_mean:.2%}")
    print(f"稳定性分数: {final_stability:.3f}")
    print(f"综合分数: {final_score:.3f}")
    print(f"各随机种子结果: {[f'{x:.2%}' for x in final_ratios]}")

    return best_params


def main():
    """主函数"""
    # 加载数据
    processed_data = load_and_process_data()
    if processed_data is None:
        return

    # 运营参数
    operational_params = {
        'Q': 25,
        'τ_N': 0.25,
        'τ_A': 0.5,
        'T': 270
    }

    # 初始化仿真器
    simulator = BikeSharingSimulator(
        network_data=processed_data,
        initial_state={
            'node_bikes': processed_data['initial_node_bikes'],
            'arc_bikes': processed_data['initial_arc_bikes']
        },
        params=operational_params
    )

    # 执行优化
    optimize_pfa_params(simulator)


if __name__ == "__main__":
    main()