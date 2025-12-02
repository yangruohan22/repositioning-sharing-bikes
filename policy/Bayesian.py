import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
import math
import random
from datetime import datetime, timedelta
import warnings

# 添加 Optuna 优化库和可视化库
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

warnings.filterwarnings('ignore')


class EnhancedConvergenceAnalyzer:
    """增强版收敛性分析器"""

    def __init__(self):
        self.optimization_history = []
        self.convergence_metrics = {}
        self.parameter_importance = {}
        self.convergence_speed_metrics = {}

    def record_trial(self, trial_number: int, performance: float, params: Dict[str, Any],
                     trial_duration: float = None, additional_metrics: Dict[str, Any] = None):
        """记录每次试验的结果"""
        record = {
            'trial_number': trial_number,
            'performance': performance,
            'params': params.copy(),
            'timestamp': datetime.now()
        }

        if trial_duration is not None:
            record['trial_duration'] = trial_duration

        if additional_metrics is not None:
            record.update(additional_metrics)

        self.optimization_history.append(record)

    def analyze_parameter_importance(self, top_k: int = 5) -> Dict[str, float]:
        """分析参数重要性"""
        if len(self.optimization_history) < 10:
            return {}

        # 准备数据
        performances = [trial['performance'] for trial in self.optimization_history]
        param_names = list(self.optimization_history[0]['params'].keys())

        # 计算每个参数与性能的相关性
        importance_scores = {}

        for param in param_names:
            param_values = [trial['params'][param] for trial in self.optimization_history]

            # 计算Pearson相关系数
            if len(set(param_values)) > 1:  # 确保参数有变化
                correlation = np.corrcoef(param_values, performances)[0, 1]
                importance_scores[param] = abs(correlation)
            else:
                importance_scores[param] = 0

        # 归一化重要性分数
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {k: v / total_importance for k, v in importance_scores.items()}

        # 排序并选择前k个
        sorted_importance = dict(sorted(importance_scores.items(),
                                        key=lambda x: x[1], reverse=True)[:top_k])

        self.parameter_importance = sorted_importance
        return sorted_importance

    def analyze_convergence_speed(self) -> Dict[str, Any]:
        """分析收敛速度"""
        if len(self.optimization_history) < 10:
            return {}

        performances = [trial['performance'] for trial in self.optimization_history]
        best_performance = max(performances)

        # 计算达到不同性能水平所需的试验次数
        convergence_thresholds = {
            '50%': 0.5 * best_performance,
            '80%': 0.8 * best_performance,
            '90%': 0.9 * best_performance,
            '95%': 0.95 * best_performance,
            '99%': 0.99 * best_performance
        }

        convergence_speed = {}
        for threshold_name, threshold_value in convergence_thresholds.items():
            trial_to_reach = next((i for i, perf in enumerate(performances)
                                   if perf >= threshold_value), None)
            convergence_speed[f'trials_to_{threshold_name}'] = trial_to_reach

        # 计算收敛速度指标
        if convergence_speed['trials_to_95%']:
            efficiency_95 = convergence_speed['trials_to_95%'] / len(performances)
        else:
            efficiency_95 = 1.0

        # 计算改进速度（前50%试验 vs 后50%试验）
        mid_point = len(performances) // 2
        if mid_point > 0:
            early_improvement = (max(performances[:mid_point]) - performances[0]) / performances[0] if performances[
                                                                                                           0] > 0 else 0
            late_improvement = (performances[-1] - max(performances[:mid_point])) / max(
                performances[:mid_point]) if max(performances[:mid_point]) > 0 else 0
        else:
            early_improvement = late_improvement = 0

        self.convergence_speed_metrics = {
            **convergence_speed,
            'efficiency_95': efficiency_95,
            'early_improvement_rate': early_improvement,
            'late_improvement_rate': late_improvement,
            'overall_improvement': (performances[-1] - performances[0]) / performances[0] if performances[0] > 0 else 0
        }

        return self.convergence_speed_metrics

    def analyze_convergence(self, window_size: int = 50) -> Dict[str, Any]:
        """分析收敛性 - 增强版本"""
        if len(self.optimization_history) < 10:
            return {"error": "数据不足进行收敛性分析"}

        performances = [trial['performance'] for trial in self.optimization_history]
        trial_numbers = [trial['trial_number'] for trial in self.optimization_history]

        # 计算基本收敛指标
        best_performance = max(performances)
        final_performance = performances[-1]
        initial_performance = performances[0]

        # 找到性能达到不同阈值的试验次数
        thresholds = {
            '90%': 0.9 * best_performance,
            '95%': 0.95 * best_performance,
            '99%': 0.99 * best_performance
        }

        trials_to_threshold = {}
        for name, threshold in thresholds.items():
            trials_to_threshold[f'trials_to_{name}'] = next(
                (i for i, perf in enumerate(performances) if perf >= threshold), None)

        # 计算性能改进统计
        improvements = []
        for i in range(1, len(performances)):
            if performances[i - 1] > 0:
                improvement = (performances[i] - performances[i - 1]) / performances[i - 1]
                improvements.append(improvement)
            else:
                improvements.append(0)

        # 计算稳定性指标
        stability_window = min(30, len(performances) // 3)
        if stability_window > 5:
            final_performances = performances[-stability_window:]
            performance_stability = np.std(final_performances) / np.mean(final_performances) if np.mean(
                final_performances) > 0 else 0
        else:
            performance_stability = 0

        # 检测收敛平台
        convergence_detected = self._detect_convergence_plateau(performances, window_size=20, tolerance=0.005)

        # 计算收敛质量指标
        if trials_to_threshold['trials_to_95%']:
            convergence_efficiency = trials_to_threshold['trials_to_95%'] / len(performances)
        else:
            convergence_efficiency = 1.0

        # 计算探索-利用平衡
        exploration_ratio = self._calculate_exploration_ratio()

        convergence_metrics = {
            'total_trials': len(performances),
            'initial_performance': initial_performance,
            'best_performance': best_performance,
            'final_performance': final_performance,
            'overall_improvement': (
                                               best_performance - initial_performance) / initial_performance if initial_performance > 0 else 0,
            'performance_gap': best_performance - final_performance,
            'relative_gap': (best_performance - final_performance) / best_performance if best_performance > 0 else 0,
            'mean_improvement_rate': np.mean(improvements) if improvements else 0,
            'std_improvement_rate': np.std(improvements) if improvements else 0,
            'performance_stability': performance_stability,
            'convergence_detected': convergence_detected,
            'convergence_efficiency': convergence_efficiency,
            'exploration_ratio': exploration_ratio,
            **trials_to_threshold
        }

        self.convergence_metrics = convergence_metrics

        # 分析参数重要性和收敛速度
        self.analyze_parameter_importance()
        self.analyze_convergence_speed()

        return convergence_metrics

    def _calculate_exploration_ratio(self) -> float:
        """计算探索-利用平衡比率"""
        if len(self.optimization_history) < 20:
            return 0.5

        # 计算参数空间的探索程度
        param_exploration_scores = []
        param_names = list(self.optimization_history[0]['params'].keys())

        for param in param_names:
            param_values = [trial['params'][param] for trial in self.optimization_history]
            # 计算参数值的变异系数作为探索指标
            if np.mean(param_values) > 0:
                cv = np.std(param_values) / np.mean(param_values)
                param_exploration_scores.append(min(cv, 1.0))  # 限制在0-1范围内
            else:
                param_exploration_scores.append(0)

        return np.mean(param_exploration_scores) if param_exploration_scores else 0.5

    def _detect_convergence_plateau(self, performances: List[float], window_size: int = 20,
                                    tolerance: float = 0.01) -> bool:
        """检测收敛平台 - 增强版本"""
        if len(performances) < window_size:
            return False

        # 使用多个窗口大小检测平台
        window_sizes = [10, 20, 30]
        plateau_detected_count = 0

        for ws in window_sizes:
            if len(performances) >= ws:
                recent_performances = performances[-ws:]
                max_recent = max(recent_performances)
                min_recent = min(recent_performances)

                if max_recent > 0:
                    relative_variation = (max_recent - min_recent) / max_recent
                    if relative_variation <= tolerance:
                        plateau_detected_count += 1

        # 如果多个窗口都检测到平台，则认为真正收敛
        return plateau_detected_count >= 2

    def plot_enhanced_convergence(self, save_path: str = None):
        """绘制增强版收敛分析图"""
        if len(self.optimization_history) < 5:
            print("数据不足绘制收敛曲线")
            return

        performances = [trial['performance'] for trial in self.optimization_history]
        trial_numbers = [trial['trial_number'] for trial in self.optimization_history]

        # 计算滚动统计量
        rolling_best = []
        current_best = performances[0]
        for perf in performances:
            if perf > current_best:
                current_best = perf
            rolling_best.append(current_best)

        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('贝叶斯优化收敛性综合分析', fontsize=16, fontweight='bold')

        # 子图1: 性能收敛曲线
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(trial_numbers, performances, 'b-', alpha=0.4, label='每次试验性能', linewidth=1)
        ax1.plot(trial_numbers, rolling_best, 'r-', linewidth=2, label='历史最佳性能')

        # 标记关键收敛点
        colors = ['green', 'orange', 'red']
        thresholds = ['90%', '95%', '99%']
        for i, threshold in enumerate(thresholds):
            trial_num = self.convergence_metrics.get(f'trials_to_{threshold}')
            if trial_num is not None:
                ax1.axvline(x=trial_num, color=colors[i], linestyle='--', alpha=0.7,
                            label=f'达到{threshold}最佳性能')

        ax1.set_xlabel('试验次数')
        ax1.set_ylabel('性能（满足需求）')
        ax1.set_title('性能收敛曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 性能改进率分析
        ax2 = plt.subplot(2, 3, 2)
        improvements = []
        for i in range(1, len(rolling_best)):
            if rolling_best[i - 1] > 0:
                improvement = (rolling_best[i] - rolling_best[i - 1]) / rolling_best[i - 1]
                improvements.append(improvement * 100)
            else:
                improvements.append(0)

        # 使用移动平均
        window = min(10, len(improvements) // 10)
        if window > 1:
            improvements_smooth = pd.Series(improvements).rolling(window=window, center=True).mean()
            ax2.plot(trial_numbers[1:], improvements_smooth, 'g-', linewidth=2, label='平滑改进率')
        else:
            ax2.plot(trial_numbers[1:], improvements, 'g-', linewidth=2, label='改进率')

        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.5)
        ax2.set_xlabel('试验次数')
        ax2.set_ylabel('性能改进率 (%)')
        ax2.set_title('性能改进率分析')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 子图3: 参数重要性
        ax3 = plt.subplot(2, 3, 3)
        if self.parameter_importance:
            params = list(self.parameter_importance.keys())
            importance_scores = list(self.parameter_importance.values())

            y_pos = np.arange(len(params))
            bars = ax3.barh(y_pos, importance_scores, align='center', alpha=0.6, color='steelblue')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(params)
            ax3.set_xlabel('相对重要性')
            ax3.set_title('参数重要性分析')

            # 在条形上添加数值
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{width:.2f}', ha='left', va='center')

            ax3.grid(True, alpha=0.3, axis='x')

        # 子图4: 参数空间探索（3D散点图，选择前3个重要参数）
        ax4 = plt.subplot(2, 3, 4, projection='3d')
        if len(self.optimization_history) > 0 and len(self.parameter_importance) >= 3:
            top_params = list(self.parameter_importance.keys())[:3]

            param1_vals = [trial['params'][top_params[0]] for trial in self.optimization_history]
            param2_vals = [trial['params'][top_params[1]] for trial in self.optimization_history]
            param3_vals = [trial['params'][top_params[2]] for trial in self.optimization_history]

            scatter = ax4.scatter(param1_vals, param2_vals, param3_vals,
                                  c=performances, cmap='viridis', alpha=0.6)
            ax4.set_xlabel(top_params[0])
            ax4.set_ylabel(top_params[1])
            ax4.set_zlabel(top_params[2])
            ax4.set_title('参数空间探索 (3D)')
            plt.colorbar(scatter, ax=ax4, label='性能')

        # 子图5: 收敛速度分析
        ax5 = plt.subplot(2, 3, 5)
        convergence_data = []
        labels = []

        for threshold in ['50%', '80%', '90%', '95%', '99%']:
            trial_num = self.convergence_speed_metrics.get(f'trials_to_{threshold}')
            if trial_num is not None:
                convergence_data.append(trial_num)
                labels.append(threshold)

        if convergence_data:
            bars = ax5.bar(labels, convergence_data, color=['lightblue', 'lightgreen', 'gold', 'orange', 'coral'])
            ax5.set_xlabel('性能阈值')
            ax5.set_ylabel('达到阈值的试验次数')
            ax5.set_title('收敛速度分析')

            # 在条形上添加数值
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}', ha='center', va='bottom')

            ax5.grid(True, alpha=0.3, axis='y')

        # 子图6: 收敛指标总结
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        # 创建指标表格
        metrics_data = [
            ['总试验次数', f"{self.convergence_metrics.get('total_trials', 0)}"],
            ['收敛效率', f"{self.convergence_metrics.get('convergence_efficiency', 0):.2%}"],
            ['最终性能差距', f"{self.convergence_metrics.get('relative_gap', 0):.2%}"],
            ['性能稳定性', f"{self.convergence_metrics.get('performance_stability', 0):.4f}"],
            ['探索比率', f"{self.convergence_metrics.get('exploration_ratio', 0):.2f}"],
            ['检测到收敛', '是' if self.convergence_metrics.get('convergence_detected') else '否']
        ]

        table = ax6.table(cellText=metrics_data,
                          colLabels=['指标', '值'],
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax6.set_title('收敛指标总结')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"增强版收敛分析图已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_parameter_evolution(self, save_path: str = None):
        """绘制参数演化图"""
        if len(self.optimization_history) < 10:
            print("数据不足绘制参数演化图")
            return

        param_names = list(self.optimization_history[0]['params'].keys())
        trial_numbers = [trial['trial_number'] for trial in self.optimization_history]

        n_params = len(param_names)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, param in enumerate(param_names):
            if i < len(axes):
                param_values = [trial['params'][param] for trial in self.optimization_history]

                # 绘制参数值
                axes[i].plot(trial_numbers, param_values, 'b-', alpha=0.7, linewidth=1)
                axes[i].set_xlabel('number of trials')
                axes[i].set_ylabel(param)
                axes[i].set_title(f'{param} parameter evolution')
                axes[i].grid(True, alpha=0.3)

                # 添加滚动平均线
                window = min(20, len(param_values) // 5)
                if window > 1:
                    rolling_mean = pd.Series(param_values).rolling(window=window).mean()
                    axes[i].plot(trial_numbers, rolling_mean, 'r-', linewidth=2,
                                 label=f'{window} rolling average')
                    axes[i].legend()

        # 隐藏多余的子图
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"参数演化图已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def print_detailed_convergence_report(self):
        """打印详细收敛性报告"""
        if not self.convergence_metrics:
            print("尚未进行收敛性分析")
            return

        metrics = self.convergence_metrics
        speed_metrics = self.convergence_speed_metrics

        print("\n" + "=" * 70)
        print("                详细收敛性分析报告")
        print("=" * 70)

        print(f"\n📊 基本统计:")
        print(f"   总试验次数: {metrics['total_trials']}")
        print(f"   初始性能: {metrics['initial_performance']:.2f}")
        print(f"   最佳性能: {metrics['best_performance']:.2f}")
        print(f"   最终性能: {metrics['final_performance']:.2f}")
        print(f"   总体改进: {metrics['overall_improvement']:.2%}")

        print(f"\n🎯 收敛性能:")
        print(f"   性能差距: {metrics['relative_gap']:.2%}")
        if metrics['trials_to_90%']:
            print(f"   达到90%最佳性能: 第{metrics['trials_to_90%']}次试验")
        if metrics['trials_to_95%']:
            print(f"   达到95%最佳性能: 第{metrics['trials_to_95%']}次试验")
        if metrics['trials_to_99%']:
            print(f"   达到99%最佳性能: 第{metrics['trials_to_99%']}次试验")

        print(f"\n📈 收敛质量:")
        print(f"   收敛效率: {metrics['convergence_efficiency']:.2%}")
        print(f"   性能稳定性: {metrics['performance_stability']:.4f}")
        print(f"   探索比率: {metrics['exploration_ratio']:.2f}")
        print(f"   检测到收敛: {'是' if metrics['convergence_detected'] else '否'}")

        print(f"\n⚡ 收敛速度:")
        if speed_metrics:
            print(f"   早期改进率: {speed_metrics.get('early_improvement_rate', 0):.2%}")
            print(f"   后期改进率: {speed_metrics.get('late_improvement_rate', 0):.2%}")
            print(f"   95%效率: {speed_metrics.get('efficiency_95', 0):.2%}")

        print(f"\n🔍 参数重要性 (前5名):")
        if self.parameter_importance:
            for param, importance in list(self.parameter_importance.items())[:5]:
                print(f"   {param}: {importance:.3f}")

        # 收敛性评级
        convergence_score = self._calculate_convergence_score()
        if convergence_score >= 0.8:
            convergence_rating = "优秀 🎉"
        elif convergence_score >= 0.6:
            convergence_rating = "良好 👍"
        elif convergence_score >= 0.4:
            convergence_rating = "一般 ✅"
        else:
            convergence_rating = "较差 ⚠️"

        print(f"\n🏆 收敛性综合评级: {convergence_rating} (得分: {convergence_score:.2f})")
        print("=" * 70)

    def _calculate_convergence_score(self) -> float:
        """计算收敛性综合得分"""
        if not self.convergence_metrics:
            return 0.0

        metrics = self.convergence_metrics

        # 评分组件
        efficiency_score = 1.0 - metrics['convergence_efficiency']  # 效率越高得分越高
        stability_score = 1.0 - min(metrics['performance_stability'] * 10, 1.0)  # 稳定性
        gap_score = 1.0 - metrics['relative_gap']  # 性能差距
        exploration_score = metrics['exploration_ratio']  # 探索程度

        # 加权平均
        weights = {
            'efficiency': 0.3,
            'stability': 0.25,
            'gap': 0.25,
            'exploration': 0.2
        }

        total_score = (efficiency_score * weights['efficiency'] +
                       stability_score * weights['stability'] +
                       gap_score * weights['gap'] +
                       exploration_score * weights['exploration'])

        return min(max(total_score, 0.0), 1.0)

    def save_convergence_report(self, filepath: str):
        """保存收敛性报告到文件"""
        if not self.convergence_metrics:
            print("没有收敛性数据可保存")
            return

        report = {
            'convergence_metrics': self.convergence_metrics,
            'parameter_importance': self.parameter_importance,
            'convergence_speed_metrics': self.convergence_speed_metrics,
            'optimization_summary': {
                'total_trials': len(self.optimization_history),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"收敛性报告已保存到: {filepath}")


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


# 修改后的 OptunaOptimizer 类，集成增强的收敛性分析
class EnhancedOptunaOptimizer:
    def __init__(self, parameter_space: Dict[str, List[Any]], optuna_params: Dict[str, Any]):
        self.parameter_space = parameter_space
        self.optuna_params = optuna_params
        self.study = None
        self.convergence_analyzer = EnhancedConvergenceAnalyzer()
        self.optimization_start_time = None

    def optimize(self, simulator: BikeSharingSimulator) -> Dict[str, Any]:
        """执行Optuna优化 - 增强版本，集成收敛性分析"""
        n_trials = self.optuna_params.get('n_trials', 100)
        n_startup_trials = self.optuna_params.get('n_startup_trials', 20)
        random_state = self.optuna_params.get('random_state', 42)

        print(f"开始Optuna优化，总试验次数: {n_trials}")
        self.optimization_start_time = datetime.now()

        # 创建Optuna研究
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(n_startup_trials=n_startup_trials, seed=random_state)
        )

        # 定义目标函数
        def objective(trial):
            trial_start = datetime.now()

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
            n_replications = 3
            performances = []
            satisfaction_ratios = []

            for rep in range(n_replications):
                result = simulator.simulate(policy_params, random_seed=rep)
                performances.append(result['total_satisfied_demand'])
                satisfaction_ratios.append(result['demand_satisfaction_ratio'])

            avg_performance = np.mean(performances)
            avg_satisfaction = np.mean(satisfaction_ratios)

            trial_duration = (datetime.now() - trial_start).total_seconds()

            # 记录到收敛性分析器
            additional_metrics = {
                'satisfaction_ratio': avg_satisfaction,
                'n_replications': n_replications,
                'performance_std': np.std(performances)
            }

            self.convergence_analyzer.record_trial(
                trial.number, avg_performance, policy_params,
                trial_duration, additional_metrics
            )

            # 定期打印进度
            if trial.number % 10 == 0:
                current_time = datetime.now()
                elapsed = (current_time - self.optimization_start_time).total_seconds()
                estimated_total = elapsed / (trial.number + 1) * n_trials
                remaining = estimated_total - elapsed

                print(f"进度: {trial.number + 1}/{n_trials} | "
                      f"当前性能: {avg_performance:.2f} | "
                      f"满足率: {avg_satisfaction:.1%} | "
                      f"预计剩余: {remaining / 60:.1f}分钟")

            return avg_performance

        # 执行优化
        self.study.optimize(objective, n_trials=n_trials)

        # 进行收敛性分析
        print("\n正在进行收敛性分析...")
        self.convergence_analyzer.analyze_convergence()

        # 生成报告和可视化
        self.generate_convergence_report()

        # 提取最佳参数
        best_params = self.study.best_params

        optimization_duration = (datetime.now() - self.optimization_start_time).total_seconds()
        print(f"\n🎉 Optuna优化完成！")
        print(f"   总耗时: {optimization_duration / 60:.1f}分钟")
        print(f"   最佳试验: {self.study.best_trial.number}")
        print(f"   最佳性能: {self.study.best_value:.2f}")
        print(f"   最佳参数: {best_params}")

        return best_params

    def generate_convergence_report(self):
        """生成完整的收敛性报告"""
        # 打印详细报告
        self.convergence_analyzer.print_detailed_convergence_report()

        # 生成可视化
        print("\n生成收敛性可视化...")
        self.convergence_analyzer.plot_enhanced_convergence('enhanced_convergence_analysis.png')
        self.convergence_analyzer.plot_parameter_evolution('parameter_evolution.png')

        # 保存报告
        self.convergence_analyzer.save_convergence_report('convergence_report.json')

        # 保存优化历史
        self.save_enhanced_optimization_history()

    def save_enhanced_optimization_history(self):
        """保存增强的优化历史 - 修复版"""
        if self.study is None:
            return None

        # 直接从收敛性分析器中获取历史数据
        history_data = []

        for record in self.convergence_analyzer.optimization_history:
            history_record = {
                'trial_number': record['trial_number'],
                'performance': record['performance'],
                'params': json.dumps(record['params']),  # 将参数字典转为JSON字符串
                'timestamp': record['timestamp'].isoformat(),
                'satisfaction_ratio': record.get('satisfaction_ratio'),
                'performance_std': record.get('performance_std'),
                'trial_duration': record.get('trial_duration'),
                'n_replications': record.get('n_replications')
            }
            history_data.append(history_record)

        history_df = pd.DataFrame(history_data)
        history_df.to_excel('enhanced_optimization_history.xlsx', index=False)
        print("增强版优化历史已保存到 'enhanced_optimization_history.xlsx'")

        return history_df

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化总结"""
        if self.study is None:
            return {}

        return {
            'best_performance': self.study.best_value,
            'best_params': self.study.best_params,
            'best_trial_number': self.study.best_trial.number,
            'total_trials': len(self.study.trials),
            'convergence_metrics': self.convergence_analyzer.convergence_metrics,
            'parameter_importance': self.convergence_analyzer.parameter_importance,
            'optimization_duration': (datetime.now() - self.optimization_start_time).total_seconds()
            if self.optimization_start_time else 0
        }

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


# 在 main 函数中使用增强的优化器
def main():
    """主函数 - 使用增强的收敛性分析"""
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

        # Optuna优化参数
        optuna_params = {
            'n_trials': 5000,  # 为了演示，减少试验次数
            'n_startup_trials': 20,
            'random_state': 42
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

        # 数据验证
        print("验证数据率合理性...")
        simulator.validate_data_rates()

        print("验证需求计算...")
        simulator.validate_demand_calculation()

        print("开始增强版Optuna优化...")
        # 使用增强版优化器
        optimizer = EnhancedOptunaOptimizer(pfa_parameter_space, optuna_params)
        best_params = optimizer.optimize(simulator)

        # 最终验证
        print("\n进行最终验证...")
        final_results = []
        for i in range(5):
            result = simulator.simulate(best_params, random_seed=1000 + i)
            final_results.append(result)
            print(f"验证运行 {i + 1}: 满足需求={result['total_satisfied_demand']:.2f}, "
                  f"满足率={result['demand_satisfaction_ratio']:.1%}")

        avg_performance = np.mean([r['total_satisfied_demand'] for r in final_results])
        avg_satisfaction = np.mean([r['demand_satisfaction_ratio'] for r in final_results])

        print(f"\n最终验证结果:")
        print(
            f"平均需求满足量: {avg_performance:.2f} ± {np.std([r['total_satisfied_demand'] for r in final_results]):.2f}")
        print(f"平均需求满足率: {avg_satisfaction:.1%}")

        # 保存最佳参数
        best_params_df = pd.DataFrame([best_params])
        best_params_df.to_excel('best_pfa_parameters_enhanced.xlsx', index=False)
        print("最佳参数已保存到 'best_pfa_parameters_enhanced.xlsx'")

        # 获取优化总结
        summary = optimizer.get_optimization_summary()
        print(f"\n优化总结:")
        print(f"总试验次数: {summary['total_trials']}")
        print(f"最佳性能: {summary['best_performance']:.2f}")
        print(f"优化耗时: {summary['optimization_duration'] / 60:.1f}分钟")

    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()