# reinforcement_learning.py

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple
import pandas as pd
from collections import deque
import random

class RewardCalculator:
    """奖励计算器"""
    def __init__(self,
                position_reward_weight: float = 0.5,
                trading_reward_weight: float = 0.5,
                risk_free_rate: float = 0.02,
                transaction_cost: float = 0.0003):
        self.position_reward_weight = position_reward_weight
        self.trading_reward_weight = trading_reward_weight
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
    
    def calculate_position_reward(self,
                                action: Dict[str, float],
                                state: Dict[str, float],
                                next_state: Dict[str, float]) -> float:
        """计算仓位调整奖励"""
        # 收益奖励
        profit = (next_state['portfolio_value'] - state['portfolio_value']) / state['portfolio_value']
        excess_return = profit - self.risk_free_rate / 252  # 超额收益
        
        # 风险控制奖励
        risk_penalty = -abs(action['position_change']) * 0.1
        
        # 仓位稳定性奖励
        stability_reward = -abs(action['position_change']) * 0.05
        
        # 长期趋势一致性奖励
        trend_reward = 0.1 if np.sign(action['position_change']) == np.sign(state['long_trend']) else -0.1
        
        total_reward = (
            0.5 * excess_return +
            0.2 * risk_penalty +
            0.2 * stability_reward +
            0.1 * trend_reward
        )
        
        return total_reward
    
    def calculate_trading_reward(self,
                            action: Dict[str, float],
                            state: Dict[str, float],
                            next_state: Dict[str, float]) -> float:
        """计算做T操作奖励"""
        # 短期收益奖励
        profit = (next_state['portfolio_value'] - state['portfolio_value']) / state['portfolio_value']
        
        # 交易成本惩罚
        cost_penalty = -abs(action['amount']) * self.transaction_cost
        
        # 持仓时间奖励
        holding_reward = 0.05 if action['type'] == 'hold' and state['volatility'] < 0.01 else 0
        
        # 市场时机奖励
        timing_reward = 0.1 if (action['type'] == 'buy' and state['short_trend'] > 0) or \
                              (action['type'] == 'sell' and state['short_trend'] < 0) else -0.1
        
        total_reward = (
            0.4 * profit +
            0.3 * cost_penalty +
            0.2 * holding_reward +
            0.1 * timing_reward
        )
        
        return total_reward

class ExperienceBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Tuple):
        """添加经验"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """采样经验"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class DualExplorationStrategy:
    """双重探索策略"""
    def __init__(self,
                epsilon_short_start: float = 0.9,
                epsilon_short_end: float = 0.1,
                epsilon_long_start: float = 0.5,
                epsilon_long_end: float = 0.05,
                decay_rate: float = 0.995):
        self.epsilon_short = epsilon_short_start
        self.epsilon_short_end = epsilon_short_end
        self.epsilon_long = epsilon_long_start
        self.epsilon_long_end = epsilon_long_end
        self.decay_rate = decay_rate
    
    def select_action(self,
                    q_values: np.ndarray,
                    trading_mode: str = 'short') -> int:
        """选择动作"""
        epsilon = self.epsilon_short if trading_mode == 'short' else self.epsilon_long
        
        if np.random.random() < epsilon:
            return np.random.randint(len(q_values))
        else:
            return np.argmax(q_values)
    
    def update_epsilon(self, performance_metrics: Dict[str, float]):
        """更新探索率"""
        # 短期探索率更新
        if performance_metrics['short_term_reward'] > 0:
            self.epsilon_short = max(
                self.epsilon_short_end,
                self.epsilon_short * self.decay_rate
            )
        else:
            self.epsilon_short = min(
                0.9,
                self.epsilon_short / self.decay_rate
            )
        
        # 长期探索率更新
        if performance_metrics['long_term_reward'] > 0:
            self.epsilon_long = max(
                self.epsilon_long_end,
                self.epsilon_long * self.decay_rate
            )
        else:
            self.epsilon_long = min(
                0.5,
                self.epsilon_long / self.decay_rate
            )

class OnlineAdapter:
    """在线自适应微调"""
    def __init__(self,
                learning_rate: float = 0.001,
                min_learning_rate: float = 0.0001,
                adaptation_threshold: float = 0.1):
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.adaptation_threshold = adaptation_threshold
        self.performance_history = []
    
    def adapt_parameters(self,
                        model,
                        current_performance: float):
        """参数自适应"""
        self.performance_history.append(current_performance)
        
        if len(self.performance_history) >= 100:
            recent_performance = np.mean(self.performance_history[-100:])
            historical_performance = np.mean(self.performance_history[:-100])
            
            # 性能下降时调整学习率
            if recent_performance < historical_performance - self.adaptation_threshold:
                self.learning_rate = max(
                    self.min_learning_rate,
                    self.learning_rate * 0.9
                )
                self._update_model_learning_rate(model)
    
    def _update_model_learning_rate(self, model):
        """更新模型学习率"""
        model.optimizer.learning_rate.assign(self.learning_rate)

class RLOptimizer:
    """强化学习优化器"""
    def __init__(self,
                model,
                reward_calculator: RewardCalculator,
                exploration_strategy: DualExplorationStrategy,
                online_adapter: OnlineAdapter,
                batch_size: int = 32,
                gamma: float = 0.99):
        self.model = model
        self.reward_calculator = reward_calculator
        self.exploration_strategy = exploration_strategy
        self.online_adapter = online_adapter
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.position_buffer = ExperienceBuffer()
        self.trading_buffer = ExperienceBuffer()
        
    def optimize_step(self,
                    state: Dict[str, np.ndarray],
                    action: Dict[str, float],
                    reward: float,
                    next_state: Dict[str, np.ndarray],
                    done: bool,
                    mode: str = 'short'):
        """优化步骤"""
        # 存储经验
        experience = (state, action, reward, next_state, done)
        if mode == 'short':
            self.trading_buffer.add(experience)
        else:
            self.position_buffer.add(experience)
        
        # 训练
        if len(self.trading_buffer) >= self.batch_size and mode == 'short':
            self._train_trading_network()
        
        if len(self.position_buffer) >= self.batch_size and mode == 'long':
            self._train_position_network()
        
        # 更新探索策略
        performance = {
            'short_term_reward': reward if mode == 'short' else 0,
            'long_term_reward': reward if mode == 'long' else 0
        }
        self.exploration_strategy.update_epsilon(performance)
        
        # 在线适应
        self.online_adapter.adapt_parameters(self.model, reward)
    
    def _train_trading_network(self):
        """训练做T网络"""
        self._train_network(self.trading_buffer, 'trading')
    
    def _train_position_network(self):
        """训练仓位网络"""
        self._train_network(self.position_buffer, 'position')
    
    def _train_network(self, buffer: ExperienceBuffer, network_type: str):
        """网络训练"""
        batch = buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为numpy数组
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # 计算目标Q值
        next_q_values = self.model.predict(next_states)
        target_q_values = rewards + (1 - dones) * self.gamma * np.max(next_q_values, axis=1)
        
        # 更新模型
        with tf.GradientTape() as tape:
            current_q_values = self.model(states)
            action_indices = [self._get_action_index(a) for a in actions]
            q_values = tf.gather(current_q_values, action_indices, axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        
        # 应用梯度
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    def _get_action_index(self, action: Dict[str, float]) -> int:
        """获取动作索引"""
        if action['type'] == 'buy':
            return 0
        elif action['type'] == 'sell':
            return 1
        else:
            return 2

def main():
    # 示例使用
    model = None  # 载入模型
    reward_calculator = RewardCalculator()
    exploration_strategy = DualExplorationStrategy()
    online_adapter = OnlineAdapter()
    
    optimizer = RLOptimizer(
        model=model,
        reward_calculator=reward_calculator,
        exploration_strategy=exploration_strategy,
        online_adapter=online_adapter
    )
    
    # 这里添加训练循环
    pass

if __name__ == "__main__":
    main()