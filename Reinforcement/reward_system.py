# reinforcement/reward_system.py

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RewardConfig:
    """奖励系统配置"""
    # 仓位调整相关
    position_reward_scale: float = 1.0
    risk_reward_scale: float = 0.3
    stability_reward_scale: float = 0.2
    max_position_change: float = 0.3
    
    # 做T相关
    trading_reward_scale: float = 1.0
    min_profit_threshold: float = 0.002
    max_holding_time: int = 20
    
    # 交易成本
    transaction_cost: float = 0.001
    slippage: float = 0.0001
    
    # 风险控制
    max_drawdown_threshold: float = 0.1
    volatility_threshold: float = 0.3
    
    # 奖励衰减
    gamma: float = 0.99

class RewardCalculator:
    """
    强化学习奖励计算系统
    """
    def __init__(self, config: RewardConfig):
        self.config = config
        self.position_history = []
        self.trade_history = []
        self.market_states = []
        
    def calculate_position_reward(self,
                                old_position: float,
                                new_position: float,
                                price_change: float,
                                market_state: Dict[str, float]) -> Dict[str, float]:
        """
        计算仓位调整的奖励
        """
        # 基础收益奖励
        position_return = new_position * price_change * self.config.position_reward_scale
        
        # 交易成本
        position_change = abs(new_position - old_position)
        transaction_cost = position_change * self.config.transaction_cost
        slippage_cost = position_change * self.config.slippage * \
                       (1 + market_state.get('volatility', 0))
        
        # 风险控制奖励
        risk_reward = self._calculate_risk_reward(
            new_position, market_state, position_change)
        
        # 稳定性奖励
        stability_reward = self._calculate_stability_reward(
            position_change, market_state)
        
        # 更新历史记录
        self.position_history.append({
            'old_position': old_position,
            'new_position': new_position,
            'price_change': price_change,
            'market_state': market_state
        })
        
        return {
            'total_reward': position_return - transaction_cost - slippage_cost + 
                          risk_reward + stability_reward,
            'position_return': position_return,
            'transaction_cost': transaction_cost,
            'slippage_cost': slippage_cost,
            'risk_reward': risk_reward,
            'stability_reward': stability_reward
        }
    
    def _calculate_risk_reward(self,
                             position: float,
                             market_state: Dict[str, float],
                             position_change: float) -> float:
        """
        计算风险控制奖励
        """
        risk_reward = 0.0
        
        # 波动率风险
        volatility = market_state.get('volatility', 0)
        if volatility > self.config.volatility_threshold:
            # 在高波动时减仓给予奖励
            if position_change < 0:
                risk_reward += abs(position_change) * self.config.risk_reward_scale
        
        # 回撤风险
        drawdown = market_state.get('drawdown', 0)
        if drawdown > self.config.max_drawdown_threshold:
            # 在大回撤时减仓给予奖励
            if position_change < 0:
                risk_reward += abs(position_change) * self.config.risk_reward_scale
        
        # 趋势一致性奖励
        trend_strength = market_state.get('trend_strength', 0)
        trend_direction = np.sign(trend_strength)
        position_direction = np.sign(position_change)
        if trend_direction == position_direction:
            risk_reward += abs(position_change) * self.config.risk_reward_scale * \
                          abs(trend_strength)
        
        return risk_reward
    
    def _calculate_stability_reward(self,
                                  position_change: float,
                                  market_state: Dict[str, float]) -> float:
        """
        计算稳定性奖励
        """
        # 基础稳定性惩罚
        stability_penalty = abs(position_change) * self.config.stability_reward_scale
        
        # 根据市场状态调整惩罚力度
        market_volatility = market_state.get('volatility', 0)
        trend_strength = abs(market_state.get('trend_strength', 0))
        
        # 在低波动和强趋势时，频繁调整的惩罚更大
        if market_volatility < 0.2 and trend_strength > 0.7:
            stability_penalty *= 2
        
        return -stability_penalty

    def calculate_trading_reward(self,
                               action: str,
                               entry_price: float,
                               exit_price: float,
                               holding_time: int,
                               market_state: Dict[str, float]) -> Dict[str, float]:
        """
        计算做T操作的奖励
        """
        # 基础交易收益
        if action == 'buy':
            profit = (exit_price - entry_price) / entry_price
        elif action == 'sell':
            profit = (entry_price - exit_price) / entry_price
        else:  # hold
            profit = 0.0
            
        trading_return = profit * self.config.trading_reward_scale
        
        # 交易成本
        transaction_cost = 2 * self.config.transaction_cost if action != 'hold' else 0
        slippage_cost = 2 * self.config.slippage * \
                       (1 + market_state.get('volatility', 0)) if action != 'hold' else 0
        
        # 持仓时间惩罚
        time_penalty = self._calculate_time_penalty(holding_time, market_state)
        
        # 最小收益阈值奖励
        threshold_reward = self._calculate_threshold_reward(profit)
        
        # 更新交易历史
        self.trade_history.append({
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'holding_time': holding_time,
            'market_state': market_state
        })
        
        return {
            'total_reward': trading_return - transaction_cost - slippage_cost - 
                          time_penalty + threshold_reward,
            'trading_return': trading_return,
            'transaction_cost': transaction_cost,
            'slippage_cost': slippage_cost,
            'time_penalty': time_penalty,
            'threshold_reward': threshold_reward
        }
    
    def _calculate_time_penalty(self,
                              holding_time: int,
                              market_state: Dict[str, float]) -> float:
        """
        计算持仓时间惩罚
        """
        # 基础时间惩罚
        base_penalty = holding_time / self.config.max_holding_time
        
        # 根据市场状态调整惩罚
        volatility = market_state.get('volatility', 0)
        trend_strength = abs(market_state.get('trend_strength', 0))
        
        # 高波动时增加持仓时间惩罚
        if volatility > self.config.volatility_threshold:
            base_penalty *= 1.5
        
        # 弱趋势时增加持仓时间惩罚
        if trend_strength < 0.3:
            base_penalty *= 1.3
            
        return base_penalty
    
    def _calculate_threshold_reward(self, profit: float) -> float:
        """
        计算最小收益阈值奖励
        """
        if abs(profit) > self.config.min_profit_threshold:
            return 0.1 * np.sign(profit)
        return 0.0
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取奖励统计信息
        """
        if not self.position_history:
            return {}
            
        position_changes = [h['new_position'] - h['old_position'] 
                          for h in self.position_history]
        trade_profits = [h['exit_price'] - h['entry_price'] 
                        for h in self.trade_history if h['action'] != 'hold']
        
        return {
            'avg_position_change': np.mean(np.abs(position_changes)),
            'max_position_change': np.max(np.abs(position_changes)),
            'avg_trade_profit': np.mean(trade_profits) if trade_profits else 0,
            'win_rate': np.mean(np.array(trade_profits) > 0) if trade_profits else 0,
            'position_change_frequency': len(position_changes) / \
                                      max(1, len(self.market_states)),
            'trade_frequency': len(trade_profits) / max(1, len(self.market_states))
        }

class DualRewardNormalizer:
    """
    双重奖励标准化器
    """
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.position_rewards = []
        self.trading_rewards = []
        
    def normalize_position_reward(self, reward: float) -> float:
        """
        标准化仓位调整奖励
        """
        self.position_rewards.append(reward)
        if len(self.position_rewards) > self.window_size:
            self.position_rewards.pop(0)
            
        if not self.position_rewards:
            return reward
            
        mean = np.mean(self.position_rewards)
        std = np.std(self.position_rewards) + 1e-8
        return (reward - mean) / std
    
    def normalize_trading_reward(self, reward: float) -> float:
        """
        标准化交易奖励
        """
        self.trading_rewards.append(reward)
        if len(self.trading_rewards) > self.window_size:
            self.trading_rewards.pop(0)
            
        if not self.trading_rewards:
            return reward
            
        mean = np.mean(self.trading_rewards)
        std = np.std(self.trading_rewards) + 1e-8
        return (reward - mean) / std

class AdaptiveRewardAdjuster:
    """
    自适应奖励调整器
    """
    def __init__(self, 
                 base_config: RewardConfig,
                 adjustment_frequency: int = 1000):
        self.base_config = base_config
        self.adjustment_frequency = adjustment_frequency
        self.step_counter = 0
        self.performance_history = []
        
    def adjust_rewards(self,
                      performance_metrics: Dict[str, float]) -> RewardConfig:
        """
        根据性能指标调整奖励配置
        """
        self.step_counter += 1
        self.performance_history.append(performance_metrics)
        
        if self.step_counter % self.adjustment_frequency == 0:
            config = self.base_config
            
            # 根据胜率调整交易奖励比例
            win_rate = performance_metrics.get('win_rate', 0.5)
            if win_rate < 0.4:
                config.trading_reward_scale *= 0.9
            elif win_rate > 0.6:
                config.trading_reward_scale *= 1.1
            
            # 根据波动率调整风险奖励比例
            volatility = performance_metrics.get('volatility', 0)
            if volatility > self.base_config.volatility_threshold:
                config.risk_reward_scale *= 1.2
            
            # 根据回撤调整稳定性奖励比例
            drawdown = performance_metrics.get('drawdown', 0)
            if drawdown > self.base_config.max_drawdown_threshold:
                config.stability_reward_scale *= 1.2
            
            return config
            
        return self.base_config