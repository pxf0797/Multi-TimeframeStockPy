# reinforcement/adaptive_learning.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque, defaultdict

@dataclass
class AdaptiveLearningConfig:
    """自适应学习配置"""
    # 基础学习参数
    base_learning_rate: float = 0.001
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 0.01
    
    # 经验回放配置
    buffer_size: int = 100000
    batch_size: int = 64
    min_experiences: int = 1000
    
    # 探索配置
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.01
    epsilon_decay: float = 0.995
    
    # 优先级采样
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_increment: float = 0.001
    
    # 市场适应参数
    market_state_dim: int = 10
    adaptation_momentum: float = 0.9
    volatility_scaling: float = 2.0
    
    # 奖励调整参数
    reward_scaling: float = 1.0
    reward_momentum: float = 0.95
    max_reward_adjustment: float = 2.0

class PrioritizedExperienceBuffer:
    """
    优先级经验回放缓冲区
    """
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.experiences = []
        self.priorities = []
        self.position = 0
        
    def add(self, experience: Dict[str, Any], priority: float = None):
        """
        添加经验
        """
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
            
        if len(self.experiences) < self.config.buffer_size:
            self.experiences.append(experience)
            self.priorities.append(priority)
        else:
            self.experiences[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.config.buffer_size
        
    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        优先级采样
        """
        if len(self.experiences) < self.config.min_experiences:
            return None, None, None
            
        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.config.priority_alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(
            len(self.experiences),
            batch_size,
            p=probs,
            replace=False
        )
        
        # 计算重要性权重
        weights = (len(self.experiences) * probs[indices]) ** \
                 (-self.config.priority_beta)
        weights /= weights.max()
        
        # 获取经验
        batch = [self.experiences[idx] for idx in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新优先级
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.experiences)

class MarketStateAdapter:
    """
    市场状态适应器
    """
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.market_stats = defaultdict(lambda: deque(maxlen=1000))
        
    def adapt_state(self, 
                   state: Dict[str, torch.Tensor], 
                   market_condition: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        根据市场状况调整状态
        """
        adapted_state = {}
        
        # 更新市场统计
        for key, value in market_condition.items():
            self.market_stats[key].append(value)
        
        # 计算市场波动性因子
        volatility_factor = self._calculate_volatility_factor(market_condition)
        
        # 调整状态
        for key, tensor in state.items():
            # 标准化
            normalized = self._normalize_state(key, tensor)
            
            # 根据波动性调整
            adapted_state[key] = normalized * volatility_factor
            
        return adapted_state
    
    def _normalize_state(self, 
                        key: str, 
                        tensor: torch.Tensor) -> torch.Tensor:
        """
        标准化状态
        """
        if len(self.market_stats[key]) > 0:
            mean = np.mean(self.market_stats[key])
            std = np.std(self.market_stats[key]) + 1e-8
            return (tensor - mean) / std
        return tensor
    
    def _calculate_volatility_factor(self,
                                   market_condition: Dict[str, float]) -> float:
        """
        计算波动性因子
        """
        volatility = market_condition.get('volatility', 0.0)
        return 1.0 + (volatility * self.config.volatility_scaling)

class AdaptiveRewardScaler:
    """
    自适应奖励缩放器
    """
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.reward_history = deque(maxlen=1000)
        self.scaling_factor = 1.0
        
    def scale_reward(self, 
                    reward: float,
                    market_condition: Dict[str, float]) -> float:
        """
        缩放奖励
        """
        # 更新奖励历史
        self.reward_history.append(reward)
        
        # 计算新的缩放因子
        target_scale = self._calculate_target_scale(market_condition)
        
        # 使用动量更新缩放因子
        self.scaling_factor = (self.config.reward_momentum * self.scaling_factor +
                             (1 - self.config.reward_momentum) * target_scale)
        
        # 应用缩放
        scaled_reward = reward * self.scaling_factor
        
        return scaled_reward
    
    def _calculate_target_scale(self,
                              market_condition: Dict[str, float]) -> float:
        """
        计算目标缩放因子
        """
        if len(self.reward_history) < 10:
            return 1.0
            
        reward_std = np.std(self.reward_history)
        volatility = market_condition.get('volatility', 0.0)
        
        # 根据市场波动性调整目标缩放
        target_scale = 1.0 / (reward_std + 1e-8)
        target_scale *= (1.0 + volatility)
        
        # 限制缩放范围
        return np.clip(target_scale, 
                      1.0 / self.config.max_reward_adjustment,
                      self.config.max_reward_adjustment)

class AdaptiveLearningSystem:
    """
    自适应学习系统
    """
    def __init__(self, 
                 model: nn.Module,
                 config: AdaptiveLearningConfig):
        self.model = model
        self.config = config
        
        # 初始化组件
        self.experience_buffer = PrioritizedExperienceBuffer(config)
        self.market_adapter = MarketStateAdapter(config)
        self.reward_scaler = AdaptiveRewardScaler(config)
        
        # 学习参数
        self.learning_rate = config.base_learning_rate
        self.epsilon = config.initial_epsilon
        
        # 性能追踪
        self.performance_history = defaultdict(list)
        
    def adapt_learning_rate(self, performance_metrics: Dict[str, float]):
        """
        自适应调整学习率
        """
        # 记录性能
        for key, value in performance_metrics.items():
            self.performance_history[key].append(value)
            
        # 根据性能趋势调整学习率
        if len(self.performance_history['reward']) > 10:
            recent_rewards = self.performance_history['reward'][-10:]
            reward_trend = (recent_rewards[-1] - recent_rewards[0]) / 10
            
            if reward_trend > 0:
                # 性能改善，可能增加学习率
                self.learning_rate = min(
                    self.learning_rate * 1.1,
                    self.config.max_learning_rate
                )
            else:
                # 性能下降，减少学习率
                self.learning_rate = max(
                    self.learning_rate * 0.9,
                    self.config.min_learning_rate
                )
    
    def update_exploration(self, market_condition: Dict[str, float]):
        """
        更新探索策略
        """
        # 基础衰减
        self.epsilon = max(
            self.config.final_epsilon,
            self.epsilon * self.config.epsilon_decay
        )
        
        # 根据市场状态调整
        volatility = market_condition.get('volatility', 0.0)
        trend_strength = market_condition.get('trend_strength', 0.0)
        
        # 高波动时增加探索
        if volatility > 0.5:
            self.epsilon = min(self.epsilon * 1.2, 0.5)
        
        # 强趋势时减少探索
        if trend_strength > 0.7:
            self.epsilon = max(self.epsilon * 0.8, 0.1)
    
    def process_experience(self,
                         state: Dict[str, torch.Tensor],
                         action: torch.Tensor,
                         reward: float,
                         next_state: Dict[str, torch.Tensor],
                         market_condition: Dict[str, float]) -> Dict[str, float]:
        """
        处理经验
        """
        # 调整状态
        adapted_state = self.market_adapter.adapt_state(state, market_condition)
        adapted_next_state = self.market_adapter.adapt_state(next_state, market_condition)
        
        # 缩放奖励
        scaled_reward = self.reward_scaler.scale_reward(reward, market_condition)
        
        # 存储经验
        experience = {
            'state': adapted_state,
            'action': action,
            'reward': scaled_reward,
            'next_state': adapted_next_state,
            'market_condition': market_condition
        }
        
        # 计算经验优先级
        priority = self._calculate_priority(experience)
        self.experience_buffer.add(experience, priority)
        
        return {
            'original_reward': reward,
            'scaled_reward': scaled_reward,
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'priority': priority
        }
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        执行一步训练
        """
        if len(self.experience_buffer) < self.config.min_experiences:
            return None
            
        # 采样经验
        batch, indices, weights = self.experience_buffer.sample(
            self.config.batch_size)
        
        if batch is None:
            return None
            
        # 计算损失
        losses = self._calculate_losses(batch, weights)
        
        # 更新优先级
        new_priorities = losses['td_errors'].detach().cpu().numpy()
        self.experience_buffer.update_priorities(indices, new_priorities)
        
        return {
            'total_loss': losses['total_loss'].item(),
            'policy_loss': losses['policy_loss'].item(),
            'value_loss': losses['value_loss'].item(),
            'entropy_loss': losses['entropy_loss'].item()
        }
    
    def _calculate_priority(self, experience: Dict[str, Any]) -> float:
        """
        计算经验优先级
        """
        with torch.no_grad():
            # 使用TD误差作为优先级
            current_q = self.model(experience['state'])
            next_q = self.model(experience['next_state'])
            
            td_error = abs(experience['reward'] + 
                         self.config.reward_scaling * next_q.max() -
                         current_q[experience['action']]).item()
            
            return td_error + 1e-6
    
    def _calculate_losses(self,
                         batch: List[Dict[str, Any]],
                         weights: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        计算训练损失
        """
        # 准备批次数据
        states = {k: torch.stack([b['state'][k] for b in batch]) 
                 for k in batch[0]['state'].keys()}
        actions = torch.stack([b['action'] for b in batch])
        rewards = torch.tensor([b['reward'] for b in batch])
        next_states = {k: torch.stack([b['next_state'][k] for b in batch]) 
                      for k in batch[0]['next_state'].keys()}
        
        # 转换为张量
        weights = torch.FloatTensor(weights)
        
        # 计算当前Q值
        current_q = self.model(states)
        next_q = self.model(next_states)
        
        # 计算目标Q值
        target_q = rewards + self.config.reward_scaling * next_q.max(dim=1)[0]
        
        # TD误差
        td_errors = target_q - current_q.gather(1, actions.unsqueeze(1)).squeeze()
        
        # 策略损失
        policy_loss = (weights * td_errors.pow(2)).mean()
        
        # 价值损失
        value_loss = F.smooth_l1_loss(current_q, target_q.detach())
        
        # 熵损失（用于探索）
        entropy_loss = -(current_q * torch.log_softmax(current_q, dim=1)).sum(dim=1).mean()
        
        # 总损失
        total_loss = policy_loss + value_loss - 0.01 * entropy_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'td_errors': td_errors.abs()
        }
    
# reinforcement/adaptive_learning.py (continued)

    def get_learning_stats(self) -> Dict[str, Any]:
        """
        获取学习统计信息
        """
        recent_history = {
            k: v[-100:] for k, v in self.performance_history.items()
        }
        
        return {
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'buffer_size': len(self.experience_buffer),
            'recent_metrics': {
                k: {
                    'mean': np.mean(v),
                    'std': np.std(v),
                    'trend': (v[-1] - v[0]) / len(v) if len(v) > 1 else 0
                }
                for k, v in recent_history.items()
            },
            'scaling_factor': self.reward_scaler.scaling_factor
        }

class OnlineLearningOptimizer:
    """
    在线学习优化器
    """
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.performance_window = deque(maxlen=100)
        self.market_stats = defaultdict(lambda: deque(maxlen=1000))
        
    def optimize_hyperparameters(self,
                               performance_metrics: Dict[str, float],
                               market_condition: Dict[str, float]) -> Dict[str, float]:
        """
        优化超参数
        """
        # 更新性能窗口
        self.performance_window.append(performance_metrics)
        
        # 更新市场统计
        for key, value in market_condition.items():
            self.market_stats[key].append(value)
        
        # 计算性能趋势
        performance_trend = self._calculate_performance_trend()
        
        # 计算市场状态
        market_state = self._analyze_market_state()
        
        # 优化超参数
        optimized_params = self._optimize_parameters(
            performance_trend,
            market_state
        )
        
        return optimized_params
    
    def _calculate_performance_trend(self) -> Dict[str, float]:
        """
        计算性能趋势
        """
        if len(self.performance_window) < 2:
            return {'trend': 0.0, 'volatility': 0.0}
            
        # 转换为numpy数组
        metrics = np.array([[m[k] for k in m.keys()] 
                          for m in self.performance_window])
        
        # 计算趋势
        trend = np.mean(metrics[-10:] - metrics[-11:-1], axis=0)
        
        # 计算波动性
        volatility = np.std(metrics, axis=0)
        
        return {
            'trend': float(np.mean(trend)),
            'volatility': float(np.mean(volatility))
        }
    
    def _analyze_market_state(self) -> Dict[str, float]:
        """
        分析市场状态
        """
        market_state = {}
        
        for key, values in self.market_stats.items():
            if len(values) > 0:
                market_state[f'{key}_mean'] = np.mean(values)
                market_state[f'{key}_std'] = np.std(values)
                if len(values) > 1:
                    market_state[f'{key}_trend'] = (values[-1] - values[0]) / len(values)
                
        return market_state
    
    def _optimize_parameters(self,
                           performance_trend: Dict[str, float],
                           market_state: Dict[str, float]) -> Dict[str, float]:
        """
        优化参数
        """
        params = {}
        
        # 根据性能趋势调整学习率
        base_lr = self.config.base_learning_rate
        if performance_trend['trend'] > 0:
            # 性能提升，保持或略微增加学习率
            lr_multiplier = min(1.1 + performance_trend['trend'], 2.0)
        else:
            # 性能下降，减少学习率
            lr_multiplier = max(0.9 + performance_trend['trend'], 0.5)
            
        params['learning_rate'] = np.clip(
            base_lr * lr_multiplier,
            self.config.min_learning_rate,
            self.config.max_learning_rate
        )
        
        # 根据市场状态调整探索率
        volatility = market_state.get('volatility_mean', 0.0)
        trend_strength = abs(market_state.get('trend_mean', 0.0))
        
        if volatility > 0.5 or trend_strength < 0.3:
            # 高波动或弱趋势时增加探索
            params['epsilon'] = min(self.config.initial_epsilon,
                                  self.config.final_epsilon * 2)
        else:
            # 低波动或强趋势时减少探索
            params['epsilon'] = max(self.config.final_epsilon,
                                  self.config.initial_epsilon * 0.5)
        
        # 调整奖励缩放
        params['reward_scaling'] = self._optimize_reward_scaling(
            market_state,
            performance_trend
        )
        
        return params
    
    def _optimize_reward_scaling(self,
                               market_state: Dict[str, float],
                               performance_trend: Dict[str, float]) -> float:
        """
        优化奖励缩放因子
        """
        base_scale = self.config.reward_scaling
        
        # 根据市场波动性调整
        volatility = market_state.get('volatility_mean', 0.0)
        volatility_adjustment = 1.0 + (volatility * self.config.volatility_scaling)
        
        # 根据性能波动性调整
        performance_volatility = performance_trend['volatility']
        volatility_penalty = np.exp(-performance_volatility)
        
        # 计算最终缩放因子
        scaling = base_scale * volatility_adjustment * volatility_penalty
        
        return np.clip(scaling, 
                      1.0 / self.config.max_reward_adjustment,
                      self.config.max_reward_adjustment)

class AdaptivePolicyUpdater:
    """
    自适应策略更新器
    """
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.update_history = deque(maxlen=1000)
        
    def compute_update_frequency(self,
                               market_condition: Dict[str, float]) -> int:
        """
        计算策略更新频率
        """
        # 基础更新频率
        base_frequency = 10
        
        # 根据市场波动性调整
        volatility = market_condition.get('volatility', 0.0)
        if volatility > 0.5:
            # 高波动时增加更新频率
            frequency_multiplier = 2.0
        else:
            # 低波动时降低更新频率
            frequency_multiplier = 1.0
            
        return max(1, int(base_frequency * frequency_multiplier))
    
    def compute_update_magnitude(self,
                               performance_metrics: Dict[str, float]) -> float:
        """
        计算策略更新幅度
        """
        # 记录更新历史
        self.update_history.append(performance_metrics)
        
        if len(self.update_history) < 2:
            return 1.0
            
        # 计算性能改善
        recent_performance = np.mean([m['reward'] for m in self.update_history[-10:]])
        previous_performance = np.mean([m['reward'] for m in self.update_history[-20:-10]])
        
        performance_improvement = (recent_performance - previous_performance) / \
                                (abs(previous_performance) + 1e-8)
        
        # 调整更新幅度
        if performance_improvement > 0:
            # 性能提升时增加更新幅度
            magnitude = 1.0 + min(performance_improvement, 0.5)
        else:
            # 性能下降时减少更新幅度
            magnitude = 1.0 + max(performance_improvement, -0.5)
            
        return magnitude

def build_adaptive_learning_system(model: nn.Module,
                                 config: AdaptiveLearningConfig) -> AdaptiveLearningSystem:
    """
    构建自适应学习系统
    """
    learning_system = AdaptiveLearningSystem(model, config)
    optimizer = OnlineLearningOptimizer(config)
    policy_updater = AdaptivePolicyUpdater(config)
    
    return learning_system, optimizer, policy_updater

# reinforcement/meta_learning.py

class MetaLearningOptimizer:
    """
    元学习优化器
    用于在线调整学习策略
    """
    def __init__(self, config: AdaptiveLearningConfig):
        self.config = config
        self.strategy_pool = []
        self.strategy_performance = {}
        
    def generate_learning_strategy(self,
                                 market_condition: Dict[str, float],
                                 performance_history: List[Dict[str, float]]
                                 ) -> Dict[str, Any]:
        """
        生成学习策略
        """
        # 分析历史性能
        performance_patterns = self._analyze_performance_patterns(
            performance_history)
        
        # 生成新策略
        new_strategy = self._create_strategy(
            market_condition,
            performance_patterns
        )
        
        # 评估策略
        strategy_score = self._evaluate_strategy(
            new_strategy,
            performance_patterns
        )
        
        # 更新策略池
        self._update_strategy_pool(new_strategy, strategy_score)
        
        return new_strategy
    
    def _analyze_performance_patterns(self,
                                    performance_history: List[Dict[str, float]]
                                    ) -> Dict[str, Any]:
        """
        分析性能模式
        """
        if not performance_history:
            return {}
            
        patterns = {}
        
        # 计算性能趋势
        rewards = [p['reward'] for p in performance_history]
        patterns['reward_trend'] = np.polyfit(
            np.arange(len(rewards)),
            rewards,
            deg=1
        )[0]
        
        # 分析波动性
        patterns['reward_volatility'] = np.std(rewards)
        
        # 分析策略效果
        for metric in ['learning_rate', 'epsilon', 'reward_scaling']:
            if all(metric in p for p in performance_history):
                values = [p[metric] for p in performance_history]
                patterns[f'{metric}_effectiveness'] = np.corrcoef(
                    values[:-1],
                    rewards[1:]
                )[0, 1]
                
        return patterns
    
    def _create_strategy(self,
                        market_condition: Dict[str, float],
                        performance_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建新的学习策略
        """
        strategy = {}
        
        # 基于性能模式调整学习参数
        if 'reward_trend' in performance_patterns:
            trend = performance_patterns['reward_trend']
            
            # 调整学习率
            if trend > 0:
                strategy['learning_rate_multiplier'] = 1.1
            else:
                strategy['learning_rate_multiplier'] = 0.9
            
            # 调整探索率
            if trend > 0:
                strategy['epsilon_decay'] = self.config.epsilon_decay
            else:
                strategy['epsilon_decay'] = self.config.epsilon_decay * 0.9
        
        # 基于市场条件调整策略
        volatility = market_condition.get('volatility', 0.0)
        trend_strength = market_condition.get('trend_strength', 0.0)
        
        # 设置奖励调整
        strategy['reward_adjustment'] = 1.0 + (volatility * trend_strength)
        
        # 设置更新频率
        strategy['update_frequency'] = int(10 / (volatility + 0.1))
        
        return strategy
    
    def _evaluate_strategy(self,
                         strategy: Dict[str, Any],
                         performance_patterns: Dict[str, Any]) -> float:
        """
        评估策略质量
        """
        score = 0.0
        
        # 评估学习率调整
        if 'learning_rate_effectiveness' in performance_patterns:
            score += performance_patterns['learning_rate_effectiveness'] * \
                    strategy['learning_rate_multiplier']
        
        # 评估探索策略
        if 'reward_volatility' in performance_patterns:
            exploration_score = 1.0 / (1.0 + performance_patterns['reward_volatility'])
            score += exploration_score * strategy['epsilon_decay']
        
        # 评估更新频率
        frequency_score = np.exp(-abs(strategy['update_frequency'] - 10) / 10)
        score += frequency_score
        
        return score
    
    def _update_strategy_pool(self,
                            strategy: Dict[str, Any],
                            score: float):
        """
        更新策略池
        """
        self.strategy_pool.append(strategy)
        self.strategy_performance[str(strategy)] = score
        
        # 保持策略池大小
        if len(self.strategy_pool) > 100:
            # 移除最差策略
            worst_strategy = min(
                self.strategy_pool,
                key=lambda s: self.strategy_performance[str(s)]
            )
            self.strategy_pool.remove(worst_strategy)
            del self.strategy_performance[str(worst_strategy)]

def build_meta_learning_optimizer(config: AdaptiveLearningConfig) -> MetaLearningOptimizer:
    """
    构建元学习优化器
    """
    return MetaLearningOptimizer(config)