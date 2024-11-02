# models/prediction_layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class PredictionConfig:
    """预测层配置"""
    # 网络配置
    input_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 3
    dropout_rate: float = 0.2
    
    # 预测目标配置
    price_periods: List[str] = None
    trend_classes: int = 3
    signal_classes: int = 3
    
    # 损失函数权重
    price_loss_weight: float = 1.0
    trend_loss_weight: float = 1.0
    signal_loss_weight: float = 1.0
    
    def __post_init__(self):
        if self.price_periods is None:
            self.price_periods = ['5min', '1h', '1d']

class PricePredictionBranch(nn.Module):
    """
    价格预测分支，负责多周期价格预测
    """
    def __init__(self, config: PredictionConfig):
        super(PricePredictionBranch, self).__init__()
        self.config = config
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 各周期独立预测头
        self.prediction_heads = nn.ModuleDict({
            period: self._build_prediction_head()
            for period in config.price_periods
        })
        
        # 波动率预测
        self.volatility_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus()  # 确保波动率为正
        )
        
    def _build_prediction_head(self) -> nn.Module:
        """构建预测头"""
        return nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim // 2, 2)  # 预测均值和标准差
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 共享特征提取
        shared_features = self.shared_layers(features)
        
        # 多周期价格预测
        predictions = {}
        for period in self.config.price_periods:
            head_output = self.prediction_heads[period](shared_features)
            predictions[f'{period}_mean'] = head_output[:, 0]
            predictions[f'{period}_std'] = F.softplus(head_output[:, 1])  # 确保标准差为正
            
        # 波动率预测
        predictions['volatility'] = self.volatility_head(shared_features)
        
        return predictions

class TrendPredictionBranch(nn.Module):
    """
    趋势预测分支，负责趋势方向和强度预测
    """
    def __init__(self, config: PredictionConfig):
        super(TrendPredictionBranch, self).__init__()
        self.config = config
        
        # 趋势特征提取
        self.trend_features = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 趋势方向预测（上涨、下跌、盘整）
        self.direction_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.trend_classes)
        )
        
        # 趋势强度预测
        self.strength_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 趋势持续性预测
        self.persistence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 提取趋势特征
        trend_features = self.trend_features(features)
        
        # 预测趋势方向
        direction_logits = self.direction_head(trend_features)
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        # 预测趋势强度
        strength = self.strength_head(trend_features)
        
        # 预测趋势持续性
        persistence = self.persistence_head(trend_features)
        
        return {
            'direction_logits': direction_logits,
            'direction_probs': direction_probs,
            'strength': strength,
            'persistence': persistence
        }

class SignalPredictionBranch(nn.Module):
    """
    买卖点信号预测分支
    """
    def __init__(self, config: PredictionConfig):
        super(SignalPredictionBranch, self).__init__()
        self.config = config
        
        # 信号特征提取
        self.signal_features = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 交易信号预测（买入、卖出、观望）
        self.signal_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.signal_classes)
        )
        
        # 趋势拐点预测
        self.reversal_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 信号强度预测
        self.strength_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 提取信号特征
        signal_features = self.signal_features(features)
        
        # 预测交易信号
        signal_logits = self.signal_head(signal_features)
        signal_probs = F.softmax(signal_logits, dim=-1)
        
        # 预测拐点概率
        reversal_prob = self.reversal_head(signal_features)
        
        # 预测信号强度
        strength = self.strength_head(signal_features)
        
        return {
            'signal_logits': signal_logits,
            'signal_probs': signal_probs,
            'reversal_prob': reversal_prob,
            'strength': strength
        }

class PredictionModule(nn.Module):
    """
    预测模块整合器
    """
    def __init__(self, config: PredictionConfig):
        super(PredictionModule, self).__init__()
        self.config = config
        
        # 各预测分支
        self.price_branch = PricePredictionBranch(config)
        self.trend_branch = TrendPredictionBranch(config)
        self.signal_branch = SignalPredictionBranch(config)
        
        # 预测结果融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 获取各分支预测
        price_predictions = self.price_branch(features)
        trend_predictions = self.trend_branch(features)
        signal_predictions = self.signal_branch(features)
        
        return {
            'price': price_predictions,
            'trend': trend_predictions,
            'signal': signal_predictions
        }
    
    def calculate_loss(self, 
                      predictions: Dict[str, torch.Tensor],
                      targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        losses = {}
        
        # 价格预测损失
        price_loss = 0
        for period in self.config.price_periods:
            mean_key = f'{period}_mean'
            std_key = f'{period}_std'
            if mean_key in predictions['price'] and period in targets:
                # 使用负对数似然损失
                price_loss += self._negative_gaussian_log_likelihood(
                    predictions['price'][mean_key],
                    predictions['price'][std_key],
                    targets[period]
                )
        losses['price_loss'] = price_loss * self.config.price_loss_weight
        
        # 趋势预测损失
        if 'direction' in targets:
            trend_loss = F.cross_entropy(
                predictions['trend']['direction_logits'],
                targets['direction']
            )
            losses['trend_loss'] = trend_loss * self.config.trend_loss_weight
        
        # 信号预测损失
        if 'signal' in targets:
            signal_loss = F.cross_entropy(
                predictions['signal']['signal_logits'],
                targets['signal']
            )
            losses['signal_loss'] = signal_loss * self.config.signal_loss_weight
        
        # 总损失
        losses['total_loss'] = sum(losses.values())
        
        return losses
    
    def _negative_gaussian_log_likelihood(self,
                                        mu: torch.Tensor,
                                        sigma: torch.Tensor,
                                        target: torch.Tensor) -> torch.Tensor:
        """计算负高斯对数似然损失"""
        return 0.5 * torch.log(2 * np.pi * sigma**2) + \
               0.5 * (target - mu)**2 / sigma**2

class PredictionAnalyzer:
    """
    预测结果分析器
    """
    def __init__(self):
        self.prediction_history = []
        
    def analyze_predictions(self, 
                          predictions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """分析预测结果"""
        # 记录预测历史
        self.prediction_history.append({
            k: v.detach().cpu().numpy() 
            for k, v in predictions.items()
        })
        
        # 分析预测一致性
        direction_probs = predictions['trend']['direction_probs']
        signal_probs = predictions['signal']['signal_probs']
        
        direction_entropy = -torch.sum(
            direction_probs * torch.log(direction_probs + 1e-8),
            dim=-1
        ).mean()
        
        signal_entropy = -torch.sum(
            signal_probs * torch.log(signal_probs + 1e-8),
            dim=-1
        ).mean()
        
        return {
            'direction_entropy': direction_entropy.item(),
            'signal_entropy': signal_entropy.item(),
            'prediction_confidence': self._calculate_confidence(predictions)
        }
    
    def _calculate_confidence(self, predictions: Dict[str, torch.Tensor]) -> float:
        """计算预测置信度"""
        # 综合考虑各个预测分支的置信度
        direction_conf = torch.max(
            predictions['trend']['direction_probs'],
            dim=-1
        )[0].mean()
        
        signal_conf = torch.max(
            predictions['signal']['signal_probs'],
            dim=-1
        )[0].mean()
        
        return (direction_conf + signal_conf).item() / 2

def build_prediction_module(config: PredictionConfig) -> nn.Module:
    """构建预测模块"""
    return PredictionModule(config)