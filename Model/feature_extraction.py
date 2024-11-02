# models/feature_extraction.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class FeatureExtractionConfig:
    """特征提取配置"""
    input_dim: int = 29  # 根据设计文档2.1.4节
    hidden_dim: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.2
    num_layers: int = 3
    num_periods: int = 6
    activation: str = 'gelu'
    
    # 注意力配置
    attention_dropout: float = 0.1
    attention_temperature: float = 0.1
    
    # 自适应权重配置
    adaptation_momentum: float = 0.9
    min_period_weight: float = 0.05

class MultiHeadPeriodAttention(nn.Module):
    """
    增强的多头周期注意力机制
    """
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        assert self.head_dim * config.num_heads == config.hidden_dim
        
        # Q/K/V投影矩阵
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # 注意力温度参数
        self.temperature = nn.Parameter(torch.ones(1) * config.attention_temperature)
        
        # 相对位置编码
        self.rel_pos_embedding = nn.Parameter(
            torch.randn(2 * config.num_periods - 1, self.head_dim)
        )
        
        # 周期重要性嵌入
        self.period_importance = nn.Parameter(
            torch.ones(config.num_periods) / config.num_periods
        )
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # 投影查询、键、值
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 添加相对位置信息
        rel_pos_bias = self._get_relative_positions(x)
        scores = scores + rel_pos_bias
        
        # 应用温度缩放
        scores = scores * self.temperature
        
        # 应用周期重要性
        period_weights = F.softmax(self.period_importance, dim=0)
        scores = scores * period_weights.view(1, 1, -1, 1)
        
        # 应用mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        return self.out_proj(output), attention_weights
    
    def _get_relative_positions(self, x: torch.Tensor) -> torch.Tensor:
        """计算相对位置编码"""
        seq_length = x.size(1)
        range_vec = torch.arange(seq_length)
        range_mat = range_vec.unsqueeze(-1) - range_vec.unsqueeze(0)
        range_mat = range_mat + self.config.num_periods - 1
        rel_pos = self.rel_pos_embedding[range_mat]
        return rel_pos.unsqueeze(0).unsqueeze(0)

class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合层
    """
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__()
        self.config = config
        
        # 市场状态感知层
        self.market_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            self._get_activation(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout_rate)
        )
        
        # 周期权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.num_periods),
            nn.Softplus()
        )
        
        # 特征转换层
        self.feature_transform = nn.ModuleDict({
            f'period_{i}': nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                self._get_activation(),
                nn.LayerNorm(config.hidden_dim)
            )
            for i in range(config.num_periods)
        })
        
        # 动态权重历史
        self.register_buffer('weight_history', 
                           torch.ones(config.num_periods) / config.num_periods)
        self.register_buffer('weight_momentum', 
                           torch.zeros(config.num_periods))
        
    def _get_activation(self) -> nn.Module:
        if self.config.activation == 'gelu':
            return nn.GELU()
        return nn.ReLU()
        
    def forward(self, 
                features_dict: Dict[str, torch.Tensor],
                market_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 编码市场状态
        if market_state is None:
            market_state = torch.zeros(
                features_dict[list(features_dict.keys())[0]].size(0),
                self.config.hidden_dim,
                device=next(iter(features_dict.values())).device
            )
        
        market_encoding = self.market_encoder(market_state)
        
        # 生成动态权重
        raw_weights = self.weight_generator(market_encoding)
        
        # 应用动量更新
        self.weight_momentum = (self.config.adaptation_momentum * self.weight_momentum + 
                              (1 - self.config.adaptation_momentum) * (raw_weights.mean(0) - self.weight_history))
        self.weight_history = self.weight_history + self.weight_momentum
        
        # 确保最小权重并归一化
        weights = F.softmax(self.weight_history.clamp(min=self.config.min_period_weight), dim=0)
        
        # 转换和融合特征
        transformed_features = {}
        weighted_sum = 0
        
        for i, (period, features) in enumerate(features_dict.items()):
            # 特征转换
            transformed = self.feature_transform[f'period_{i}'](features)
            transformed_features[period] = transformed
            
            # 加权求和
            weighted_sum = weighted_sum + transformed * weights[i].view(1, 1, 1)
        
        return weighted_sum, {
            'weights': weights,
            'transformed_features': transformed_features,
            'weight_momentum': self.weight_momentum
        }

class EnhancedFeatureExtractor(nn.Module):
    """
    增强的特征提取器
    """
    def __init__(self, config: FeatureExtractionConfig):
        super().__init__()
        self.config = config
        
        # 特征编码层
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            self._get_activation(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout_rate)
        )
        
        # 多头注意力层
        self.attention_layers = nn.ModuleList([
            MultiHeadPeriodAttention(config)
            for _ in range(config.num_layers)
        ])
        
        # 前馈网络层
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                self._get_activation(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dim * 4, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim)
            )
            for _ in range(config.num_layers)
        ])
        
        # 特征融合层
        self.feature_fusion = AdaptiveFeatureFusion(config)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            self._get_activation(),
            nn.LayerNorm(config.hidden_dim)
        )
        
    def _get_activation(self) -> nn.Module:
        if self.config.activation == 'gelu':
            return nn.GELU()
        return nn.ReLU()
    
    def forward(self, 
                features_dict: Dict[str, torch.Tensor],
                market_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # 编码特征
        encoded_features = {}
        attention_weights = []
        
        for period, features in features_dict.items():
            # 初始编码
            encoded = self.feature_encoder(features)
            
            # 应用注意力层
            for attention_layer, ff_layer in zip(self.attention_layers, 
                                               self.feed_forward_layers):
                attended, weights = attention_layer(encoded)
                attended = ff_layer(attended)
                encoded = encoded + attended
                attention_weights.append((period, weights))
            
            encoded_features[period] = encoded
        
        # 特征融合
        fused_features, fusion_info = self.feature_fusion(encoded_features, market_state)
        
        # 输出处理
        output = self.output_layer(fused_features)
        
        return output, {
            'attention_weights': attention_weights,
            'fusion_info': fusion_info,
            'encoded_features': encoded_features
        }

def build_feature_extractor(config: FeatureExtractionConfig) -> nn.Module:
    """构建特征提取器"""
    return EnhancedFeatureExtractor(config)