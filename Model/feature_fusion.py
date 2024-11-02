# models/feature_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FusionConfig:
    """特征融合配置"""
    # 编码配置
    max_periods: int = 7
    encoding_dim: int = 64
    num_attention_heads: int = 8
    
    # 网络配置
    hidden_dim: int = 256
    dropout_rate: float = 0.2
    activation: str = 'gelu'
    
    # 周期权重配置
    period_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.period_weights is None:
            self.period_weights = {
                '5min': 0.15,
                '15min': 0.15,
                '1h': 0.15,
                '1d': 0.25,
                '1w': 0.20,
                '1M': 0.10
            }

class PeriodEncoder(nn.Module):
    """
    周期编码模块
    """
    def __init__(self, config: FusionConfig):
        super(PeriodEncoder, self).__init__()
        self.config = config
        
        # 位置编码矩阵
        self.register_buffer(
            'position_encoding',
            self._create_position_encoding()
        )
        
        # 周期嵌入层
        self.period_embedding = nn.Embedding(
            config.max_periods,
            config.encoding_dim
        )
        
        # 编码增强层
        self.encoding_enhancement = nn.Sequential(
            nn.Linear(config.encoding_dim * 2, config.encoding_dim),
            self._get_activation(),
            nn.LayerNorm(config.encoding_dim),
            nn.Dropout(config.dropout_rate)
        )
        
    def _create_position_encoding(self) -> torch.Tensor:
        """
        创建位置编码矩阵
        """
        position = torch.arange(self.config.max_periods).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.config.encoding_dim, 2) * 
            (-math.log(10000.0) / self.config.encoding_dim)
        )
        
        pos_encoding = torch.zeros(self.config.max_periods, self.config.encoding_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
        
    def _get_activation(self) -> nn.Module:
        """
        获取激活函数
        """
        if self.config.activation == 'gelu':
            return nn.GELU()
        return nn.ReLU()
    
    def forward(self, period_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 获取位置编码
        position_codes = self.position_encoding[period_ids]
        
        # 获取周期嵌入
        period_codes = self.period_embedding(period_ids)
        
        # 合并编码
        combined = torch.cat([position_codes, period_codes], dim=-1)
        enhanced = self.encoding_enhancement(combined)
        
        return enhanced

class MultiPeriodAttention(nn.Module):
    """
    多周期注意力机制
    """
    def __init__(self, config: FusionConfig):
        super(MultiPeriodAttention, self).__init__()
        self.config = config
        
        # 多头注意力层
        self.self_attention = nn.MultiheadAttention(
            config.encoding_dim,
            config.num_attention_heads,
            dropout=config.dropout_rate
        )
        
        # 层标准化
        self.norm1 = nn.LayerNorm(config.encoding_dim)
        self.norm2 = nn.LayerNorm(config.encoding_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(config.encoding_dim, config.hidden_dim),
            self._get_activation(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.encoding_dim)
        )
        
        # 周期注意力分数
        self.period_attention = nn.Sequential(
            nn.Linear(config.encoding_dim, config.num_attention_heads),
            nn.Softmax(dim=-1)
        )
        
    def _get_activation(self) -> nn.Module:
        if self.config.activation == 'gelu':
            return nn.GELU()
        return nn.ReLU()
    
    def forward(self, 
                features: torch.Tensor,
                period_encoding: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        """
        # 自注意力
        attended_features, attention_weights = self.self_attention(
            features, features, features,
            key_padding_mask=mask,
            need_weights=True
        )
        
        # 残差连接和标准化
        features = self.norm1(features + attended_features)
        
        # 前馈网络
        ff_output = self.feed_forward(features)
        features = self.norm2(features + ff_output)
        
        # 计算周期注意力分数
        period_scores = self.period_attention(period_encoding)
        
        # 应用周期注意力
        weighted_features = features * period_scores.unsqueeze(-1)
        
        return weighted_features, attention_weights

class FeatureFusion(nn.Module):
    """
    特征融合模块
    """
    def __init__(self, config: FusionConfig):
        super(FeatureFusion, self).__init__()
        self.config = config
        
        # 周期编码器
        self.period_encoder = PeriodEncoder(config)
        
        # 多周期注意力
        self.attention = MultiPeriodAttention(config)
        
        # 特征增强层
        self.feature_enhancement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.encoding_dim, config.hidden_dim),
                self._get_activation(),
                nn.LayerNorm(config.hidden_dim),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dim, config.encoding_dim)
            )
            for _ in range(3)  # 使用3层特征增强
        ])
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.encoding_dim * 2, config.hidden_dim),
            self._get_activation(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.encoding_dim)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(config.encoding_dim, config.hidden_dim),
            self._get_activation(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.encoding_dim)
        )
        
    def _get_activation(self) -> nn.Module:
        if self.config.activation == 'gelu':
            return nn.GELU()
        return nn.ReLU()
    
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        batch_size = next(iter(features_dict.values())).size(0)
        device = next(iter(features_dict.values())).device
        
        # 准备周期ID
        period_ids = torch.tensor(
            [self._period_to_id(p) for p in features_dict.keys()],
            device=device
        )
        
        # 获取周期编码
        period_encoding = self.period_encoder(period_ids)
        
        # 堆叠特征
        stacked_features = torch.stack(list(features_dict.values()), dim=1)
        
        # 应用注意力机制
        attended_features, attention_weights = self.attention(
            stacked_features, period_encoding
        )
        
        # 特征增强
        enhanced_features = attended_features
        for enhancement_layer in self.feature_enhancement:
            enhanced_features = enhancement_layer(enhanced_features)
        
        # 融合短期和长期特征
        short_term_features = self._fuse_period_features(
            enhanced_features,
            ['5min', '15min', '1h'],
            features_dict
        )
        
        long_term_features = self._fuse_period_features(
            enhanced_features,
            ['1d', '1w', '1M'],
            features_dict
        )
        
        # 最终融合
        combined_features = torch.cat([short_term_features, long_term_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        
        # 输出处理
        output_features = self.output_layer(fused_features)
        
        return {
            'short_term': short_term_features,
            'long_term': long_term_features,
            'fused': output_features,
            'attention_weights': attention_weights
        }
    
    def _fuse_period_features(self,
                            features: torch.Tensor,
                            periods: List[str],
                            features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        融合特定周期的特征
        """
        period_indices = [i for i, p in enumerate(features_dict.keys()) if p in periods]
        if not period_indices:
            return torch.zeros_like(features[:, 0])
            
        period_features = features[:, period_indices]
        
        # 应用周期权重
        weights = torch.tensor(
            [self.config.period_weights[p] for p in periods if p in features_dict],
            device=features.device
        )
        weighted_features = period_features * weights.view(1, -1, 1)
        
        return weighted_features.mean(dim=1)
    
    def _period_to_id(self, period: str) -> int:
        """
        将周期转换为ID
        """
        period_mapping = {
            '5min': 0,
            '15min': 1,
            '1h': 2,
            '1d': 3,
            '1w': 4,
            '1M': 5,
            '1Q': 6
        }
        return period_mapping.get(period, -1)

class AdaptiveFeatureFusion(FeatureFusion):
    """
    自适应特征融合模块
    """
    def __init__(self, config: FusionConfig):
        super(AdaptiveFeatureFusion, self).__init__(config)
        
        # 市场状态感知层
        self.market_state_layer = nn.Sequential(
            nn.Linear(config.encoding_dim, config.hidden_dim),
            self._get_activation(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, len(config.period_weights))
        )
        
        # 权重学习历史
        self.weight_history = []
        
    def forward(self, 
                features_dict: Dict[str, torch.Tensor],
                market_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播，包含市场状态自适应
        """
        # 基础特征融合
        fusion_output = super().forward(features_dict)
        
        if market_state is not None:
            # 根据市场状态调整周期权重
            adaptive_weights = self.market_state_layer(market_state)
            adaptive_weights = F.softmax(adaptive_weights, dim=-1)
            
            # 记录权重变化
            self.weight_history.append(adaptive_weights.detach().cpu().numpy())
            
            # 重新加权融合特征
            weighted_features = fusion_output['fused'] * adaptive_weights.unsqueeze(-1)
            fusion_output['fused'] = weighted_features.mean(dim=1)
        
        return fusion_output
    
    def get_weight_statistics(self) -> Dict[str, np.ndarray]:
        """
        获取权重变化统计信息
        """
        if not self.weight_history:
            return {}
            
        weight_history = np.stack(self.weight_history)
        return {
            'mean_weights': np.mean(weight_history, axis=0),
            'std_weights': np.std(weight_history, axis=0),
            'weight_evolution': weight_history
        }

def build_feature_fusion(config: FusionConfig) -> nn.Module:
    """
    构建特征融合模块
    """
    return AdaptiveFeatureFusion(config)