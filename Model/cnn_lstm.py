# models/cnn_lstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class ParallelCNNLSTM(nn.Module):
    """
    用于处理短周期数据的并行CNN-LSTM结构
    按照设计文档2.2.3节实现
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(ParallelCNNLSTM, self).__init__()
        
        # CNN部分，严格按照设计文档参数
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_dim, 64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Layer 2
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Layer 3
            nn.Conv1d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # LSTM部分，按照设计文档参数
        self.lstm = nn.Sequential(
            nn.LSTM(input_dim, 128, num_layers=2, batch_first=True, 
                   bidirectional=False, dropout=0.2),
            nn.LSTM(128, 64, num_layers=1, batch_first=True, 
                   bidirectional=False, dropout=0.2)
        )
        
        # 特征融合层
        self.fusion = self._build_fusion_layer(256 + 64)
        
    def _build_fusion_layer(self, input_dim: int) -> nn.Module:
        """
        构建特征融合层，使用注意力机制
        """
        return nn.Sequential(
            # 自注意力层
            nn.MultiheadAttention(input_dim, num_heads=8, dropout=0.1),
            nn.LayerNorm(input_dim),
            
            # 前馈网络
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 4, input_dim),
            nn.LayerNorm(input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # CNN分支
        cnn_input = x.transpose(1, 2)  # [batch, features, seq_len]
        cnn_out = self.cnn(cnn_input)
        cnn_out = F.adaptive_avg_pool1d(cnn_out, 1).view(batch_size, -1)
        
        # LSTM分支
        lstm_out, _ = self.lstm[0](x)
        lstm_out, _ = self.lstm[1](lstm_out)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步
        
        # 特征融合
        combined = torch.cat([cnn_out, lstm_out], dim=1).unsqueeze(0)
        fused, _ = self.fusion[0](combined, combined, combined)
        fused = fused.squeeze(0)
        
        # 应用其余的融合层
        for layer in self.fusion[1:]:
            fused = layer(fused)
            
        return fused

class SerialCNNLSTM(nn.Module):
    """
    用于处理长周期数据的串行CNN-LSTM结构
    按照设计文档2.2.4节实现
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(SerialCNNLSTM, self).__init__()
        
        # CNN前处理，按照设计文档参数
        self.cnn = nn.ModuleList([
            # Layer 1
            nn.Sequential(
                nn.Conv1d(input_dim, 32, kernel_size=5, padding='same'),
                nn.BatchNorm1d(32),
                nn.ReLU()
            ),
            # Layer 2
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=5, padding='same'),
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
        ])
        
        # 残差连接
        self.residual = nn.Conv1d(input_dim, 64, kernel_size=1)
        
        # LSTM主体，按照设计文档参数
        self.lstm_layers = nn.ModuleList([
            # Bidirectional LSTM
            nn.LSTM(64, 128, bidirectional=True, batch_first=True),
            # Regular LSTM layers
            nn.LSTM(256, 96, batch_first=True),
            nn.LSTM(96, 64, batch_first=True)
        ])
        
        # 注意力机制
        self.attention = self._build_attention_layer(64)
        
    def _build_attention_layer(self, dim: int) -> nn.Module:
        """
        构建自注意力层
        """
        return nn.Sequential(
            nn.MultiheadAttention(dim, num_heads=4, dropout=0.1),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN前处理
        cnn_input = x.transpose(1, 2)
        
        # 应用CNN层
        cnn_out = cnn_input
        for cnn_layer in self.cnn:
            cnn_out = cnn_layer(cnn_out)
        
        # 残差连接
        residual = self.residual(cnn_input)
        cnn_out = cnn_out + residual
        
        # 转回序列格式
        lstm_input = cnn_out.transpose(1, 2)
        
        # 应用LSTM层
        # Bidirectional LSTM
        lstm_out, _ = self.lstm_layers[0](lstm_input)
        
        # Regular LSTM layers
        for lstm_layer in self.lstm_layers[1:]:
            lstm_out, _ = lstm_layer(lstm_out)
        
        # 应用注意力机制
        lstm_out = lstm_out.transpose(0, 1)  # [seq_len, batch, features]
        attended_out, _ = self.attention[0](lstm_out, lstm_out, lstm_out)
        attended_out = attended_out.transpose(0, 1)  # [batch, seq_len, features]
        
        # 应用剩余的注意力层组件
        for layer in self.attention[1:]:
            attended_out = layer(attended_out)
            
        return attended_out

class MultiPeriodFeatureExtractor(nn.Module):
    """
    多周期特征提取器，整合并行和串行CNN-LSTM
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(MultiPeriodFeatureExtractor, self).__init__()
        
        self.parallel_cnn_lstm = ParallelCNNLSTM(input_dim, hidden_dim)
        self.serial_cnn_lstm = SerialCNNLSTM(input_dim, hidden_dim)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features_dict = {}
        
        # 处理短周期数据
        for period in ['5min', '15min', '1h']:
            if period in data_dict:
                features = self.parallel_cnn_lstm(data_dict[period])
                features_dict[period] = features
        
        # 处理长周期数据
        for period in ['1d', '1w', '1M']:
            if period in data_dict:
                features = self.serial_cnn_lstm(data_dict[period])
                features_dict[period] = features
        
        # 特征融合
        combined_features = torch.cat([
            torch.stack(list(features_dict.values()), dim=1).mean(1),
            torch.stack([features_dict[p] for p in ['1d', '1w', '1M'] if p in features_dict], dim=1).mean(1)
        ], dim=1)
        
        fused_features = self.feature_fusion(combined_features)
        
        return {
            'short_term': torch.stack([features_dict[p] for p in ['5min', '15min', '1h'] if p in features_dict], dim=1),
            'long_term': torch.stack([features_dict[p] for p in ['1d', '1w', '1M'] if p in features_dict], dim=1),
            'fused': fused_features
        }

class FeatureExtractorConfig:
    """
    特征提取器配置类
    """
    def __init__(self):
        # CNN配置
        self.cnn_short_filters = [64, 128, 256]
        self.cnn_long_filters = [32, 64]
        self.cnn_short_kernel_size = 3
        self.cnn_long_kernel_size = 5
        
        # LSTM配置
        self.lstm_short_units = [128, 64]
        self.lstm_long_units = [128, 96, 64]
        
        # 注意力配置
        self.attention_heads_short = 8
        self.attention_heads_long = 4
        
        # Dropout率
        self.dropout_rate = 0.2
        
        # 激活函数
        self.activation = 'relu'

def init_weights(m: nn.Module):
    """
    初始化模型权重
    """
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)