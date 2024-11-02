# models/cnn_lstm.py

import torch
import torch.nn as nn

class ParallelCNNLSTM(nn.Module):
    """
    用于处理短周期数据的并行CNN-LSTM结构
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(ParallelCNNLSTM, self).__init__()
        
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM部分
        self.lstm = nn.Sequential(
            nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True,
                   dropout=0.2, bidirectional=True),
            nn.Dropout(0.2)
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(256 + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        # CNN分支
        # 调整输入维度 [batch, seq_len, features] -> [batch, features, seq_len]
        x_cnn = x.transpose(1, 2)
        cnn_out = self.cnn(x_cnn)
        cnn_out = cnn_out.mean(2)  # 全局平均池化
        
        # LSTM分支
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出
        
        # 特征融合
        combined = torch.cat([cnn_out, lstm_out], dim=1)
        out = self.fusion(combined)
        
        return out

class SerialCNNLSTM(nn.Module):
    """
    用于处理长周期数据的串行CNN-LSTM结构
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(SerialCNNLSTM, self).__init__()
        
        # CNN前处理
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 残差连接
        self.residual = nn.Conv1d(input_dim, 64, kernel_size=1)
        
        # LSTM主体
        self.lstm = nn.Sequential(
            nn.LSTM(64, hidden_dim, num_layers=3, batch_first=True,
                   dropout=0.2, bidirectional=True),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        # CNN前处理
        x_cnn = x.transpose(1, 2)
        cnn_out = self.cnn(x_cnn)
        
        # 残差连接
        residual = self.residual(x_cnn)
        cnn_out = cnn_out + residual
        
        # 转换回序列格式
        cnn_out = cnn_out.transpose(1, 2)
        
        # LSTM处理
        lstm_out, _ = self.lstm(cnn_out)
        
        return lstm_out

# models/attention.py

class MultiPeriodAttention(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super(MultiPeriodAttention, self).__init__()
        
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
    def forward(self, features_dict):
        # 将不同周期的特征堆叠在一起
        features = torch.stack(list(features_dict.values()), dim=0)
        
        # Self-attention
        attended_features, _ = self.attention(features, features, features)
        features = self.norm1(features + attended_features)
        
        # Feed forward
        ff_out = self.ff(features)
        features = self.norm2(features + ff_out)
        
        # 重新分解为字典格式
        output_dict = {
            period: features[i] 
            for i, period in enumerate(features_dict.keys())
        }
        
        return output_dict

# models/period_encoder.py

class PeriodEncoder(nn.Module):
    def __init__(self, num_periods: int, encoding_dim: int):
        super(PeriodEncoder, self).__init__()
        
        self.period_embeddings = nn.Embedding(num_periods, encoding_dim)
        
    def forward(self, period_ids):
        return self.period_embeddings(period_ids)