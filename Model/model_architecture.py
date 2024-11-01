# model_architecture.py

import tensorflow as tf
from tensorflow.keras import Model, layers, initializers
from typing import Dict, List, Tuple, Optional
import numpy as np

class PeriodicEncoding(layers.Layer):
    """周期编码层"""
    def __init__(self, max_periods: int = 7, encoding_dim: int = 32):
        super(PeriodicEncoding, self).__init__()
        self.max_periods = max_periods
        self.encoding_dim = encoding_dim
        
    def build(self, input_shape):
        # 创建可训练的周期嵌入矩阵
        self.period_embedding = self.add_weight(
            name='period_embedding',
            shape=(self.max_periods, self.encoding_dim),
            initializer='uniform',
            trainable=True
        )
    
    def call(self, period_ids):
        # 获取周期编码
        period_encodings = tf.gather(self.period_embedding, period_ids)
        return period_encodings

class AttentionBlock(layers.Layer):
    """注意力模块"""
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super(AttentionBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads
        )
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(hidden_dim * 2, activation='relu'),
            layers.Dense(hidden_dim)
        ])
        
    def call(self, x, training=False):
        # 多头注意力
        attn_output = self.mha(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class ShortPeriodNetwork(Model):
    """短周期并行CNN-LSTM网络"""
    def __init__(self, 
                input_dim: int,
                conv_filters: List[int] = [64, 128, 256],
                lstm_units: List[int] = [128, 64],
                dropout_rate: float = 0.3):
        super(ShortPeriodNetwork, self).__init__()
        
        # CNN部分
        self.conv_layers = []
        self.bn_layers = []
        for filters in conv_filters:
            self.conv_layers.append(layers.Conv1D(
                filters=filters,
                kernel_size=3,
                padding='same',
                activation='relu'
            ))
            self.bn_layers.append(layers.BatchNormalization())
        
        # LSTM部分
        self.lstm_layers = []
        for units in lstm_units:
            self.lstm_layers.append(layers.LSTM(
                units=units,
                return_sequences=True
            ))
        
        self.dropout = layers.Dropout(dropout_rate)
        self.fusion = layers.Dense(256, activation='relu')
    
    def call(self, inputs, training=False):
        # CNN路径
        x_cnn = inputs
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x_cnn = conv(x_cnn)
            x_cnn = bn(x_cnn, training=training)
        
        # LSTM路径
        x_lstm = inputs
        for lstm in self.lstm_layers:
            x_lstm = lstm(x_lstm)
            x_lstm = self.dropout(x_lstm, training=training)
        
        # 特征融合
        concat = layers.concatenate([x_cnn, x_lstm])
        output = self.fusion(concat)
        
        return output

class LongPeriodNetwork(Model):
    """长周期串行CNN-LSTM网络"""
    def __init__(self,
                input_dim: int,
                conv_filters: List[int] = [32, 64],
                lstm_units: List[int] = [128, 96, 64],
                dropout_rate: float = 0.3):
        super(LongPeriodNetwork, self).__init__()
        
        # CNN预处理
        self.conv_layers = []
        for filters in conv_filters:
            self.conv_layers.append(layers.Conv1D(
                filters=filters,
                kernel_size=5,
                padding='same',
                activation='relu'
            ))
        
        # 双向LSTM
        self.bilstm = layers.Bidirectional(
            layers.LSTM(lstm_units[0], return_sequences=True)
        )
        
        # LSTM层
        self.lstm_layers = []
        for units in lstm_units[1:]:
            self.lstm_layers.append(layers.LSTM(
                units=units,
                return_sequences=True
            ))
        
        self.dropout = layers.Dropout(dropout_rate)
        self.attention = AttentionBlock()
    
    def call(self, inputs, training=False):
        # CNN预处理
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
            x = self.dropout(x, training=training)
        
        # 双向LSTM
        x = self.bilstm(x)
        
        # LSTM处理
        for lstm in self.lstm_layers:
            x = lstm(x)
            x = self.dropout(x, training=training)
        
        # 注意力处理
        x = self.attention(x)
        
        return x

class PredictionHead(Model):
    """预测层"""
    def __init__(self,
                input_dim: int,
                hidden_dims: List[int] = [256, 128, 64]):
        super(PredictionHead, self).__init__()
        
        # 价格预测分支
        self.price_layers = []
        for dim in hidden_dims:
            self.price_layers.extend([
                layers.Dense(dim, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3)
            ])
        self.price_output = layers.Dense(3)  # 短中长期价格预测
        
        # 趋势强度预测分支
        self.trend_layers = []
        for dim in hidden_dims:
            self.trend_layers.extend([
                layers.Dense(dim, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3)
            ])
        self.trend_output = layers.Dense(3, activation='softmax')  # 上涨、下跌、盘整
        
        # 买卖点信号预测分支
        self.signal_layers = []
        for dim in hidden_dims:
            self.signal_layers.extend([
                layers.Dense(dim, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3)
            ])
        self.signal_output = layers.Dense(3, activation='softmax')  # 买入、卖出、观望
    
    def call(self, inputs, training=False):
        # 价格预测
        x_price = inputs
        for layer in self.price_layers:
            x_price = layer(x_price, training=training)
        price_pred = self.price_output(x_price)
        
        # 趋势预测
        x_trend = inputs
        for layer in self.trend_layers:
            x_trend = layer(x_trend, training=training)
        trend_pred = self.trend_output(x_trend)
        
        # 信号预测
        x_signal = inputs
        for layer in self.signal_layers:
            x_signal = layer(x_signal, training=training)
        signal_pred = self.signal_output(x_signal)
        
        return {
            'price': price_pred,
            'trend': trend_pred,
            'signal': signal_pred
        }

class MultiPeriodTradingModel(Model):
    """多周期交易模型"""
    def __init__(self,
                input_dim: int,
                num_periods: int,
                periodic_encoding_dim: int = 32):
        super(MultiPeriodTradingModel, self).__init__()
        
        self.period_encoding = PeriodicEncoding(
            max_periods=num_periods,
            encoding_dim=periodic_encoding_dim
        )
        
        self.short_network = ShortPeriodNetwork(input_dim)
        self.long_network = LongPeriodNetwork(input_dim)
        
        self.feature_fusion = layers.Dense(512, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.attention = AttentionBlock()
        
        self.prediction_head = PredictionHead(512)
    
    def call(self, inputs, training=False):
        # 解包输入
        short_period_data, long_period_data, period_ids = inputs
        
        # 获取周期编码
        period_encodings = self.period_encoding(period_ids)
        
        # 特征提取
        short_features = self.short_network(short_period_data, training=training)
        long_features = self.long_network(long_period_data, training=training)
        
        # 添加周期编码
        short_features = short_features + period_encodings[:, None, :]
        long_features = long_features + period_encodings[:, None, :]
        
        # 特征融合
        combined = layers.concatenate([short_features, long_features])
        fused = self.feature_fusion(combined)
        fused = self.dropout(fused, training=training)
        
        # 注意力处理
        attended = self.attention(fused)
        
        # 生成预测
        predictions = self.prediction_head(attended, training=training)
        
        return predictions

class TradingLoss:
    """交易模型损失函数"""
    def __init__(self,
                price_loss_weight: float = 1.0,
                trend_loss_weight: float = 1.0,
                signal_loss_weight: float = 1.0):
        self.price_loss_weight = price_loss_weight
        self.trend_loss_weight = trend_loss_weight
        self.signal_loss_weight = signal_loss_weight
        
        self.price_loss_fn = tf.keras.losses.MeanSquaredError()
        self.trend_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.signal_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    def __call__(self,
                y_true: Dict[str, tf.Tensor],
                y_pred: Dict[str, tf.Tensor]) -> tf.Tensor:
        # 计算各分支损失
        price_loss = self.price_loss_fn(y_true['price'], y_pred['price'])
        trend_loss = self.trend_loss_fn(y_true['trend'], y_pred['trend'])
        signal_loss = self.signal_loss_fn(y_true['signal'], y_pred['signal'])
        
        # 加权总损失
        total_loss = (
            self.price_loss_weight * price_loss +
            self.trend_loss_weight * trend_loss +
            self.signal_loss_weight * signal_loss
        )
        
        return total_loss

if __name__ == "__main__":
    # 示例使用
    model = MultiPeriodTradingModel(
        input_dim=29,  # 根据特征维度设置
        num_periods=5  # 5个周期
    )
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=TradingLoss(),
        metrics=['accuracy']
    )