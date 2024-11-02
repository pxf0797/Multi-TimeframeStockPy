# models/decision_layers.py

class DecisionModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(DecisionModule, self).__init__()
        
        # 仓位调整网络
        self.position_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出0-1之间的仓位比例
        )
        
        # 做T操作网络
        self.trading_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 买入、卖出、观望三种操作
            nn.Softmax(dim=1)
        )
        
    def forward(self, predictions, market_state):
        # 合并预测结果和市场状态
        combined_input = torch.cat([
            predictions['price'],
            predictions['trend'],
            predictions['signal'],
            market_state
        ], dim=1)
        
        # 生成决策
        position_decision = self.position_net(combined_input)
        trading_decision = self.trading_net(combined_input)
        
        return {
            'position': position_decision,
            'trading': trading_decision
        }

# 完整的模型类
class MultiPeriodTradingModel(nn.Module):
    def __init__(self, input_dim: int, num_periods: int):
        super(MultiPeriodTradingModel, self).__init__()
        
        # 特征提取
        self.short_period_net = ParallelCNNLSTM(input_dim)
        self.long_period_net = SerialCNNLSTM(input_dim)
        
        # 周期编码
        self.period_encoder = PeriodEncoder(num_periods, input_dim)
        
        # 注意力机制
        self.attention = MultiPeriodAttention(input_dim)
        
        # 预测层
        self.prediction = PredictionModule(input_dim * 2)  # 2倍是因为特征拼接
        
        # 决策层
        self.decision = DecisionModule(input_dim * 4)  # 4倍是因为包含预测结果
        
    def forward(self, data_dict):
        features_dict = {}
        
        # 处理短周期数据
        for period in ['5min', '15min', '1h']:
            features = self.short_period_net(data_dict[period])
            features_dict[period] = features
        
        # 处理长周期数据
        for period in ['1d', '1w', '1M']:
            features = self.long_period_net(data_dict[period])
            features_dict[period] = features
        
        # 添加周期编码
        for period, features in features_dict.items():
            period_code = torch.tensor([self._period_to_id(period)], 
                                     device=features.device)
            period_embedding = self.period_encoder(period_code)
            features_dict[period] = features + period_embedding
        
        # 注意力融合
        attended_features = self.attention(features_dict)
        
        # 合并特征
        combined_features = torch.cat([
            torch.stack(list(attended_features.values()), dim=1).mean(1),
            torch.stack(list(features_dict.values()), dim=1).mean(1)
        ], dim=1)
        
        # 预测
        predictions = self.prediction(combined_features)
        
        # 决策
        market_state = self._get_market_state()
            
# models/decision_layers.py (continued)

    def _get_market_state(self, data_dict):
        """
        生成市场状态向量
        """
        # 计算市场波动率
        volatility = torch.std(data_dict['5min'][:, :, 3], dim=1, keepdim=True)  # 使用收盘价
        
        # 计算趋势强度
        trend = (data_dict['1d'][:, -1, 3] - data_dict['1d'][:, 0, 3]) / data_dict['1d'][:, 0, 3]
        trend = trend.unsqueeze(1)
        
        # 计算成交量变化
        volume_change = (data_dict['5min'][:, -1, 4] - data_dict['5min'][:, 0, 4]) / data_dict['5min'][:, 0, 4]
        volume_change = volume_change.unsqueeze(1)
        
        return torch.cat([volatility, trend, volume_change], dim=1)
    
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
            '1M': 5
        }
        return period_mapping.get(period, -1)



