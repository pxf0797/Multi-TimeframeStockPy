# models/prediction_layers.py

class PredictionModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(PredictionModule, self).__init__()
        
        # 价格预测分支
        self.price_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 3)  # 预测短期、中期、长期价格
        )
        
        # 趋势强度预测分支
        self.trend_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 4)  # 预测方向(3类)和持续性(1类)
        )
        
        # 买卖点信号预测分支
        self.signal_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 4)  # 预测信号类型(3类)和拐点概率(1类)
        )
        
    def forward(self, x):
        price_pred = self.price_branch(x)
        trend_pred = self.trend_branch(x)
        signal_pred = self.signal_branch(x)
        
        return {
            'price': price_pred,
            'trend': trend_pred,
            'signal': signal_pred
        }

