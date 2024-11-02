# evaluation/risk_control.py

class RiskController:
    def __init__(self,
                 max_position: float = 1.0,
                 min_position: float = 0.0,
                 max_drawdown: float = 0.2,
                 volatility_threshold: float = 0.3):
        self.max_position = max_position
        self.min_position = min_position
        self.max_drawdown = max_drawdown
        self.volatility_threshold = volatility_threshold
        
        self.current_drawdown = 0.0
        self.peak_value = 0.0
        
    def check_position(self,
                      position: float,
                      portfolio_value: float,
                      volatility: float) -> float:
        """
        检查并调整仓位
        """
        # 更新回撤
        self.peak_value = max(self.peak_value, portfolio_value)
        self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        
        # 基本约束
        position = np.clip(position, self.min_position, self.max_position)
        
        # 回撤控制
        if self.current_drawdown > self.max_drawdown:
            position = min(position, 0.5)
        
        # 波动率控制
        if volatility > self.volatility_threshold:
            position = min(position, 0.7)
        
        return position
    
    def check_trade(self,
                   trade_action: str,
                   current_position: float,
                   volatility: float) -> bool:
        """
        检查交易是否允许
        """
        # 波动率过高时禁止开新仓
        if volatility > self.volatility_threshold and trade_action == 'buy':
            return False
        
        # 回撤过大时禁止加仓
        if self.current_drawdown > self.max_drawdown and trade_action == 'buy':
            return False
        
        # 仓位限制
        if trade_action == 'buy' and current_position >= self.max_position:
            return False
        if trade_action == 'sell' and current_position <= self.min_position:
            return False
        
        return True