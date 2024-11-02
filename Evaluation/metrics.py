# evaluation/metrics.py

class PerformanceMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.portfolio_values = []
        self.returns = []
        self.positions = []
        self.trades = []
    
    def update(self, portfolio_value: float, position: float, trade_type: str):
        self.portfolio_values.append(portfolio_value)
        
        if len(self.portfolio_values) > 1:
            daily_return = (portfolio_value / self.portfolio_values[-2]) - 1
            self.returns.append(daily_return)
        
        self.positions.append(position)
        if trade_type != 'hold':
            self.trades.append({
                'type': trade_type,
                'price': portfolio_value,
                'position': position,
                'timestamp': len(self.portfolio_values) - 1
            })
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        计算性能指标
        """
        returns = np.array(self.returns)
        
        # 收益率指标
        total_return = (self.portfolio_values[-1] / self.portfolio_values[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # 风险指标
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown()
        
        # 交易相关指标
        trade_count = len(self.trades)
        win_rate = self._calculate_win_rate()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trade_count': trade_count,
            'win_rate': win_rate
        }
    
    def _calculate_max_drawdown(self) -> float:
        """
        计算最大回撤
        """
        portfolio_values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        return drawdown.max()
    
    def _calculate_win_rate(self) -> float:
        """
        计算胜率
        """
        if not self.trades:
            return 0.0
            
        winning_trades = sum(1 for i in range(1, len(self.trades))
                           if self.trades[i]['price'] > self.trades[i-1]['price'])
        return winning_trades / (len(self.trades) - 1) if len(self.trades) > 1 else 0.0

