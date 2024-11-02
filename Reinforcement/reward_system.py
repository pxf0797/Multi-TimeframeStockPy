# reinforcement/reward_system.py

class RewardCalculator:
    def __init__(self, transaction_cost: float = 0.001):
        self.transaction_cost = transaction_cost
        
    def calculate_position_reward(self, 
                                old_position: float,
                                new_position: float,
                                price_change: float,
                                volatility: float) -> float:
        """
        计算仓位调整的奖励
        """
        # 基础收益
        position_return = new_position * price_change
        
        # 交易成本
        transaction_cost = abs(new_position - old_position) * self.transaction_cost
        
        # 风险控制奖励
        risk_reward = 0.0
        if volatility > 0.5 and new_position < old_position:
            risk_reward = 0.001
        
        # 稳定性奖励
        stability_reward = -0.0005 * abs(new_position - old_position)
        
        total_reward = position_return - transaction_cost + risk_reward + stability_reward
        return total_reward
    
    def calculate_trading_reward(self,
                               action: int,
                               price_change: float,
                               holding_time: int) -> float:
        """
        计算做T操作的奖励
        """
        # 基础收益
        if action == 0:  # 买入
            trading_return = price_change
        elif action == 1:  # 卖出
            trading_return = -price_change
        else:  # 观望
            trading_return = 0
        
        # 交易成本
        transaction_cost = self.transaction_cost if action != 2 else 0
        
        # 持仓时间惩罚
        time_penalty = 0.0001 * holding_time if action != 2 else 0
        
        total_reward = trading_return - transaction_cost - time_penalty
        return total_reward