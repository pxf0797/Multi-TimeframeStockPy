import numpy as np
from utils import adaptive_stop_loss, position_sizing

class RiskManager:
    def __init__(self, config):
        self.config = config

    def apply_risk_management(self, signals, processed_data):
        managed_signals = {}
        for tf, signal in signals.items():
            risk = self.calculate_risk(processed_data[tf])
            managed_signal = self.adjust_signal_by_risk(signal, risk)
            managed_signal = self.apply_position_sizing(managed_signal, processed_data[tf])
            managed_signals[tf] = managed_signal
        return managed_signals

    def calculate_risk(self, data):
        volatility_risk = np.std(data['returns']) * np.sqrt(252)
        liquidity_risk = 1 / (data['Volume'] / data['Volume'].rolling(20).mean())
        price_deviation_risk = np.abs(data['Close'] - data['MA_20']) / data['MA_20']
        
        total_risk = 0.4 * volatility_risk + 0.3 * liquidity_risk + 0.3 * price_deviation_risk
        return np.clip(total_risk, 0, 1)

    def adjust_signal_by_risk(self, signal, risk):
        signal['signal_strength'] *= (1 - risk)
        signal['entry_level'] *= (1 - risk)
        signal['stop_loss'] = adaptive_stop_loss(signal['entry_price'], signal['ATR'], risk)
        return signal

    def apply_position_sizing(self, signal, data):
        account_balance = self.config.get('account_balance', 100000)  # Default to 100,000
        risk_per_trade = self.config.get('risk_per_trade', 0.02)  # Default to 2%
        
        position_size = position_sizing(account_balance, risk_per_trade, 
                                        signal['entry_price'], signal['stop_loss'])
        
        max_position = self.config.get('max_position_size', 1.0)
        signal['position_size'] = np.clip(position_size, -max_position, max_position)
        return signal

    def calculate_total_risk(self, processed_data):
        risks = [self.calculate_risk(df) for df in processed_data.values()]
        return np.mean(risks)  # Simple average of risks across timeframes

    def implement_risk_limits(self, current_loss, max_daily_loss):
        if current_loss > max_daily_loss:
            return True  # Halt trading
        return False

    def dynamic_risk_adjustment(self, base_risk, market_volatility, recent_performance):
        volatility_factor = 1 + (market_volatility - self.config['avg_volatility']) / self.config['avg_volatility']
        performance_factor = 1 + (recent_performance - self.config['avg_performance']) / self.config['avg_performance']
        
        adjusted_risk = base_risk * volatility_factor * performance_factor
        return np.clip(adjusted_risk, self.config['min_risk'], self.config['max_risk'])
