import numpy as np
from Utils.utils import calculate_sharpe_ratio, calculate_max_drawdown

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.risk_per_trade = config['risk_per_trade']
        self.max_position_size = config['max_position_size']
        self.stop_loss = config['stop_loss']
        self.take_profit = config['take_profit']
        self.max_daily_loss = config['max_daily_loss']

    def calculate_position_sizes(self, signals, dynamic_weights):
        position_sizes = {}
        for tf, signal in signals.items():
            weight = dynamic_weights[tf]
            position_size = self.risk_per_trade * weight * signal
            position_size = np.clip(position_size, -self.max_position_size, self.max_position_size)
            position_sizes[tf] = position_size
        return position_sizes

    def apply_risk_management(self, signals, processed_data):
        managed_signals = {}
        for tf, signal in signals.items():
            managed_signal = {
                'position_size': self.calculate_position_sizes({tf: signal}, {tf: 1})[tf],
                'entry_level': processed_data[tf]['close'].iloc[-1],
                'stop_loss': self.calculate_stop_loss(signal, processed_data[tf]),
                'take_profit': self.calculate_take_profit(signal, processed_data[tf])
            }
            managed_signals[tf] = managed_signal
        return managed_signals

    def calculate_stop_loss(self, signal, data):
        current_price = data['close'].iloc[-1]
        if signal > 0:
            return current_price * (1 - self.stop_loss)
        elif signal < 0:
            return current_price * (1 + self.stop_loss)
        else:
            return None

    def calculate_take_profit(self, signal, data):
        current_price = data['close'].iloc[-1]
        if signal > 0:
            return current_price * (1 + self.take_profit)
        elif signal < 0:
            return current_price * (1 - self.take_profit)
        else:
            return None

    def implement_risk_limits(self, current_loss, max_daily_loss):
        return current_loss >= max_daily_loss

    def evaluate_risk_metrics(self, returns, trades):
        sharpe_ratio = calculate_sharpe_ratio(returns)
        max_drawdown = calculate_max_drawdown(returns.cumsum())
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'risk_adjusted_return': sharpe_ratio / (1 + abs(max_drawdown))
        }

    def adjust_risk_parameters(self, risk_metrics):
        # This is a placeholder for a more sophisticated risk adjustment method
        if risk_metrics['sharpe_ratio'] < 0.5:
            self.risk_per_trade *= 0.9
        elif risk_metrics['max_drawdown'] > 0.2:
            self.max_position_size *= 0.9
        
        self.risk_per_trade = max(self.risk_per_trade, 0.001)
        self.max_position_size = max(self.max_position_size, 0.01)

    def calculate_value_at_risk(self, returns, confidence_level=0.95):
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        return abs(sorted_returns[index])

    def calculate_expected_shortfall(self, returns, confidence_level=0.95):
        var = self.calculate_value_at_risk(returns, confidence_level)
        return abs(returns[returns <= -var].mean())

    def stress_test(self, model, data, num_scenarios=1000):
        stress_results = []
        for _ in range(num_scenarios):
            stressed_data = self.generate_stress_scenario(data)
            signals, _ = model.generate_signals(stressed_data)
            performance = self.simulate_performance(signals, stressed_data)
            stress_results.append(performance)
        
        return {
            'worst_case_loss': min(stress_results),
            'average_stress_performance': np.mean(stress_results),
            '5th_percentile_performance': np.percentile(stress_results, 5)
        }

    def generate_stress_scenario(self, data):
        # This is a placeholder for a more sophisticated stress scenario generation
        stress_factor = np.random.uniform(0.8, 1.2, size=len(data))
        return data * stress_factor

    def simulate_performance(self, signals, data):
        # This is a placeholder for a more sophisticated performance simulation
        return np.sum(signals * data['returns'])
