import pandas as pd
import numpy as np
from utils import calculate_sharpe_ratio, calculate_maximum_drawdown

class Backtester:
    def __init__(self, config):
        self.config = config

    def run_backtest(self, signals, data, dynamic_weights):
        results = {}
        for tf, signal in signals.items():
            pnl = self.calculate_pnl(signal, data[tf])
            sharpe = calculate_sharpe_ratio(pnl, self.config['risk_free_rate'])
            max_drawdown = calculate_maximum_drawdown(pnl.cumsum())
            win_rate = self.calculate_win_rate(pnl)
            profit_loss_ratio = self.calculate_profit_loss_ratio(pnl)
            
            results[tf] = {
                'Total Return': pnl.sum(),
                'Sharpe Ratio': sharpe,
                'Max Drawdown': max_drawdown,
                'Win Rate': win_rate,
                'Profit/Loss Ratio': profit_loss_ratio,
                'Dynamic Weight': dynamic_weights[tf]
            }
        return results

    def calculate_pnl(self, signal, data):
        position = signal['position_size']
        returns = data['returns']
        stop_loss_returns = np.minimum(returns, -signal['stop_loss'])
        pnl = position * np.maximum(returns, stop_loss_returns)
        return pnl

    def calculate_win_rate(self, pnl):
        return (pnl > 0).mean()

    def calculate_profit_loss_ratio(self, pnl):
        profits = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        if len(losses) == 0:
            return np.inf
        return profits.mean() / abs(losses.mean())

    def monte_carlo_simulation(self, signals, data, num_simulations=1000):
        results = []
        for _ in range(num_simulations):
            shuffled_data = self.shuffle_data(data)
            sim_results = self.run_backtest(signals, shuffled_data, {tf: 1/len(data) for tf in data})
            results.append(sim_results)
        return results

    def shuffle_data(self, data):
        shuffled_data = {}
        for tf, df in data.items():
            shuffled_data[tf] = df.sample(frac=1).reset_index(drop=True)
        return shuffled_data

    def sensitivity_analysis(self, model, featured_data, processed_data, param_ranges):
        results = {}
        for param, range_values in param_ranges.items():
            param_results = []
            for value in range_values:
                self.config[param] = value
                signals, _ = model.generate_signals(featured_data)
                backtest_results = self.run_backtest(signals, processed_data, {tf: 1/len(processed_data) for tf in processed_data})
                param_results.append((value, backtest_results))
            results[param] = param_results
        return results
