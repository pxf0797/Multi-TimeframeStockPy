import numpy as np
import pandas as pd

class Backtester:
    def __init__(self, config):
        self.config = config

    def run_backtest(self, signals, data, dynamic_weights):
        results = {}
        for tf in signals.keys():
            if tf in data:
                pnl = self.calculate_pnl(signals[tf], data[tf])
                sharpe_ratio = self.calculate_sharpe_ratio(pnl)
                max_drawdown = self.calculate_max_drawdown(pnl)
                results[tf] = {
                    'PnL': pnl.sum(),
                    'Sharpe Ratio': sharpe_ratio,
                    'Max Drawdown': max_drawdown
                }
        return results

    def calculate_pnl(self, signal, data):
        # Ensure the signal and data have the same length
        min_length = min(len(signal['signal_strength']), len(data))
        
        # Use the last min_length elements
        position = signal['position_size'][-min_length:]
        returns = data['returns'].iloc[-min_length:].values
        
        pnl = position * returns
        return pd.Series(pnl)

    def calculate_sharpe_ratio(self, pnl, risk_free_rate=0.02):
        returns = pnl.pct_change().dropna()
        excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def calculate_max_drawdown(self, pnl):
        cumulative_returns = (1 + pnl).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        return max_drawdown
