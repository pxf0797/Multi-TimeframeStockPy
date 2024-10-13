import numpy as np
import pandas as pd
from Utils.utils import calculate_sharpe_ratio, calculate_maximum_drawdown

class Backtester:
    def __init__(self, config):
        self.config = config

    def run_backtest(self, signals, data, dynamic_weights):
        results = {}
        for tf in self.config['timeframes']:
            if tf not in signals or tf not in data:
                print(f"Skipping backtest for timeframe {tf} due to missing data")
                continue

            df = data[tf].copy()
            df['Signal'] = signals[tf]
            df['Weight'] = dynamic_weights[tf]

            df['Position'] = df['Signal'] * df['Weight']
            df['PnL'] = df['Position'].shift(1) * df['returns']

            # Debugging information
            print(f"\nDebugging information for timeframe {tf}:")
            print(f"Signal range: {df['Signal'].min()} to {df['Signal'].max()}")
            print(f"Weight range: {df['Weight'].min()} to {df['Weight'].max()}")
            print(f"Position range: {df['Position'].min()} to {df['Position'].max()}")
            print(f"Returns range: {df['returns'].min()} to {df['returns'].max()}")
            print(f"PnL range: {df['PnL'].min()} to {df['PnL'].max()}")
            print(f"Number of inf values in PnL: {np.isinf(df['PnL']).sum()}")
            print(f"Number of nan values in PnL: {np.isnan(df['PnL']).sum()}")

            # Handle potential division by zero or invalid values
            returns = df['PnL'].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns) == 0:
                print(f"Warning: No valid returns for timeframe {tf}")
                results[tf] = {'PnL': 0, 'Sharpe Ratio': np.nan, 'Max Drawdown': np.nan}
                continue

            total_pnl = returns.sum()
            sharpe = calculate_sharpe_ratio(returns)
            max_drawdown = calculate_maximum_drawdown(returns.cumsum())

            results[tf] = {
                'PnL': total_pnl,
                'Sharpe Ratio': sharpe,
                'Max Drawdown': max_drawdown
            }

        return results

    def calculate_overall_performance(self, results):
        overall_pnl = sum(result['PnL'] for result in results.values())
        overall_sharpe = np.mean([result['Sharpe Ratio'] for result in results.values() if not np.isnan(result['Sharpe Ratio'])])
        overall_max_drawdown = max(result['Max Drawdown'] for result in results.values() if not np.isnan(result['Max Drawdown']))

        return {
            'Overall PnL': overall_pnl,
            'Overall Sharpe Ratio': overall_sharpe,
            'Overall Max Drawdown': overall_max_drawdown
        }
