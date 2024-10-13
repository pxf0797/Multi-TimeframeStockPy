import numpy as np
import pandas as pd
from Utils.utils import calculate_sharpe_ratio, calculate_max_drawdown, calculate_profit_factor, calculate_expectancy

class Backtester:
    def __init__(self, config):
        self.config = config

    def run_backtest(self, signals, featured_data, dynamic_weights):
        results = {}
        for tf in self.config['timeframes']:
            print(f"\nDebugging information for timeframe {tf}:")
            signal = signals[tf]
            df = featured_data[tf]
            weight = dynamic_weights[tf]

            # Ensure signal and weight are the same length as df
            if len(signal) < len(df):
                signal = np.pad(signal, (len(df) - len(signal), 0), 'constant')
            if len(weight) < len(df):
                weight = np.pad(weight, (len(df) - len(weight), 0), 'constant')

            df['Signal'] = signal
            df['Weight'] = weight
            df['Position'] = np.where(df['Signal'] > self.config['entry_threshold'], 1, 
                                      np.where(df['Signal'] < -self.config['entry_threshold'], -1, 0))
            df['Position'] = df['Position'] * df['Weight']

            print(f"Signal range: {df['Signal'].min()} to {df['Signal'].max()}")
            print(f"Weight range: {df['Weight'].min()} to {df['Weight'].max()}")
            print(f"Position range: {df['Position'].min()} to {df['Position'].max()}")
            print(f"Returns range: {df['returns'].min()} to {df['returns'].max()}")

            df['PnL'] = df['Position'].shift(1) * df['returns']
            print(f"PnL range: {df['PnL'].min()} to {df['PnL'].max()}")
            print(f"Number of inf values in PnL: {np.isinf(df['PnL']).sum()}")
            print(f"Number of nan values in PnL: {np.isnan(df['PnL']).sum()}")

            total_return = df['PnL'].sum()
            sharpe_ratio = calculate_sharpe_ratio(df['PnL'])
            max_drawdown = calculate_max_drawdown(df['PnL'].cumsum())
            win_rate = (df['PnL'] > 0).mean()
            profit_factor = calculate_profit_factor(df['PnL'])
            expectancy = calculate_expectancy(df['PnL'])

            results[tf] = {
                'Total Return': total_return,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Win Rate': win_rate,
                'Profit Factor': profit_factor,
                'Expectancy': expectancy
            }

            print(f"Results for {tf}:")
            for key, value in results[tf].items():
                print(f"{key}: {value}")

        return results

    def calculate_overall_performance(self, results):
        print("\nCalculating overall performance:")
        for tf, result in results.items():
            print(f"{tf}: {result}")

        overall_return = sum(result['Total Return'] for result in results.values())
        overall_sharpe = np.mean([result['Sharpe Ratio'] for result in results.values() if not np.isnan(result['Sharpe Ratio'])])
        
        max_drawdowns = [result['Max Drawdown'] for result in results.values() if not np.isnan(result['Max Drawdown'])]
        overall_max_drawdown = max(max_drawdowns) if max_drawdowns else np.nan
        
        overall_win_rate = np.mean([result['Win Rate'] for result in results.values() if not np.isnan(result['Win Rate'])])
        overall_profit_factor = np.mean([result['Profit Factor'] for result in results.values() if not np.isnan(result['Profit Factor'])])
        overall_expectancy = np.mean([result['Expectancy'] for result in results.values() if not np.isnan(result['Expectancy'])])

        return {
            'Overall Return': overall_return,
            'Overall Sharpe Ratio': overall_sharpe,
            'Overall Max Drawdown': overall_max_drawdown,
            'Overall Win Rate': overall_win_rate,
            'Overall Profit Factor': overall_profit_factor,
            'Overall Expectancy': overall_expectancy
        }
