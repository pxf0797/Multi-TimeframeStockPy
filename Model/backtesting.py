import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, signals, initial_capital=100000):
        self.signals = signals
        self.initial_capital = initial_capital
        self.positions = self.generate_positions()
        self.portfolio = self.backtest_portfolio()

    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions['Position'] = 0
        positions.loc[self.signals['Buy'] == 1, 'Position'] = 1
        positions.loc[self.signals['Sell'] == 1, 'Position'] = -1
        return positions

    def backtest_portfolio(self):
        portfolio = pd.DataFrame(index=self.signals.index).fillna(0.0)
        portfolio['Position'] = self.positions['Position']
        portfolio['Close'] = self.signals['Close']
        portfolio['Returns'] = portfolio['Close'].pct_change()
        portfolio['Strategy'] = portfolio['Position'].shift(1) * portfolio['Returns']
        
        portfolio['Equity'] = (1 + portfolio['Strategy']).cumprod() * self.initial_capital
        portfolio['DrawDown'] = (portfolio['Equity'] - portfolio['Equity'].cummax()) / portfolio['Equity'].cummax()
        
        # Add debugging information
        print("Backtest Summary:")
        print(f"Total trades: {(portfolio['Position'].diff() != 0).sum()}")
        print(f"Profitable trades: {(portfolio['Strategy'] > 0).sum()}")
        print(f"Unprofitable trades: {(portfolio['Strategy'] < 0).sum()}")
        print(f"Win rate: {(portfolio['Strategy'] > 0).sum() / (portfolio['Strategy'] != 0).sum():.2%}")
        print(f"Average profit per trade: ${portfolio['Strategy'][portfolio['Strategy'] > 0].mean():.2f}")
        print(f"Average loss per trade: ${portfolio['Strategy'][portfolio['Strategy'] < 0].mean():.2f}")
        
        return portfolio

    def calculate_performance_metrics(self):
        total_return = (self.portfolio['Equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        sharpe_ratio = np.sqrt(252) * self.portfolio['Strategy'].mean() / self.portfolio['Strategy'].std() if self.portfolio['Strategy'].std() != 0 else np.nan
        max_drawdown = self.portfolio['DrawDown'].min()
        
        return {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }

def run_backtest(signals):
    backtester = Backtester(signals)
    performance_metrics = backtester.calculate_performance_metrics()
    
    print("\nBacktest Results:")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value:.2f}")
    
    return backtester.portfolio

if __name__ == "__main__":
    # Example usage
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    close_prices = np.random.randn(len(dates)).cumsum() + 100
    signals = pd.DataFrame({
        'Close': close_prices,
        'Signal': np.random.randn(len(dates)),
        'Buy': np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1]),
        'Sell': np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])
    }, index=dates)
    
    portfolio = run_backtest(signals)
    
    # Plot equity curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio.index, portfolio['Equity'])
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.savefig('equity_curve.png')
    plt.close()
    print("Equity curve plot saved as equity_curve.png")
