import matplotlib.pyplot as plt
import pandas as pd

def plot_trading_signals(signals: pd.DataFrame, symbol: str):
    plt.figure(figsize=(12, 6))
    plt.plot(signals.index, signals['Close'], label='Close Price', alpha=0.5)
    plt.scatter(signals.index[signals['Buy'] == 1], signals['Close'][signals['Buy'] == 1], 
                color='green', label='Buy Signal', marker='^', alpha=1)
    plt.scatter(signals.index[signals['Sell'] == 1], signals['Close'][signals['Sell'] == 1], 
                color='red', label='Sell Signal', marker='v', alpha=1)
    
    plt.title(f'Trading Signals for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'trading_signals_{symbol}.png')
    plt.close()

    print(f"Trading signals plot saved as trading_signals_{symbol}.png")

def plot_multi_timeframe_data(data_dict: dict, symbol: str):
    num_timeframes = len(data_dict)
    fig, axes = plt.subplots(num_timeframes, 1, figsize=(12, 4*num_timeframes), sharex=True)
    
    for i, (timeframe, data) in enumerate(data_dict.items()):
        ax = axes[i] if num_timeframes > 1 else axes
        ax.plot(data.index, data['Close'], label='Close Price')
        ax.set_title(f'{symbol} - {timeframe} Timeframe')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True)
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(f'multi_timeframe_data_{symbol}.png')
    plt.close()

    print(f"Multi-timeframe data plot saved as multi_timeframe_data_{symbol}.png")

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    close_prices = np.random.randn(len(dates)).cumsum() + 100
    signals = pd.DataFrame({
        'Close': close_prices,
        'Signal': np.random.randn(len(dates)),
        'Buy': np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1]),
        'Sell': np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1])
    }, index=dates)
    
    plot_trading_signals(signals, "EXAMPLE")
    
    # Example multi-timeframe data
    data_dict = {
        '1d': pd.DataFrame({'Close': close_prices}, index=dates),
        '1wk': pd.DataFrame({'Close': close_prices[::7]}, index=dates[::7]),
        '1mo': pd.DataFrame({'Close': close_prices[::30]}, index=dates[::30])
    }
    
    plot_multi_timeframe_data(data_dict, "EXAMPLE")
