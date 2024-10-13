import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import pandas as pd
import numpy as np
from Data.data_acquisition import DataAcquisition
from Model.lstm_model import MultiTimeframeLSTM
from Model.dynamic_weight import DynamicWeightModule, generate_signal
from Visualization.plot_utils import plot_trading_signals, plot_multi_timeframe_data
from Model.backtesting import run_backtest

class MultiTimeframeTrader:
    def __init__(self, symbol, timeframes, start_date, end_date):
        print(f"Initializing MultiTimeframeTrader for {symbol}")
        self.symbol = symbol
        self.timeframes = timeframes
        self.start_date = start_date
        self.end_date = end_date
        
        self.data_acq = DataAcquisition(symbol, timeframes)
        print("Fetching data...")
        self.data_acq.fetch_data(start_date, end_date)
        self.data_acq.align_data()
        print("Data fetched and aligned")
        
        self.input_sizes = [11] * len(timeframes)  # Updated to 11 features
        self.hidden_size = 64
        self.num_layers = 2
        self.output_size = 1
        
        print("Initializing LSTM model and Dynamic Weight module")
        self.lstm_model = MultiTimeframeLSTM(self.input_sizes, self.hidden_size, self.num_layers, self.output_size, len(timeframes))
        self.dynamic_weight_module = DynamicWeightModule(self.output_size, len(timeframes), 32)
        
    def prepare_data(self):
        print("Preparing data...")
        data_list = []
        for tf in self.timeframes:
            df = self.data_acq.get_data(tf)
            if df.empty:
                print(f"No data available for timeframe {tf}")
                continue
            # Here you would typically calculate your features
            # For simplicity, we'll just use some basic features
            df['returns'] = df['Close'].pct_change()
            df['ma_5'] = df['Close'].rolling(window=5).mean()
            df['ma_20'] = df['Close'].rolling(window=20).mean()
            df['rsi'] = self.calculate_rsi(df['Close'], period=14)
            df['rsi_slow'] = self.calculate_rsi(df['Close'], period=21)
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'ma_5', 'ma_20', 'rsi', 'rsi_slow', 'volatility']
            data_list.append(df[features].dropna())
        
        print("Data preparation complete")
        return data_list
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train_model(self, data_list):
        print("Training model (placeholder)...")
        # This is a placeholder for model training
        # In a real scenario, you would split your data, create DataLoaders, and train the model
        pass
    
    def generate_trading_signals(self, data_list):
        print("Generating trading signals...")
        if not data_list:
            print("No data available to generate trading signals")
            return pd.DataFrame()
        
        signals = pd.DataFrame(index=data_list[0].index)
        signals['Close'] = data_list[0]['Close']
        
        x_list = [torch.tensor(df.values, dtype=torch.float32).unsqueeze(0) for df in data_list]
        
        with torch.no_grad():
            lstm_out = self.lstm_model(x_list)
            
            num_timeframes = len(data_list)
            accuracy = torch.rand(num_timeframes, dtype=torch.float32)  # Placeholder for actual accuracy calculation
            trend_strength = torch.rand(num_timeframes, dtype=torch.float32) * 2 - 1  # Placeholder
            volatility = torch.tensor([df['volatility'].iloc[-1] for df in data_list], dtype=torch.float32)
            
            # Print shapes for debugging
            print(f"LSTM output shape: {lstm_out.shape}")
            print(f"Accuracy shape: {accuracy.shape}")
            print(f"Trend strength shape: {trend_strength.shape}")
            print(f"Volatility shape: {volatility.shape}")
            
            weights = self.dynamic_weight_module(lstm_out, accuracy, trend_strength, volatility)
            weighted_output = (weights * lstm_out).sum()
            
            signal_strength, _, _ = generate_signal(weighted_output)
            
            # Convert signal_strength to a pandas Series
            signal_strength_series = pd.Series(signal_strength.item(), index=signals.index)
            
            # Generate buy and sell signals based on multiple factors
            signals['Signal'] = signal_strength_series
            signals['RSI'] = data_list[0]['rsi']
            signals['RSI_slow'] = data_list[0]['rsi_slow']
            signals['MA_5'] = data_list[0]['ma_5']
            signals['MA_20'] = data_list[0]['ma_20']
            
            # Buy conditions: LSTM signal > -0.1, RSI < 50, slow RSI < 50, MA_5 > MA_20
            signals['Buy'] = ((signal_strength_series > -0.1) & 
                              (signals['RSI'] < 50) & 
                              (signals['RSI_slow'] < 50) & 
                              (signals['MA_5'] > signals['MA_20'])).astype(int)
            
            # Sell conditions: LSTM signal < 0.1, RSI > 50, slow RSI > 50, MA_5 < MA_20
            signals['Sell'] = ((signal_strength_series < 0.1) & 
                               (signals['RSI'] > 50) & 
                               (signals['RSI_slow'] > 50) & 
                               (signals['MA_5'] < signals['MA_20'])).astype(int)
            
            # Position sizing based on signal strength
            signals['Position'] = np.where(signals['Buy'] == 1, 1 + 2*abs(signal_strength_series),
                                           np.where(signals['Sell'] == 1, -1 - 2*abs(signal_strength_series), 0))
        
        print("Trading signals generated")
        print(signals.head())
        print(f"Number of buy signals: {signals['Buy'].sum()}")
        print(f"Number of sell signals: {signals['Sell'].sum()}")
        return signals
    
    def run_backtest(self):
        print("Running backtest...")
        data_list = self.prepare_data()
        if not data_list:
            print("No data available for backtesting")
            return None, None
        self.train_model(data_list)
        signals = self.generate_trading_signals(data_list)
        if signals.empty:
            print("No signals generated for backtesting")
            return None, None
        portfolio = run_backtest(signals)
        print("Backtest complete")
        return signals, portfolio

def main():
    print("Starting main function")
    symbol = "MSFT"
    timeframes = ["1d", "1wk", "1mo"]
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    
    trader = MultiTimeframeTrader(symbol, timeframes, start_date, end_date)
    signals, portfolio = trader.run_backtest()
    
    if signals is None or portfolio is None:
        print("Backtesting failed. Unable to generate plots or summary.")
        return
    
    print("Generating plots...")
    # Generate plots
    plot_trading_signals(signals, symbol)
    plot_multi_timeframe_data({tf: trader.data_acq.get_data(tf) for tf in timeframes if not trader.data_acq.get_data(tf).empty}, symbol)
    
    # Plot equity curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio.index, portfolio['Equity'])
    plt.title(f'Equity Curve - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.savefig(f'equity_curve_{symbol}.png')
    plt.close()
    print(f"Equity curve plot saved as equity_curve_{symbol}.png")
    
    # Print summary
    print("\nTrading Strategy Summary:")
    print(f"Symbol: {symbol}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"\nFinal Portfolio Value: ${portfolio['Equity'].iloc[-1]:.2f}")
    total_return = (portfolio['Equity'].iloc[-1] / portfolio['Equity'].iloc[0] - 1) * 100
    print(f"Total Return: {total_return:.2f}%")
    
    strategy_returns = portfolio['Strategy']
    if strategy_returns.std() != 0:
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print("Sharpe Ratio: N/A (strategy returns have zero standard deviation)")
    
    print(f"Max Drawdown: {portfolio['DrawDown'].min() * 100:.2f}%")

if __name__ == "__main__":
    main()
