# Multi-Timeframe Stock Trading System

This project implements a multi-timeframe stock trading system using machine learning techniques. It supports various modes of operation including training, backtesting, optimization, and live trading.

## Features

- Multi-timeframe data processing and feature engineering
- Machine learning model for signal generation
- Backtesting functionality
- Parameter optimization
- Live trading with real-time data fetching and trade execution
- Risk management and position sizing
- Comprehensive logging and error handling

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/multi-timeframe-stock-trading.git
   cd multi-timeframe-stock-trading
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your configuration in `config.yaml`. Make sure to update the following:
   - API keys for your chosen exchange (for live trading)
   - Asset and timeframes you want to trade
   - Risk management parameters
   - Other configuration options as needed

## Usage

The system supports four modes of operation: train, backtest, optimize, and live_trading. You can set the mode in the `config.yaml` file.

To run the system:

```
python main.py
```

### Training

Set `mode: 'train'` in `config.yaml` to train the model. The trained model will be saved to the path specified in `model_save_path`.

### Backtesting

Set `mode: 'backtest'` in `config.yaml` to run a backtest using historical data. The results will be logged and can be analyzed for performance evaluation.

### Optimization

Set `mode: 'optimize'` in `config.yaml` to run the parameter optimization process. This will search for the best parameters for your model and strategy.

### Live Trading

Set `mode: 'live_trading'` in `config.yaml` to run the system in live trading mode. Make sure you have set up your exchange API keys correctly before running live trading.

**IMPORTANT:** Always test the live trading functionality using a testnet or paper trading account before using real funds.

## Configuration

The `config.yaml` file contains all the configuration options for the system. Here are some key parameters:

- `mode`: The operation mode ('train', 'backtest', 'optimize', or 'live_trading')
- `asset`: The trading pair (e.g., 'BTC/USD')
- `timeframes`: List of timeframes to use
- `risk_per_trade`: Maximum risk per trade as a fraction of account balance
- `max_position_size`: Maximum position size as a fraction of account balance
- `exchange`: The name of the exchange to use (must be supported by ccxt)
- `api_key` and `api_secret`: Your exchange API credentials

## Safety and Security

- Never commit your `config.yaml` file with real API keys to version control.
- Always start with small trade sizes and conservative risk parameters when going live.
- Monitor your trading system regularly and be prepared to shut it down if it's not performing as expected.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
