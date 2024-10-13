import torch
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from model_building import ModelBuilder
from signal_generation import SignalGenerator
from risk_management import RiskManager
from backtesting import Backtester
from optimization import Optimizer
from live_trading import LiveTrader
from utils import calculate_trend_consistency, identify_market_regime

def main():
    config = {
        'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M'],
        'ma_periods': [3, 5, 10, 20],
        'macd_params': (5, 10, 5),
        'input_size': 20,
        'hidden_size': 64,
        'num_layers': 2,
        'num_heads': 4,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'risk_free_rate': 0.02,
        'max_daily_loss': 0.02,
        'account_balance': 100000,
        'risk_per_trade': 0.02,
        'max_position_size': 1.0,
    }

    data_processor = DataProcessor(config)
    feature_engineer = FeatureEngineer(config)
    model_builder = ModelBuilder(config)
    signal_generator = SignalGenerator(config)
    risk_manager = RiskManager(config)
    backtester = Backtester(config)
    optimizer = Optimizer(config)
    live_trader = LiveTrader(config)

    # Data processing and feature engineering
    raw_data = data_processor.load_data()
    processed_data = data_processor.process_data(raw_data)
    featured_data = feature_engineer.engineer_features(processed_data)

    # Identify market regime
    featured_data = {tf: identify_market_regime(df) for tf, df in featured_data.items()}

    # Model building and training
    model = model_builder.build_model(featured_data)
    trained_model = model_builder.train_model(model, featured_data)

    # Signal generation and risk management
    signals, dynamic_weights = signal_generator.generate_signals(trained_model, featured_data)
    trend_cons = calculate_trend_consistency(featured_data)
    s_comprehensive = signal_generator.generate_comprehensive_signal(signals, dynamic_weights, trend_cons)
    
    risk_total = risk_manager.calculate_total_risk(processed_data)
    entry_strategy = signal_generator.determine_entry_strategy(s_comprehensive, signals['1d']['entry_level'], trend_cons, risk_total)
    
    managed_signals = risk_manager.apply_risk_management(signals, processed_data)

    # Backtesting
    backtest_results = backtester.run_backtest(managed_signals, processed_data, dynamic_weights)
    print("Backtest Results:", backtest_results)

    # Optimization
    optimized_params = optimizer.optimize(model, featured_data, processed_data)
    print("Optimized Parameters:", optimized_params)

    # Live trading simulation
    live_trader.start_trading(trained_model, optimized_params)

if __name__ == "__main__":
    main()
