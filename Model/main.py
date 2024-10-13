import sys
import os
import logging
import torch
import pandas as pd
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from model_building import ModelBuilder, print_model_summary
from signal_generation import SignalGenerator
from risk_management import RiskManager
from backtesting import Backtester
from optimization import Optimizer
from live_trading import LiveTrader
from Utils.utils import calculate_trend_consistency, identify_market_regime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        config = {
            'asset': 'BTC-USD',
            'timeframes': ['1m', '5m', '15m', '1h', '1d', '1M'],
            'ma_periods': [3, 5, 10, 20],
            'macd_params': (5, 10, 5),
            'input_size': 33,  # Will be updated based on actual feature count
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
            'sequence_length': 100,
        }

        logger.info("Initializing components...")
        data_processor = DataProcessor(config)
        feature_engineer = FeatureEngineer(config)
        model_builder = ModelBuilder(config)
        signal_generator = SignalGenerator(config)
        risk_manager = RiskManager(config)
        backtester = Backtester(config)
        optimizer = Optimizer(config)
        live_trader = LiveTrader(config)

        # Add signal_generator to config for use in optimization
        config['signal_generator'] = signal_generator

        # Data processing and feature engineering
        logger.info("Loading and processing data...")
        raw_data = data_processor.load_data()
        processed_data = data_processor.process_data(raw_data)
        featured_data = feature_engineer.engineer_features(processed_data)

        # Print data shapes and check for NaN or infinite values
        logger.info("Checking data quality...")
        for tf, df in featured_data.items():
            logger.info(f"{tf}: shape={df.shape}, NaN={df.isna().sum().sum()}, Inf={np.isinf(df).sum().sum()}")
            featured_data[tf] = df.replace([np.inf, -np.inf], np.nan).dropna()
            logger.info(f"After handling NaN and inf, {tf} shape: {featured_data[tf].shape}")

        # Ensure all timeframes have the same number of features
        feature_count = len(featured_data[list(featured_data.keys())[0]].columns)
        for tf in featured_data:
            if len(featured_data[tf].columns) != feature_count:
                logger.warning(f"{tf} has {len(featured_data[tf].columns)} features, expected {feature_count}")
                featured_data[tf] = featured_data[tf].reindex(columns=featured_data[list(featured_data.keys())[0]].columns, fill_value=0)

        # Update input_size based on actual feature count
        config['input_size'] = feature_count - 1  # Subtract 1 to exclude the target variable
        logger.info(f"Updated input_size: {config['input_size']}")

        # Identify market regime
        logger.info("Identifying market regime...")
        featured_data = {tf: identify_market_regime(df) for tf, df in featured_data.items()}

        # Model building and training
        logger.info("Building and training model...")
        model = model_builder.build_model(featured_data)
        print_model_summary(model, config)
        trained_model = model_builder.train_model(model, featured_data)

        # Signal generation and risk management
        logger.info("Generating signals and managing risk...")
        signals, dynamic_weights = signal_generator.generate_signals(trained_model, featured_data)
        trend_cons = calculate_trend_consistency(featured_data)
        s_comprehensive = signal_generator.generate_comprehensive_signal(signals, dynamic_weights, trend_cons)
        
        risk_total = risk_manager.calculate_total_risk(processed_data)
        entry_strategy = signal_generator.determine_entry_strategy(s_comprehensive, signals['1d']['entry_level'], trend_cons, risk_total)
        
        managed_signals = risk_manager.apply_risk_management(signals, processed_data)

        # Backtesting
        logger.info("Running backtests...")
        backtest_results = backtester.run_backtest(managed_signals, processed_data, dynamic_weights)
        logger.info(f"Backtest Results: {backtest_results}")

        # Optimization
        logger.info("Optimizing parameters...")
        optimized_params = optimizer.optimize(model, featured_data, processed_data)
        logger.info(f"Optimized Parameters: {optimized_params}")

        # Live trading simulation
        logger.info("Starting live trading simulation...")
        live_trader.start_trading(trained_model, optimized_params)

        logger.info("Multi-timeframe stock trading process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
