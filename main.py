import sys
import os
import logging
import torch
import argparse
import pandas as pd
from datetime import datetime, timedelta

# Add the Model directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model'))

# Import custom modules
from Model.data_processing import DataProcessor
from Model.feature_engineering import FeatureEngineer
from Model.model_building import ModelBuilder, print_model_summary
from Model.signal_generation import SignalGenerator
from Model.risk_management import RiskManager
from Model.backtesting import Backtester
from Model.optimization import Optimizer
from Model.live_trading import LiveTrader
from Utils.utils import calculate_trend_consistency, identify_market_regime
from Data.data_acquisition import DataAcquisition

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    # Load configuration from a file or environment variables
    config = {
        'asset': 'sz000001',
        'end_date_count' : ['2024-10-18', '13000'], # for lesss len 1d, count is the date, others is period counts
        'timeframes': ['5m', '15m', '60m', '1d', '1m', '1q'],
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
        'sequence_length': 20,  # Increased to capture more historical data
        'model_save_path': 'model_params.pth',
        'data_dir': 'Data/csv_files',
    }
    return config

def acquire_data(config):
    logger.info("Starting data acquisition process...")
    
    # Initialize DataAcquisition
    data_acquisition = DataAcquisition(config['asset'], config['end_date_count'], config['timeframes'], config['data_dir'])
    
    # acquire data and verify
    data_acquisition.fetch_data_all()
    data_acquisition.verify_csv_files()
    

def process_data(config, raw_data):
    data_processor = DataProcessor(config)
    feature_engineer = FeatureEngineer(config)

    logger.info("Processing and engineering features...")
    
    try:
        processed_data = data_processor.process_data(raw_data)
        
        # Log the shape of processed data for each timeframe
        for tf, data in processed_data.items():
            logger.info(f"Processed data shape for {tf}: {data.shape}")

        featured_data = feature_engineer.engineer_features(processed_data)

        # Log the shape of featured data for each timeframe
        for tf, data in featured_data.items():
            logger.info(f"Featured data shape for {tf}: {data.shape}")

        logger.info("Identifying market regime...")
        featured_data = {tf: identify_market_regime(df) for tf, df in featured_data.items()}

        # Update input_size based on actual feature count
        if featured_data:
            config['input_size'] = featured_data[list(featured_data.keys())[0]].shape[1]
            logger.info(f"Updated input_size to {config['input_size']}")

        return processed_data, featured_data
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        return None, None

def train_model(config, featured_data, continue_training=False):
    try:
        model_builder = ModelBuilder(config)
        model = model_builder.build_model(featured_data)

        if continue_training and os.path.exists(config['model_save_path']):
            logger.info("Loading existing model parameters...")
            model.load_state_dict(torch.load(config['model_save_path'], map_location=config['device']))

        logger.info("Training model...")
        trained_model = model_builder.train_model(model, featured_data)

        logger.info("Saving model parameters...")
        torch.save(trained_model.state_dict(), config['model_save_path'])

        return trained_model
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return None

def generate_signals(config, model, featured_data):
    try:
        signal_generator = SignalGenerator(config)
        logger.info("Generating signals...")
        signals, dynamic_weights = signal_generator.generate_signals(model, featured_data)
        trend_cons = calculate_trend_consistency(featured_data)
        s_comprehensive = signal_generator.generate_comprehensive_signal(signals, dynamic_weights, trend_cons)
        return signals, dynamic_weights, s_comprehensive, trend_cons
    except Exception as e:
        logger.error(f"Error in signal generation: {str(e)}")
        return None, None, None, None

def manage_risk(config, signals, processed_data, s_comprehensive, trend_cons):
    try:
        risk_manager = RiskManager(config)
        logger.info("Managing risk...")
        risk_total = risk_manager.calculate_total_risk(processed_data)
        entry_strategy = SignalGenerator(config).determine_entry_strategy(s_comprehensive, signals['1d']['entry_level'], trend_cons, risk_total)
        managed_signals = risk_manager.apply_risk_management(signals, processed_data)
        return managed_signals, entry_strategy
    except Exception as e:
        logger.error(f"Error in risk management: {str(e)}")
        return None, None

def run_backtest(config, managed_signals, processed_data, dynamic_weights):
    try:
        backtester = Backtester(config)
        logger.info("Running backtests...")
        backtest_results = backtester.run_backtest(managed_signals, processed_data, dynamic_weights)
        logger.info(f"Backtest Results: {backtest_results}")
        return backtest_results
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        return None

def optimize_parameters(config, model, featured_data, processed_data):
    try:
        optimizer = Optimizer(config)
        logger.info("Optimizing parameters...")
        optimized_params = optimizer.optimize(model, featured_data, processed_data)
        logger.info(f"Optimized Parameters: {optimized_params}")
        return optimized_params
    except Exception as e:
        logger.error(f"Error in parameter optimization: {str(e)}")
        return None

def start_live_trading(config, model, optimized_params):
    try:
        live_trader = LiveTrader(config)
        logger.info("Starting live trading simulation...")
        live_trader.start_trading(model, optimized_params)
    except Exception as e:
        logger.error(f"Error in live trading: {str(e)}")

def model_to_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Multi-timeframe stock trading system")
    parser.add_argument('--mode', choices=['acquire_data', 'train', 'backtest', 'optimize', 'live', 'full_cycle'], required=True, help="Operation mode")
    parser.add_argument('--continue_training', action='store_true', help="Continue training from saved parameters")
    args = parser.parse_args()

    config = load_config()
    config['signal_generator'] = SignalGenerator(config)  # Add signal_generator to config

    try:
        if args.mode == 'acquire_data' or args.mode == 'full_cycle':
            raw_data = acquire_data(config)
            if raw_data is None:
                logger.error("Data acquisition failed. Exiting.")
                return
            
            processed_data, featured_data = process_data(config, raw_data)
            if not featured_data:
                logger.error("Data processing failed. Exiting.")
                return
        
        if args.mode == 'full_cycle' or args.mode in ['train', 'backtest', 'optimize', 'live']:
            if 'processed_data' not in locals() or 'featured_data' not in locals():
                logger.info("Loading previously acquired data...")
                raw_data = acquire_data(config)
                processed_data, featured_data = process_data(config, raw_data)

            model_builder = ModelBuilder(config)
            if args.mode == 'train' or args.mode == 'full_cycle' or not os.path.exists(config['model_save_path']):
                logger.info("Training new model...")
                trained_model = train_model(config, featured_data, args.continue_training)
            else:
                logger.info("Loading existing model...")
                trained_model = model_builder.build_model(featured_data)
                try:
                    trained_model.load_state_dict(torch.load(config['model_save_path'], map_location=config['device']))
                except RuntimeError as e:
                    logger.warning(f"Failed to load existing model: {e}")
                    logger.info("Training new model...")
                    trained_model = train_model(config, featured_data, False)

            if trained_model is None:
                logger.error("Model training or loading failed. Exiting.")
                return

            signals, dynamic_weights, s_comprehensive, trend_cons = generate_signals(config, trained_model, featured_data)
            managed_signals, entry_strategy = manage_risk(config, signals, processed_data, s_comprehensive, trend_cons)

            if args.mode == 'backtest' or args.mode == 'full_cycle':
                run_backtest(config, managed_signals, processed_data, dynamic_weights)
            
            if args.mode == 'optimize' or args.mode == 'full_cycle':
                optimize_parameters(config, trained_model, featured_data, processed_data)
            
            if args.mode == 'live' or args.mode == 'full_cycle':
                flat_params = model_to_flat_params(trained_model)
                start_live_trading(config, trained_model, flat_params)

        logger.info("Multi-timeframe stock trading process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
