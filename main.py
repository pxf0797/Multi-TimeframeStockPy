import sys
import os
import logging
import torch
import argparse

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

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    # Load configuration from a file or environment variables
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
        'model_save_path': 'model_params.pth',
        'data_dir': 'Data/csv_files',  # Add this line for the data directory
    }
    return config

def process_data(config):
    data_processor = DataProcessor(config)
    feature_engineer = FeatureEngineer(config)

    logger.info("Loading and processing data...")
    raw_data = data_processor.load_data()
    processed_data = data_processor.process_data(raw_data)
    featured_data = feature_engineer.engineer_features(processed_data)

    logger.info("Identifying market regime...")
    featured_data = {tf: identify_market_regime(df) for tf, df in featured_data.items()}

    return processed_data, featured_data

def train_model(config, featured_data, continue_training=False):
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

def generate_signals(config, model, featured_data):
    signal_generator = SignalGenerator(config)
    logger.info("Generating signals...")
    signals, dynamic_weights = signal_generator.generate_signals(model, featured_data)
    trend_cons = calculate_trend_consistency(featured_data)
    s_comprehensive = signal_generator.generate_comprehensive_signal(signals, dynamic_weights, trend_cons)
    return signals, dynamic_weights, s_comprehensive, trend_cons

def manage_risk(config, signals, processed_data, s_comprehensive, trend_cons):
    risk_manager = RiskManager(config)
    logger.info("Managing risk...")
    risk_total = risk_manager.calculate_total_risk(processed_data)
    entry_strategy = SignalGenerator(config).determine_entry_strategy(s_comprehensive, signals['1d']['entry_level'], trend_cons, risk_total)
    managed_signals = risk_manager.apply_risk_management(signals, processed_data)
    return managed_signals, entry_strategy

def run_backtest(config, managed_signals, processed_data, dynamic_weights):
    backtester = Backtester(config)
    logger.info("Running backtests...")
    backtest_results = backtester.run_backtest(managed_signals, processed_data, dynamic_weights)
    logger.info(f"Backtest Results: {backtest_results}")
    return backtest_results

def optimize_parameters(config, model, featured_data, processed_data):
    optimizer = Optimizer(config)
    logger.info("Optimizing parameters...")
    optimized_params = optimizer.optimize(model, featured_data, processed_data)
    logger.info(f"Optimized Parameters: {optimized_params}")
    return optimized_params

def start_live_trading(config, model, optimized_params):
    live_trader = LiveTrader(config)
    logger.info("Starting live trading simulation...")
    live_trader.start_trading(model, optimized_params)

def model_to_flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Multi-timeframe stock trading system")
    parser.add_argument('--mode', choices=['train', 'backtest', 'optimize', 'live'], required=True, help="Operation mode")
    parser.add_argument('--continue_training', action='store_true', help="Continue training from saved parameters")
    args = parser.parse_args()

    config = load_config()
    config['signal_generator'] = SignalGenerator(config)  # Add signal_generator to config

    try:
        processed_data, featured_data = process_data(config)

        if args.mode == 'train' or args.continue_training:
            trained_model = train_model(config, featured_data, args.continue_training)
        else:
            model_builder = ModelBuilder(config)
            trained_model = model_builder.build_model(featured_data)
            trained_model.load_state_dict(torch.load(config['model_save_path'], map_location=config['device']))

        signals, dynamic_weights, s_comprehensive, trend_cons = generate_signals(config, trained_model, featured_data)
        managed_signals, entry_strategy = manage_risk(config, signals, processed_data, s_comprehensive, trend_cons)

        if args.mode == 'backtest':
            run_backtest(config, managed_signals, processed_data, dynamic_weights)
        elif args.mode == 'optimize':
            optimize_parameters(config, trained_model, featured_data, processed_data)
        elif args.mode == 'live':
            flat_params = model_to_flat_params(trained_model)
            start_live_trading(config, trained_model, flat_params)

        logger.info("Multi-timeframe stock trading process completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
