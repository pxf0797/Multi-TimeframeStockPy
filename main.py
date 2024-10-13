import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from Model.data_processing import DataProcessor
from Model.feature_engineering import FeatureEngineer
from Model.model_building import build_model, train_model
from Model.signal_generation import SignalGenerator
from Model.risk_management import RiskManager
from Model.backtesting import Backtester
from Model.optimization import Optimizer
from Utils.utils import load_config, prepare_data_for_training

def process_data(config):
    data_processor = DataProcessor(config)
    feature_engineer = FeatureEngineer(config)
    
    raw_data = data_processor.load_data()
    processed_data = data_processor.process_data(raw_data)
    featured_data = feature_engineer.engineer_features(processed_data)
    
    return processed_data, featured_data

def generate_signals(config, model, featured_data):
    signal_generator = SignalGenerator(config)
    signals, dynamic_weights = signal_generator.generate_signals(model, featured_data)
    s_comprehensive = signal_generator.generate_comprehensive_signal(signals, dynamic_weights, trend_consistency=1)
    trend_cons = np.mean([np.sign(s) for s in signals.values()])
    return signals, dynamic_weights, s_comprehensive, trend_cons

def manage_risk(config, signals, dynamic_weights):
    risk_manager = RiskManager(config)
    position_sizes = risk_manager.calculate_position_sizes(signals, dynamic_weights)
    return position_sizes

def run_backtest(config, signals, dynamic_weights, featured_data):
    backtester = Backtester(config)
    results = backtester.run_backtest(signals, featured_data, dynamic_weights)
    overall_performance = backtester.calculate_overall_performance(results)
    return results, overall_performance

def optimize_parameters(config, featured_data):
    optimizer = Optimizer(config)
    best_params = optimizer.optimize(featured_data)
    return best_params

def main():
    config = load_config()
    
    processed_data, featured_data = process_data(config)
    
    if config['mode'] == 'train':
        # Prepare data for training
        train_data, val_data = prepare_data_for_training(featured_data, config)
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'])
        
        # Build and train model
        model = build_model(config)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, config['num_epochs'], config['device'])
        
        # Save the trained model
        torch.save(trained_model.state_dict(), config['model_save_path'])
        print("Model training completed and saved.")
    
    elif config['mode'] == 'backtest':
        model = build_model(config)
        model.load_state_dict(torch.load(config['model_save_path'], map_location=config['device']))
        model.eval()
        
        signals, dynamic_weights, s_comprehensive, trend_cons = generate_signals(config, model, featured_data)
        position_sizes = manage_risk(config, signals, dynamic_weights)
        results, overall_performance = run_backtest(config, signals, dynamic_weights, featured_data)
        
        print("Backtest Results:", results)
        print("Overall Performance:", overall_performance)
    
    elif config['mode'] == 'optimize':
        best_params = optimize_parameters(config, featured_data)
        print("Optimized Parameters:", best_params)
    
    else:
        print("Invalid mode specified in config.")

if __name__ == "__main__":
    main()
