import numpy as np
from scipy.optimize import minimize
from Model.model_building import build_model, train_model
from Model.signal_generation import SignalGenerator
from Model.backtesting import Backtester
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from Utils.utils import prepare_data_for_training

class Optimizer:
    def __init__(self, config):
        self.config = config
        self.signal_generator = SignalGenerator(config)
        self.backtester = Backtester(config)

    def objective(self, params):
        # Unpack parameters
        learning_rate, hidden_size, num_layers, num_heads = params
        
        # Update config with new parameters
        self.config['learning_rate'] = learning_rate
        self.config['hidden_size'] = int(hidden_size)
        self.config['num_layers'] = int(num_layers)
        self.config['num_heads'] = int(num_heads)
        
        # Build and train model
        model = build_model(self.config)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Prepare data for training
        train_data, val_data = prepare_data_for_training(self.featured_data, self.config)
        train_loader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config['batch_size'])
        
        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, self.config['num_epochs'], self.config['device'])
        
        # Generate signals
        signals, dynamic_weights = self.signal_generator.generate_signals(trained_model, self.featured_data)
        
        # Run backtest
        results = self.backtester.run_backtest(signals, self.featured_data, dynamic_weights)
        overall_performance = self.backtester.calculate_overall_performance(results)
        
        # Return negative Sharpe ratio (we want to maximize it)
        return -overall_performance['Overall Sharpe Ratio']

    def optimize(self, featured_data):
        self.featured_data = featured_data
        
        # Initial guess
        x0 = [self.config['learning_rate'], self.config['hidden_size'], self.config['num_layers'], self.config['num_heads']]
        
        # Bounds for parameters
        bounds = [(1e-5, 1e-2), (32, 256), (1, 5), (1, 8)]
        
        # Run optimization
        result = minimize(self.objective, x0, method='L-BFGS-B', bounds=bounds)
        
        # Return optimized parameters
        return {
            'learning_rate': result.x[0],
            'hidden_size': int(result.x[1]),
            'num_layers': int(result.x[2]),
            'num_heads': int(result.x[3])
        }
