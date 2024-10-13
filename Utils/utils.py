import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import yaml

def handle_nan_inf(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.interpolate(method='time', inplace=True, limit_direction='both')
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_maximum_drawdown(cumulative_returns):
    return np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))

def prepare_data_for_training(featured_data, config):
    all_data = []
    for tf in config['timeframes']:
        df = featured_data[tf]
        features = df.values
        
        # Extract the last three columns as volatility, accuracy, and trend_strength
        volatility = features[:, -3]
        accuracy = features[:, -2]
        trend_strength = features[:, -1]
        
        # Use all columns except the last three as input features
        input_features = features[:, :-3]
        
        # For target, we'll use the first column (assuming it's returns or a similar target variable)
        target = features[:, 0]
        
        all_data.append((input_features[:-1], volatility[:-1], accuracy[:-1], trend_strength[:-1], target[1:]))
    
    # Combine data from all timeframes
    combined_data = [np.concatenate(d) for d in zip(*all_data)]
    
    # Convert to PyTorch tensors
    tensors = [torch.FloatTensor(d) for d in combined_data]
    
    # Create dataset
    dataset = TensorDataset(*tensors)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
