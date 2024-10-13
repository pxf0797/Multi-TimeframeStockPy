import yaml
import logging
import csv
from datetime import datetime
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data_for_training(featured_data, config):
    # Print column names for debugging
    print("Available columns:")
    for tf, df in featured_data.items():
        print(f"Timeframe {tf}: {df.columns.tolist()}")

    # Combine data from all timeframes
    combined_data = []
    for tf, df in featured_data.items():
        df_values = df.values
        combined_data.append(df_values)
    
    # Stack the data from different timeframes
    X = np.stack(combined_data, axis=1)
    
    # Extract required features
    inputs = X[:, :, :39]  # Assuming the first 39 columns are the input features
    volatility = X[:, :, 39]  # Assuming 'Volatility' is the 40th column
    accuracy = X[:, :, 40]  # Assuming 'Accuracy' is the 41st column
    trend_strength = X[:, :, 41]  # Assuming 'Trend_Strength' is the 42nd column
    
    # Use 'returns' as the target variable
    y = featured_data[config['timeframes'][0]]['returns'].values[:-1]
    
    # Remove the last row to align all data
    inputs = inputs[:-1]
    volatility = volatility[:-1]
    accuracy = accuracy[:-1]
    trend_strength = trend_strength[:-1]
    
    # Remove any rows with NaN values
    mask = ~np.isnan(y)
    inputs = inputs[mask]
    volatility = volatility[mask]
    accuracy = accuracy[mask]
    trend_strength = trend_strength[mask]
    y = y[mask]
    
    # Split the data into training and validation sets
    X_train, X_val, v_train, v_val, a_train, a_val, t_train, t_val, y_train, y_val = train_test_split(
        inputs, volatility, accuracy, trend_strength, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    train_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(v_train),
        torch.FloatTensor(a_train),
        torch.FloatTensor(t_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    val_data = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(v_val),
        torch.FloatTensor(a_val),
        torch.FloatTensor(t_val),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    
    return train_data, val_data

def log_trade(trade_info, log_file):
    with open(log_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=trade_info.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(trade_info)

def implement_circuit_breaker(price_change, threshold):
    return abs(price_change) > threshold

def setup_logging(config):
    logger = logging.getLogger()
    logger.setLevel(config['log_level'])

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(config['log_file'])
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(equity_curve):
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def calculate_win_rate(trades):
    winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
    return winning_trades / len(trades) if trades else 0

def calculate_profit_factor(trades):
    gross_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
    gross_loss = abs(sum(trade['profit'] for trade in trades if trade['profit'] < 0))
    return gross_profit / gross_loss if gross_loss != 0 else float('inf')

def calculate_expectancy(trades):
    if not trades:
        return 0
    total_profit = sum(trade['profit'] for trade in trades)
    return total_profit / len(trades)

def calculate_average_trade_duration(trades):
    if not trades:
        return 0
    durations = [(trade['exit_time'] - trade['entry_time']).total_seconds() for trade in trades]
    return sum(durations) / len(durations)

def calculate_risk_reward_ratio(trades):
    if not trades:
        return 0
    total_risk = sum(trade['risk'] for trade in trades)
    total_reward = sum(trade['reward'] for trade in trades)
    return total_reward / total_risk if total_risk != 0 else float('inf')

def handle_nan_inf(data):
    """
    Handle NaN and Inf values in the data.
    
    :param data: numpy array or pandas DataFrame
    :return: data with NaN and Inf values handled
    """
    if isinstance(data, np.ndarray):
        # Replace inf with large finite numbers
        data = np.nan_to_num(data, nan=0.0, posinf=1e30, neginf=-1e30)
    else:  # Assume it's a pandas DataFrame
        # Replace inf with large finite numbers
        data = data.replace([np.inf, -np.inf], [1e30, -1e30])
        # Fill NaN values with 0
        data = data.fillna(0)
    
    return data
