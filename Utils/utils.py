import yaml
import logging
import csv
from datetime import datetime

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data_for_training(featured_data, config):
    # Implement data preparation logic here
    # This is a placeholder function
    return featured_data, featured_data

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
