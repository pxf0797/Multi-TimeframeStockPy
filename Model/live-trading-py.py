import time
import torch
import numpy as np
import pandas as pd
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from signal_generation import SignalGenerator
from risk_management import RiskManager
from utils import log_trade, implement_circuit_breaker

class LiveTrader:
    def __init__(self, config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config)

    def start_trading(self, model, optimized_params):
        self.update_model_params(model, optimized_params)
        model.eval()
        
        while True:
            try:
                live_data = self.get_live_data()
                processed_data = self.data_processor.process_data(live_data)
                featured_data = self.feature_engineer.engineer_features(processed_data)
                signals, dynamic_weights = self.signal_generator.generate_signals(model, featured_data)
                managed_signals = self.risk_manager.apply_risk_management(signals, processed_data)
                
                self.execute_trades(managed_signals, dynamic_weights)
                
                if self.check_stop_conditions():
                    break
                
                time.sleep(self.config['trading_interval'])
            except Exception as e:
                print(f"Error in live trading: {e}")
                time.sleep(300)  # Wait for 5 minutes if there's an error

    def get_live_data(self):
        # In a real implementation, this would fetch live market data
        # For simulation purposes, we'll generate random data
        live_data = {}
        for tf in self.config['timeframes']:
            live_data[tf] = self.generate_random_data()
        return live_data

    def generate_random_data(self):
        # Generate random OHLCV data for simulation
        return pd.DataFrame({
            'Open': np.random.randn(100),
            'High': np.random.randn(100),
            'Low': np.random.randn(100),
            'Close': np.random.randn(100),
            'Volume': np.abs(np.random.randn(100)) * 1000
        }, index=pd.date_range(end=pd.Timestamp.now(), periods=100))

    def execute_trades(self, managed_signals, dynamic_weights):
        for tf, signal in