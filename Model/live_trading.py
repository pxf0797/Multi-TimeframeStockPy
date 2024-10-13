import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from Model.data_processing import DataProcessor
from Model.feature_engineering import FeatureEngineer
from Model.signal_generation import SignalGenerator
from Model.risk_management import RiskManager
from Utils.utils import log_trade, implement_circuit_breaker, setup_logging
import ccxt
import logging
import os
import signal

class LiveTrader:
    def __init__(self, config, model):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config)
        self.exchange = self.setup_exchange()
        self.logger = setup_logging(config)
        self.is_running = True
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_exchange(self):
        exchange_class = getattr(ccxt, self.config['exchange'])
        return exchange_class({
            'apiKey': self.config['api_key'],
            'secret': self.config['api_secret'],
            'enableRateLimit': True,
        })

    def signal_handler(self, signum, frame):
        self.logger.info("Received shutdown signal. Stopping live trading...")
        self.is_running = False

    def start_trading(self, optimized_params=None):
        if optimized_params:
            self.update_model_params(optimized_params)
        self.load_model_state()
        self.model.eval()
        
        self.logger.info("Starting live trading...")
        while self.is_running:
            try:
                live_data = self.get_live_data()
                processed_data = self.data_processor.process_data(live_data)
                featured_data = self.feature_engineer.engineer_features(processed_data)
                signals, dynamic_weights = self.signal_generator.generate_signals(self.model, featured_data)
                managed_signals = self.risk_manager.apply_risk_management(signals, processed_data)
                
                self.execute_trades(managed_signals, dynamic_weights)
                
                if self.check_stop_conditions():
                    break
                
                self.save_model_state()
                time.sleep(self.config['trading_interval'])
            except ccxt.NetworkError as e:
                self.logger.error(f"Network error: {str(e)}. Retrying in 60 seconds...", exc_info=True)
                time.sleep(60)
            except ccxt.ExchangeError as e:
                self.logger.error(f"Exchange error: {str(e)}. Retrying in 300 seconds...", exc_info=True)
                time.sleep(300)
            except Exception as e:
                self.logger.error(f"Unexpected error in live trading: {str(e)}", exc_info=True)
                self.is_running = False

        self.logger.info("Live trading stopped.")
        self.cleanup()

    def get_live_data(self):
        live_data = {}
        for tf in self.config['timeframes']:
            try:
                ohlcv = self.exchange.fetch_ohlcv(self.config['asset'], tf, limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                live_data[tf] = df
            except Exception as e:
                self.logger.error(f"Error fetching data for timeframe {tf}: {str(e)}", exc_info=True)
                raise
        return live_data

    def execute_trades(self, managed_signals, dynamic_weights):
        for tf, signal in managed_signals.items():
            position_size = signal['position_size']
            entry_level = signal['entry_level']
            stop_loss = signal['stop_loss']
            weight = dynamic_weights[tf]
            
            trade_info = {
                'timeframe': tf,
                'position_size': position_size,
                'entry_level': entry_level,
                'stop_loss': stop_loss,
                'dynamic_weight': weight
            }
            
            try:
                if position_size > 0:
                    order = self.exchange.create_market_buy_order(self.config['asset'], position_size)
                elif position_size < 0:
                    order = self.exchange.create_market_sell_order(self.config['asset'], abs(position_size))
                else:
                    continue  # No trade to execute
                
                trade_info['order'] = order
                self.logger.info(f"Executed trade: {trade_info}")
            except Exception as e:
                self.logger.error(f"Failed to execute trade: {str(e)}", exc_info=True)
            
            log_trade(trade_info, self.config['trade_log_file'])

    def update_model_params(self, params):
        start = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data = torch.FloatTensor(params[start:start+numel]).reshape(p.shape).to(self.config['device'])
            start += numel

    def check_stop_conditions(self):
        current_loss = self.calculate_current_loss()
        if self.risk_manager.implement_risk_limits(current_loss, self.config['max_daily_loss']):
            self.logger.warning("Maximum daily loss reached. Stopping trading.")
            return True
        
        price_change = self.calculate_price_change()
        if implement_circuit_breaker(price_change, self.config['circuit_breaker_threshold']):
            self.logger.warning("Circuit breaker triggered. Pausing trading.")
            time.sleep(self.config['circuit_breaker_pause_time'])
        
        return False

    def calculate_current_loss(self):
        try:
            positions = self.exchange.fetch_positions([self.config['asset']])
            open_pnl = sum(float(position['unrealizedPnl']) for position in positions)
            
            since = int(time.time() * 1000) - 86400000  # Last 24 hours
            trades = self.exchange.fetch_my_trades(self.config['asset'], since=since)
            realized_pnl = sum(float(trade['realized_pnl']) for trade in trades if 'realized_pnl' in trade)
            
            return open_pnl + realized_pnl
        except Exception as e:
            self.logger.error(f"Error calculating current loss: {str(e)}", exc_info=True)
            return 0

    def calculate_price_change(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.config['asset'], '1h', limit=2)
            if len(ohlcv) < 2:
                return 0
            
            previous_close = ohlcv[0][4]
            current_close = ohlcv[1][4]
            return (current_close - previous_close) / previous_close
        except Exception as e:
            self.logger.error(f"Error calculating price change: {str(e)}", exc_info=True)
            return 0

    def save_model_state(self):
        try:
            state = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }
            torch.save(state, self.config['model_state_path'])
            self.logger.info("Model state saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving model state: {str(e)}", exc_info=True)

    def load_model_state(self):
        try:
            if os.path.exists(self.config['model_state_path']):
                state = torch.load(self.config['model_state_path'])
                self.model.load_state_dict(state['model_state_dict'])
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                self.config.update(state['config'])
                self.logger.info("Model state loaded successfully.")
            else:
                self.logger.info("No saved model state found. Starting with a fresh model.")
        except Exception as e:
            self.logger.error(f"Error loading model state: {str(e)}", exc_info=True)

    def cleanup(self):
        try:
            # Close all open positions
            positions = self.exchange.fetch_positions([self.config['asset']])
            for position in positions:
                if float(position['contracts']) != 0:
                    self.exchange.create_market_order(
                        self.config['asset'],
                        'sell' if float(position['contracts']) > 0 else 'buy',
                        abs(float(position['contracts'])),
                        {'reduce_only': True}
                    )
            self.logger.info("All positions closed.")
            
            # Cancel all open orders
            open_orders = self.exchange.fetch_open_orders(self.config['asset'])
            for order in open_orders:
                self.exchange.cancel_order(order['id'], self.config['asset'])
            self.logger.info("All open orders canceled.")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
