import pandas as pd
import numpy as np
import yfinance as yf

class DataProcessor:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        data = {}
        for tf in self.config['timeframes']:
            data[tf] = yf.download("BTC-USD", period="2y", interval=tf)
        return data

    def process_data(self, data):
        processed_data = {}
        for tf, df in data.items():
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df.dropna(inplace=True)
            processed_data[tf] = df
        return processed_data

    def clean_data(self, df):
        # Remove outliers
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
        
        # Handle missing values
        df.interpolate(method='time', inplace=True)
        
        return df

    def align_timeframes(self, data):
        aligned_data = {}
        base_tf = self.config['timeframes'][-1]  # Use the largest timeframe as base
        base_index = data[base_tf].index

        for tf, df in data.items():
            aligned_data[tf] = df.reindex(base_index, method='ffill')

        return aligned_data
