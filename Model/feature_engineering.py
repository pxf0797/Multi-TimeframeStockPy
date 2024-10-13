import numpy as np
import pandas as pd
from Utils.utils import handle_nan_inf

class FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def engineer_features(self, processed_data):
        featured_data = {}
        for tf, df in processed_data.items():
            featured_data[tf] = self.engineer_timeframe_features(df)
        return featured_data

    def engineer_timeframe_features(self, df):
        df = df.copy()
        
        # Ensure we have 'Close' column
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        
        # Basic price and volume features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volume_change'] = df['Volume'].pct_change() if 'Volume' in df.columns else 0
        
        # Moving averages
        for period in [5, 10, 20]:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
        
        # Exponential moving averages
        for period in [5, 10, 20]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min() if 'Low' in df.columns else df['Close'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max() if 'High' in df.columns else df['Close'].rolling(window=14).max()
        df['%K'] = (df['Close'] - low_14) * 100 / (high_14 - low_14)
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Additional features
        df['Volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['Momentum'] = df['Close'] - df['Close'].shift(4)
        df['Rate_of_Change'] = (df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12)
        
        # Handle NaN and inf values
        df = handle_nan_inf(df)
        
        # Select features
        feature_columns = [
            'returns', 'log_returns', 'volume_change',
            'MA_5', 'MA_10', 'MA_20',
            'EMA_5', 'EMA_10', 'EMA_20',
            'MACD', 'MACD_signal', 'MACD_hist',
            'RSI',
            'BB_middle', 'BB_std', 'BB_upper', 'BB_lower',
            '%K', '%D',
            'Volatility', 'Momentum', 'Rate_of_Change'
        ]
        
        # Add placeholder columns if needed
        for i in range(39 - len(feature_columns)):
            placeholder_name = f'Placeholder_{i}'
            df[placeholder_name] = 0
            feature_columns.append(placeholder_name)
        
        # Add the last three columns
        feature_columns += ['Volatility', 'Accuracy', 'Trend_Strength']
        
        # Ensure Accuracy and Trend_Strength exist (use placeholder if not)
        if 'Accuracy' not in df.columns:
            df['Accuracy'] = 0
        if 'Trend_Strength' not in df.columns:
            df['Trend_Strength'] = 0
        
        assert len(feature_columns) == 42, f"Expected 42 features, but got {len(feature_columns)}"
        
        return df[feature_columns]
