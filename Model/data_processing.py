import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        data = {}
        end_date = datetime.now()
        
        for tf in self.config['timeframes']:
            print(f"Downloading data for timeframe: {tf}")
            
            if tf == '1m':
                start_date = end_date - timedelta(days=7)
            elif tf in ['5m', '15m']:
                start_date = end_date - timedelta(days=60)
            else:
                start_date = end_date - timedelta(days=730)  # 2 years
            
            try:
                df = yf.download(self.config['asset'], start=start_date, end=end_date, interval=tf)
                if df.empty:
                    print(f"Warning: Empty DataFrame for timeframe {tf}")
                else:
                    print(f"Downloaded {len(df)} rows for timeframe {tf}")
                    # Convert index to UTC
                    df.index = df.index.tz_localize('UTC')
                    data[tf] = df
            except Exception as e:
                print(f"Error downloading data for timeframe {tf}: {e}")
        
        return data

    def process_data(self, data):
        processed_data = {}
        for tf, df in data.items():
            if df.empty:
                print(f"Skipping empty DataFrame for timeframe {tf}")
                continue
            
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df = self.clean_data(df)
            df.dropna(inplace=True)
            
            # Pad or truncate the sequence to the specified length
            df = self.pad_sequence(df)
            
            processed_data[tf] = df
            print(f"Processed {len(df)} rows for timeframe {tf}")
        
        # Align timeframes
        processed_data = self.align_timeframes(processed_data)
        
        return processed_data

    def clean_data(self, df):
        # Remove outliers
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
        
        # Handle missing values
        df.interpolate(method='time', inplace=True)
        
        return df

    def pad_sequence(self, df):
        seq_length = self.config['sequence_length']
        if len(df) > seq_length:
            return df.iloc[-seq_length:]
        elif len(df) < seq_length:
            pad_length = seq_length - len(df)
            pad_df = pd.DataFrame(index=range(pad_length), columns=df.columns)
            return pd.concat([pad_df, df]).reset_index(drop=True)
        else:
            return df

    def align_timeframes(self, data):
        aligned_data = {}
        base_tf = max(data.keys(), key=lambda x: len(data[x]))  # Use the timeframe with most data as base
        base_index = data[base_tf].index

        for tf, df in data.items():
            # Ensure all indexes are in UTC
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            elif df.index.tz != 'UTC':
                df.index = df.index.tz_convert('UTC')
            
            aligned_df = df.reindex(base_index, method='ffill')
            aligned_df = self.pad_sequence(aligned_df)
            aligned_data[tf] = aligned_df
            print(f"Aligned {len(aligned_data[tf])} rows for timeframe {tf}")

        return aligned_data
