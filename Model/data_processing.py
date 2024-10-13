import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
from Utils.utils import handle_nan_inf

class DataProcessor:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        data = {}
        end_date = datetime.now()
        max_retries = 5
        retry_delay = 5  # seconds

        for tf in self.config['timeframes']:
            print(f"Downloading data for timeframe: {tf}")
            
            if tf == '1m':
                start_date = end_date - timedelta(days=7)
            elif tf in ['5m', '15m']:
                start_date = end_date - timedelta(days=60)
            else:
                start_date = end_date - timedelta(days=730)  # 2 years
            
            for attempt in range(max_retries):
                try:
                    df = yf.download(self.config['asset'], start=start_date, end=end_date, interval=tf)
                    if df.empty:
                        print(f"Warning: Empty DataFrame for timeframe {tf}")
                    else:
                        print(f"Downloaded {len(df)} rows for timeframe {tf}")
                        if df.index.tz is not None:
                            df.index = df.index.tz_convert('UTC')
                        else:
                            df.index = df.index.tz_localize('UTC', nonexistent='shift_forward')
                        data[tf] = df
                    break
                except Exception as e:
                    print(f"Error downloading data for timeframe {tf} (Attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"Failed to download data for timeframe {tf} after {max_retries} attempts")
        
        if not data:
            raise ValueError("No data could be downloaded for any timeframe")
        
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
            
            df = self.pad_sequence(df)
            
            processed_data[tf] = df
            print(f"Processed {len(df)} rows for timeframe {tf}")
        
        if not processed_data:
            raise ValueError("No data could be processed for any timeframe")
        
        processed_data = self.align_timeframes(processed_data)
        
        return processed_data

    def clean_data(self, df):
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
        
        df = handle_nan_inf(df)
        
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
        if not data:
            raise ValueError("No data to align")
        
        aligned_data = {}
        base_tf = max(data.keys(), key=lambda x: len(data[x]))
        base_index = data[base_tf].index

        for tf, df in data.items():
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC', nonexistent='shift_forward')
            elif df.index.tz != 'UTC':
                df.index = df.index.tz_convert('UTC')
            
            aligned_df = df.reindex(base_index, method='ffill')
            aligned_df = self.pad_sequence(aligned_df)
            aligned_data[tf] = aligned_df
            print(f"Aligned {len(aligned_data[tf])} rows for timeframe {tf}")

        return aligned_data
