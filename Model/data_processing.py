import pandas as pd
import numpy as np
import os
from Utils.utils import handle_nan_inf

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1h',
            '1d': 'D'
        }

    def load_data(self):
        data = {}
        for tf in self.config['timeframes']:
            file_path = os.path.join('Data', f"{self.config['asset']}_{tf}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                # Ensure index is timezone-aware UTC
                if df.index.tz is None:
                    df.index = pd.to_datetime(df.index, utc=True)
                else:
                    df.index = df.index.tz_convert('UTC')
                # Remove duplicate indices
                df = df[~df.index.duplicated(keep='first')]
                # Set the frequency
                df = df.asfreq(self.freq_map[tf])
                data[tf] = df
                print(f"Loaded {len(df)} rows for timeframe {tf}")
            else:
                print(f"Warning: No data file found for timeframe {tf}")
        return data

    def process_data(self, data):
        processed_data = {}
        for tf, df in data.items():
            if df.empty:
                print(f"Skipping empty DataFrame for timeframe {tf}")
                continue
            
            df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df = self.clean_data(df)
            df.dropna(inplace=True)
            
            df = self.pad_sequence(df)
            
            processed_data[tf] = df
            print(f"Processed {len(df)} rows for timeframe {tf}")
        
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
            pad_index = pd.date_range(end=df.index[0], periods=pad_length, freq=df.index.freq)
            pad_df = pd.DataFrame(index=pad_index, columns=df.columns).fillna(0)
            return pd.concat([pad_df, df])
        else:
            return df

    def align_timeframes(self, data):
        aligned_data = {}
        base_tf = max(data.keys(), key=lambda x: len(data[x]))
        base_index = data[base_tf].index

        for tf, df in data.items():
            if tf == base_tf:
                aligned_data[tf] = df
            else:
                resampled_df = df.resample(self.freq_map[base_tf]).last().ffill()
                aligned_df = resampled_df.reindex(base_index, method='ffill')
                aligned_df = self.pad_sequence(aligned_df)
                aligned_data[tf] = aligned_df
            print(f"Aligned {len(aligned_data[tf])} rows for timeframe {tf}")

        return aligned_data
