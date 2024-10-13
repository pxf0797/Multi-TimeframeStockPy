import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
from Utils.utils import handle_nan_inf
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config

    def load_data(self):
        data = {}
        end_date = datetime.now()
        max_retries = 5
        retry_delay = 5  # seconds

        for tf in self.config['timeframes']:
            logger.info(f"Downloading data for timeframe: {tf}")
            
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
                        logger.warning(f"Empty DataFrame for timeframe {tf}")
                    else:
                        logger.info(f"Downloaded {len(df)} rows for timeframe {tf}")
                        if df.index.tz is not None:
                            df.index = df.index.tz_convert('UTC')
                        else:
                            df.index = df.index.tz_localize('UTC', nonexistent='shift_forward')
                        data[tf] = df
                    break
                except Exception as e:
                    logger.error(f"Error downloading data for timeframe {tf} (Attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to download data for timeframe {tf} after {max_retries} attempts")
        
        if not data:
            raise ValueError("No data could be downloaded for any timeframe")
        
        return data

    def process_data(self, data):
        processed_data = {}
        for tf, df in data.items():
            if df.empty:
                logger.warning(f"Skipping empty DataFrame for timeframe {tf}")
                continue
            
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df = self.clean_data(df)
            df.dropna(inplace=True)
            
            logger.info(f"Returns range for {tf}: [{df['returns'].min()}, {df['returns'].max()}]")
            logger.info(f"Log returns range for {tf}: [{df['log_returns'].min()}, {df['log_returns'].max()}]")
            
            df = self.pad_sequence(df)
            
            processed_data[tf] = df
            logger.info(f"Processed {len(df)} rows for timeframe {tf}")
        
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
            return df  # Return the original DataFrame without padding
        else:
            return df

    def align_timeframes(self, data):
        if not data:
            raise ValueError("No data to align")
        
        aligned_data = {}
        base_tf = min(data.keys(), key=lambda x: pd.Timedelta(x))  # Use the shortest timeframe as base
        base_df = data[base_tf]

        for tf, df in data.items():
            if tf == base_tf:
                aligned_data[tf] = df
            else:
                # Resample to the base timeframe
                resampled = df.resample(base_tf).last()
                # Forward fill missing values, but only within the original timeframe
                filled = resampled.fillna(method='ffill', limit=int(pd.Timedelta(tf) / pd.Timedelta(base_tf)) - 1)
                # Align with the base DataFrame
                aligned = filled.reindex(base_df.index, method=None)
                aligned_data[tf] = aligned

            logger.info(f"Aligned {len(aligned_data[tf])} rows for timeframe {tf}")

        return aligned_data
