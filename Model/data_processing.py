import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Utils.utils import handle_nan_inf
import logging
from Data.data_acquisition import DataAcquisition

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.get('data_dir', 'Data/csv_files')  # Default to 'Data/csv_files' if not specified

    def load_data(self):
        data = {}
        end_date = datetime.now().strftime('%Y-%m-%d')
        max_retries = 5
        retry_delay = 5  # seconds

        data_acquisition = DataAcquisition(self.config['asset'], self.config['timeframes'], self.data_dir)

        for tf in self.config['timeframes']:
            logger.info(f"Loading data for timeframe: {tf}")
            
            if tf == '1m':
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            elif tf in ['5m', '15m']:
                start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            else:
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
            
            for attempt in range(max_retries):
                try:
                    data_acquisition.fetch_data(start_date, end_date)
                    df = data_acquisition.get_data(tf)
                    if df.empty:
                        logger.warning(f"Empty DataFrame for timeframe {tf}")
                    else:
                        logger.info(f"Loaded {len(df)} rows for timeframe {tf}")
                        if df.index.tz is not None:
                            df.index = df.index.tz_convert('UTC')
                        else:
                            df.index = df.index.tz_localize('UTC', nonexistent='shift_forward')
                        data[tf] = df
                    break
                except Exception as e:
                    logger.error(f"Error loading data for timeframe {tf} (Attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to load data for timeframe {tf} after {max_retries} attempts")
        
        if not data:
            raise ValueError("No data could be loaded for any timeframe")
        
        return data

    def process_data(self, data):
        processed_data = {}
        for tf, df in data.items():
            if df.empty:
                logger.warning(f"Skipping empty DataFrame for timeframe {tf}")
                continue
            
            logger.info(f"Processing {len(df)} rows for timeframe {tf}")
            df = self.convert_to_numeric(df)
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df = self.clean_data(df)
            
            logger.info(f"After cleaning: {len(df)} rows for timeframe {tf}")
            logger.info(f"Returns range for {tf}: [{df['returns'].min()}, {df['returns'].max()}]")
            logger.info(f"Log returns range for {tf}: [{df['log_returns'].min()}, {df['log_returns'].max()}]")
            
            df = self.pad_sequence(df)
            
            processed_data[tf] = df
            logger.info(f"Processed {len(df)} rows for timeframe {tf}")
            
            # Log data types of each column
            for column in df.columns:
                logger.info(f"Column {column} dtype: {df[column].dtype}")
            
            # Log any remaining object columns
            object_columns = df.select_dtypes(include=['object']).columns
            if not object_columns.empty:
                logger.warning(f"Object columns remaining in {tf}: {list(object_columns)}")
                for col in object_columns:
                    logger.warning(f"Sample values for {col}: {df[col].head()}")
        
        if not processed_data:
            logger.warning("No data could be processed for any timeframe")
            return {}
        
        processed_data = self.align_timeframes(processed_data)
        
        return processed_data

    def convert_to_numeric(self, df):
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"Converted column {col} to numeric")
                except Exception as e:
                    logger.error(f"Error converting column {col} to numeric: {str(e)}")
        return df

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

    def timeframe_to_offset(self, timeframe):
        if timeframe == '1M':
            return pd.offsets.MonthEnd()
        return pd.Timedelta(timeframe)

    def timeframe_to_minutes(self, timeframe):
        if timeframe == '1M':
            return 43200  # Approximate a month as 30 days
        return int(pd.Timedelta(timeframe).total_seconds() / 60)

    def align_timeframes(self, data):
        if not data:
            logger.warning("No data to align")
            return {}
        
        aligned_data = {}
        base_tf = min(data.keys(), key=lambda x: self.timeframe_to_minutes(x))  # Use the shortest timeframe as base
        base_df = data[base_tf]

        for tf, df in data.items():
            if tf == base_tf:
                aligned_data[tf] = df
            else:
                # Resample to the base timeframe
                resampled = df.resample(self.timeframe_to_offset(base_tf)).last()
                # Forward fill missing values, but only within the original timeframe
                filled = resampled.ffill(limit=int(self.timeframe_to_minutes(tf) / self.timeframe_to_minutes(base_tf)) - 1)
                # Align with the base DataFrame
                aligned = filled.reindex(base_df.index, method=None)
                aligned_data[tf] = aligned

            logger.info(f"Aligned {len(aligned_data[tf])} rows for timeframe {tf}")

        return aligned_data
