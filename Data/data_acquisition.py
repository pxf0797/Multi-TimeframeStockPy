import pandas as pd
import numpy as np
from typing import List, Dict
import os
import logging
from datetime import datetime, timedelta
from GetStock import GetStock

logger = logging.getLogger(__name__)

class DataAcquisition:
    def __init__(self, symbol: str, end_date_count: List[str], timeframes: List[str], data_dir: str):
        self.symbol = symbol
        end_date = datetime.strptime(end_date_count[0], '%Y-%m-%d')
        count = int(end_date_count[1])
        start_date = (end_date - timedelta(days=count)).strftime('%Y-%m-%d')
        # calculate weeks count
        if count > 6000:
            start_data_week = (end_date - timedelta(days=7*6000)).strftime('%Y-%m-%d')
        else:
            start_data_week = (end_date - timedelta(days=7*count)).strftime('%Y-%m-%d')
        # calculate monthes count
        if count > 2000:
            start_data_month = (end_date - timedelta(days=30*2000)).strftime('%Y-%m-%d')
        else:
            start_data_month = (end_date - timedelta(days=30*count)).strftime('%Y-%m-%d')
        # calculate quatres count
        if count > 500:
            start_data_quatre = (end_date - timedelta(days=90*500)).strftime('%Y-%m-%d')
        else:
            start_data_quatre = (end_date - timedelta(days=90*count)).strftime('%Y-%m-%d')
        self.period_item = {
            '5m': [start_date, end_date.strftime('%Y-%m-%d')],
            '15m': [start_date, end_date.strftime('%Y-%m-%d')],
            '60m': [start_date, end_date.strftime('%Y-%m-%d')],
            '1d': [start_date, end_date.strftime('%Y-%m-%d')],
            '1w': [start_data_week, end_date.strftime('%Y-%m-%d')],
            '1m': [start_data_month, end_date.strftime('%Y-%m-%d')],
            '1q': [start_data_quatre, end_date.strftime('%Y-%m-%d')]
        }
        self.timeframes = sorted(timeframes)
        self.data_dir = data_dir
        self.data: Dict[str, pd.DataFrame] = {}
        self.gs = GetStock()
        self.gs.set_stock_name(self.symbol)
        
        # Check if the data directory exists, if not, create it
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")

    def fetch_data(self, start_date: str, end_date: str):
        logger.info(f"Fetching data for {self.symbol} from {start_date} to {end_date}")

        for tf in self.timeframes:
            try:
                logger.info(f"Processing timeframe: {tf}")
                df = self.gs.get_stock_data(start_date, end_date, tf, self.data_dir)
                if df is not None and not df.empty:
                    self.data[tf] = df
                    logger.info(f"Fetched data shape for {tf}: {df.shape}")
                else:
                    logger.warning(f"No data fetched for timeframe {tf}")
            except Exception as e:
                logger.error(f"Error fetching data for timeframe {tf}: {str(e)}")
        
        if not self.data:
            raise ValueError("No data could be fetched for any timeframe")
        
    def fetch_data_all(self):
        logger.info(f"Fetching data for {self.symbol} for all timeframes")

        for tf in self.timeframes:
            try:
                start_date, end_date = self.period_item[tf]
                logger.info(f"Processing timeframe: {tf} from {start_date} to {end_date}")
                df = self.gs.get_stock_data(start_date, end_date, tf, self.data_dir)
                if df is not None and not df.empty:
                    self.data[tf] = df
                    logger.info(f"Fetched data shape for {tf}: {df.shape}")
                else:
                    logger.warning(f"No data fetched for timeframe {tf}")
            except Exception as e:
                logger.error(f"Error fetching data for timeframe {tf}: {str(e)}")
            
        if not self.data:
            raise ValueError("No data could be fetched for any timeframe")

    def validate_csv_data(self, df: pd.DataFrame, timeframe: str) -> bool:
        if df.empty:
            logger.error(f"DataFrame for {timeframe} is empty")
            return False

        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')

        if timeframe in ['5m', '15m', '60m']:
            # Check for continuous data within trading hours
            df['time'] = df['datetime'].dt.time
            trading_hours = (pd.to_datetime('09:30:00').time(), pd.to_datetime('15:00:00').time())
            df_trading = df[(df['time'] >= trading_hours[0]) & (df['time'] <= trading_hours[1])]
            expected_intervals = {'5m': 5, '15m': 15, '60m': 60}
            time_diff = df_trading['datetime'].diff().dt.total_seconds() / 60
            if not np.allclose(time_diff.dropna(), expected_intervals[timeframe], atol=1):
                logger.error(f"Inconsistent time intervals detected for {timeframe}")
                return False
        elif timeframe == '1d':
            # Check for continuous daily data
            date_diff = df['datetime'].dt.date.diff().dt.days
            if not np.allclose(date_diff.dropna(), 1, atol=1):
                logger.error(f"Missing daily data detected for {timeframe}")
                return False
        elif timeframe == '1w':
            # Check for continuous weekly data
            week_diff = df['datetime'].dt.to_period('W').diff()
            if not np.allclose(week_diff.dropna(), 1, atol=1):
                logger.error(f"Missing weekly data detected for {timeframe}")
                return False
        elif timeframe == '1m':
            # Check for continuous monthly data
            month_diff = df['datetime'].dt.to_period('M').diff()
            if not np.allclose(month_diff.dropna(), 1, atol=1):
                logger.error(f"Missing monthly data detected for {timeframe}")
                return False
        elif timeframe == '1q':
            # Check for continuous quarterly data
            quarter_diff = df['datetime'].dt.to_period('Q').diff()
            if not np.allclose(quarter_diff.dropna(), 1, atol=1):
                logger.error(f"Missing quarterly data detected for {timeframe}")
                return False

        logger.info(f"Data validation successful for {timeframe}")
        return True

    def verify_csv_files(self):
        logger.info("Verifying CSV files for all timeframes")
        for tf in self.timeframes:
            start_date, end_date = self.period_item[tf]
            filename = f'{self.symbol}_{tf}_{start_date}_{end_date}.csv'
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"CSV file for {tf} exists at {file_path}")
                logger.info(f"File size: {file_size} bytes")
                if file_size == 0:
                    logger.error(f"CSV file for {tf} is empty")
                else:
                    # Read and validate the CSV data
                    df = pd.read_csv(file_path)
                    if self.validate_csv_data(df, tf):
                        logger.info(f"CSV data for {tf} is valid")
                    else:
                        logger.error(f"CSV data for {tf} is invalid")
            else:
                logger.error(f"CSV file for {tf} is missing at {file_path}")

    def validate_data(self):
        logger.info("Performing cross-timeframe data validation")
        
        # Load all data if not already loaded
        if not self.data:
            self.fetch_data_all()
        
        # Sort timeframes from smallest to largest
        timeframes = sorted(self.timeframes, key=lambda x: self._timeframe_to_minutes(x))
        
        for i in range(len(timeframes) - 1):
            lower_tf = timeframes[i]
            higher_tf = timeframes[i + 1]
            
            lower_df = self.data[lower_tf]
            higher_df = self.data[higher_tf]
            
            # Ensure datetime columns are datetime type
            lower_df['datetime'] = pd.to_datetime(lower_df['datetime'])
            higher_df['datetime'] = pd.to_datetime(higher_df['datetime'])
            
            # Check alignment of data
            if not self._check_data_alignment(lower_df, higher_df, lower_tf, higher_tf):
                logger.error(f"Data misalignment detected between {lower_tf} and {higher_tf}")
                return False
            
            # Check consistency of open and close prices
            if not self._check_price_consistency(lower_df, higher_df, lower_tf, higher_tf):
                logger.error(f"Price inconsistency detected between {lower_tf} and {higher_tf}")
                return False
            
            # Check consistency of volume data
            if not self._check_volume_consistency(lower_df, higher_df, lower_tf, higher_tf):
                logger.error(f"Volume inconsistency detected between {lower_tf} and {higher_tf}")
                return False
        
        logger.info("Cross-timeframe data validation completed successfully")
        return True

    def _timeframe_to_minutes(self, timeframe):
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe == '1d':
            return 1440  # minutes in a day
        elif timeframe == '1w':
            return 10080  # minutes in a week
        elif timeframe == '1m':
            return 43200  # approximate minutes in a month
        elif timeframe == '1q':
            return 129600  # approximate minutes in a quarter
        else:
            raise ValueError(f"Unknown timeframe: {timeframe}")

    def _check_data_alignment(self, lower_df, higher_df, lower_tf, higher_tf):
        # Check if the start and end dates align
        lower_start = lower_df['datetime'].min()
        lower_end = lower_df['datetime'].max()
        higher_start = higher_df['datetime'].min()
        higher_end = higher_df['datetime'].max()
        
        if lower_start != higher_start or lower_end != higher_end:
            logger.warning(f"Date range mismatch between {lower_tf} and {higher_tf}")
            return False
        
        return True

    def _check_price_consistency(self, lower_df, higher_df, lower_tf, higher_tf):
        # Group lower timeframe data to match higher timeframe
        grouped = lower_df.groupby(pd.Grouper(key='datetime', freq=higher_tf))
        
        for date, group in grouped:
            higher_row = higher_df[higher_df['datetime'] == date]
            if higher_row.empty:
                continue
            
            if group['open'].iloc[0] != higher_row['open'].iloc[0]:
                logger.warning(f"Open price mismatch at {date} between {lower_tf} and {higher_tf}")
                return False
            
            if group['close'].iloc[-1] != higher_row['close'].iloc[0]:
                logger.warning(f"Close price mismatch at {date} between {lower_tf} and {higher_tf}")
                return False
        
        return True

    def _check_volume_consistency(self, lower_df, higher_df, lower_tf, higher_tf):
        # Group lower timeframe data to match higher timeframe
        grouped = lower_df.groupby(pd.Grouper(key='datetime', freq=higher_tf))
        
        for date, group in grouped:
            higher_row = higher_df[higher_df['datetime'] == date]
            if higher_row.empty:
                continue
            
            lower_volume = group['volume'].sum()
            higher_volume = higher_row['volume'].iloc[0]
            
            if not np.isclose(lower_volume, higher_volume, rtol=1e-5):
                logger.warning(f"Volume mismatch at {date} between {lower_tf} and {higher_tf}")
                return False
        
        return True

    def simulate_fetch(self):
        logger.info(f"Simulating data fetch for {self.symbol}")
        self.fetch_data_all()
        logger.info("Data fetch simulation completed")
        self.verify_csv_files()
        if self.validate_data():
            logger.info("Data validation successful")
        else:
            logger.error("Data validation failed")

if __name__ == "__main__":
    # This section will only run if the script is executed directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    symbol = "sz000001"
    end_date_count = ['2024-10-18', '15000']  # end_date and count of days
    timeframes = ["5m", "15m", "60m", "1d", "1w", "1m", "1q"]
    data_dir = "Data/csv_files"  # Update this to the actual path of your CSV files
    data_acq = DataAcquisition(symbol, end_date_count, timeframes, data_dir)
    data_acq.simulate_fetch()
