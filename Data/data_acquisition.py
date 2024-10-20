import pandas as pd
import numpy as np
from typing import List, Dict
import os
import logging
from datetime import datetime, timedelta
from GetStock import GetStock
from data_integrity_check import *

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

    def validate_csv_data(self, file_name: str, timeframe: str) -> bool:
        logger.info(f"Start for {timeframe} check.")

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
        if timeframe == '5m':
            check_5min_data_integrity(file_name,'chinese_holidays.csv')
        elif timeframe == '15m':
            check_15min_data_integrity(file_name,'chinese_holidays.csv')
        elif timeframe == '60m':
            check_60min_data_integrity(file_name,'chinese_holidays.csv')
        elif timeframe == '1d':
            # Check for continuous daily data
            check_daily_data_integrity(file_name,'chinese_holidays.csv')
        elif timeframe == '1w':
            # Check for continuous weekly data
            check_weekly_data_integrity(file_name,'chinese_holidays.csv')
        elif timeframe == '1m':
            # Check for continuous monthly data
            check_60min_data_integrity(file_name,'chinese_holidays.csv')
        elif timeframe == '1q':
            # Check for continuous quarterly data
            check_quarterly_data_integrity(file_name,'chinese_holidays.csv')

        logger.info(f"Data validation successful for {timeframe}")
        return True

    def validate_data(self):
        logger.info("Validating data for all timeframes")


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
