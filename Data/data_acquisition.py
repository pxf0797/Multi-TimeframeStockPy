import pandas as pd
import numpy as np
from typing import List, Dict
import os
import logging
from datetime import datetime, timedelta
import glob
from GetStock import GetStock
from data_integrity_check import check_intraday_data_integrity, check_period_data_integrity

def setup_logging(log_file):
    """Set up logging to file and console"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

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
        # calculate months count
        if count > 2000:
            start_data_month = (end_date - timedelta(days=30*2000)).strftime('%Y-%m-%d')
        else:
            start_data_month = (end_date - timedelta(days=30*count)).strftime('%Y-%m-%d')
        # calculate quarters count
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
            print(f"Created data directory: {self.data_dir}")

    def fetch_data(self, start_date: str, end_date: str):
        print(f"Fetching data for {self.symbol} from {start_date} to {end_date}")

        for tf in self.timeframes:
            try:
                print(f"Processing timeframe: {tf}")
                df = self.gs.get_stock_data(start_date, end_date, tf, self.data_dir)
                if df is not None and not df.empty:
                    self.data[tf] = df
                    print(f"Fetched data shape for {tf}: {df.shape}")
                else:
                    print(f"No data fetched for timeframe {tf}")
            except Exception as e:
                print(f"Error fetching data for timeframe {tf}: {str(e)}")
        
        if not self.data:
            raise ValueError("No data could be fetched for any timeframe")
        
    def fetch_data_all(self):
        print(f"Fetching data for {self.symbol} for all timeframes")

        for tf in self.timeframes:
            try:
                start_date, end_date = self.period_item[tf]
                print(f"Processing timeframe: {tf} from {start_date} to {end_date}")
                df = self.gs.get_stock_data(start_date, end_date, tf, self.data_dir)
                if df is not None and not df.empty:
                    self.data[tf] = df
                    print(f"Fetched data shape for {tf}: {df.shape}")
                else:
                    print(f"No data fetched for timeframe {tf}")
            except Exception as e:
                print(f"Error fetching data for timeframe {tf}: {str(e)}")
            
        if not self.data:
            raise ValueError("No data could be fetched for any timeframe")

    def validate_csv_data(self, file_path: str, timeframe: str) -> bool:
        print(f"Start for {timeframe} check.")
        
        timeframe_types = {
            '5m': '5min',
            '15m': '15min',
            '60m': '60min',
            '1d': 'daily',
            '1w': 'weekly', 
            '1m': 'monthly',
            '1q': 'quarterly'
        }
        
        # Use an absolute path for the holiday file
        holiday_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chinese_holidays.csv'))
        print(f"Holiday file path: {holiday_file}")
        
        if timeframe in ['5m', '15m', '60m']:
            check_intraday_data_integrity(file_path=file_path, holiday_file=holiday_file, period=timeframe_types[timeframe], logger=logging.getLogger())
        else:
            check_period_data_integrity(file_path=file_path, holiday_file=holiday_file, period=timeframe_types[timeframe], logger=logging.getLogger())
        
        print(f"Data validation successful for {timeframe}")
        return True

    def validate_data(self):
        logging.info("Validating data for all timeframes")
        all_valid = True
        for tf in self.timeframes:
            file_pattern = os.path.join(self.data_dir, f'{self.symbol}_{tf}*.csv')
            matching_files = glob.glob(file_pattern)
            if matching_files:
                file_path = matching_files[0]  # Use the first matching file
                try:
                    if self.validate_csv_data(file_path, tf):
                        logging.info(f"Data validation successful for {tf}")
                    else:
                        logging.error(f"Data validation failed for {tf}")
                        all_valid = False
                except Exception as e:
                    logging.error(f"Error during data validation for {tf}: {str(e)}")
                    all_valid = False
            else:
                logging.error(f"No CSV file found for {tf} matching pattern: {file_pattern}")
                all_valid = False
        return all_valid
            
    def verify_csv_files(self):
        print("Verifying CSV files for all timeframes")
        for tf in self.timeframes:
            file_pattern = os.path.join(self.data_dir, f'{self.symbol}_{tf}*.csv')
            matching_files = glob.glob(file_pattern)
            if matching_files:
                file_path = matching_files[0]  # Use the first matching file
                file_size = os.path.getsize(file_path)
                print(f"CSV file for {tf} exists at {file_path}")
                print(f"File size: {file_size} bytes")
                if file_size == 0:
                    print(f"CSV file for {tf} is empty")
                else:
                    # Read and validate the CSV data
                    if self.validate_csv_data(file_path, tf):
                        print(f"CSV data for {tf} is valid")
                    else:
                        print(f"CSV data for {tf} is invalid")
            else:
                print(f"No CSV file found for {tf} matching pattern: {file_pattern}")
                
    def simulate_fetch(self):
        logging.info(f"Simulating data fetch for {self.symbol}")
        self.fetch_data_all()
        logging.info("Data fetch simulation completed")
        self.verify_csv_files()
        if self.validate_data():
            logging.info("Data validation successful")
        else:
            logging.error("Data validation failed")

if __name__ == "__main__":
    # This section will only run if the script is executed directly
    symbol = "sz000001"
    end_date_count = ['2024-10-18', '15000']  # end_date and count of days
    timeframes = ["5m", "15m", "60m", "1d", "1w", "1m", "1q"]
    data_dir = "Data/csv_files"  # Update this to the actual path of your CSV files
    
    # Set up logging
    log_file = f"data_acquisition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_file)
    
    data_acq = DataAcquisition(symbol, end_date_count, timeframes, data_dir)
    data_acq.simulate_fetch()