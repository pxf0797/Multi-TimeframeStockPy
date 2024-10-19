import pandas as pd
import numpy as np
from typing import List, Dict
import os
import logging
from datetime import datetime, timedelta
from GetStock import GetStock

logger = logging.getLogger(__name__)

class DataAcquisition:
    def __init__(self, symbol: str, timeframes: List[str], data_dir: str):
        self.symbol = symbol
        self.timeframes = sorted(timeframes)
        self.data_dir = data_dir
        self.data: Dict[str, pd.DataFrame] = {}
        self.gs = GetStock()
        self.gs.set_stock_name(self.symbol)
        
        # Check if the data directory exists, if not, create it
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")

    def fetch_data(self, start_date: str, end_date: str, max_retries=3):
        logger.info(f"Fetching data for {self.symbol} from {start_date} to {end_date}")

        for tf in self.timeframes:
            logger.info(f"Processing timeframe: {tf}")
            df = self.gs.get_stock_data(start_date, end_date, tf, self.data_dir)
            logger.info(f"Fetched data shape for {tf}: {df.shape}")
            
        if not self.data:
            raise ValueError("No data could be fetched for any timeframe")

    def verify_csv_files(self, start_date: str, end_date: str):
        logger.info("Verifying CSV files for all timeframes")
        for tf in self.timeframes:
            filename = f'{self.symbol}_{tf}_{start_date}_{end_date}.csv'
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"CSV file for {tf} exists at {file_path}")
                logger.info(f"File size: {file_size} bytes")
                if file_size == 0:
                    logger.error(f"CSV file for {tf} is empty")
                else:
                    # Read and log the first few lines of the file
                    with open(file_path, 'r') as f:
                        first_lines = ''.join(f.readlines()[:5])
                    logger.info(f"First few lines of {tf} CSV file:\n{first_lines}")
            else:
                logger.error(f"CSV file for {tf} is missing at {file_path}")

    def simulate_fetch(self, start_date: str, end_date: str):
        logger.info(f"Simulating data fetch for {self.symbol} from {start_date} to {end_date}")
        self.fetch_data(start_date, end_date)
        logger.info("Data fetch simulation completed")
        self.verify_csv_files(start_date, end_date)  # Pass start_date and end_date

if __name__ == "__main__":
    # This section will only run if the script is executed directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    symbol = "sz000001"
    timeframes = ["5m", "15m", "60m", "1d", "1m", "1q"]
    data_dir = "Data/csv_files"  # Update this to the actual path of your CSV files
    data_acq = DataAcquisition(symbol, timeframes, data_dir)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # Fetch one year of data
    data_acq.simulate_fetch(start_date, end_date)
