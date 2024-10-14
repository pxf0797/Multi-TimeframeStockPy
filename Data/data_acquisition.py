import pandas as pd
import numpy as np
from typing import List, Dict
import os
import logging
from datetime import datetime, timedelta
import time
from Data.Ashare import get_price

logger = logging.getLogger(__name__)

class DataAcquisition:
    def __init__(self, symbol: str, timeframes: List[str], data_dir: str):
        self.symbol = symbol
        self.timeframes = sorted(timeframes)
        self.data_dir = data_dir
        self.data: Dict[str, pd.DataFrame] = {}

    def fetch_data(self, start_date: str, end_date: str):
        logger.info(f"Fetching data for {self.symbol} from {start_date} to {end_date}")
        for timeframe in self.timeframes:
            try:
                file_path = os.path.join(self.data_dir, f"{self.symbol}_{timeframe}.csv")
                logger.info(f"Attempting to read file: {file_path}")
                if not os.path.exists(file_path):
                    logger.warning(f"CSV file not found for {self.symbol} with timeframe {timeframe}")
                    logger.info(f"Fetching data from API for {timeframe}")
                    df = get_price(self.symbol, frequency=timeframe, count=1000)
                else:
                    df = pd.read_csv(file_path)
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    else:
                        logger.warning(f"'Date' column not found in {file_path}. Using default index.")
                    df = df.loc[start_date:end_date]
                
                if df.empty:
                    logger.warning(f"No data available for {self.symbol} with timeframe {timeframe} from {start_date} to {end_date}")
                    logger.info(f"Fetching data from API for {timeframe}")
                    df = get_price(self.symbol, frequency=timeframe, count=1000)
                else:
                    # Ensure all columns except 'Date' are numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Remove rows with NaN values
                    df = df.dropna()
                    
                    if len(df) < 100:  # If we have less than 100 data points, fetch more data
                        logger.warning(f"Insufficient data for {self.symbol} with timeframe {timeframe}. Fetching more data.")
                        df = get_price(self.symbol, frequency=timeframe, count=1000)
                
                # Check if the DataFrame has the required columns
                required_columns = ['open', 'close', 'high', 'low', 'volume']
                if not all(col in df.columns for col in required_columns):
                    logger.error(f"Required columns {required_columns} not found in the DataFrame for {timeframe}. Fetching data again.")
                    df = get_price(self.symbol, frequency=timeframe, count=1000)
                    if not all(col in df.columns for col in required_columns):
                        logger.error(f"Failed to fetch data with required columns for {timeframe}. Skipping this timeframe.")
                        continue
                
                # Ensure all column names are lowercase
                df.columns = df.columns.str.lower()
                
                self.data[timeframe] = df
                logger.info(f"Successfully fetched/generated data for {self.symbol} with timeframe {timeframe}. Shape: {df.shape}")
                
                # Save the data to CSV
                self.save_to_csv(timeframe, df)
            except Exception as e:
                logger.error(f"Error fetching data for {self.symbol} with timeframe {timeframe}: {str(e)}")

        if not self.data:
            raise ValueError("No data could be fetched for any timeframe")

        # Synthesize data for larger timeframes if needed
        self.synthesize_larger_timeframes()

        # Verify all CSV files exist
        self.verify_csv_files()

    def synthesize_larger_timeframes(self):
        logger.info("Starting synthesis of larger timeframes")
        for tf in self.timeframes:
            if tf not in self.data or self.data[tf].empty:
                # Find the nearest smaller timeframe with data
                for smaller_tf in self.timeframes:
                    if smaller_tf in self.data and not self.data[smaller_tf].empty:
                        logger.info(f"Synthesizing data for {tf} from {smaller_tf}")
                        self.data[tf] = self.data[smaller_tf].resample(tf).agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                        logger.info(f"Synthesized data for {tf}. Shape: {self.data[tf].shape}")
                        # Save the synthesized data to CSV
                        self.save_to_csv(tf, self.data[tf])
                        break
                else:
                    logger.warning(f"Unable to synthesize data for {tf}. No smaller timeframe data available.")
        logger.info("Finished synthesis of larger timeframes")

    def save_to_csv(self, timeframe: str, df: pd.DataFrame, max_retries=3):
        file_path = os.path.join(self.data_dir, f"{self.symbol}_{timeframe}.csv")
        logger.info(f"Attempting to save {timeframe} data to {file_path}")
        for attempt in range(max_retries):
            try:
                df.to_csv(file_path)
                logger.info(f"Saved data for {self.symbol} with timeframe {timeframe} to {file_path}")
                
                # Additional check for all timeframes
                if os.path.exists(file_path):
                    logger.info(f"{timeframe} CSV file successfully created at {file_path}")
                    logger.info(f"{timeframe} CSV file size: {os.path.getsize(file_path)} bytes")
                    
                    # Read the first few lines of the saved file to verify its content
                    with open(file_path, 'r') as f:
                        first_lines = ''.join(f.readlines()[:5])
                    logger.info(f"First few lines of {timeframe} CSV file:\n{first_lines}")
                    return  # Successfully saved, exit the function
                else:
                    logger.error(f"Failed to create {timeframe} CSV file at {file_path}")
            except Exception as e:
                logger.error(f"Error saving {timeframe} data to CSV (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait for 1 second before retrying
                else:
                    logger.error(f"Failed to save {timeframe} data after {max_retries} attempts")

    def verify_csv_files(self):
        logger.info("Verifying CSV files for all timeframes")
        for tf in self.timeframes:
            file_path = os.path.join(self.data_dir, f"{self.symbol}_{tf}.csv")
            if os.path.exists(file_path):
                logger.info(f"CSV file for {tf} exists at {file_path}")
                logger.info(f"File size: {os.path.getsize(file_path)} bytes")
            else:
                logger.error(f"CSV file for {tf} is missing at {file_path}")

    def simulate_fetch(self, start_date: str, end_date: str):
        logger.info(f"Simulating data fetch for {self.symbol} from {start_date} to {end_date}")
        self.fetch_data(start_date, end_date)
        logger.info("Data fetch simulation completed")
        self.verify_csv_files()  # Add a final verification step

    def get_data(self, timeframe: str) -> pd.DataFrame:
        return self.data.get(timeframe, pd.DataFrame())

    def align_data(self):
        if not self.data:
            logger.warning("No data to align. Please fetch data first.")
            return

        available_timeframes = list(self.data.keys())
        
        if not available_timeframes:
            logger.warning("No data available for alignment.")
            return

        shortest_tf = min(available_timeframes)
        shortest_df = self.data[shortest_tf]

        for tf in self.timeframes:
            if tf not in self.data or self.data[tf].empty:
                logger.warning(f"No data available for timeframe {tf}. Skipping alignment for this timeframe.")
            elif tf != shortest_tf:
                self.data[tf] = self.data[tf].reindex(shortest_df.index, method='ffill')
                # Ensure all columns are float type
                self.data[tf] = self.data[tf].astype(float)
                # Save the aligned data to CSV
                self.save_to_csv(tf, self.data[tf])

        logger.info("Data aligned successfully")
        self.verify_csv_files()  # Add a final verification step after alignment

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    symbol = "sh000001"
    timeframes = ["1m", "5m", "15m", "1h", "1d", "1M"]
    data_dir = "Data/csv_files"  # Update this to the actual path of your CSV files
    data_acq = DataAcquisition(symbol, timeframes, data_dir)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # Fetch one year of data
    data_acq.simulate_fetch(start_date, end_date)
    data_acq.align_data()
    for tf in timeframes:
        logger.info(f"\nData for {tf}:")
        logger.info(data_acq.get_data(tf).head())
        logger.info(f"CSV file saved: {os.path.exists(os.path.join(data_dir, f'{symbol}_{tf}.csv'))}")
    
    # Final verification
    data_acq.verify_csv_files()
