import pandas as pd
import numpy as np
from typing import List, Dict
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def timeframe_to_minutes(tf: str) -> int:
    if tf.endswith('m'):
        return int(tf[:-1])
    elif tf.endswith('h'):
        return int(tf[:-1]) * 60
    elif tf.endswith('d'):
        return int(tf[:-1]) * 1440  # 24 * 60
    elif tf.endswith('M'):
        return int(tf[:-1]) * 43200  # 30 * 24 * 60
    else:
        raise ValueError(f"Unsupported timeframe: {tf}")

class DataAcquisition:
    def __init__(self, symbol: str, timeframes: List[str], data_dir: str):
        self.symbol = symbol
        self.timeframes = timeframes
        self.data_dir = data_dir
        self.data: Dict[str, pd.DataFrame] = {}

    def fetch_data(self, start_date: str, end_date: str):
        logger.info(f"Fetching data for {self.symbol} from {start_date} to {end_date}")
        for timeframe in self.timeframes:
            try:
                file_path = os.path.join(self.data_dir, f"{self.symbol}_{timeframe}.csv")
                if not os.path.exists(file_path):
                    logger.warning(f"CSV file not found for {self.symbol} with timeframe {timeframe}")
                    df = self.generate_synthetic_data(timeframe, start_date, end_date)
                else:
                    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                    df = df.loc[start_date:end_date]
                
                if df.empty:
                    logger.warning(f"No data available for {self.symbol} with timeframe {timeframe} from {start_date} to {end_date}")
                    df = self.generate_synthetic_data(timeframe, start_date, end_date)
                else:
                    # Ensure all columns except 'Date' are numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Remove rows with NaN values
                    df = df.dropna()
                    
                    if len(df) < 100:  # If we have less than 100 data points, generate synthetic data
                        logger.warning(f"Insufficient data for {self.symbol} with timeframe {timeframe}. Generating synthetic data.")
                        df = self.generate_synthetic_data(timeframe, start_date, end_date, df)
                
                self.data[timeframe] = df
                logger.info(f"Successfully fetched/generated data for {self.symbol} with timeframe {timeframe}. Shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error fetching data for {self.symbol} with timeframe {timeframe}: {str(e)}")

        if not self.data:
            raise ValueError("No data could be fetched for any timeframe")

    def generate_synthetic_data(self, timeframe: str, start_date: str, end_date: str, seed_data: pd.DataFrame = None):
        logger.info(f"Generating synthetic data for {timeframe} from {start_date} to {end_date}")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if seed_data is None or len(seed_data) < 2:
            # If we don't have seed data, create some reasonable starting values
            last_close = 16000  # Starting price
            daily_volatility = 0.02  # 2% daily volatility
        else:
            last_close = seed_data['Close'].iloc[-1]
            daily_volatility = seed_data['Close'].pct_change().std()

        # Generate daily data
        days = pd.date_range(start=start, end=end, freq='D')
        daily_returns = np.random.normal(0, daily_volatility, size=len(days))
        closes = last_close * (1 + daily_returns).cumprod()
        
        df = pd.DataFrame({
            'Open': closes,
            'High': closes * (1 + abs(np.random.normal(0, daily_volatility/2, size=len(days)))),
            'Low': closes * (1 - abs(np.random.normal(0, daily_volatility/2, size=len(days)))),
            'Close': closes,
            'Volume': np.random.randint(100000, 2000000, size=len(days))
        }, index=days)
        
        # Resample to the desired timeframe
        if timeframe != '1d':
            df = df.resample(timeframe).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        return df

    def simulate_fetch(self, start_date: str, end_date: str):
        logger.info(f"Simulating data fetch for {self.symbol} from {start_date} to {end_date}")
        self.fetch_data(start_date, end_date)
        logger.info("Data fetch simulation completed")

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

        shortest_tf = min(available_timeframes, key=timeframe_to_minutes)
        shortest_df = self.data[shortest_tf]

        for tf in available_timeframes:
            if tf != shortest_tf:
                self.data[tf] = self.data[tf].reindex(shortest_df.index, method='ffill')
                # Ensure all columns are float type
                self.data[tf] = self.data[tf].astype(float)

        logger.info("Data aligned successfully")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    symbol = "BTC-USD"
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
