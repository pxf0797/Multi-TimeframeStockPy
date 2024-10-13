import pandas as pd
from typing import List, Dict
import os

class DataAcquisition:
    def __init__(self, symbol: str, timeframes: List[str], data_dir: str):
        self.symbol = symbol
        self.timeframes = timeframes
        self.data_dir = data_dir
        self.data: Dict[str, pd.DataFrame] = {}

    def fetch_data(self, start_date: str, end_date: str):
        for timeframe in self.timeframes:
            try:
                file_path = os.path.join(self.data_dir, f"{self.symbol}_{timeframe}.csv")
                if not os.path.exists(file_path):
                    print(f"CSV file not found for {self.symbol} with timeframe {timeframe}")
                    continue
                
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                df = df.loc[start_date:end_date]
                
                if df.empty:
                    print(f"No data available for {self.symbol} with timeframe {timeframe} from {start_date} to {end_date}")
                else:
                    self.data[timeframe] = df
                    print(f"Successfully fetched data for {self.symbol} with timeframe {timeframe}")
            except Exception as e:
                print(f"Error fetching data for {self.symbol} with timeframe {timeframe}: {str(e)}")

    def simulate_fetch(self, start_date: str, end_date: str):
        print(f"Simulating data fetch for {self.symbol} from {start_date} to {end_date}")
        self.fetch_data(start_date, end_date)
        print("Data fetch simulation completed")

    def get_data(self, timeframe: str) -> pd.DataFrame:
        return self.data.get(timeframe, pd.DataFrame())

    def align_data(self):
        if not self.data:
            print("No data to align. Please fetch data first.")
            return

        # Filter out timeframes with no data
        available_timeframes = [tf for tf in self.timeframes if tf in self.data]

        if not available_timeframes:
            print("No data available for any timeframe.")
            return

        # Align data for all available timeframes to the longest timeframe
        longest_tf = max(available_timeframes, key=lambda x: len(self.data[x]))
        for tf in available_timeframes:
            if tf != longest_tf:
                self.data[tf] = self.data[tf].reindex(self.data[longest_tf].index, method='ffill')

        print("Data aligned successfully")

if __name__ == "__main__":
    # Example usage
    symbol = "AAPL"
    timeframes = ["1d", "1wk", "1mo"]
    data_dir = "path/to/csv/files"  # Update this to the actual path of your CSV files
    data_acq = DataAcquisition(symbol, timeframes, data_dir)
    data_acq.simulate_fetch("2023-01-01", "2023-12-31")
    data_acq.align_data()
    for tf in timeframes:
        print(f"\nData for {tf}:")
        print(data_acq.get_data(tf).head())
