import yfinance as yf
import pandas as pd
from typing import List, Dict

class DataAcquisition:
    def __init__(self, symbol: str, timeframes: List[str]):
        self.symbol = symbol
        self.timeframes = timeframes
        self.data: Dict[str, pd.DataFrame] = {}

    def fetch_data(self, start_date: str, end_date: str):
        for timeframe in self.timeframes:
            try:
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(start=start_date, end=end_date, interval=timeframe)
                if df.empty:
                    print(f"No data available for {self.symbol} with timeframe {timeframe} from {start_date} to {end_date}")
                else:
                    df.index = pd.to_datetime(df.index)
                    self.data[timeframe] = df
                    print(f"Successfully fetched data for {self.symbol} with timeframe {timeframe}")
            except Exception as e:
                print(f"Error fetching data for {self.symbol} with timeframe {timeframe}: {str(e)}")

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
    data_acq = DataAcquisition(symbol, timeframes)
    data_acq.fetch_data("2023-01-01", "2023-12-31")
    data_acq.align_data()
    for tf in timeframes:
        print(f"\nData for {tf}:")
        print(data_acq.get_data(tf).head())
