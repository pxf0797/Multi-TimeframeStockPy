import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_and_save_data(symbol, timeframes, start_date, end_date):
    for tf in timeframes:
        print(f"Downloading data for timeframe: {tf}")
        try:
            df = yf.download(symbol, start=start_date, end=end_date, interval=tf)
            if df.empty:
                print(f"Warning: Empty DataFrame for timeframe {tf}")
            else:
                print(f"Downloaded {len(df)} rows for timeframe {tf}")
                # Ensure index is timezone-aware UTC
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC', nonexistent='shift_forward')
                else:
                    df.index = df.index.tz_convert('UTC')
                
                # Save to CSV
                filename = f"{symbol}_{tf}.csv"
                df.to_csv(os.path.join('Data', filename))
                print(f"Saved data to {filename}")
        except Exception as e:
            print(f"Error downloading data for timeframe {tf}: {e}")

if __name__ == "__main__":
    symbol = "BTC-USD"
    timeframes = ['1m', '5m', '15m', '1h', '1d']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Adjust as needed

    download_and_save_data(symbol, timeframes, start_date, end_date)
