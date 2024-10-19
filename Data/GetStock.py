#!/usr/bin/env python3
# coding: utf-8
# GetStock.py

from Ashare import *
import csvfile
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GetStock:
    def __init__(self):
        self.__stock_name = 'sh000001'
        self.__header = 'day,open,high,low,close,volume'
        self.__csvfile = csvfile.csvfile(['pyScript', '/', 'test_data.csv'])
        self.__df = pd.DataFrame()

    def set_stock_name(self, name):
        self.__stock_name = name

    def show_df(self):
        print(self.__df)

    def get_stock(self, frequency='1d', count=1000, end_date=None):
        """
        Get stock data for the specified frequency and count.
        
        :param frequency: '5m','15m','30m','60m','1d','1w','1m','1q'
        :param count: Number of data points to retrieve
        :param end_date: End date for the data (optional)
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f"Attempting to fetch {frequency} data for {self.__stock_name}")
            
            if frequency == '1q':
                # For quarterly data, fetch monthly data and resample
                self.__df = get_price(self.__stock_name, frequency='1m', count=count*3, end_date=end_date)
                if not self.__df.empty:
                    self.__df = self.__df.resample('QE').last()
                    logger.info(f"Successfully resampled monthly data to quarterly for {self.__stock_name}")
            else:
                self.__df = get_price(self.__stock_name, frequency=frequency, count=count, end_date=end_date)

            if self.__df.empty:
                logger.warning(f"No data retrieved for {self.__stock_name} with frequency {frequency}")
            else:
                logger.info(f"Successfully retrieved {len(self.__df)} rows of {frequency} data for {self.__stock_name}")

        except Exception as e:
            logger.error(f"Error retrieving data: {str(e)}")

    def filter_date_range(self, start_date):
        """
        Filter the dataframe to include only data from the start_date onwards.
        """
        self.__df = self.__df[self.__df.index >= start_date]

    def save_stock_csv(self, frequency, start_date, end_date):
        if self.__df.empty:
            logger.warning("No data to save. Please fetch data first.")
            return

        filename = f'{self.__stock_name}_{frequency}_{start_date}_{end_date}.csv'
        self.__df.to_csv(filename, index=True)
        logger.info(f'Data saved to {filename}')

    def get_stock_data(self, start_date, end_date, period):
        """
        Interface for retrieving stock data for different time periods.
        
        :param start_date: Start date for the data (YYYY-MM-DD)
        :param end_date: End date for the data (YYYY-MM-DD)
        :param period: '5m','15m','30m','60m','1d','1w','1m','1q'
        :return: DataFrame with the requested stock data
        """
        try:
            # Convert dates to datetime objects
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

            # Calculate the number of days between start and end dates
            days_diff = (end_date - start_date).days

            # Calculate count based on period and date range
            if period in ['5m', '15m', '30m', '60m']:
                # Assuming market is open 4 hours a day (simplified)
                count = (days_diff * 4 * 60) // int(period[:-1])
            elif period == '1d':
                count = days_diff
            elif period == '1w':
                count = days_diff // 7
            elif period == '1m':
                count = days_diff // 30
            elif period == '1q':
                count = (days_diff // 90) + 1
            else:
                raise ValueError(f"Invalid period: {period}")

            # Fetch data
            self.get_stock(frequency=period, count=count, end_date=end_date.strftime('%Y-%m-%d'))

            # Filter date range
            self.filter_date_range(start_date)

            # Save to CSV
            self.save_stock_csv(frequency=period, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))

            return self.__df

        except Exception as e:
            logger.error(f"Error in get_stock_data: {str(e)}")
            return pd.DataFrame()

if __name__ == '__main__':
    gs = GetStock()
    gs.set_stock_name('sh000001')

    # Test the new interface
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Test 5-minute data (past week)
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    df_5m = gs.get_stock_data(start_date, end_date, '5m')
    print("5-minute data shape:", df_5m.shape)

    # Test 60-minute data (past month)
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    df_60m = gs.get_stock_data(start_date, end_date, '60m')
    print("60-minute data shape:", df_60m.shape)

    # Test daily data (past year)
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    df_1d = gs.get_stock_data(start_date, end_date, '1d')
    print("Daily data shape:", df_1d.shape)

    # Test monthly data (past 5 years)
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    df_1m = gs.get_stock_data(start_date, end_date, '1m')
    print("Monthly data shape:", df_1m.shape)

    # Test quarterly data (past 10 years)
    start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    df_1q = gs.get_stock_data(start_date, end_date, '1q')
    print("Quarterly data shape:", df_1q.shape)
