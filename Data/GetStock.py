#!/usr/bin/env python3
# coding: utf-8
# GetStock.py

"""
This module provides a class for fetching and managing stock data.
It uses the Ashare module to retrieve stock data and offers various
methods for data manipulation and storage.

Usage:
1. Create an instance of GetStock
2. Set the stock name using set_stock_name method
3. Use get_stock_data method to fetch and process stock data

Example:
    gs = GetStock()
    gs.set_stock_name('sh000001')
    df = gs.get_stock_data('2023-01-01', '2023-12-31', '1d')
"""

from Data.Ashare import *
import os
from Data import csvfile
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GetStock:
    def __init__(self):
        """
        Initialize GetStock instance with default values.
        """
        self.__stock_name = 'sh000001'
        self.__header = 'day,open,high,low,close,volume'
        self.__csvfile = csvfile.csvfile(['pyScript', '/', 'test_data.csv'])
        self.__df = pd.DataFrame()

    def set_stock_name(self, name):
        """
        Set the stock name for data retrieval.
        
        :param name: Stock code (e.g., 'sh000001' for Shanghai Composite Index)
        """
        self.__stock_name = name

    def show_df(self):
        """
        Print the current DataFrame containing stock data.
        """
        print(self.__df)

    def get_stock(self, frequency='1d', count=1000, end_date=None):
        """
        Get stock data for the specified frequency and count.
        
        :param frequency: Data frequency ('5m','15m','30m','60m','1d','1w','1m','1q')
        :param count: Number of data points to retrieve
        :param end_date: End date for the data (optional)
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f"Attempting to fetch {frequency} data for {self.__stock_name}")
            
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
        
        :param start_date: Start date for filtering (datetime object)
        """
        self.__df = self.__df[self.__df.index >= start_date]

    def save_stock_csv(self, frequency, start_date, end_date,data_dir=''):
        """
        Save the current DataFrame to a CSV file.
        
        :param frequency: Data frequency used for filename
        :param start_date: Start date used for filename
        :param end_date: End date used for filename
        """
        if self.__df.empty:
            logger.warning("No data to save. Please fetch data first.")
            return

        filename = f'{self.__stock_name}_{frequency}_{start_date}_{end_date}.csv'
        file_path = os.path.join(data_dir, filename)
        self.__df.to_csv(file_path, index=True)
        logger.info(f'Data saved to {file_path}')

    def get_stock_data(self, start_date, end_date, period,data_dir=''):
        """
        Interface for retrieving stock data for different time periods.
        This method fetches data, filters it based on the date range,
        saves it to a CSV file, and returns the resulting DataFrame.
        
        :param start_date: Start date for the data (YYYY-MM-DD)
        :param end_date: End date for the data (YYYY-MM-DD)
        :param period: Data frequency ('5m','15m','30m','60m','1d','1w','1m','1q')
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
            self.save_stock_csv(frequency=period, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'),data_dir=data_dir)

            return self.__df

        except Exception as e:
            logger.error(f"Error in get_stock_data: {str(e)}")
            return pd.DataFrame()

if __name__ == '__main__':
    # Test cases for the GetStock class
    gs = GetStock()
    gs.set_stock_name('sz000001')

    # Test the new interface
    #end_date = datetime.now().strftime('%Y-%m-%d')
    end_date = datetime.strptime('2024-09-15', '%Y-%m-%d')

    # Test 5-minute data, only can fetch the data from now to few months age, 5000 about 5 months
    start_date = (end_date - timedelta(days=1000)).strftime('%Y-%m-%d')
    df_5m = gs.get_stock_data(start_date, end_date.strftime('%Y-%m-%d'), '5m')
    print("5-minute data shape:", df_5m.shape)
    
    # Test 15-minute data, only can fetch the data from now to few months age, 5000 about 1 yearr
    start_date = (end_date - timedelta(days=1000)).strftime('%Y-%m-%d')
    df_5m = gs.get_stock_data(start_date, end_date.strftime('%Y-%m-%d'), '15m')
    print("5-minute data shape:", df_5m.shape)

    # Test 60-minute data, only can fetch the data from now to few years age, 2000 about 2years
    start_date = (end_date - timedelta(days=1000)).strftime('%Y-%m-%d')
    df_60m = gs.get_stock_data(start_date, end_date.strftime('%Y-%m-%d'), '60m')
    print("60-minute data shape:", df_60m.shape)

    # Test daily data
    start_date = (end_date - timedelta(days=365)).strftime('%Y-%m-%d')
    df_1d = gs.get_stock_data(start_date, end_date.strftime('%Y-%m-%d'), '1d')
    print("Daily data shape:", df_1d.shape)

    # Test monthly data (past 5 years)
    start_date = (end_date - timedelta(days=365*5)).strftime('%Y-%m-%d')
    df_1m = gs.get_stock_data(start_date, end_date.strftime('%Y-%m-%d'), '1m')
    print("Monthly data shape:", df_1m.shape)

    # Test quarterly data (past 10 years)
    start_date = (end_date - timedelta(days=365*10)).strftime('%Y-%m-%d')
    df_1q = gs.get_stock_data(start_date, end_date.strftime('%Y-%m-%d'), '1q')
    print("Quarterly data shape:", df_1q.shape)
