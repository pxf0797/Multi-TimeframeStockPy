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

if __name__ == '__main__':
    gs = GetStock()
    gs.set_stock_name('sh000001')

    end_date = datetime.now().strftime('%Y-%m-%d')

    # Test 5-minute data (past week)
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    gs.get_stock(frequency='5m', count=2000, end_date=end_date)
    gs.filter_date_range(start_date)
    gs.save_stock_csv(frequency='5m', start_date=start_date, end_date=end_date)
    
    # Test 60-minute data (past month)
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    gs.get_stock(frequency='60m', count=1000, end_date=end_date)
    gs.filter_date_range(start_date)
    gs.save_stock_csv(frequency='60m', start_date=start_date, end_date=end_date)

    # Test daily data (past year)
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    gs.get_stock(frequency='1d', count=500, end_date=end_date)
    gs.filter_date_range(start_date)
    gs.save_stock_csv(frequency='1d', start_date=start_date, end_date=end_date)
    
    # Test monthly data (past 5 years)
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    gs.get_stock(frequency='1m', count=100, end_date=end_date)
    gs.filter_date_range(start_date)
    gs.save_stock_csv(frequency='1m', start_date=start_date, end_date=end_date)

    # Test quarterly data (past 10 years)
    start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    gs.get_stock(frequency='1q', count=50, end_date=end_date)
    gs.filter_date_range(start_date)
    gs.save_stock_csv(frequency='1q', start_date=start_date, end_date=end_date)
