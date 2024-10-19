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
        
        :param frequency: '1m','5m','15m','30m','60m','1d','1w','1M','1q'
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

    def save_stock_csv(self, frequency):
        if self.__df.empty:
            logger.warning("No data to save. Please fetch data first.")
            return

        filename = f'{frequency}_{self.__stock_name}_data.csv'
        self.__df.to_csv(filename, index=True)
        logger.info(f'Data saved to {filename}')

if __name__ == '__main__':
    gs = GetStock()
    gs.set_stock_name('sh000001')

    # Test 1-minute data
    gs.get_stock(frequency='1m', count=1000)
    gs.save_stock_csv(frequency='1m')

    # Test daily data
    gs.get_stock(frequency='1d', count=100)
    gs.save_stock_csv(frequency='1d')

    # Test quarterly data
    gs.get_stock(frequency='1q', count=20)
    gs.save_stock_csv(frequency='1q')
