#!/usr/bin/env python3
# coding: utf-8
# GetStock.py

from Ashare2 import *
import csvfile
import pandas as pd
from datetime import datetime, timedelta

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
        
        :param frequency: '1m','5m','15m','30m','60m','1d','1w','1M'
        :param count: Number of data points to retrieve
        :param end_date: End date for the data (optional)
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            if frequency == '1m':
                # For 1-minute data, we need to fetch data day by day
                self.__df = self._get_1m_data(count, end_date)
            else:
                self.__df = get_price(self.__stock_name, frequency=frequency, count=count, end_date=end_date)

            if self.__df.empty:
                print(f"No data retrieved for {self.__stock_name} with frequency {frequency}")
            else:
                print(f"Successfully retrieved {len(self.__df)} rows of {frequency} data for {self.__stock_name}")

        except Exception as e:
            print(f"Error retrieving data: {str(e)}")

    def _get_1m_data(self, count, end_date):
        """
        Helper method to get 1-minute data by fetching day by day
        """
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        all_data = []
        days_to_fetch = count // 240 + 1  # Assuming 240 minutes in a trading day

        for i in range(days_to_fetch):
            date = (end_date - timedelta(days=i)).strftime('%Y-%m-%d')
            day_data = get_price(self.__stock_name, frequency='1m', count=240, end_date=date)
            if not day_data.empty:
                all_data.append(day_data)

        if all_data:
            combined_data = pd.concat(all_data, axis=0)
            return combined_data.sort_index().tail(count)
        else:
            return pd.DataFrame()

    def save_stock_csv(self, frequency):
        if self.__df.empty:
            print("No data to save. Please fetch data first.")
            return

        filename = f'{frequency}_{self.__stock_name}_data.csv'
        self.__df.to_csv(filename, index=True)
        print(f'Data saved to {filename}')

if __name__ == '__main__':
    gs = GetStock()
    gs.set_stock_name('sh000001')

    # Test 1-minute data
    gs.get_stock(frequency='1m', count=1000)
    gs.save_stock_csv(frequency='1m')

    # Test daily data
    gs.get_stock(frequency='1d', count=100)
    gs.save_stock_csv(frequency='1d')
