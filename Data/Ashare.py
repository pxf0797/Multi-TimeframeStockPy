# -*- coding: utf-8 -*-
"""
Ashare 股票行情数据双核心版 (https://github.com/mpquant/Ashare)
Optimized and restructured version
"""

import json
from typing import Optional, Union
from datetime import datetime, date
import requests
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """Base class for fetching stock data"""
    
    @staticmethod
    def process_end_date(end_date: Union[str, date, datetime]) -> str:
        """Process the end_date parameter"""
        if isinstance(end_date, (date, datetime)):
            return end_date.strftime('%Y-%m-%d')
        elif isinstance(end_date, str):
            return end_date.split(' ')[0]
        return ''

    @staticmethod
    def create_dataframe(data: list, columns: list) -> pd.DataFrame:
        """Create and process a DataFrame from the fetched data"""
        df = pd.DataFrame(data, columns=columns)
        for col in df.columns:
            if col != 'time' and col != 'day':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['time'] = pd.to_datetime(df['time'] if 'time' in df.columns else df['day'])
        df.set_index('time', inplace=True)
        df.index.name = ''
        return df

class TencentDataFetcher(StockDataFetcher):
    """Class for fetching stock data from Tencent"""

    @staticmethod
    def get_price_day(code: str, end_date: str = '', count: int = 10, frequency: str = '1d') -> pd.DataFrame:
        """Fetch daily price data from Tencent"""
        unit = 'week' if frequency == '1w' else 'month' if frequency == '1M' else 'day'
        end_date = TencentDataFetcher.process_end_date(end_date)
        end_date = '' if end_date == datetime.now().strftime('%Y-%m-%d') else end_date

        url = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},{unit},,{end_date},{count},qfq'
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = json.loads(response.content)
            stock_data = data['data'][code]
            ms = f'qfq{unit}'
            buf = stock_data[ms] if ms in stock_data else stock_data[unit]
            return TencentDataFetcher.create_dataframe(buf, ['time', 'open', 'close', 'high', 'low', 'volume'])
        except requests.RequestException as e:
            logger.error(f"Error fetching data from Tencent: {e}")
            raise

    @staticmethod
    def get_price_min(code: str, end_date: Optional[str] = None, count: int = 10, frequency: str = '1d') -> pd.DataFrame:
        """Fetch minute price data from Tencent"""
        ts = int(frequency[:-1]) if frequency[:-1].isdigit() else 1
        end_date = TencentDataFetcher.process_end_date(end_date) if end_date else None

        url = f'http://ifzq.gtimg.cn/appstock/app/kline/mkline?param={code},m{ts},,{count}'
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = json.loads(response.content)
            buf = data['data'][code][f'm{ts}']
            df = TencentDataFetcher.create_dataframe(buf, ['time', 'open', 'close', 'high', 'low', 'volume', 'n1', 'n2'])
            df = df[['open', 'close', 'high', 'low', 'volume']]
            df['close'].iloc[-1] = float(data['data'][code]['qt'][code][3])
            return df
        except requests.RequestException as e:
            logger.error(f"Error fetching data from Tencent: {e}")
            raise

class SinaDataFetcher(StockDataFetcher):
    """Class for fetching stock data from Sina"""

    @staticmethod
    def get_price(code: str, end_date: str = '', count: int = 10, frequency: str = '60m') -> pd.DataFrame:
        """Fetch price data from Sina"""
        frequency = frequency.replace('1d', '240m').replace('1w', '1200m').replace('1M', '7200m')
        ts = int(frequency[:-1]) if frequency[:-1].isdigit() else 1
        
        if end_date and frequency in ['240m', '1200m', '7200m']:
            end_date = pd.to_datetime(end_date) if not isinstance(end_date, date) else end_date
            unit = 4 if frequency == '1200m' else 29 if frequency == '7200m' else 1
            count += (datetime.now() - end_date).days // unit

        url = f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale={ts}&ma=5&datalen={count}'
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = json.loads(response.content)
            df = SinaDataFetcher.create_dataframe(data, ['day', 'open', 'high', 'low', 'close', 'volume'])
            if end_date and frequency in ['240m', '1200m', '7200m']:
                return df[df.index <= end_date].tail(count)
            return df
        except requests.RequestException as e:
            logger.error(f"Error fetching data from Sina: {e}")
            raise

def get_price(code: str, end_date: str = '', count: int = 10, frequency: str = '1d', fields: list = []) -> pd.DataFrame:
    """Main function to get stock price data"""
    xcode = code.replace('.XSHG', '').replace('.XSHE', '')
    xcode = f"sh{xcode}" if 'XSHG' in code else f"sz{xcode}" if 'XSHE' in code else code

    if frequency in ['1d', '1w', '1M']:
        try:
            return SinaDataFetcher.get_price(xcode, end_date=end_date, count=count, frequency=frequency)
        except Exception as e:
            logger.warning(f"Failed to fetch data from Sina, trying Tencent: {e}")
            return TencentDataFetcher.get_price_day(xcode, end_date=end_date, count=count, frequency=frequency)

    if frequency in ['1m', '5m', '15m', '30m', '60m']:
        if frequency == '1m':
            return TencentDataFetcher.get_price_min(xcode, end_date=end_date, count=count, frequency=frequency)
        try:
            return SinaDataFetcher.get_price(xcode, end_date=end_date, count=count, frequency=frequency)
        except Exception as e:
            logger.warning(f"Failed to fetch data from Sina, trying Tencent: {e}")
            return TencentDataFetcher.get_price_min(xcode, end_date=end_date, count=count, frequency=frequency)

if __name__ == '__main__':
    # Test cases
    df = get_price('sh000001', frequency='1d', count=10)
    print('上证指数日线行情\n', df)

    df = get_price('000001.XSHG', frequency='15m', count=10)
    print('上证指数分钟线\n', df)

    df = get_price('000001.XSHG', end_date='2024-09-30', frequency='5m', count=10)
    print('上证指数分钟线\n', df)
