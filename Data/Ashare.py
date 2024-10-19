# -*- coding: utf-8 -*-
"""
Ashare 股票行情数据双核心版 (https://github.com/mpquant/Ashare)
Optimized and restructured version
tushare 1b298c8f2a7b7a0c929ae7552434213df054ab4fb49cb7676d14e0f9
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
            if col not in ['time', 'day']:
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
        unit = 'week' if frequency == '1w' else 'month' if frequency == '1m' else 'day'
        end_date = TencentDataFetcher.process_end_date(end_date)
        end_date = '' if end_date == datetime.now().strftime('%Y-%m-%d') else end_date

        url = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},{unit},,{end_date},{count},qfq'
        logger.info(f"Fetching data from Tencent: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = json.loads(response.content)
            logger.debug(f"Received data from Tencent: {data}")
            stock_data = data['data'][code]
            ms = f'qfq{unit}'
            buf = stock_data[ms] if ms in stock_data else stock_data[unit]
            logger.info(f"Successfully fetched {len(buf)} rows of {frequency} data for {code}")
            return TencentDataFetcher.create_dataframe(buf, ['time', 'open', 'close', 'high', 'low', 'volume'])
        except requests.RequestException as e:
            logger.error(f"Error fetching data from Tencent: {e}")
            raise
        except KeyError as e:
            logger.error(f"KeyError in Tencent data: {e}")
            logger.debug(f"Tencent data structure: {data}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Tencent data fetching: {e}")
            raise

    @staticmethod
    def get_price_min(code: str, end_date: Optional[str] = None, count: int = 10, frequency: str = '1d') -> pd.DataFrame:
        """Fetch minute price data from Tencent"""
        ts = int(frequency[:-1]) if frequency[:-1].isdigit() else 1
        end_date = TencentDataFetcher.process_end_date(end_date) if end_date else None

        url = f'http://ifzq.gtimg.cn/appstock/app/kline/mkline?param={code},m{ts},,{count}'
        logger.info(f"Fetching minute data from Tencent: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = json.loads(response.content)
            logger.debug(f"Received minute data from Tencent: {data}")
            buf = data['data'][code][f'm{ts}']
            df = TencentDataFetcher.create_dataframe(buf, ['time', 'open', 'close', 'high', 'low', 'volume', 'n1', 'n2'])
            df = df[['open', 'close', 'high', 'low', 'volume']]
            df.loc[df.index[-1], 'close'] = float(data['data'][code]['qt'][code][3])
            logger.info(f"Successfully fetched {len(df)} rows of {frequency} minute data for {code}")
            return df
        except requests.RequestException as e:
            logger.error(f"Error fetching minute data from Tencent: {e}")
            raise
        except KeyError as e:
            logger.error(f"KeyError in Tencent minute data: {e}")
            logger.debug(f"Tencent minute data structure: {data}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Tencent minute data fetching: {e}")
            raise

class SinaDataFetcher(StockDataFetcher):
    """Class for fetching stock data from Sina"""

    @staticmethod
    def get_price(code: str, end_date: str = '', count: int = 10, frequency: str = '60m') -> pd.DataFrame:
        """Fetch price data from Sina"""
        frequency = frequency.replace('1d', '240m').replace('1w', '1200m').replace('1m', '7200m')
        ts = int(frequency[:-1]) if frequency[:-1].isdigit() else 1
        
        if end_date and frequency in ['240m', '1200m', '7200m']:
            end_date = pd.to_datetime(end_date) if not isinstance(end_date, date) else end_date
            unit = 4 if frequency == '1200m' else 29 if frequency == '7200m' else 1
            count += (datetime.now() - end_date).days // unit

        url = f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale={ts}&ma=5&datalen={count}'
        logger.info(f"Fetching data from Sina: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = json.loads(response.content)
            logger.debug(f"Received data from Sina: {data}")
            if not data:
                logger.warning(f"No data received from Sina for {code} with frequency {frequency}")
                return pd.DataFrame()
            df = SinaDataFetcher.create_dataframe(data, ['day', 'open', 'high', 'low', 'close', 'volume'])
            if end_date and frequency in ['240m', '1200m', '7200m']:
                df = df[df.index <= end_date].tail(count)
            logger.info(f"Successfully fetched {len(df)} rows of {frequency} data for {code}")
            return df
        except requests.RequestException as e:
            logger.error(f"Error fetching data from Sina: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from Sina: {e}")
            logger.debug(f"Sina response content: {response.content}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Sina data fetching: {e}")
            raise

def get_price(code: str, end_date: str = '', count: int = 10, frequency: str = '1d', fields: list = []) -> pd.DataFrame:
    """Main function to get stock price data"""
    xcode = code.replace('.XSHG', '').replace('.XSHE', '')
    xcode = f"sh{xcode}" if 'XSHG' in code else f"sz{xcode}" if 'XSHE' in code else code

    logger.info(f"Attempting to fetch {frequency} data for {xcode}")

    if frequency in ['1d', '1w', '1m']:
        try:
            logger.info(f"Trying Sina for {frequency} data")
            return SinaDataFetcher.get_price(xcode, end_date=end_date, count=count, frequency=frequency)
        except Exception as e:
            logger.warning(f"Failed to fetch data from Sina, trying Tencent: {e}")
            try:
                return TencentDataFetcher.get_price_day(xcode, end_date=end_date, count=count, frequency=frequency)
            except Exception as e:
                logger.error(f"Failed to fetch data from Tencent: {e}")
                return pd.DataFrame()

    if frequency == '1q':
        try:
            logger.info("Fetching monthly data for quarterly resampling")
            monthly_df = get_price(xcode, end_date=end_date, count=count*3, frequency='1m')
            if monthly_df is not None and not monthly_df.empty:
                df = monthly_df.resample('Q').last()
                logger.info(f"Successfully resampled monthly data to quarterly, got {len(df)} rows")
                return df
            else:
                logger.warning(f"Failed to fetch monthly data for {xcode}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error while fetching or resampling data for quarterly frequency: {e}")
            return pd.DataFrame()

    if frequency in ['5m', '15m', '30m', '60m']:
        try:
            logger.info(f"Trying Sina for {frequency} data")
            return SinaDataFetcher.get_price(xcode, end_date=end_date, count=count, frequency=frequency)
        except Exception as e:
            logger.warning(f"Failed to fetch data from Sina, trying Tencent: {e}")
            try:
                return TencentDataFetcher.get_price_min(xcode, end_date=end_date, count=count, frequency=frequency)
            except Exception as e:
                logger.error(f"Failed to fetch data from Tencent: {e}")
                return pd.DataFrame()

if __name__ == '__main__':
    # Test cases
    df = get_price('sh000001', frequency='1d', count=10)
    print('上证指数日线行情\n', df)

    df = get_price('000001.XSHG', frequency='15m', count=10)
    print('上证指数分钟线\n', df)

    df = get_price('000001.XSHG', end_date='2024-09-30', frequency='60m', count=10)
    print('上证指数分钟线\n', df)
    
    df = get_price('sh000001', frequency='1w', count=10)
    print('上证指数周线行情\n', df)
    
    df = get_price('sh000001', frequency='1m', count=10)
    print('上证指数月线行情\n', df)

    df = get_price('sh000001', frequency='1q', count=10)
    print('上证指数季度线行情\n', df)
