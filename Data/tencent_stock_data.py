import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TencentStockData:
    """ 
    count: 
        big period, only use to count the data, and start_date will be igored 
        small period, count will be ignored, start_date and 
        default value is 10
    start_date: 
        the start date of the data in "YYYY-MM-DD" format. 
    end_date: 
        the end date of the data in "YYYY-MM-DD" format. 
    """
    
    """
    Fetches stock data from a financial API.

    Args:
        code (str): The stock code. Format should be {market}{code}, where market is:
                    - "sh" for Shanghai stocks
                    - "sz" for Shenzhen stocks
                    For example, "sh600519" for Kweichow Moutai.
        unit (str): The time unit for the data. Possible values are:
                    - "day" for daily data
                    - "week" for weekly data
                    - "month" for monthly data
                    - "quarter" for quarterly data
                    - "year" for yearly data
        start_date (str): The start date for the data in "YYYY-MM-DD" format. 
                          Use empty string "" to ignore this parameter.
        end_date (str): The end date for the data in "YYYY-MM-DD" format.
        count (int): The number of data points to retrieve, counting backwards from the end date.
        adjust (str, optional): The price adjustment method. Defaults to 'qfq'. Options:
                    - "qfq" for forward adjustment
                    - "hfq" for backward adjustment
                    - "" (empty string) for no adjustment

    Returns:
        Dict[str, Any]: A dictionary containing the requested stock data. The structure is:
            {
                "code": str,  # The stock code
                "name": str,  # The stock name
                "data": List[List[Any]],  # The historical data
                "qt": Dict[str, Any]  # Real-time quotes and other information
            }
            Each item in the "data" list typically contains:
            [date, open, close, high, low, volume, ...]

    Raises:
        requests.RequestException: If there's an error in making the API request.
        ValueError: If the API returns an error or invalid data.

    Notes:
        - Ensure you have the necessary permissions to access this API and comply with its terms of use.
        - The actual data returned may vary based on the stock and the time range requested.
        - For "quarter" and "year" units, the count parameter behavior might differ from shorter time units.
        - If both start_date and count are specified, the API might prioritize one over the other.

    Example:
        >>> data = get_stock_data("sh600519", "day", "", "2023-10-18", 30)
        This will fetch 30 days of daily data for Kweichow Moutai up to 2023-10-18.

        >>> quarterly_data = get_stock_data("sz000001", "quarter", "2023-01-01", "2023-12-31", 4)
        This will fetch quarterly data for Ping An Bank from 2023-01-01 to 2023-12-31, up to 4 quarters.
    """
    #base_url = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    #url = f"{base_url}?param={code},{unit},{start_date},{end_date},{count},{adjust}"

    """
    Fetches minute-level stock data from a financial API.

    Args:
        code (str): The stock code. Format should be {market}{code}, where market is:
                    - "sh" for Shanghai stocks
                    - "sz" for Shenzhen stocks
                    For example, "sh600519" for Kweichow Moutai.
        ts (int): The time scale in minutes. Common values include:
                  - 1 for 1-minute data
                  - 5 for 5-minute data
                  - 15 for 15-minute data
                  - 30 for 30-minute data
                  - 60 for 60-minute data
        start_date (str): The start date for the data in "YYYY-MM-DD" format.
                          Use empty string "" to start from the earliest available data.
        count (int): The number of data points to retrieve.

    Returns:
        Dict[str, Any]: A dictionary containing the requested stock data. The structure may include:
            {
                "code": str,  # The stock code
                "data": List[List[Any]],  # The historical minute-level data
                "qt": Dict[str, Any]  # Real-time quotes and other information
            }
            Each item in the "data" list typically contains:
            [timestamp, open, close, high, low, volume, ...]

    Raises:
        requests.RequestException: If there's an error in making the API request.
        ValueError: If the API returns an error or invalid data.

    Notes:
        - Ensure you have the necessary permissions to access this API and comply with its terms of use.
        - The actual data returned may vary based on the stock and the time range requested.
        - This function retrieves minute-level data, which is more granular than daily data.
        - The API might have limitations on how far back in time you can request data or how many data points you can retrieve at once.
        - If both start_date and count are specified, the API will return 'count' number of data points starting from 'start_date'.
        - If start_date is an empty string, the API will return the most recent 'count' number of data points.

    Example:
        >>> data = get_stock_minute_data("sh600519", 5, "2023-10-18", 100)
        This will fetch 100 data points of 5-minute interval data for Kweichow Moutai, starting from 2023-10-18.

        >>> data = get_stock_minute_data("sz000001", 1, "", 60)
        This will fetch the most recent 60 data points of 1-minute interval data for Ping An Bank.
    """
    #base_url = "http://ifzq.gtimg.cn/appstock/app/kline/mkline"
    #url = f"{base_url}?param={code},m{ts},{start_date},{count}"
    
    def __init__(self, code):
        self.code = code
        self.base_url = "http://ifzq.gtimg.cn/appstock/app/kline/mkline"
        self.base_url2 = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        self.data = {}
        self.periods = {
            '1m': 'm1', '5m': 'm5', '15m': 'm15', '30m': 'm30', '1h': 'm60',
            '1d': 'day', '1w': 'week', '1M': 'month'
        }

    def fetch_data(self, start_date, end_date, period='5m'):
        if period not in self.periods:
            raise ValueError(f"Invalid period: {period}. Supported periods are: {', '.join(self.periods.keys())}")
        
        api_period = self.periods[period]
        all_data = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # 确保结束日期不超过今天
        today = datetime.now().date()
        if end_date.date() > today:
            end_date = datetime.combine(today, datetime.min.time())
            logging.warning(f"End date adjusted to today: {end_date.strftime('%Y-%m-%d')}")
        
        # Adjust URL construction and date range based on period
        if period in ['1d', '1w', '1M']:
            # 调整开始日期，确保有足够的历史数据
            if period == '1d':
                start_date = (end_date - timedelta(days=30)).strftime('%Y-%m-%d')
            elif period == '1w':
                start_date = (end_date - timedelta(weeks=8)).strftime('%Y-%m-%d')
            else:  # '1M'
                start_date = (end_date - timedelta(days=180)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url2}?param={self.code},{api_period},,{start_date},{end_date.strftime('%Y-%m-%d')},640,qfq"
        else:
            url = f"{self.base_url}?param={self.code},{api_period},,{start_date}"
        
        logging.debug(f"Requesting URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            logging.debug(f"API response: {data}")
            
            if data['code'] != 0 or 'data' not in data or self.code not in data['data']:
                logging.error(f"Error in API response: {data}")
                return pd.DataFrame()
            
            if period in ['1d', '1w', '1M']:
                stock_data = data['data'][self.code].get(f'qfq{api_period}', data['data'][self.code].get(api_period, []))
            elif isinstance(data['data'][self.code], dict) and api_period in data['data'][self.code]:
                stock_data = data['data'][self.code][api_period]
            elif isinstance(data['data'][self.code], list):
                stock_data = data['data'][self.code]
            else:
                logging.warning(f"Unexpected data structure: {data['data'][self.code]}")
                return pd.DataFrame()
            
            if stock_data:
                parsed_data = self._parse_stock_data(stock_data, period)
                all_data.extend(parsed_data)
            else:
                logging.info(f"No data available for the specified period")
        except requests.RequestException as e:
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['date'] = df['datetime'].dt.date
            df = df[(df['date'] >= current_date.date()) & (df['date'] <= end_date.date())]
            df.set_index('datetime', inplace=True)
            df.index.name = ''
        else:
            logging.warning(f"No data retrieved for period {period} between {start_date} and {end_date}")
        self.data[period] = df
        return df

    def _parse_stock_data(self, data, period):
        parsed_data = []
        for item in data:
            try:
                if isinstance(item, str):
                    # Split the string into a list
                    item = item.split(' ')
                if isinstance(item, list):
                    parsed_item = {
                        'datetime': item[0],
                        'open': float(item[1]),
                        'close': float(item[2]),
                        'high': float(item[3]),
                        'low': float(item[4]),
                        'volume': float(item[5]),
                    }
                    if len(item) > 7:
                        parsed_item['amount'] = float(item[7])
                    parsed_data.append(parsed_item)
                elif isinstance(item, dict):
                    parsed_item = {
                        'datetime': item.get('dt'),
                        'open': float(item.get('open', item.get('o', 0))),
                        'close': float(item.get('close', item.get('c', 0))),
                        'high': float(item.get('high', item.get('h', 0))),
                        'low': float(item.get('low', item.get('l', 0))),
                        'volume': float(item.get('volume', item.get('v', 0))),
                        'amount': float(item.get('amount', item.get('a', 0)))
                    }
                    parsed_data.append(parsed_item)
            except (ValueError, IndexError) as e:
                logging.error(f"Error parsing data item: {item}. Error: {e}")
        return parsed_data

    def analyze_data(self, period='5m'):
        if period not in self.data:
            raise ValueError(f"No data available for period {period}. Please fetch data first.")

        df = self.data[period]
        if df.empty:
            logging.warning(f"No data available for analysis in period {period}")
            return

        # 按日期分组的统计
        daily_stats = df.groupby(df.index.date).agg({
            'open': 'first',
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum',
            'amount': 'sum'
        })
        daily_stats['daily_return'] = (daily_stats['close'] - daily_stats['open']) / daily_stats['open']
        daily_stats['volatility'] = (daily_stats['high'] - daily_stats['low']) / daily_stats['open']
        
        logging.info(f"\n{period} 数据分析结果：")
        logging.info("每日统计：")
        logging.info(daily_stats)
        
        # 计算整体统计
        logging.info("\n整体统计：")
        logging.info(f"平均每日交易量：{daily_stats['volume'].mean():.2f}")
        logging.info(f"平均每日交易金额：{daily_stats['amount'].mean():.2f}万元")
        logging.info(f"平均每日收益率：{daily_stats['daily_return'].mean():.2%}")
        logging.info(f"平均每日波动率：{daily_stats['volatility'].mean():.2%}")
        
        if period in ['1m', '5m', '15m', '30m', '1h']:
            # 找出交易最活跃的时段
            df['hour'] = df.index.hour
            hourly_volume = df.groupby('hour')['volume'].mean()
            most_active_hour = hourly_volume.idxmax()
            logging.info(f"\n交易最活跃的时段：{most_active_hour}时，平均交易量：{hourly_volume.max():.2f}")
        
        # 计算移动平均线
        if period in ['1m', '5m', '15m', '30m', '1h']:
            df['MA5'] = df['close'].rolling(window=5*int(8*60/int(period[:-1]))).mean()
            df['MA10'] = df['close'].rolling(window=10*int(8*60/int(period[:-1]))).mean()
        else:
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA10'] = df['close'].rolling(window=10).mean()
        
        logging.info("\n当前5日和10日移动平均线：")
        logging.info(f"5日MA: {df['MA5'].iloc[-1]:.2f}")
        logging.info(f"10日MA: {df['MA10'].iloc[-1]:.2f}")

    def save_to_csv(self, start_date, end_date, period='5m'):
        if period not in self.data:
            raise ValueError(f"No data available for period {period}. Please fetch data first.")

        df = self.data[period]
        if df.empty:
            logging.warning(f"No data to save for period {period}")
            return

        # 确保 'data' 文件夹存在
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # 创建文件名
        filename = f"data/{self.code}_{period}_{start_date}_{end_date}.csv"
        
        # 保存数据到CSV文件
        df.to_csv(filename)
        logging.info(f"Data saved to file: {filename}")