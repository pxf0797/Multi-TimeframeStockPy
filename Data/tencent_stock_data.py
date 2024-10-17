import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TencentStockData:
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
            url = f"{self.base_url2}?param={self.code},{api_period},,{start_date},{end_date.strftime('%Y-%m-%d')},640"
        else:
            url = f"{self.base_url}?param={self.code},{api_period},,{start_date}"
        
        logging.debug(f"Requesting URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            logging.debug(f"API response: {data}")
        except requests.RequestException as e:
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()

        if 'data' in data and self.code in data['data']:
            if period in ['1d', '1w', '1M']:
                stock_data = data['data'][self.code][api_period]
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
        else:
            logging.info(f"No data available for the specified stock code and period")
        
        df = pd.DataFrame(all_data)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['date'] = df['datetime'].dt.date
            df = df[(df['date'] >= current_date.date()) & (df['date'] <= end_date.date())]
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
                        'open': float(item.get('o', 0)),
                        'close': float(item.get('c', 0)),
                        'high': float(item.get('h', 0)),
                        'low': float(item.get('l', 0)),
                        'volume': float(item.get('v', 0)),
                        'amount': float(item.get('a', 0))
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
        daily_stats = df.groupby('date').agg({
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
            df['hour'] = df['datetime'].dt.hour
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
        df.to_csv(filename, index=False)
        logging.info(f"Data saved to file: {filename}")