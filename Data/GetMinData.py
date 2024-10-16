import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class TencentStockData:
    def __init__(self, code):
        self.code = code
        self.base_url = "http://ifzq.gtimg.cn/appstock/app/kline/mkline"
        self.data = None

    def fetch_data(self, start_date, end_date, period='m5'):
        all_data = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            url = f"{self.base_url}?param={self.code},{period},,{date_str}"
            
            response = requests.get(url)
            data = response.json()
            
            stock_data = data['data'][self.code][period]
            parsed_data = self._parse_stock_data(stock_data)
            all_data.extend(parsed_data)
            
            current_date += timedelta(days=1)
        
        self.data = pd.DataFrame(all_data)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'], format='%Y%m%d%H%M')
        self.data['date'] = self.data['datetime'].dt.date
        return self.data

    def _parse_stock_data(self, data):
        parsed_data = []
        for item in data:
            if isinstance(item, list) and len(item) >= 7:
                parsed_item = {
                    'datetime': item[0],
                    'open': float(item[1]),
                    'close': float(item[2]),
                    'high': float(item[3]),
                    'low': float(item[4]),
                    'volume': float(item[5]),
                    'amount': float(item[7]) if len(item) > 7 else None
                }
                parsed_data.append(parsed_item)
        return parsed_data

    def analyze_data(self):
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")

        # 按日期分组的统计
        daily_stats = self.data.groupby('date').agg({
            'open': 'first',
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum',
            'amount': 'sum'
        })
        daily_stats['daily_return'] = (daily_stats['close'] - daily_stats['open']) / daily_stats['open']
        daily_stats['volatility'] = (daily_stats['high'] - daily_stats['low']) / daily_stats['open']
        
        print("每日统计：")
        print(daily_stats)
        
        # 计算整体统计
        print("\n整体统计：")
        print(f"平均每日交易量：{daily_stats['volume'].mean():.2f}")
        print(f"平均每日交易金额：{daily_stats['amount'].mean():.2f}万元")
        print(f"平均每日收益率：{daily_stats['daily_return'].mean():.2%}")
        print(f"平均每日波动率：{daily_stats['volatility'].mean():.2%}")
        
        # 找出交易最活跃的时段
        self.data['hour'] = self.data['datetime'].dt.hour
        hourly_volume = self.data.groupby('hour')['volume'].mean()
        most_active_hour = hourly_volume.idxmax()
        print(f"\n交易最活跃的时段：{most_active_hour}时，平均交易量：{hourly_volume.max():.2f}")
        
        # 计算5日和10日移动平均线
        self.data['MA5'] = self.data['close'].rolling(window=5*48).mean()  # 每天48个5分钟周期
        self.data['MA10'] = self.data['close'].rolling(window=10*48).mean()
        
        print("\n当前5日和10日移动平均线：")
        print(f"5日MA: {self.data['MA5'].iloc[-1]:.2f}")
        print(f"10日MA: {self.data['MA10'].iloc[-1]:.2f}")

    def save_to_csv(self, start_date, end_date):
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")

        # 确保 'data' 文件夹存在
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # 创建文件名
        filename = f"data/{self.code}_{start_date}_{end_date}.csv"
        
        # 保存数据到CSV文件
        self.data.to_csv(filename, index=False)
        print(f"\n数据已保存到文件：{filename}")

# 使用示例
if __name__ == "__main__":
    code = "sz000001"  # 平安银行的股票代码
    start_date = "2024-10-08"
    end_date = "2024-10-16"

    try:
        stock_data = TencentStockData(code)
        stock_data.fetch_data(start_date, end_date)
        stock_data.analyze_data()
        stock_data.save_to_csv(start_date, end_date)
    except Exception as e:
        print(f"An error occurred: {e}")