from tencent_stock_data import TencentStockData
import unittest
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestTencentStockData(unittest.TestCase):
    def setUp(self):
        self.stock_codes = ["sz000001"]  # 平安银行

    def test_fetch_data(self):
        periods = ['1m', '5m', '15m', '30m', '1h', '1d', '1w', '1M']
        for stock_code in self.stock_codes:
            stock = TencentStockData(stock_code)
            for period in periods:
                with self.subTest(stock_code=stock_code, period=period):
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    if period in ['1m', '5m', '15m', '30m', '1h']:
                        start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    elif period == '1d':
                        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                    elif period == '1w':
                        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
                    else:  # '1M'
                        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
                    
                    logging.info(f"Testing {stock_code} {period} data from {start_date} to {end_date}")
                    try:
                        df = stock.fetch_data(start_date, end_date, period=period)
                        self.assertIsNotNone(df)
                        if not df.empty:
                            logging.info(f"Retrieved {len(df)} rows of data for {stock_code} {period}")
                            logging.info(f"Date range: from {df['date'].min()} to {df['date'].max()}")
                            stock.analyze_data(period)
                            stock.save_to_csv(start_date, end_date, period)
                        else:
                            logging.warning(f"No data retrieved for {stock_code} {period}")
                    except Exception as e:
                        logging.error(f"Error processing {stock_code} {period}: {e}")
                        self.fail(f"Error processing {stock_code} {period}: {e}")

if __name__ == '__main__':
    unittest.main()
