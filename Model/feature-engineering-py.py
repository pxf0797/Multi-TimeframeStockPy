import pandas as pd
import numpy as np
import talib

class FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def engineer_features(self, data):
        featured_data = {}
        for tf, df in data.items():
            df = self.calculate_technical_indicators(df)
            df = self.calculate_volatility(df)
            df = self.calculate_trend_strength(df)
            df = self.calculate_volume_indicators(df)
            featured_data[tf] = df
        return featured_data

    def calculate_technical_indicators(self, df):
        # Existing indicators
        for period in self.config['ma_periods']:
            df[f'MA_{period}'] = talib.SMA(df['Close'], timeperiod=period)
        
        fast, slow, signal = self.config['macd_params']
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=fast, slowperiod=slow, signalperiod=signal)
        
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # New indicators
        df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
        df['DIFF'] = df['EMA_12'] - df['EMA_26']
        df['DEA'] = talib.EMA(df['DIFF'], timeperiod=9)
        df['MACD'] = 2 * (df['DIFF'] - df['DEA'])
        
        df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
        
        df['TSI'] = self.calculate_tsi(df['Close'])
        
        return df

    def calculate_volatility(self, df):
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        return df

    def calculate_trend_strength(self, df):
        df['Trend_Strength'] = (df['MA_5'] - df['MA_20']) / df['MA_20']
        return df

    def calculate_volume_indicators(self, df):
        df['VOL_MA_5'] = talib.SMA(df['Volume'], timeperiod=5)
        df['VOL_RATE'] = (df['Volume'] - df['VOL_MA_5']) / df['VOL_MA_5']
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        return df

    def calculate_tsi(self, close, r=25, s=13):
        diff = close - close.shift(1)
        abs_diff = abs(diff)
        
        smooth_diff = talib.EMA(talib.EMA(diff, timeperiod=r), timeperiod=s)
        smooth_abs_diff = talib.EMA(talib.EMA(abs_diff, timeperiod=r), timeperiod=s)
        
        tsi = 100 * smooth_diff / smooth_abs_diff
        return tsi

    def calculate_wave_trend(self, df, n1=10, n2=21):
        ap = (df['High'] + df['Low'] + df['Close']) / 3
        esa = talib.EMA(ap, timeperiod=n1)
        d = talib.EMA(abs(ap - esa), timeperiod=n1)
        ci = (ap - esa) / (0.015 * d)
        wt1 = talib.EMA(ci, timeperiod=n2)
        wt2 = talib.SMA(wt1, timeperiod=4)
        df['WaveTrend'] = wt1
        df['WaveTrend_Signal'] = wt2
        return df
