# features/macd_features.py
import numpy as np
import pandas as pd
from typing import Dict

class MACDFeatureCalculator:
    def __init__(self, fast_period: int = 5, slow_period: int = 10, signal_period: int = 5):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate_macd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算MACD相关特征
        """
        df = df.copy()
        
        # 计算快速和慢速EMA
        df['EMA_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # 计算DIFF线
        df['DIFF'] = df['EMA_fast'] - df['EMA_slow']
        
        # 计算DEA线
        df['DEA'] = df['DIFF'].ewm(span=self.signal_period, adjust=False).mean()
        
        # 计算MACD柱
        df['MACD'] = 2 * (df['DIFF'] - df['DEA'])
        
        # 计算DEA偏离
        df['DEA_DV'] = df['DEA'] / df['Y']
        
        # 计算DEA斜率
        df['DEA_SLP'] = (df['DEA'] - df['DEA'].shift(1)) / (df['DEA'].shift(1) * df['Y'])
        
        # 计算MACD动量
        df['MACD_MOM'] = (df['MACD'] - df['MACD'].shift(1)) / \
                        (df['MACD'].shift(1).abs() * df['Y'])
        
        return df
