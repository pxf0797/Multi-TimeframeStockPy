# features/ma_features.py

import numpy as np
import pandas as pd
from typing import Dict

class MAFeatureCalculator:
    def __init__(self, ma_periods: List[int] = [3, 5, 10, 20]):
        self.ma_periods = ma_periods
        
    def calculate_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算MA均线相关特征
        """
        df = df.copy()
        
        # 计算日内均值
        df['AVE'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # 计算历史均值Y
        window = 100
        df['Y'] = df['AVE'].ewm(span=window).mean()
        
        for period in self.ma_periods:
            # 计算移动平均
            df[f'MA{period}'] = df['close'].rolling(window=period).mean()
            df[f'MA{period}_norm'] = df[f'MA{period}'] / df['Y']
            
            # 计算斜率
            df[f'SLP{period}'] = (df[f'MA{period}'] - df[f'MA{period}'].shift(1)) / \
                                (df[f'MA{period}'].shift(1) * df['Y'])
            
            # 计算偏离度
            df[f'DV{period}'] = (df[f'MA{period}'] - df['AVE']) / \
                               (df[f'MA{period}'] * df['Y'])
            
            # 计算偏离斜率
            df[f'DV_SLP{period}'] = (df[f'DV{period}'] - df[f'DV{period}'].shift(1)) / \
                                   (df[f'DV{period}'].shift(1) * df['Y'])
        
        # 计算均线排列状态
        df['MA_STATE'] = self._calculate_ma_state(df)
        
        return df
    
    def _calculate_ma_state(self, df: pd.DataFrame) -> pd.Series:
        """
        计算均线排列状态
        """
        ma_state = pd.Series(index=df.index, data=0)
        
        # 金叉死叉判断
        ma_state = np.where(
            (df['MA3'] > df['MA5']) & (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20']), 2,
            np.where((df['MA3'] > df['MA5']) & (df['MA10'] > df['MA20']), 1,
            np.where((df['MA3'] < df['MA5']) & (df['MA10'] < df['MA20']), -1,
            np.where((df['MA3'] < df['MA5']) & (df['MA5'] < df['MA10']) & (df['MA10'] < df['MA20']), -2, 0))))
        
        return pd.Series(ma_state, index=df.index)




