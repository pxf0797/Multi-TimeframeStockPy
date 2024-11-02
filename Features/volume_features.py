# features/volume_features.py
import numpy as np
import pandas as pd
from typing import Dict

class VolumeFeatureCalculator:
    def __init__(self, volume_ma_period: int = 5):
        self.volume_ma_period = volume_ma_period
        
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交量相关特征
        """
        df = df.copy()
        
        # 计算成交量移动平均
        df['VOL_MA'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        
        # 计算成交量相对变化率
        df['VOL_RATE'] = (df['volume'] - df['VOL_MA']) / df['VOL_MA']
        
        # 计算量价关系
        df['VOL_PRICE_RATIO'] = df['VOL_RATE'] / df['Y']
        
        # 计算成交量趋势
        df['VOL_TREND'] = self._calculate_volume_trend(df)
        
        return df
    
# features/volume_features.py (continued)

    def _calculate_volume_trend(self, df: pd.DataFrame) -> pd.Series:
        """
        计算成交量趋势
        """
        vol_trend = pd.Series(index=df.index, data=0)
        
        # 判断成交量趋势
        vol_trend = np.where(
            (df['VOL_RATE'] > 0.5) & (df['volume'] > df['volume'].shift(1)), 2,
            np.where(df['VOL_RATE'] > 0, 1,
            np.where(df['VOL_RATE'] < 0, -1,
            np.where((df['VOL_RATE'] < -0.5) & (df['volume'] < df['volume'].shift(1)), -2, 0))))
        
        return pd.Series(vol_trend, index=df.index)