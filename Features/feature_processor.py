# features/feature_processor.py
import numpy as np
import pandas as pd
from typing import Dict

class FeatureProcessor:
    def __init__(self):
        self.ma_calculator = MAFeatureCalculator()
        self.macd_calculator = MACDFeatureCalculator()
        self.volume_calculator = VolumeFeatureCalculator()
        
    def process_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        处理所有周期的特征
        """
        processed_data = {}
        
        for period, df in data_dict.items():
            # 计算各类特征
            df = self.ma_calculator.calculate_ma_features(df)
            df = self.macd_calculator.calculate_macd_features(df)
            df = self.volume_calculator.calculate_volume_features(df)
            
            # 选择需要的特征列
            feature_columns = [
                'period_code', 'open', 'high', 'low', 'close', 'volume',
                'MA3_norm', 'SLP3', 'DV3', 'DV_SLP3',
                'MA5_norm', 'SLP5', 'DV5', 'DV_SLP5',
                'MA10_norm', 'SLP10', 'DV10', 'DV_SLP10',
                'MA20_norm', 'SLP20', 'DV20', 'DV_SLP20',
                'MA_STATE',
                'DEA_DV', 'DEA_SLP', 'MACD_MOM',
                'VOL_RATE', 'VOL_PRICE_RATIO', 'VOL_TREND'
            ]
            
            # 添加周期编码
            df['period_code'] = self._encode_period(period)
            
            processed_data[period] = df[feature_columns]
            
        return processed_data
    
    def _encode_period(self, period: str) -> int:
        """
        将周期转换为数值编码
        """
        period_mapping = {
            '5min': 0,
            '15min': 1,
            '1h': 2,
            '1d': 3,
            '1w': 4,
            '1M': 5
        }
        return period_mapping.get(period, -1)