# data_processing.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class DataNormalizer:
    """数据标准化处理类"""
    
    def __init__(self):
        self.history_mean = {}
        self.history_std = {}
        
    def normalize_price(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        价格数据标准化
        """
        normalized = price_data.copy()
        for col in ['open', 'high', 'low', 'close']:
            normalized[col] = (price_data[col] - price_data[col].mean()) / price_data[col].std()
        return normalized
    
    def normalize_volume(self, volume_data: pd.Series) -> pd.Series:
        """
        成交量数据标准化
        """
        return (volume_data - volume_data.mean()) / volume_data.std()
    
    def normalize_by_history(self, data: pd.Series, window: int = 100) -> pd.Series:
        """
        使用历史数据进行标准化
        """
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        return (data - rolling_mean) / rolling_std

class MACalculator:
    """MA均线系统计算类"""
    
    def __init__(self, ma_periods: List[int] = [3, 5, 10, 20]):
        self.ma_periods = ma_periods
        
    def calculate_ma_basic(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """计算基础MA指标"""
        df = price_data.copy()
        
        # 计算日内均值
        df['AVE'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # 计算历史均值Y
        window = 100
        alpha = 1/window
        df['Y'] = df['AVE'].ewm(alpha=alpha, adjust=False).mean()
        
        # 计算各周期MA
        for period in self.ma_periods:
            df[f'MA_{period}'] = df['close'].rolling(window=period).mean()
            df[f'MA_{period}_norm'] = df[f'MA_{period}'] / df['Y']
            
        return df
    
    def calculate_ma_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MA状态量"""
        for period in self.ma_periods:
            # 计算斜率
            df[f'SLP_{period}'] = (df[f'MA_{period}'] - df[f'MA_{period}'].shift(1)) / \
                                 (df[f'MA_{period}'].shift(1) * df['Y'])
            
            # 计算偏离度
            df[f'DV_{period}'] = (df[f'MA_{period}'] - df['AVE']) / \
                                (df[f'MA_{period}'] * df['Y'])
            
            # 计算偏离斜率
            df[f'DV_SLP_{period}'] = (df[f'DV_{period}'] - df[f'DV_{period}'].shift(1)) / \
                                    (df[f'DV_{period}'].shift(1) * df['Y'])
        
        # 计算均线排列状态
        df['MA_STATE'] = 0
        
        # 多头排列
        bull_mask = (df['MA_3'] > df['MA_5']) & (df['MA_5'] > df['MA_10']) & (df['MA_10'] > df['MA_20'])
        df.loc[bull_mask, 'MA_STATE'] = 2
        
        # 部分多头
        partial_bull_mask = (df['MA_3'] > df['MA_5']) & (df['MA_10'] > df['MA_20'])
        df.loc[partial_bull_mask & ~bull_mask, 'MA_STATE'] = 1
        
        # 空头排列
        bear_mask = (df['MA_3'] < df['MA_5']) & (df['MA_5'] < df['MA_10']) & (df['MA_10'] < df['MA_20'])
        df.loc[bear_mask, 'MA_STATE'] = -2
        
        # 部分空头
        partial_bear_mask = (df['MA_3'] < df['MA_5']) & (df['MA_10'] < df['MA_20'])
        df.loc[partial_bear_mask & ~bear_mask, 'MA_STATE'] = -1
        
        return df

class MACDCalculator:
    """MACD指标系统计算类"""
    
    def __init__(self, fast_period: int = 5, slow_period: int = 10, signal_period: int = 5):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate_macd_basic(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """计算基础MACD指标"""
        df = price_data.copy()
        
        # 计算快速和慢速EMA
        df['EMA_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # 计算DIFF和DEA
        df['DIFF'] = df['EMA_fast'] - df['EMA_slow']
        df['DEA'] = df['DIFF'].ewm(span=self.signal_period, adjust=False).mean()
        
        # 计算MACD柱
        df['MACD'] = 2 * (df['DIFF'] - df['DEA'])
        
        return df
    
    def calculate_macd_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD状态量"""
        # DEA偏离
        df['DEA_DV'] = df['DEA'] / df['Y']
        
        # DEA斜率
        df['DEA_SLP'] = (df['DEA'] - df['DEA'].shift(1)) / (df['DEA'].shift(1) * df['Y'])
        
        # MACD动量
        df['MACD_MOM'] = (df['MACD'] - df['MACD'].shift(1)) / (abs(df['MACD'].shift(1)) * df['Y'])
        
        return df

class VolumeAnalyzer:
    """成交量分析类"""
    
    def __init__(self, volume_ma_period: int = 5):
        self.volume_ma_period = volume_ma_period
        
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量指标"""
        # 成交量移动平均
        df['VOL_MA5'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        
        # 成交量相对变化率
        df['VOL_RATE'] = (df['volume'] - df['VOL_MA5']) / df['VOL_MA5']
        
        # 量价关系
        df['VOL_PRICE_RATIO'] = df['VOL_RATE'] / df['Y']
        
        # 成交量趋势
        df['VOL_TREND'] = 0
        
        # 设置成交量趋势状态
        df.loc[df['VOL_RATE'] > 0.5, 'VOL_TREND'] = 2  # 强势放量
        df.loc[(df['VOL_RATE'] > 0) & (df['VOL_RATE'] <= 0.5), 'VOL_TREND'] = 1  # 温和放量
        df.loc[(df['VOL_RATE'] < 0) & (df['VOL_RATE'] >= -0.5), 'VOL_TREND'] = -1  # 温和缩量
        df.loc[df['VOL_RATE'] < -0.5, 'VOL_TREND'] = -2  # 强势缩量
        
        return df

class MultiPeriodDataProcessor:
    """多周期数据处理主类"""
    
    def __init__(self, periods: List[str]):
        self.periods = periods  # ['5min', '15min', '1h', '1d', '1w']
        self.normalizer = DataNormalizer()
        self.ma_calculator = MACalculator()
        self.macd_calculator = MACDCalculator()
        self.volume_analyzer = VolumeAnalyzer()
        
    def process_single_period(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        """处理单个周期的数据"""
        df = data.copy()
        
        # 添加周期标识
        df['period_code'] = self.periods.index(period)
        
        # 标准化处理
        df = self.normalizer.normalize_price(df)
        df['volume'] = self.normalizer.normalize_volume(df['volume'])
        
        # 计算技术指标
        df = self.ma_calculator.calculate_ma_basic(df)
        df = self.ma_calculator.calculate_ma_states(df)
        df = self.macd_calculator.calculate_macd_basic(df)
        df = self.macd_calculator.calculate_macd_states(df)
        df = self.volume_analyzer.calculate_volume_indicators(df)
        
        return df
    
    def align_multi_period_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """对齐多周期数据"""
        aligned_data = {}
        base_period = min(self.periods)  # 使用最小周期作为基准
        
        for period in self.periods:
            if period == base_period:
                aligned_data[period] = data_dict[period]
            else:
                # 重采样到基准周期
                df = data_dict[period]
                resampled = df.resample(base_period).ffill()
                aligned_data[period] = resampled
        
        return aligned_data
    
    def process_all_periods(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """处理所有周期的数据"""
        processed_data = {}
        
        # 处理每个周期的数据
        for period in self.periods:
            processed_data[period] = self.process_single_period(data_dict[period], period)
        
        # 对齐多周期数据
        aligned_data = self.align_multi_period_data(processed_data)
        
        return aligned_data

def create_feature_matrix(processed_data: Dict[str, pd.DataFrame]) -> np.ndarray:
    """创建特征矩阵"""
    feature_columns = [
        'period_code', 'open', 'high', 'low', 'close', 'volume',
        'MA_3_norm', 'SLP_3', 'DV_3', 'DV_SLP_3',
        'MA_5_norm', 'SLP_5', 'DV_5', 'DV_SLP_5',
        'MA_10_norm', 'SLP_10', 'DV_10', 'DV_SLP_10',
        'MA_20_norm', 'SLP_20', 'DV_20', 'DV_SLP_20',
        'MA_STATE',
        'DEA_DV', 'DEA_SLP', 'MACD_MOM',
        'VOL_RATE', 'VOL_PRICE_RATIO', 'VOL_TREND'
    ]
    
    all_features = []
    for period_data in processed_data.values():
        features = period_data[feature_columns].values
        all_features.append(features)
    
    return np.stack(all_features, axis=1)  # [samples, periods, features]

if __name__ == "__main__":
    # 示例使用
    periods = ['5min', '15min', '1h', '1d', '1w']
    processor = MultiPeriodDataProcessor(periods)
    
    # 假设我们有测试数据
    test_data = {
        '5min': pd.DataFrame(),  # 添加实际数据
        '15min': pd.DataFrame(),
        '1h': pd.DataFrame(),
        '1d': pd.DataFrame(),
        '1w': pd.DataFrame()
    }
    
    # 处理数据
    processed_data = processor.process_all_periods(test_data)
    
    # 创建特征矩阵
    feature_matrix = create_feature_matrix(processed_data)