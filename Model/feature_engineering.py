# feature_engineering.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis

class FeatureEngineer:
    """特征工程主类"""
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20]):
        self.lookback_periods = lookback_periods
        self.scaler = StandardScaler()
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间特征"""
        df = df.copy()
        
        # 时间特征
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        
        # 交易时段特征
        df['is_morning'] = ((df['hour'] >= 9) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 13) & (df['hour'] < 15)).astype(int)
        
        return df
    
class FeatureEngineer:
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建价格特征"""
        df = df.copy()
        
        for period in self.lookback_periods:
            # 价格动量
            df[f'price_momentum_{period}'] = df['close'].pct_change(period)
            
            # 价格波动率
            df[f'price_volatility_{period}'] = df['close'].rolling(period).std() / df['close'].rolling(period).mean()
            
            # 价格趋势
            df[f'price_trend_{period}'] = (df['close'] - df['close'].rolling(period).mean()) / df['close'].rolling(period).std()
            
            # 价格区间
            df[f'price_range_{period}'] = (df['high'].rolling(period).max() - df['low'].rolling(period).min()) / df['close']
            
            # 价格分位数
            df[f'price_quantile_{period}'] = (df['close'] - df['low'].rolling(period).min()) / \
                (df['high'].rolling(period).max() - df['low'].rolling(period).min())
        
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建成交量特征"""
        df = df.copy()
        
        for period in self.lookback_periods:
            # 成交量趋势
            df[f'volume_trend_{period}'] = df['volume'].rolling(period).mean().pct_change()
            
            # 成交量波动率
            df[f'volume_volatility_{period}'] = df['volume'].rolling(period).std() / df['volume'].rolling(period).mean()
            
            # 量价相关性
            df[f'volume_price_corr_{period}'] = df['volume'].rolling(period).corr(df['close'])
            
            # 成交量比率
            df[f'volume_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
            
            # 成交量分布
            df[f'volume_skew_{period}'] = df['volume'].rolling(period).apply(lambda x: skew(x))
            df[f'volume_kurt_{period}'] = df['volume'].rolling(period).apply(lambda x: kurtosis(x))
        
        return df
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建技术指标特征"""
        df = df.copy()
        
        for period in self.lookback_periods:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # Stochastic Oscillator
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            df[f'K_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df[f'D_{period}'] = df[f'K_{period}'].rolling(3).mean()
            
            # Bollinger Bands
            middle = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'BB_upper_{period}'] = middle + 2 * std
            df[f'BB_lower_{period}'] = middle - 2 * std
            df[f'BB_width_{period}'] = (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}']) / middle
            
            # Average True Range (ATR)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df[f'ATR_{period}'] = true_range.rolling(period).mean()
        
        return df
    
    def create_cross_period_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """创建跨周期特征"""
        result_df = pd.DataFrame(index=data_dict[min(data_dict.keys())].index)
        periods = list(data_dict.keys())
        
        for i in range(len(periods)-1):
            current_period = periods[i]
            next_period = periods[i+1]
            
            # 价格趋势一致性
            result_df[f'price_trend_consistency_{current_period}_{next_period}'] = \
                np.sign(data_dict[current_period]['close'].pct_change()) == \
                np.sign(data_dict[next_period]['close'].pct_change())
            
            # 成交量趋势一致性
            result_df[f'volume_trend_consistency_{current_period}_{next_period}'] = \
                np.sign(data_dict[current_period]['volume'].pct_change()) == \
                np.sign(data_dict[next_period]['volume'].pct_change())
            
            # MA趋势一致性
            result_df[f'ma_trend_consistency_{current_period}_{next_period}'] = \
                np.sign(data_dict[current_period]['MA_STATE']) == \
                np.sign(data_dict[next_period]['MA_STATE'])
        
        return result_df
    
    def create_derivative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建衍生特征"""
        df = df.copy()
        
        # 趋势强度指标
        df['trend_strength'] = abs(df['MA_STATE']) * df['VOL_TREND'].abs()
        
        # 综合动量指标
        df['momentum_index'] = df['MACD_MOM'] * df['VOL_RATE']
        
        # 价格突破指标
        for period in self.lookback_periods:
            ma_col = f'MA_{period}'
            df[f'price_breakthrough_{period}'] = (df['close'] > df[ma_col]).astype(int)
        
        # 波动率预警指标
        df['volatility_alert'] = ((df['MA_3_norm'] - df['MA_20_norm']).abs() > 0.05).astype(int)
        
        return df
    
    def process_features(self, df: pd.DataFrame, data_dict: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """特征处理主函数"""
        # 创建基础特征
        df = self.create_time_features(df)
        df = self.create_price_features(df)
        df = self.create_volume_features(df)
        df = self.create_technical_features(df)
        
        # 如果提供了多周期数据，创建跨周期特征
        if data_dict is not None:
            cross_period_features = self.create_cross_period_features(data_dict)
            df = pd.concat([df, cross_period_features], axis=1)
        
        # 创建衍生特征
        df = self.create_derivative_features(df)
        
        # 处理缺失值
        df = self.handle_missing_values(df)
        
        # 特征标准化
        df = self.standardize_features(df)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 对于时间序列数据，使用前向填充
        df = df.ffill()
        # 仍然存在的缺失值使用0填充
        df = df.fillna(0)
        return df
    
    def standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征标准化"""
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        return df

class FeatureSelector:
    """特征选择类"""
    
    def __init__(self, correlation_threshold: float = 0.95, importance_threshold: float = 0.01):
        self.correlation_threshold = correlation_threshold
        self.importance_threshold = importance_threshold
    
    def remove_highly_correlated(self, df: pd.DataFrame) -> pd.DataFrame:
        """移除高相关特征"""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        return df.drop(columns=to_drop)
    
    def select_features_by_importance(self, df: pd.DataFrame, importance_scores: pd.Series) -> pd.DataFrame:
        """根据特征重要性选择特征"""
        important_features = importance_scores[importance_scores > self.importance_threshold].index
        return df[important_features]

if __name__ == "__main__":
    # 示例使用
    feature_engineer = FeatureEngineer()
    feature_selector = FeatureSelector()
    
    # 假设我们有示例数据
    sample_data = pd.DataFrame()  # 添加实际数据
    
    # 处理特征
    processed_df = feature_engineer.process_features(sample_data)
    
    # 特征选择
    selected_df = feature_selector.remove_highly_correlated(processed_df)