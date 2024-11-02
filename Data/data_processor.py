# data_processor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

class MarketDataProcessor:
    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window
        self.data_cache = {}
        self.history = {}
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        标幺化数据处理，完全按照设计文档的公式实现
        """
        normalized_data = data.copy()
        
        # 计算日内均值AVE
        normalized_data['AVE'] = (normalized_data['open'] + normalized_data['high'] + 
                                 normalized_data['low'] + normalized_data['close']) / 4
        
        # 计算历史均值Y，使用设计文档中的公式
        alpha = 1 / self.lookback_window
        normalized_data['Y'] = self._calculate_historical_mean(normalized_data['AVE'], alpha)
        
        # 对价格数据进行标幺化，使用设计文档中的公式
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            rolling_min = normalized_data[col].rolling(window=self.lookback_window, min_periods=1).min()
            rolling_max = normalized_data[col].rolling(window=self.lookback_window, min_periods=1).max()
            
            normalized_data[f'{col}_norm'] = (normalized_data[col] - rolling_min) / \
                                           (rolling_max - rolling_min + 1e-8)
            
            # 存储历史极值用于实时预测
            self.history[f'{col}_min'] = rolling_min.iloc[-1]
            self.history[f'{col}_max'] = rolling_max.iloc[-1]
        
        return normalized_data
    
    def _calculate_historical_mean(self, series: pd.Series, alpha: float) -> pd.Series:
        """
        使用设计文档中的公式计算历史均值：
        Y(t_i) = (1-α)Y(t_i-1) + α·AVE(t_i)
        """
        y_values = []
        current_y = series.iloc[0]  # 初始值
        
        for ave in series:
            current_y = (1 - alpha) * current_y + alpha * ave
            y_values.append(current_y)
            
        return pd.Series(y_values, index=series.index)

    def align_multi_period_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        多周期数据对齐，实现两种方案
        """
        aligned_data = {}
        base_period = min(self.periods)  # 使用最小周期作为基准
        
        for period, df in data_dict.items():
            if period == base_period:
                aligned_data[period] = df
            else:
                # 方案1：将长周期数据转换为短周期对应时刻值
                aligned_df_1 = self._convert_to_base_period(df, base_period)
                
                # 方案2：将短周期数据聚合为长周期
                aligned_df_2 = self._aggregate_to_longer_period(data_dict[base_period], period)
                
                # 根据数据质量选择更好的方案
                if self._evaluate_data_quality(aligned_df_1) > self._evaluate_data_quality(aligned_df_2):
                    aligned_data[period] = aligned_df_1
                else:
                    aligned_data[period] = aligned_df_2
                
        return aligned_data
    
    def _convert_to_base_period(self, df: pd.DataFrame, base_period: str) -> pd.DataFrame:
        """
        方案1：将长周期数据转换为短周期对应时刻值
        """
        converted_df = df.resample(base_period).asfreq()
        
        # 价格数据使用前向填充
        price_cols = ['open', 'high', 'low', 'close']
        converted_df[price_cols] = converted_df[price_cols].fillna(method='ffill')
        
        # 成交量数据按比例分配
        if 'volume' in converted_df.columns:
            period_ratio = self._calculate_period_ratio(base_period, df.index.freq)
            converted_df['volume'] = converted_df['volume'].fillna(0) / period_ratio
        
        return converted_df
    
    def _aggregate_to_longer_period(self, df: pd.DataFrame, target_period: str) -> pd.DataFrame:
        """
        方案2：将短周期数据聚合为长周期
        """
        aggregation_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        return df.resample(target_period).agg(aggregation_rules)
    
    def _evaluate_data_quality(self, df: pd.DataFrame) -> float:
        """
        评估数据质量，返回0-1之间的分数
        """
        # 计算数据完整性
        completeness = 1 - df.isnull().mean().mean()
        
        # 计算数据连续性
        continuity = self._calculate_data_continuity(df)
        
        # 计算数据一致性
        consistency = self._calculate_data_consistency(df)
        
        # 综合评分
        quality_score = 0.4 * completeness + 0.3 * continuity + 0.3 * consistency
        return quality_score
    
    def _calculate_data_continuity(self, df: pd.DataFrame) -> float:
        """
        计算数据连续性
        """
        if df.empty:
            return 0.0
            
        # 检查时间戳是否连续
        timestamps = df.index
        expected_diff = pd.Timedelta(timestamps.freq)
        actual_diffs = timestamps[1:] - timestamps[:-1]
        
        continuity_score = np.mean(actual_diffs == expected_diff)
        return float(continuity_score)
    
    def _calculate_data_consistency(self, df: pd.DataFrame) -> float:
        """
        计算数据一致性
        """
        if df.empty:
            return 0.0
            
        # 检查价格逻辑关系
        price_consistent = np.all(
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) & 
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close'])
        )
        
        # 检查成交量非负
        volume_consistent = np.all(df['volume'] >= 0) if 'volume' in df.columns else True
        
        return float(price_consistent and volume_consistent)
    
    def _calculate_period_ratio(self, base_period: str, target_period: str) -> int:
        """
        计算周期比率
        """
        period_minutes = {
            '5min': 5,
            '15min': 15,
            '30min': 30,
            '1H': 60,
            '4H': 240,
            '1D': 1440
        }
        
        base_minutes = period_minutes.get(base_period, 1)
        target_minutes = period_minutes.get(str(target_period), 1)
        
        return target_minutes // base_minutes

    def normalize_realtime_data(self, data: pd.Series) -> pd.Series:
        """
        实时数据标幺化，用于在线预测
        """
        normalized_data = pd.Series(index=data.index)
        
        for col in data.index:
            if col in self.history:
                min_val = self.history[f'{col}_min']
                max_val = self.history[f'{col}_max']
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val + 1e-8)
            else:
                normalized_data[col] = data[col]
        
        return normalized_data

class MultiPeriodDataset(Dataset):
    def __init__(self, 
                 data_dict: Dict[str, pd.DataFrame], 
                 sequence_length: int = 100,
                 feature_columns: List[str] = None):
        self.data_dict = data_dict
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns or ['open_norm', 'high_norm', 'low_norm', 
                                                 'close_norm', 'volume']
        self.prepare_data()
        
    def prepare_data(self):
        """
        准备数据集
        """
        self.prepared_data = {}
        for period, df in self.data_dict.items():
            # 确保所有需要的特征都存在
            available_features = [col for col in self.feature_columns if col in df.columns]
            
            # 转换为numpy数组
            features = df[available_features].values
            
            # 标准化处理
            features = self._standardize_features(features)
            
            self.prepared_data[period] = features
            
        # 确保所有周期数据长度一致
        min_length = min(len(data) for data in self.prepared_data.values())
        for period in self.prepared_data.keys():
            self.prepared_data[period] = self.prepared_data[period][:min_length]
            
        self.length = min_length - self.sequence_length + 1
        
    def _standardize_features(self, features: np.ndarray) -> np.ndarray:
        """
        标准化特征
        """
        # 对每个特征进行标准化
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        standardized = (features - mean) / (std + 1e-8)
        
        return np.nan_to_num(standardized)  # 处理可能的nan值
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        获取数据样本
        """
        sample = {}
        for period, data in self.prepared_data.items():
            # 获取序列数据
            sequence = data[idx:idx+self.sequence_length]
            sample[period] = torch.FloatTensor(sequence)
            
        # 获取目标值（使用下一时刻的收盘价）
        target = self.prepared_data['5min'][idx+self.sequence_length, 3]  # 收盘价索引为3
        return sample, torch.FloatTensor([target])

def create_dataloaders(data_dict: Dict[str, pd.DataFrame], 
                      batch_size: int = 32, 
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      sequence_length: int = 100) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    """
    dataset = MultiPeriodDataset(data_dict, sequence_length=sequence_length)
    
    # 计算数据集切分点
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 划分数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader