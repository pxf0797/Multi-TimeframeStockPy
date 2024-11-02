# data_processor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

class MarketDataProcessor:
    def __init__(self):
        self.data_cache = {}
        
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化数据处理
        """
        normalized_data = data.copy()
        
        # 计算价格的历史均值
        normalized_data['AVE'] = (normalized_data['open'] + normalized_data['high'] + 
                                 normalized_data['low'] + normalized_data['close']) / 4
        
        # 使用移动平均作为基准进行标准化
        window = 100
        normalized_data['Y'] = normalized_data['AVE'].ewm(span=window).mean()
        
        # 标准化价格数据
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            normalized_data[f'{col}_norm'] = normalized_data[col] / normalized_data['Y']
            
        return normalized_data

    def align_multi_period_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        对齐不同周期的数据
        """
        aligned_data = {}
        base_period = min(self.periods)  # 使用最小周期作为基准
        
        for period, df in data_dict.items():
            if period == base_period:
                aligned_data[period] = df
            else:
                # 将较长周期的数据转换为较短周期
                aligned_df = self._convert_to_base_period(df, base_period)
                aligned_data[period] = aligned_df
                
        return aligned_data
    
    def _convert_to_base_period(self, df: pd.DataFrame, base_period: str) -> pd.DataFrame:
        """
        将长周期数据转换为短周期对应时刻值
        """
        # 示例实现，实际使用时需要根据具体需求进行调整
        converted_df = df.resample(base_period).ffill()
        return converted_df

class MultiPeriodDataset(Dataset):
    def __init__(self, data_dict: Dict[str, pd.DataFrame], sequence_length: int = 100):
        self.data_dict = data_dict
        self.sequence_length = sequence_length
        self.prepare_data()
        
    def prepare_data(self):
        """
        准备数据集
        """
        self.prepared_data = {}
        for period, df in self.data_dict.items():
            # 转换为numpy数组
            price_data = df[['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
            volume_data = df['volume'].values.reshape(-1, 1)
            
            # 组合特征
            features = np.concatenate([price_data, volume_data], axis=1)
            self.prepared_data[period] = features
            
        # 确保所有周期数据长度一致
        min_length = min(len(data) for data in self.prepared_data.values())
        for period in self.prepared_data.keys():
            self.prepared_data[period] = self.prepared_data[period][:min_length]
            
        self.length = min_length - self.sequence_length + 1
        
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
                      train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    """
    dataset = MultiPeriodDataset(data_dict)
    
    # 划分训练集和验证集
    train_size = int(len(dataset) * train_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader