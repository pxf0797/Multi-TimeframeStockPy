# features/feature_integration.py

import torch
import numpy as np
from typing import Dict, List, Optional

class FeatureIntegrator:
    """
    特征整合器：负责将不同周期的特征进行组装和整合
    """
    def __init__(self, periods: List[str]):
        self.periods = periods
        self.short_periods = ['5min', '15min', '1h']
        self.long_periods = ['1d', '1w', '1M']
        
    def integrate_features(self, features_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        整合不同周期的特征
        """
        integrated_features = {}
        
        # 处理短周期特征
        for period in self.short_periods:
            if period in features_dict:
                features = features_dict[period]
                # 添加短期特征组合
                integrated_features[period] = self._combine_short_period_features(features)
        
        # 处理长周期特征
        for period in self.long_periods:
            if period in features_dict:
                features = features_dict[period]
                # 添加长期特征组合
                integrated_features[period] = self._combine_long_period_features(features)
        
        return integrated_features
    
    def _combine_short_period_features(self, features: np.ndarray) -> np.ndarray:
        """
        组合短周期特征
        """
        # 计算滚动统计量
        rolling_mean = np.mean(features, axis=0)
        rolling_std = np.std(features, axis=0)
        rolling_skew = self._calculate_skewness(features)
        
        # 组合特征
        combined_features = np.concatenate([
            features,
            rolling_mean.reshape(1, -1),
            rolling_std.reshape(1, -1),
            rolling_skew.reshape(1, -1)
        ], axis=0)
        
        return combined_features
    
    def _combine_long_period_features(self, features: np.ndarray) -> np.ndarray:
        """
        组合长周期特征
        """
        # 计算趋势特征
        trend = self._calculate_trend(features)
        momentum = self._calculate_momentum(features)
        volatility = self._calculate_volatility(features)
        
        # 组合特征
        combined_features = np.concatenate([
            features,
            trend.reshape(1, -1),
            momentum.reshape(1, -1),
            volatility.reshape(1, -1)
        ], axis=0)
        
        return combined_features
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> np.ndarray:
        """计算偏度"""
        return np.nan_to_num(stats.skew(data, axis=0))
    
    @staticmethod
    def _calculate_trend(data: np.ndarray) -> np.ndarray:
        """计算趋势"""
        return (data[-1] - data[0]) / (data[0] + 1e-8)
    
    @staticmethod
    def _calculate_momentum(data: np.ndarray) -> np.ndarray:
        """计算动量"""
        return data[-1] - np.mean(data, axis=0)
    
    @staticmethod
    def _calculate_volatility(data: np.ndarray) -> np.ndarray:
        """计算波动率"""
        return np.std(np.diff(data, axis=0), axis=0)

class CrossPeriodFeatures:
    """
    跨周期特征计算器：计算不同周期之间的关联特征
    """
    def __init__(self, periods: List[str]):
        self.periods = periods
        
    def calculate_cross_period_features(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        计算跨周期特征
        """
        cross_features = []
        
        # 计算周期间的相关性
        correlations = self._calculate_period_correlations(features_dict)
        cross_features.append(correlations)
        
        # 计算周期间的趋势一致性
        trend_consistency = self._calculate_trend_consistency(features_dict)
        cross_features.append(trend_consistency)
        
        # 计算周期间的波动率比率
        volatility_ratios = self._calculate_volatility_ratios(features_dict)
        cross_features.append(volatility_ratios)
        
        return np.concatenate(cross_features, axis=1)
    
    def _calculate_period_correlations(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        计算不同周期间的相关性
        """
        correlations = []
        for i, period1 in enumerate(self.periods):
            for j, period2 in enumerate(self.periods[i+1:], i+1):
                if period1 in features_dict and period2 in features_dict:
                    corr = np.corrcoef(
                        features_dict[period1].flatten(),
                        features_dict[period2].flatten()
                    )[0, 1]
                    correlations.append(corr)
        
        return np.array(correlations).reshape(1, -1)
    
    def _calculate_trend_consistency(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        计算趋势一致性
        """
        trends = []
        for period in self.periods:
            if period in features_dict:
                # 计算该周期的趋势
                trend = np.sign(features_dict[period][-1] - features_dict[period][0])
                trends.append(trend)
        
        # 计算趋势一致性得分
        trend_array = np.array(trends)
        consistency = np.sum(trend_array == trends[0]) / len(trends)
        
        return np.array([consistency]).reshape(1, -1)
    
    def _calculate_volatility_ratios(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        计算不同周期间的波动率比率
        """
        volatilities = []
        for period in self.periods:
            if period in features_dict:
                vol = np.std(features_dict[period], axis=0)
                volatilities.append(vol)
        
        # 计算波动率比率
        ratios = []
        for i in range(len(volatilities)-1):
            ratio = volatilities[i] / (volatilities[i+1] + 1e-8)
            ratios.append(ratio)
        
        return np.array(ratios).reshape(1, -1)