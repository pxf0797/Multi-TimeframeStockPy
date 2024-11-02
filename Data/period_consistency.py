# data/period_consistency.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import logging

@dataclass
class ConsistencyCheckConfig:
    """一致性检验配置"""
    # 基础配置
    base_period: str = '5min'
    check_periods: List[str] = None
    
    # 检验阈值
    correlation_threshold: float = 0.85
    deviation_threshold: float = 0.02
    volume_consistency_threshold: float = 0.90
    
    # 时间窗口
    rolling_window: int = 100
    anomaly_detection_window: int = 20
    
    # 统计显著性水平
    significance_level: float = 0.05
    
    def __post_init__(self):
        if self.check_periods is None:
            self.check_periods = ['15min', '1h', '1d', '1w']

class CrossPeriodConsistencyChecker:
    """
    跨周期数据一致性检验器
    """
    def __init__(self, config: ConsistencyCheckConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.consistency_history = []
        
    def check_consistency(self,
                         period_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        检查跨周期数据一致性
        """
        # 基准周期数据
        base_data = period_data[self.config.base_period]
        
        consistency_results = {}
        
        # 对每个检查周期进行一致性检验
        for period in self.config.check_periods:
            if period in period_data:
                period_results = self._check_period_consistency(
                    base_data,
                    period_data[period],
                    period
                )
                consistency_results[period] = period_results
        
        # 计算整体一致性得分
        overall_score = self._calculate_overall_consistency(consistency_results)
        
        # 检测异常
        anomalies = self._detect_consistency_anomalies(consistency_results)
        
        # 生成一致性报告
        report = self._generate_consistency_report(
            consistency_results,
            overall_score,
            anomalies
        )
        
        # 更新历史记录
        self.consistency_history.append({
            'timestamp': pd.Timestamp.now(),
            'results': consistency_results,
            'overall_score': overall_score,
            'anomalies': anomalies
        })
        
        return report
    
    def _check_period_consistency(self,
                                base_data: pd.DataFrame,
                                period_data: pd.DataFrame,
                                period: str) -> Dict[str, Any]:
        """
        检查单个周期的数据一致性
        """
        # 价格一致性检验
        price_consistency = self._check_price_consistency(
            base_data,
            period_data
        )
        
        # 成交量一致性检验
        volume_consistency = self._check_volume_consistency(
            base_data,
            period_data
        )
        
        # OHLC一致性检验
        ohlc_consistency = self._check_ohlc_consistency(
            base_data,
            period_data
        )
        
        # 技术指标一致性检验
        indicator_consistency = self._check_indicator_consistency(
            base_data,
            period_data
        )
        
        # 时间对齐检验
        time_alignment = self._check_time_alignment(
            base_data,
            period_data
        )
        
        return {
            'price_consistency': price_consistency,
            'volume_consistency': volume_consistency,
            'ohlc_consistency': ohlc_consistency,
            'indicator_consistency': indicator_consistency,
            'time_alignment': time_alignment,
            'period': period
        }
    
    def _check_price_consistency(self,
                               base_data: pd.DataFrame,
                               period_data: pd.DataFrame) -> Dict[str, float]:
        """
        检查价格一致性
        """
        # 重采样基准数据到目标周期
        resampled_base = self._resample_data(base_data, period_data.index)
        
        # 计算价格相关性
        price_correlation = self._calculate_price_correlation(
            resampled_base['close'],
            period_data['close']
        )
        
        # 计算价格偏离度
        price_deviation = self._calculate_price_deviation(
            resampled_base['close'],
            period_data['close']
        )
        
        # 进行统计显著性检验
        significance = self._perform_statistical_test(
            resampled_base['close'],
            period_data['close']
        )
        
        return {
            'correlation': price_correlation,
            'deviation': price_deviation,
            'significance': significance,
            'is_consistent': (
                price_correlation > self.config.correlation_threshold and
                price_deviation < self.config.deviation_threshold and
                significance['p_value'] < self.config.significance_level
            )
        }
    
    def _check_volume_consistency(self,
                                base_data: pd.DataFrame,
                                period_data: pd.DataFrame) -> Dict[str, float]:
        """
        检查成交量一致性
        """
        # 计算成交量累计值
        base_volume = base_data['volume'].resample(period_data.index.freq).sum()
        
        # 计算成交量比例
        volume_ratio = (period_data['volume'] / base_volume).dropna()
        
        # 检查成交量偏差
        volume_deviation = abs(1 - volume_ratio.mean())
        
        # 检查成交量分布一致性
        distribution_consistency = self._check_volume_distribution(
            base_volume,
            period_data['volume']
        )
        
        return {
            'volume_ratio': volume_ratio.mean(),
            'volume_deviation': volume_deviation,
            'distribution_consistency': distribution_consistency,
            'is_consistent': (
                volume_deviation < (1 - self.config.volume_consistency_threshold)
            )
        }
    
    def _check_ohlc_consistency(self,
                              base_data: pd.DataFrame,
                              period_data: pd.DataFrame) -> Dict[str, float]:
        """
        检查OHLC一致性
        """
        resampled_base = self._resample_data(base_data, period_data.index)
        
        # 检查开盘价一致性
        open_consistency = (
            resampled_base['open'].iloc[0] == period_data['open'].iloc[0]
        )
        
        # 检查最高价一致性
        high_consistency = np.allclose(
            resampled_base['high'].max(),
            period_data['high'].max(),
            rtol=1e-5
        )
        
        # 检查最低价一致性
        low_consistency = np.allclose(
            resampled_base['low'].min(),
            period_data['low'].min(),
            rtol=1e-5
        )
        
        # 检查收盘价一致性
        close_consistency = (
            resampled_base['close'].iloc[-1] == period_data['close'].iloc[-1]
        )
        
        return {
            'open_consistency': float(open_consistency),
            'high_consistency': float(high_consistency),
            'low_consistency': float(low_consistency),
            'close_consistency': float(close_consistency),
            'is_consistent': all([
                open_consistency,
                high_consistency,
                low_consistency,
                close_consistency
            ])
        }
    
    def _check_indicator_consistency(self,
                                   base_data: pd.DataFrame,
                                   period_data: pd.DataFrame) -> Dict[str, float]:
        """
        检查技术指标一致性
        """
        # 获取共同的技术指标
        common_indicators = set(base_data.columns) & set(period_data.columns)
        indicator_results = {}
        
        for indicator in common_indicators:
            if indicator not in ['open', 'high', 'low', 'close', 'volume']:
                # 重采样基准数据的指标
                resampled_indicator = self._resample_indicator(
                    base_data[indicator],
                    period_data.index
                )
                
                # 计算指标相关性
                correlation = self._calculate_indicator_correlation(
                    resampled_indicator,
                    period_data[indicator]
                )
                
                # 计算指标偏离度
                deviation = self._calculate_indicator_deviation(
                    resampled_indicator,
                    period_data[indicator]
                )
                
                indicator_results[indicator] = {
                    'correlation': correlation,
                    'deviation': deviation,
                    'is_consistent': (
                        correlation > self.config.correlation_threshold and
                        deviation < self.config.deviation_threshold
                    )
                }
        
        return {
            'indicators': indicator_results,
            'overall_consistency': np.mean([
                result['is_consistent']
                for result in indicator_results.values()
            ])
        }
    
    def _check_time_alignment(self,
                            base_data: pd.DataFrame,
                            period_data: pd.DataFrame) -> Dict[str, bool]:
        """
        检查时间对齐
        """
        # 检查时间戳对齐
        time_aligned = self._check_timestamp_alignment(
            base_data.index,
            period_data.index
        )
        
        # 检查时间间隔一致性
        interval_consistent = self._check_interval_consistency(
            base_data.index,
            period_data.index
        )
        
        # 检查时区一致性
        timezone_consistent = self._check_timezone_consistency(
            base_data.index,
            period_data.index
        )
        
        return {
            'time_aligned': time_aligned,
            'interval_consistent': interval_consistent,
            'timezone_consistent': timezone_consistent,
            'is_consistent': all([
                time_aligned,
                interval_consistent,
                timezone_consistent
            ])
        }
    
    def _calculate_overall_consistency(self,
                                    consistency_results: Dict[str, Dict[str, Any]]
                                    ) -> float:
        """
        计算整体一致性得分
        """
        scores = []
        weights = []
        
        for period, results in consistency_results.items():
            # 计算各个维度的得分
            price_score = float(results['price_consistency']['is_consistent'])
            volume_score = float(results['volume_consistency']['is_consistent'])
            ohlc_score = float(results['ohlc_consistency']['is_consistent'])
            indicator_score = results['indicator_consistency']['overall_consistency']
            time_score = float(results['time_alignment']['is_consistent'])
            
            # 计算周期权重
            period_weight = self._calculate_period_weight(period)
            
            # 计算加权得分
            period_score = np.mean([
                price_score * 0.3,
                volume_score * 0.2,
                ohlc_score * 0.2,
                indicator_score * 0.2,
                time_score * 0.1
            ])
            
            scores.append(period_score)
            weights.append(period_weight)
        
        # 计算加权平均分
        if weights:
            return np.average(scores, weights=weights)
        return 0.0
    
    def _detect_consistency_anomalies(self,
                                    consistency_results: Dict[str, Dict[str, Any]]
                                    ) -> List[Dict[str, Any]]:
        """
        检测一致性异常
        """
        anomalies = []
        
        for period, results in consistency_results.items():
            # 检查价格异常
            if not results['price_consistency']['is_consistent']:
                anomalies.append({
                    'period': period,
                    'type': 'price_inconsistency',
                    'severity': 'high',
                    'details': results['price_consistency']
                })
            
            # 检查成交量异常
            if not results['volume_consistency']['is_consistent']:
                anomalies.append({
                    'period': period,
                    'type': 'volume_inconsistency',
                    'severity': 'medium',
                    'details': results['volume_consistency']
                })
            
            # 检查OHLC异常
            if not results['ohlc_consistency']['is_consistent']:
                anomalies.append({
                    'period': period,
                    'type': 'ohlc_inconsistency',
                    'severity': 'high',
                    'details': results['ohlc_consistency']
                })
            
            # 检查时间对齐异常
            if not results['time_alignment']['is_consistent']:
                anomalies.append({
                    'period': period,
                    'type': 'time_misalignment',
                    'severity': 'critical',
                    'details': results['time_alignment']
                })
        
        return anomalies
    
    def _generate_consistency_report(self,
                                   consistency_results: Dict[str, Dict[str, Any]],
                                   overall_score: float,
                                   anomalies: List[Dict[str, Any]]
                                   ) -> Dict[str, Any]:
        """
        生成一致性检验报告
        """
        return {
            'timestamp': pd.Timestamp.now(),
            'overall_consistency_score': overall_score,
            'period_results': consistency_results,
            'anomalies': anomalies,
            'status': 'passed' if overall_score > 0.9 else 'failed',
            'recommendations': self._generate_recommendations(
                consistency_results,
                anomalies
            ),
            'metadata': {
                'base_period': self.config.base_period,
                'check_periods': self.config.check_periods,
                'config_thresholds': {
                    'correlation': self.config.correlation_threshold,
                    'deviation': self.config.deviation_threshold,
                    'volume': self.config.volume_consistency_threshold
                }
            }
        }

def build_consistency_checker(config: ConsistencyCheckConfig) -> CrossPeriodConsistencyChecker:
    """
    构建一致性检验器
    """
    return CrossPeriodConsistencyChecker(config)