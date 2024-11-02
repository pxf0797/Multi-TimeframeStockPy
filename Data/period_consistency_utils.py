# data/period_consistency_utils.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mutual_info_score
from scipy import stats

class ConsistencyUtils:
    """
    一致性检验工具类
    """
    @staticmethod
    def _resample_data(data: pd.DataFrame,
                      target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        重采样数据到目标周期
        """
        rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        resampled = data.resample(target_index.freq).agg(rules)
        resampled = resampled.reindex(target_index)
        
        return resampled
    
    @staticmethod
    def _calculate_price_correlation(series1: pd.Series,
                                  series2: pd.Series) -> float:
        """
        计算价格序列相关性
        """
        # 清理数据
        clean_data = pd.concat([series1, series2], axis=1).dropna()
        if len(clean_data) < 2:
            return 0.0
            
        # 计算Spearman相关系数
        correlation = stats.spearmanr(clean_data.iloc[:, 0],
                                    clean_data.iloc[:, 1])[0]
        return correlation
    
    @staticmethod
    def _calculate_price_deviation(series1: pd.Series,
                                series2: pd.Series) -> float:
        """
        计算价格偏离度
        """
        # 标准化价格
        norm1 = (series1 - series1.mean()) / series1.std()
        norm2 = (series2 - series2.mean()) / series2.std()
        
        # 计算均方根误差
        deviation = np.sqrt(np.mean((norm1 - norm2).dropna() ** 2))
        return deviation
    
    @staticmethod
    def _perform_statistical_test(series1: pd.Series,
                               series2: pd.Series) -> Dict[str, float]:
        """
        进行统计显著性检验
        """
        # Kolmogorov-Smirnov检验
        ks_stat, ks_pvalue = stats.ks_2samp(series1.dropna(), series2.dropna())
        
        # Mann-Whitney U检验
        mw_stat, mw_pvalue = stats.mannwhitneyu(series1.dropna(), 
                                               series2.dropna(),
                                               alternative='two-sided')
        
        return {
            'ks_statistic': ks_stat,
            'ks_p_value': ks_pvalue,
            'mw_statistic': mw_stat,
            'mw_p_value': mw_pvalue,
            'significant': (ks_pvalue < 0.05) and (mw_pvalue < 0.05)
        }
    
    @staticmethod
    def _check_volume_distribution(base_volume: pd.Series,
                                period_volume: pd.Series) -> float:
        """
        检查成交量分布一致性
        """
        # 使用互信息评分
        bins = min(len(base_volume) // 10, 50)  # 动态设置分箱数
        
        base_hist = np.histogram(base_volume, bins=bins)[0]
        period_hist = np.histogram(period_volume, bins=bins)[0]
        
        # 归一化直方图
        base_hist = base_hist / base_hist.sum()
        period_hist = period_hist / period_hist.sum()
        
        # 计算互信息分数
        mi_score = mutual_info_score(base_hist, period_hist)
        
        return mi_score
    
    @staticmethod
    def _resample_indicator(indicator: pd.Series,
                         target_index: pd.DatetimeIndex) -> pd.Series:
        """
        重采样技术指标
        """
        # 对于不同类型的指标使用不同的重采样方法
        if indicator.name.startswith('MA') or indicator.name.startswith('EMA'):
            # 移动平均类指标使用last
            resampled = indicator.resample(target_index.freq).last()
        elif indicator.name.startswith('RSI') or indicator.name.startswith('MACD'):
            # 震荡类指标使用mean
            resampled = indicator.resample(target_index.freq).mean()
        elif indicator.name.startswith('VOL'):
            # 成交量类指标使用sum
            resampled = indicator.resample(target_index.freq).sum()
        else:
            # 其他指标默认使用last
            resampled = indicator.resample(target_index.freq).last()
            
        return resampled
    
    @staticmethod
    def _calculate_indicator_correlation(indicator1: pd.Series,
                                     indicator2: pd.Series) -> float:
        """
        计算指标相关性
        """
        # 使用Pearson相关系数
        correlation = indicator1.corr(indicator2)
        return correlation if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def _calculate_indicator_deviation(indicator1: pd.Series,
                                   indicator2: pd.Series) -> float:
        """
        计算指标偏离度
        """
        # 归一化处理
        norm1 = (indicator1 - indicator1.min()) / (indicator1.max() - indicator1.min())
        norm2 = (indicator2 - indicator2.min()) / (indicator2.max() - indicator2.min())
        
        # 计算平均绝对偏差
        deviation = np.mean(np.abs(norm1 - norm2))
        return deviation
    
    @staticmethod
    def _check_timestamp_alignment(index1: pd.DatetimeIndex,
                                index2: pd.DatetimeIndex) -> bool:
        """
        检查时间戳对齐
        """
        # 检查起止时间对齐
        start_aligned = index1[0].floor('D') == index2[0].floor('D')
        end_aligned = index1[-1].floor('D') == index2[-1].floor('D')
        
        return start_aligned and end_aligned
    
    @staticmethod
    def _check_interval_consistency(index1: pd.DatetimeIndex,
                                 index2: pd.DatetimeIndex) -> bool:
        """
        检查时间间隔一致性
        """
        # 计算时间间隔
        interval1 = index1[1] - index1[0]
        interval2 = index2[1] - index2[0]
        
        # 检查间隔比例是否为整数
        ratio = interval2.total_seconds() / interval1.total_seconds()
        return abs(ratio - round(ratio)) < 1e-6
    
    @staticmethod
    def _check_timezone_consistency(index1: pd.DatetimeIndex,
                                 index2: pd.DatetimeIndex) -> bool:
        """
        检查时区一致性
        """
        return index1.tz == index2.tz
    
    @staticmethod
    def _calculate_period_weight(period: str) -> float:
        """
        计算周期权重
        """
        # 基于周期长度计算权重
        period_weights = {
            '5min': 1.0,
            '15min': 0.9,
            '1h': 0.8,
            '4h': 0.7,
            '1d': 0.6,
            '1w': 0.5,
            '1M': 0.4
        }
        return period_weights.get(period, 0.5)

class ConsistencyAnalyzer:
    """
    一致性分析器
    用于分析和可视化一致性检验结果
    """
    @staticmethod
    def analyze_consistency_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析一致性检验历史
        """
        if not history:
            return {}
            
        # 提取历史得分
        scores = [h['overall_score'] for h in history]
        
        # 计算统计指标
        analysis = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'trend': np.polyfit(range(len(scores)), scores, 1)[0],
            'stability': 1 - (np.std(scores) / np.mean(scores))
        }
        
        # 分析异常模式
        analysis['anomaly_patterns'] = ConsistencyAnalyzer._analyze_anomaly_patterns(history)
        
        # 分析周期表现
        analysis['period_performance'] = ConsistencyAnalyzer._analyze_period_performance(history)
        
        return analysis
    
    @staticmethod
    def _analyze_anomaly_patterns(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析异常模式
        """
        patterns = defaultdict(list)
        
        for record in history:
            for anomaly in record['anomalies']:
                patterns[anomaly['type']].append({
                    'timestamp': record['timestamp'],
                    'period': anomaly['period'],
                    'severity': anomaly['severity']
                })
        
        return {
            anomaly_type: {
                'count': len(occurrences),
                'most_affected_period': Counter(
                    o['period'] for o in occurrences
                ).most_common(1)[0][0],
                'severity_distribution': Counter(
                    o['severity'] for o in occurrences
                )
            }
            for anomaly_type, occurrences in patterns.items()
        }
    
    @staticmethod
    def _analyze_period_performance(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析各周期表现
        """
        period_scores = defaultdict(list)
        
        for record in history:
            for period, results in record['period_results'].items():
                period_scores[period].append({
                    'price_consistency': results['price_consistency']['is_consistent'],
                    'volume_consistency': results['volume_consistency']['is_consistent'],
                    'ohlc_consistency': results['ohlc_consistency']['is_consistent'],
                    'time_alignment': results['time_alignment']['is_consistent']
                })
        
        return {
            period: {
                'overall_reliability': np.mean([
                    all(s.values()) for s in scores
                ]),
                'dimension_reliability': {
                    dim: np.mean([s[dim] for s in scores])
                    for dim in scores[0].keys()
                }
            }
            for period, scores in period_scores.items()
        }
    
    @staticmethod
    def generate_recommendations(analysis_results: Dict[str, Any]) -> List[str]:
        """
        生成改进建议
        """
        recommendations = []
        
        # 基于整体得分的建议
        if analysis_results['mean_score'] < 0.9:
            recommendations.append(
                "建议提高整体数据一致性，当前平均得分较低"
            )
        
        # 基于稳定性的建议
        if analysis_results['stability'] < 0.8:
            recommendations.append(
                "数据一致性波动较大，建议检查数据处理流程的稳定性"
            )
        
        # 基于异常模式的建议
        for anomaly_type, pattern in analysis_results['anomaly_patterns'].items():
            if pattern['count'] > len(analysis_results['history']) * 0.1:
                recommendations.append(
                    f"'{anomaly_type}' 异常频繁发生于 {pattern['most_affected_period']} 周期，"
                    f"建议重点关注该周期的数据处理"
                )
        
        # 基于周期表现的建议
        for period, performance in analysis_results['period_performance'].items():
            if performance['overall_reliability'] < 0.85:
                weak_dimensions = [
                    dim for dim, rel in performance['dimension_reliability'].items()
                    if rel < 0.85
                ]
                if weak_dimensions:
                    recommendations.append(
                        f"{period} 周期的 {', '.join(weak_dimensions)} 维度一致性较弱，"
                        f"建议优化相关数据处理"
                    )
        
        return recommendations

def create_consistency_utils() -> ConsistencyUtils:
    """
    创建一致性检验工具实例
    """
    return ConsistencyUtils()

def create_consistency_analyzer() -> ConsistencyAnalyzer:
    """
    创建一致性分析器实例
    """
    return ConsistencyAnalyzer()