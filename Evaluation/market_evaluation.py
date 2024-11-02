# evaluation/market_evaluation.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import pandas as pd
from scipy import stats

@dataclass
class MarketEvaluationConfig:
    """市场评价系统配置"""
    # 时间窗口设置
    short_window: int = 20
    medium_window: int = 60
    long_window: int = 120
    
    # 评分阈值
    volatility_thresholds: Dict[str, float] = None
    trend_thresholds: Dict[str, float] = None
    volume_thresholds: Dict[str, float] = None
    
    # 权重配置
    dimension_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.volatility_thresholds is None:
            self.volatility_thresholds = {
                'low': 0.1,
                'medium': 0.3,
                'high': 0.5
            }
        
        if self.trend_thresholds is None:
            self.trend_thresholds = {
                'weak': 0.2,
                'medium': 0.5,
                'strong': 0.8
            }
            
        if self.volume_thresholds is None:
            self.volume_thresholds = {
                'low': 0.5,
                'medium': 1.0,
                'high': 2.0
            }
            
        if self.dimension_weights is None:
            self.dimension_weights = {
                'price_action': 0.3,
                'market_structure': 0.2,
                'momentum': 0.15,
                'volatility': 0.15,
                'volume_analysis': 0.1,
                'market_breadth': 0.1
            }

class MarketStructureAnalyzer:
    """
    市场结构分析器
    分析价格形态、支撑阻力和趋势结构
    """
    def __init__(self, config: MarketEvaluationConfig):
        self.config = config
        self.price_history = deque(maxlen=config.long_window)
        self.structure_points = []
        
    def analyze_structure(self, 
                         price_data: Dict[str, np.ndarray],
                         technical_indicators: Dict[str, np.ndarray]
                         ) -> Dict[str, float]:
        """
        分析市场结构
        """
        # 更新价格历史
        self.price_history.append(price_data['close'])
        
        if len(self.price_history) < self.config.short_window:
            return {}
            
        # 识别关键价位
        support_resistance = self._identify_support_resistance()
        
        # 分析趋势结构
        trend_structure = self._analyze_trend_structure(technical_indicators)
        
        # 识别价格形态
        price_patterns = self._identify_price_patterns()
        
        # 计算结构强度
        structure_strength = self._calculate_structure_strength(
            support_resistance,
            trend_structure,
            price_patterns
        )
        
        return {
            'support_resistance_levels': support_resistance,
            'trend_structure': trend_structure,
            'price_patterns': price_patterns,
            'structure_strength': structure_strength
        }
        
    def _identify_support_resistance(self) -> Dict[str, float]:
        """
        识别支撑阻力位
        """
        prices = np.array(list(self.price_history))
        
        # 使用核密度估计找出价格聚集区域
        kde = stats.gaussian_kde(prices)
        price_range = np.linspace(min(prices), max(prices), 100)
        density = kde(price_range)
        
        # 找出局部极大值作为潜在的支撑阻力位
        peaks = self._find_peaks(density)
        levels = price_range[peaks]
        
        # 计算各个位置的强度
        strengths = density[peaks]
        
        return {
            'levels': levels.tolist(),
            'strengths': strengths.tolist(),
            'current_distance': self._calculate_level_distance(prices[-1], levels)
        }
        
    def _analyze_trend_structure(self,
                               technical_indicators: Dict[str, np.ndarray]
                               ) -> Dict[str, float]:
        """
        分析趋势结构
        """
        # 分析移动平均线排列
        ma_structure = self._analyze_ma_structure(technical_indicators)
        
        # 分析高低点排列
        swing_structure = self._analyze_swing_structure()
        
        # 分析趋势完整性
        trend_integrity = self._analyze_trend_integrity(
            ma_structure,
            swing_structure
        )
        
        return {
            'ma_structure': ma_structure,
            'swing_structure': swing_structure,
            'trend_integrity': trend_integrity
        }
        
    def _identify_price_patterns(self) -> Dict[str, float]:
        """
        识别价格形态
        """
        prices = np.array(list(self.price_history))
        
        # 识别常见形态
        patterns = {
            'double_top': self._check_double_top(prices),
            'double_bottom': self._check_double_bottom(prices),
            'head_shoulders': self._check_head_shoulders(prices),
            'triangle': self._check_triangle(prices)
        }
        
        # 计算形态可信度
        pattern_confidence = self._calculate_pattern_confidence(patterns)
        
        return {
            'patterns': patterns,
            'confidence': pattern_confidence
        }
        
    def _calculate_structure_strength(self,
                                    support_resistance: Dict[str, float],
                                    trend_structure: Dict[str, float],
                                    price_patterns: Dict[str, float]) -> float:
        """
        计算结构强度
        """
        # 支撑阻力强度
        sr_strength = np.mean(support_resistance['strengths']) if support_resistance['strengths'] else 0
        
        # 趋势结构强度
        trend_strength = trend_structure['trend_integrity']
        
        # 形态强度
        pattern_strength = price_patterns['confidence']
        
        # 综合评分
        return 0.4 * sr_strength + 0.4 * trend_strength + 0.2 * pattern_strength

class MomentumAnalyzer:
    """
    动量分析器
    分析价格动量、趋势强度和背离
    """
    def __init__(self, config: MarketEvaluationConfig):
        self.config = config
        self.momentum_history = deque(maxlen=config.long_window)
        
    def analyze_momentum(self,
                        price_data: Dict[str, np.ndarray],
                        technical_indicators: Dict[str, np.ndarray]
                        ) -> Dict[str, float]:
        """
        分析动量
        """
        # 计算价格动量
        price_momentum = self._calculate_price_momentum(price_data)
        
        # 分析趋势强度
        trend_strength = self._analyze_trend_strength(technical_indicators)
        
        # 检测背离
        divergence = self._detect_divergence(
            price_data,
            technical_indicators
        )
        
        # 计算动量评分
        momentum_score = self._calculate_momentum_score(
            price_momentum,
            trend_strength,
            divergence
        )
        
        return {
            'price_momentum': price_momentum,
            'trend_strength': trend_strength,
            'divergence': divergence,
            'momentum_score': momentum_score
        }
        
    def _calculate_price_momentum(self, 
                                price_data: Dict[str, np.ndarray]
                                ) -> Dict[str, float]:
        """
        计算价格动量
        """
        prices = price_data['close']
        
        # 计算ROC
        roc = self._calculate_roc(prices)
        
        # 计算RSI
        rsi = self._calculate_rsi(prices)
        
        # 计算动量指标
        momentum = self._calculate_momentum_indicator(prices)
        
        return {
            'roc': roc,
            'rsi': rsi,
            'momentum': momentum
        }
        
    def _analyze_trend_strength(self,
                              technical_indicators: Dict[str, np.ndarray]
                              ) -> Dict[str, float]:
        """
        分析趋势强度
        """
        # 分析ADX
        adx_strength = self._analyze_adx(technical_indicators)
        
        # 分析MACD
        macd_strength = self._analyze_macd(technical_indicators)
        
        # 综合评估
        return {
            'adx_strength': adx_strength,
            'macd_strength': macd_strength,
            'composite_strength': 0.5 * adx_strength + 0.5 * macd_strength
        }
        
    def _detect_divergence(self,
                          price_data: Dict[str, np.ndarray],
                          technical_indicators: Dict[str, np.ndarray]
                          ) -> Dict[str, float]:
        """
        检测背离
        """
        prices = price_data['close']
        
        # 检测RSI背离
        rsi_divergence = self._check_rsi_divergence(prices, technical_indicators)
        
        # 检测MACD背离
        macd_divergence = self._check_macd_divergence(prices, technical_indicators)
        
        return {
            'rsi_divergence': rsi_divergence,
            'macd_divergence': macd_divergence,
            'divergence_strength': max(rsi_divergence, macd_divergence)
        }

class VolumeAnalyzer:
    """
    成交量分析器
    分析成交量特征和量价关系
    """
    def __init__(self, config: MarketEvaluationConfig):
        self.config = config
        self.volume_history = deque(maxlen=config.long_window)
        
    def analyze_volume(self,
                      price_data: Dict[str, np.ndarray],
                      volume_data: np.ndarray) -> Dict[str, float]:
        """
        分析成交量
        """
        # 分析成交量趋势
        volume_trend = self._analyze_volume_trend(volume_data)
        
        # 分析量价关系
        price_volume_relation = self._analyze_price_volume_relation(
            price_data,
            volume_data
        )
        
        # 计算成交量强度
        volume_strength = self._calculate_volume_strength(volume_data)
        
        # 检测成交量异常
        volume_anomalies = self._detect_volume_anomalies(volume_data)
        
        return {
            'volume_trend': volume_trend,
            'price_volume_relation': price_volume_relation,
            'volume_strength': volume_strength,
            'volume_anomalies': volume_anomalies
        }
        
    def _analyze_volume_trend(self, volume_data: np.ndarray) -> Dict[str, float]:
        """
        分析成交量趋势
        """
        # 计算成交量移动平均
        volume_ma = self._calculate_volume_ma(volume_data)
        
        # 判断趋势方向
        trend_direction = self._determine_volume_trend(volume_data, volume_ma)
        
        # 计算趋势强度
        trend_strength = self._calculate_volume_trend_strength(
            volume_data,
            volume_ma
        )
        
        return {
            'direction': trend_direction,
            'strength': trend_strength,
            'consistency': self._calculate_trend_consistency(volume_data)
        }
        
    def _analyze_price_volume_relation(self,
                                     price_data: Dict[str, np.ndarray],
                                     volume_data: np.ndarray) -> Dict[str, float]:
        """
        分析量价关系
        """
        # 计算相关性
        correlation = self._calculate_price_volume_correlation(
            price_data['close'],
            volume_data
        )
        
        # 检测量价配合
        confirmation = self._check_volume_confirmation(
            price_data,
            volume_data
        )
        
        # 分析分歧
        divergence = self._analyze_volume_divergence(
            price_data,
            volume_data
        )
        
        return {
            'correlation': correlation,
            'confirmation': confirmation,
            'divergence': divergence
        }

class MarketBreadthAnalyzer:
    """
    市场宽度分析器
    分析市场内部动能和参与度
    """
    def __init__(self, config: MarketEvaluationConfig):
        self.config = config
        
    def analyze_market_breadth(self,
                             market_data: Dict[str, np.ndarray]
                             ) -> Dict[str, float]:
        """
        分析市场宽度
        """
        # 分析上涨下跌家数比
        advance_decline = self._analyze_advance_decline(market_data)
        
        # 分析新高新低
        high_low = self._analyze_new_highs_lows(market_data)
        
        # 计算参与度
        participation = self._calculate_participation(market_data)
        
        # 分析市场动能
        momentum = self._analyze_market_momentum(market_data)
        
        return {
            'advance_decline': advance_decline,
            'high_low': high_low,
            'participation': participation,
            'momentum': momentum
        }

class EnhancedMarketEvaluator:
    """
    增强型市场评价器
    整合各个维度的分析结果
    """
    def __init__(self, config: MarketEvaluationConfig):
        self.config = config
        self.structure_analyzer = MarketStructureAnalyzer(config)
        self.momentum_analyzer = MomentumAnalyzer(config)
        self.volume_analyzer = VolumeAnalyzer(config)
        self.breadth_analyzer = MarketBreadthAnalyzer(config)
        
    def evaluate_market_state(self,
                            market_data: Dict[str, Dict[str, np.ndarray]]
                            ) -> Dict[str, float]:
        """
        评估市场状态
        """
        # 分析市场结构
        structure_analysis = self.structure_analyzer.analyze_structure(
            market_data['price'],
            market_data['indicators']
        )
        
        # 分析动量
        momentum_analysis = self.momentum_analyzer.analyze_momentum(
            market_data['price'],
            market_data['indicators']
        )
        
        # 分析成交量
        volume_analysis = self.volume_analyzer.analyze_volume(
            market_data['price'],
            market_data['volume']
        )
        
        # evaluation/market_evaluation.py (continued)

        # 分析市场宽度
        breadth_analysis = self.breadth_analyzer.analyze_market_breadth(
            market_data['market_breadth']
        )
        
        # 整合各维度分析结果
        evaluation_results = self._integrate_analysis_results(
            structure_analysis=structure_analysis,
            momentum_analysis=momentum_analysis,
            volume_analysis=volume_analysis,
            breadth_analysis=breadth_analysis
        )
        
        # 计算综合评分
        composite_score = self._calculate_composite_score(evaluation_results)
        
        # 生成市场状态描述
        market_state = self._generate_market_state(
            evaluation_results,
            composite_score
        )
        
        return {
            'detailed_analysis': evaluation_results,
            'composite_score': composite_score,
            'market_state': market_state
        }
    
    def _integrate_analysis_results(self,
                                  structure_analysis: Dict[str, float],
                                  momentum_analysis: Dict[str, float],
                                  volume_analysis: Dict[str, float],
                                  breadth_analysis: Dict[str, float]
                                  ) -> Dict[str, float]:
        """
        整合各维度分析结果
        """
        integrated_results = {
            'market_structure': {
                'score': structure_analysis['structure_strength'],
                'components': {
                    'support_resistance': structure_analysis['support_resistance_levels'],
                    'trend_structure': structure_analysis['trend_structure'],
                    'patterns': structure_analysis['price_patterns']
                }
            },
            'momentum': {
                'score': momentum_analysis['momentum_score'],
                'components': {
                    'price_momentum': momentum_analysis['price_momentum'],
                    'trend_strength': momentum_analysis['trend_strength'],
                    'divergence': momentum_analysis['divergence']
                }
            },
            'volume': {
                'score': volume_analysis['volume_strength'],
                'components': {
                    'trend': volume_analysis['volume_trend'],
                    'price_volume_relation': volume_analysis['price_volume_relation'],
                    'anomalies': volume_analysis['volume_anomalies']
                }
            },
            'market_breadth': {
                'score': breadth_analysis['momentum'],
                'components': {
                    'advance_decline': breadth_analysis['advance_decline'],
                    'high_low': breadth_analysis['high_low'],
                    'participation': breadth_analysis['participation']
                }
            }
        }
        
        # 计算各维度得分的置信度
        confidence_scores = self._calculate_confidence_scores(integrated_results)
        integrated_results['confidence_scores'] = confidence_scores
        
        return integrated_results
    
    def _calculate_composite_score(self,
                                 evaluation_results: Dict[str, Dict[str, Any]]
                                 ) -> float:
        """
        计算综合评分
        """
        # 获取各维度得分和置信度
        dimension_scores = {}
        confidence_weights = {}
        
        for dimension, results in evaluation_results.items():
            if dimension != 'confidence_scores':
                dimension_scores[dimension] = results['score']
                confidence_weights[dimension] = evaluation_results['confidence_scores'][dimension]
        
        # 应用维度权重和置信度权重
        weighted_scores = []
        total_weight = 0
        
        for dimension, score in dimension_scores.items():
            weight = (self.config.dimension_weights[dimension] * 
                     confidence_weights[dimension])
            weighted_scores.append(score * weight)
            total_weight += weight
        
        # 计算加权平均分
        if total_weight > 0:
            composite_score = sum(weighted_scores) / total_weight
        else:
            composite_score = 0.5  # 默认中性评分
            
        return composite_score
    
    def _calculate_confidence_scores(self,
                                   integrated_results: Dict[str, Dict[str, Any]]
                                   ) -> Dict[str, float]:
        """
        计算各维度评分的置信度
        """
        confidence_scores = {}
        
        for dimension, results in integrated_results.items():
            if dimension != 'confidence_scores':
                # 基于组件完整性计算置信度
                component_completeness = self._calculate_component_completeness(
                    results['components']
                )
                
                # 基于数据质量计算置信度
                data_quality = self._assess_data_quality(
                    results['components']
                )
                
                # 基于结果一致性计算置信度
                result_consistency = self._assess_result_consistency(
                    results['components']
                )
                
                # 综合置信度
                confidence_scores[dimension] = (
                    0.4 * component_completeness +
                    0.3 * data_quality +
                    0.3 * result_consistency
                )
        
        return confidence_scores
    
    def _generate_market_state(self,
                             evaluation_results: Dict[str, Dict[str, Any]],
                             composite_score: float) -> Dict[str, Any]:
        """
        生成市场状态描述
        """
        # 确定市场阶段
        market_phase = self._determine_market_phase(
            evaluation_results,
            composite_score
        )
        
        # 识别主要驱动因素
        key_drivers = self._identify_key_drivers(evaluation_results)
        
        # 评估风险水平
        risk_assessment = self._assess_risk_level(
            evaluation_results,
            composite_score
        )
        
        # 生成行动建议
        action_suggestions = self._generate_action_suggestions(
            market_phase,
            key_drivers,
            risk_assessment
        )
        
        return {
            'phase': market_phase,
            'key_drivers': key_drivers,
            'risk_assessment': risk_assessment,
            'action_suggestions': action_suggestions,
            'analysis_timestamp': pd.Timestamp.now()
        }
    
    def _determine_market_phase(self,
                              evaluation_results: Dict[str, Dict[str, Any]],
                              composite_score: float) -> Dict[str, Any]:
        """
        确定市场阶段
        """
        # 分析趋势特征
        trend_characteristics = self._analyze_trend_characteristics(
            evaluation_results
        )
        
        # 分析市场情绪
        market_sentiment = self._analyze_market_sentiment(
            evaluation_results,
            composite_score
        )
        
        # 确定市场周期位置
        cycle_position = self._determine_cycle_position(
            trend_characteristics,
            market_sentiment
        )
        
        return {
            'primary_phase': cycle_position['primary_phase'],
            'sub_phase': cycle_position['sub_phase'],
            'characteristics': trend_characteristics,
            'sentiment': market_sentiment,
            'transition_probability': cycle_position['transition_probability']
        }
    
    def _identify_key_drivers(self,
                            evaluation_results: Dict[str, Dict[str, Any]]
                            ) -> List[Dict[str, Any]]:
        """
        识别主要驱动因素
        """
        drivers = []
        
        # 分析各维度的影响力
        for dimension, results in evaluation_results.items():
            if dimension != 'confidence_scores':
                # 计算维度影响力
                impact = self._calculate_dimension_impact(
                    results,
                    evaluation_results['confidence_scores'][dimension]
                )
                
                if impact['significance'] > 0.3:  # 显著性阈值
                    drivers.append({
                        'dimension': dimension,
                        'impact_score': impact['score'],
                        'significance': impact['significance'],
                        'direction': impact['direction'],
                        'key_components': impact['key_components']
                    })
        
        # 按影响力排序
        drivers.sort(key=lambda x: x['significance'], reverse=True)
        
        return drivers
    
    def _assess_risk_level(self,
                          evaluation_results: Dict[str, Dict[str, Any]],
                          composite_score: float) -> Dict[str, Any]:
        """
        评估风险水平
        """
        # 计算基础风险指标
        volatility_risk = self._calculate_volatility_risk(evaluation_results)
        trend_risk = self._calculate_trend_risk(evaluation_results)
        liquidity_risk = self._calculate_liquidity_risk(evaluation_results)
        
        # 评估系统性风险
        systemic_risk = self._assess_systemic_risk(
            evaluation_results,
            composite_score
        )
        
        # 生成风险评估报告
        return {
            'overall_risk_level': (volatility_risk + trend_risk + 
                                 liquidity_risk + systemic_risk) / 4,
            'risk_components': {
                'volatility_risk': volatility_risk,
                'trend_risk': trend_risk,
                'liquidity_risk': liquidity_risk,
                'systemic_risk': systemic_risk
            },
            'risk_alerts': self._generate_risk_alerts(
                volatility_risk,
                trend_risk,
                liquidity_risk,
                systemic_risk
            ),
            'risk_momentum': self._calculate_risk_momentum(evaluation_results)
        }
    
    def _generate_action_suggestions(self,
                                   market_phase: Dict[str, Any],
                                   key_drivers: List[Dict[str, Any]],
                                   risk_assessment: Dict[str, Any]
                                   ) -> List[Dict[str, str]]:
        """
        生成行动建议
        """
        suggestions = []
        
        # 基于市场阶段的建议
        phase_suggestions = self._get_phase_based_suggestions(market_phase)
        suggestions.extend(phase_suggestions)
        
        # 基于关键驱动因素的建议
        driver_suggestions = self._get_driver_based_suggestions(key_drivers)
        suggestions.extend(driver_suggestions)
        
        # 基于风险评估的建议
        risk_suggestions = self._get_risk_based_suggestions(risk_assessment)
        suggestions.extend(risk_suggestions)
        
        # 综合建议优先级排序
        suggestions.sort(key=lambda x: x['priority'], reverse=True)
        
        return suggestions

def build_market_evaluator(config: MarketEvaluationConfig) -> EnhancedMarketEvaluator:
    """
    构建市场评价器
    """
    return EnhancedMarketEvaluator(config)