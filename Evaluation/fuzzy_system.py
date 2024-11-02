# evaluation/fuzzy_system.py

import numpy as np
from typing import Dict, List, Tuple, Optional
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from dataclasses import dataclass

@dataclass
class FuzzyConfig:
    """模糊系统配置"""
    # 评分范围
    score_range: Tuple[float, float] = (0.0, 1.0)
    
    # 趋势强度阈值
    trend_thresholds: Dict[str, float] = None
    
    # 波动率阈值
    volatility_thresholds: Dict[str, float] = None
    
    # 评分权重
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.trend_thresholds is None:
            self.trend_thresholds = {
                'weak': 0.3,
                'medium': 0.5,
                'strong': 0.7
            }
            
        if self.volatility_thresholds is None:
            self.volatility_thresholds = {
                'low': 0.2,
                'medium': 0.4,
                'high': 0.6
            }
            
        if self.weights is None:
            self.weights = {
                'trend': 0.4,
                'volatility': 0.3,
                'consistency': 0.3
            }

class FuzzyEvaluationSystem:
    """
    完整的模糊逻辑评价系统
    """
    def __init__(self, config: FuzzyConfig):
        self.config = config
        self.setup_fuzzy_system()
        
    def setup_fuzzy_system(self):
        """
        设置模糊逻辑系统
        """
        # 创建输入变量
        self.trend = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'trend')
        self.volatility = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'volatility')
        self.consistency = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'consistency')
        self.volume = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'volume')
        self.price_momentum = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'price_momentum')
        
        # 创建输出变量
        self.score = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'score')
        
        # 定义模糊集
        self._setup_fuzzy_sets()
        
        # 定义规则
        self.rules = self._setup_rules()
        
        # 创建控制系统
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
    def _setup_fuzzy_sets(self):
        """
        设置模糊集
        """
        # 趋势的模糊集
        self.trend['strong_down'] = fuzz.trimf(self.trend.universe, [-1, -1, -0.6])
        self.trend['down'] = fuzz.trimf(self.trend.universe, [-0.8, -0.4, 0])
        self.trend['neutral'] = fuzz.trimf(self.trend.universe, [-0.2, 0, 0.2])
        self.trend['up'] = fuzz.trimf(self.trend.universe, [0, 0.4, 0.8])
        self.trend['strong_up'] = fuzz.trimf(self.trend.universe, [0.6, 1, 1])
        
        # 波动率的模糊集
        self.volatility['very_low'] = fuzz.trimf(self.volatility.universe, [0, 0, 0.2])
        self.volatility['low'] = fuzz.trimf(self.volatility.universe, [0.1, 0.3, 0.5])
        self.volatility['medium'] = fuzz.trimf(self.volatility.universe, [0.3, 0.5, 0.7])
        self.volatility['high'] = fuzz.trimf(self.volatility.universe, [0.5, 0.7, 0.9])
        self.volatility['very_high'] = fuzz.trimf(self.volatility.universe, [0.7, 1, 1])
        
        # 一致性的模糊集
        self.consistency['low'] = fuzz.trimf(self.consistency.universe, [0, 0, 0.4])
        self.consistency['medium'] = fuzz.trimf(self.consistency.universe, [0.3, 0.5, 0.7])
        self.consistency['high'] = fuzz.trimf(self.consistency.universe, [0.6, 1, 1])
        
        # 成交量的模糊集
        self.volume['low'] = fuzz.trimf(self.volume.universe, [0, 0, 0.4])
        self.volume['medium'] = fuzz.trimf(self.volume.universe, [0.3, 0.5, 0.7])
        self.volume['high'] = fuzz.trimf(self.volume.universe, [0.6, 1, 1])
        
        # 价格动量的模糊集
        self.price_momentum['strong_negative'] = fuzz.trimf(self.price_momentum.universe, [-1, -1, -0.6])
        self.price_momentum['negative'] = fuzz.trimf(self.price_momentum.universe, [-0.8, -0.4, 0])
        self.price_momentum['neutral'] = fuzz.trimf(self.price_momentum.universe, [-0.2, 0, 0.2])
        self.price_momentum['positive'] = fuzz.trimf(self.price_momentum.universe, [0, 0.4, 0.8])
        self.price_momentum['strong_positive'] = fuzz.trimf(self.price_momentum.universe, [0.6, 1, 1])
        
        # 评分的模糊集
        self.score['very_low'] = fuzz.trimf(self.score.universe, [0, 0, 0.2])
        self.score['low'] = fuzz.trimf(self.score.universe, [0.1, 0.3, 0.5])
        self.score['medium'] = fuzz.trimf(self.score.universe, [0.3, 0.5, 0.7])
        self.score['high'] = fuzz.trimf(self.score.universe, [0.5, 0.7, 0.9])
        self.score['very_high'] = fuzz.trimf(self.score.universe, [0.7, 1, 1])
        
    def _setup_rules(self) -> List[ctrl.Rule]:
        """
        设置完整的规则集
        """
        rules = [
            # 强趋势规则
            ctrl.Rule(
                self.trend['strong_up'] & self.consistency['high'] & 
                self.volatility['low'] & self.volume['high'] & 
                self.price_momentum['positive'],
                self.score['very_high']
            ),
            
            ctrl.Rule(
                self.trend['strong_down'] & self.consistency['high'] & 
                self.volatility['low'] & self.volume['high'] & 
                self.price_momentum['negative'],
                self.score['very_low']
            ),
            
            # 震荡市规则
            ctrl.Rule(
                self.trend['neutral'] & self.volatility['high'] & 
                self.consistency['low'],
                self.score['medium']
            ),
            
            # 趋势转折规则
            ctrl.Rule(
                self.trend['up'] & self.volume['high'] & 
                self.price_momentum['strong_positive'] & 
                self.volatility['medium'],
                self.score['high']
            ),
            
            ctrl.Rule(
                self.trend['down'] & self.volume['high'] & 
                self.price_momentum['strong_negative'] & 
                self.volatility['medium'],
                self.score['low']
            ),
            
            # 高波动规则
            ctrl.Rule(
                self.volatility['very_high'] & self.consistency['low'],
                self.score['low']
            ),
            
            # 低波动规则
            ctrl.Rule(
                self.volatility['very_low'] & self.consistency['high'] & 
                self.trend['neutral'],
                self.score['medium']
            ),
            
            # 成交量规则
            ctrl.Rule(
                self.volume['high'] & self.trend['up'] & 
                self.price_momentum['positive'],
                self.score['high']
            ),
            
            ctrl.Rule(
                self.volume['low'] & self.trend['neutral'],
                self.score['medium']
            ),
            
            # 动量规则
            ctrl.Rule(
                self.price_momentum['strong_positive'] & self.volume['high'] & 
                self.consistency['high'],
                self.score['very_high']
            ),
            
            ctrl.Rule(
                self.price_momentum['strong_negative'] & self.volume['high'] & 
                self.consistency['high'],
                self.score['very_low']
            )
        ]
        
        return rules
    
    def evaluate(self, market_state: Dict[str, float]) -> float:
        """
        评估市场状态
        """
        try:
            # 设置输入
            self.simulation.input['trend'] = market_state.get('trend', 0)
            self.simulation.input['volatility'] = market_state.get('volatility', 0)
            self.simulation.input['consistency'] = market_state.get('consistency', 0)
            self.simulation.input['volume'] = market_state.get('volume', 0)
            self.simulation.input['price_momentum'] = market_state.get('price_momentum', 0)
            
            # 计算评分
            self.simulation.compute()
            return self.simulation.output['score']
            
        except Exception as e:
            print(f"Fuzzy evaluation error: {e}")
            return 0.5  # 默认中性评分

class MultiScaleEvaluator:
    """
    多尺度加权评价系统
    """
    def __init__(self, config: FuzzyConfig):
        self.config = config
        self.fuzzy_system = FuzzyEvaluationSystem(config)
        self.period_weights = self._initialize_period_weights()
        
    def _initialize_period_weights(self) -> Dict[str, float]:
        """
        初始化各周期权重
        """
        return {
            '5min': 0.15,
            '15min': 0.15,
            '1h': 0.15,
            '1d': 0.25,
            '1w': 0.20,
            '1M': 0.10
        }
    
    def evaluate_multi_period(self,
                            period_states: Dict[str, Dict[str, float]],
                            market_conditions: Dict[str, float]) -> Dict[str, float]:
        """
        多周期评价
        """
        # 计算各周期评分
        period_scores = {}
        for period, state in period_states.items():
            score = self.fuzzy_system.evaluate(state)
            period_scores[period] = score
        
        # 调整权重
        adjusted_weights = self._adjust_weights(market_conditions)
        
        # 计算加权总分
        total_score = 0
        weight_sum = 0
        for period, score in period_scores.items():
            weight = adjusted_weights.get(period, 0)
            total_score += score * weight
            weight_sum += weight
        
        final_score = total_score / weight_sum if weight_sum > 0 else 0.5
        
        return {
            'total_score': final_score,
            'period_scores': period_scores,
            'weights': adjusted_weights
        }
    
    def _adjust_weights(self, 
                       market_conditions: Dict[str, float]) -> Dict[str, float]:
        """
        根据市场条件调整权重
        """
        adjusted_weights = self.period_weights.copy()
        volatility = market_conditions.get('volatility', 0)
        trend_strength = market_conditions.get('trend_strength', 0)
        
        # 高波动时增加短周期权重
        if volatility > self.config.volatility_thresholds['high']:
            for period in ['5min', '15min', '1h']:
                adjusted_weights[period] *= 1.5
            for period in ['1d', '1w', '1M']:
                adjusted_weights[period] *= 0.5
                
        # 强趋势时增加长周期权重
        elif trend_strength > self.config.trend_thresholds['strong']:
            for period in ['1d', '1w', '1M']:
                adjusted_weights[period] *= 1.5
            for period in ['5min', '15min', '1h']:
                adjusted_weights[period] *= 0.5
        
        # 归一化权重
        weight_sum = sum(adjusted_weights.values())
        return {k: v/weight_sum for k, v in adjusted_weights.items()}

class EvaluationAnalyzer:
    """
    评价结果分析器
    """
    def __init__(self):
        self.evaluation_history = []
        
    def analyze_evaluation(self, 
                         evaluation_result: Dict[str, float]) -> Dict[str, float]:
        """
        分析评价结果
        """
        self.evaluation_history.append(evaluation_result)
        
        # 计算评分稳定性
        if len(self.evaluation_history) > 1:
            score_changes = np.diff([e['total_score'] 
                                   for e in self.evaluation_history])
            score_volatility = np.std(score_changes)
        else:
            score_volatility = 0
            
        # 计算周期一致性
        period_scores = evaluation_result['period_scores']
        score_consistency = np.std(list(period_scores.values()))
        
        return {
            'score_volatility': score_volatility,
            'score_consistency': 1 - score_consistency,  # 转换为一致性指标
            'evaluation_confidence': self._calculate_confidence(evaluation_result)
        }
    
# evaluation/evaluation_integration.py

    def _calculate_confidence(self, 
                            evaluation_result: Dict[str, float]) -> float:
        """
        计算评价置信度
        """
        period_scores = evaluation_result['period_scores']
        weights = evaluation_result['weights']
        
        # 计算加权标准差
        weighted_scores = [score * weights[period] 
                         for period, score in period_scores.items()]
        weighted_std = np.std(weighted_scores)
        
        # 置信度与标准差成反比
        confidence = 1 / (1 + weighted_std)
        
        return confidence
    
    def get_evaluation_statistics(self) -> Dict[str, float]:
        """
        获取评价统计信息
        """
        if not self.evaluation_history:
            return {}
        
        total_scores = [e['total_score'] for e in self.evaluation_history]
        
        return {
            'mean_score': np.mean(total_scores),
            'score_std': np.std(total_scores),
            'min_score': np.min(total_scores),
            'max_score': np.max(total_scores),
            'score_trend': (total_scores[-1] - total_scores[0]) / len(total_scores)
            if len(total_scores) > 1 else 0
        }

class IntegratedEvaluationSystem:
    """
    集成评价系统
    """
    def __init__(self, config: FuzzyConfig):
        self.config = config
        self.fuzzy_evaluator = FuzzyEvaluationSystem(config)
        self.multi_scale_evaluator = MultiScaleEvaluator(config)
        self.analyzer = EvaluationAnalyzer()
        
    def evaluate_market_state(self,
                            market_data: Dict[str, Dict[str, float]],
                            extra_info: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        综合评价市场状态
        """
        # 计算基础市场条件
        market_conditions = self._calculate_market_conditions(market_data)
        
        # 合并额外信息
        if extra_info:
            market_conditions.update(extra_info)
            
        # 多周期评价
        multi_scale_result = self.multi_scale_evaluator.evaluate_multi_period(
            market_data, market_conditions)
        
        # 分析评价结果
        analysis_result = self.analyzer.analyze_evaluation(multi_scale_result)
        
        return {
            **multi_scale_result,
            **analysis_result,
            'market_conditions': market_conditions
        }
        
    def _calculate_market_conditions(self,
                                   market_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        计算市场状况
        """
        conditions = {}
        
        # 计算综合波动率
        volatilities = [state.get('volatility', 0) for state in market_data.values()]
        conditions['volatility'] = np.mean(volatilities)
        
        # 计算综合趋势强度
        trend_strengths = [state.get('trend', 0) for state in market_data.values()]
        conditions['trend_strength'] = np.mean(trend_strengths)
        
        # 计算周期一致性
        conditions['period_consistency'] = 1 - np.std(trend_strengths)
        
        return conditions

class EvaluationManager:
    """
    评价系统管理器
    """
    def __init__(self, config: FuzzyConfig):
        self.evaluation_system = IntegratedEvaluationSystem(config)
        self.evaluation_history = []
        self.alerts = []
        
    def update(self,
              market_data: Dict[str, Dict[str, float]],
              extra_info: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        更新评价结果
        """
        # 获取评价结果
        evaluation_result = self.evaluation_system.evaluate_market_state(
            market_data, extra_info)
        
        # 保存历史记录
        self.evaluation_history.append(evaluation_result)
        
        # 检查是否需要发出警报
        self._check_alerts(evaluation_result)
        
        return evaluation_result
    
    def _check_alerts(self, evaluation_result: Dict[str, float]):
        """
        检查是否需要发出警报
        """
        alerts = []
        
        # 检查评分突变
        if len(self.evaluation_history) > 1:
            score_change = abs(evaluation_result['total_score'] - 
                             self.evaluation_history[-2]['total_score'])
            if score_change > 0.3:
                alerts.append({
                    'type': 'score_change',
                    'severity': 'high',
                    'message': f'Significant score change detected: {score_change:.2f}'
                })
        
        # 检查周期不一致
        period_scores = evaluation_result['period_scores']
        score_std = np.std(list(period_scores.values()))
        if score_std > 0.3:
            alerts.append({
                'type': 'period_inconsistency',
                'severity': 'medium',
                'message': f'High period score inconsistency: {score_std:.2f}'
            })
        
        # 检查市场异常
        market_conditions = evaluation_result['market_conditions']
        if market_conditions['volatility'] > 0.5:
            alerts.append({
                'type': 'high_volatility',
                'severity': 'high',
                'message': f'High market volatility: {market_conditions["volatility"]:.2f}'
            })
        
        self.alerts.extend(alerts)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取评价系统总结
        """
        if not self.evaluation_history:
            return {}
        
        recent_evaluations = self.evaluation_history[-100:]  # 最近100次评价
        
        return {
            'evaluation_stats': self.evaluation_system.analyzer.get_evaluation_statistics(),
            'recent_alerts': self.alerts[-10:],  # 最近10个警报
            'period_performance': self._calculate_period_performance(recent_evaluations),
            'system_health': self._assess_system_health(recent_evaluations)
        }
    
    def _calculate_period_performance(self,
                                    evaluations: List[Dict[str, float]]) -> Dict[str, float]:
        """
        计算各周期表现
        """
        period_scores = defaultdict(list)
        for eval_result in evaluations:
            for period, score in eval_result['period_scores'].items():
                period_scores[period].append(score)
        
        return {
            period: {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'trend': (scores[-1] - scores[0]) / len(scores) if len(scores) > 1 else 0
            }
            for period, scores in period_scores.items()
        }
    
    def _assess_system_health(self,
                            evaluations: List[Dict[str, float]]) -> Dict[str, float]:
        """
        评估系统健康状况
        """
        total_scores = [e['total_score'] for e in evaluations]
        confidence_scores = [e.get('evaluation_confidence', 0) for e in evaluations]
        
        return {
            'stability': 1 - np.std(total_scores),
            'confidence': np.mean(confidence_scores),
            'alert_frequency': len(self.alerts) / len(evaluations) if evaluations else 0,
            'system_reliability': self._calculate_reliability(evaluations)
        }
    
    def _calculate_reliability(self,
                             evaluations: List[Dict[str, float]]) -> float:
        """
        计算系统可靠性
        """
        if not evaluations:
            return 1.0
            
        # 检查评分连续性
        score_changes = np.diff([e['total_score'] for e in evaluations])
        large_changes = np.sum(np.abs(score_changes) > 0.3)
        
        # 检查置信度稳定性
        confidence_values = [e.get('evaluation_confidence', 0) for e in evaluations]
        confidence_stability = 1 - np.std(confidence_values)
        
        # 计算总体可靠性
        continuity_score = 1 - (large_changes / len(evaluations))
        reliability = 0.6 * continuity_score + 0.4 * confidence_stability
        
        return reliability

def build_evaluation_system(config: FuzzyConfig) -> EvaluationManager:
    """
    构建评价系统
    """
    return EvaluationManager(config)