# evaluation/fuzzy_evaluation.py

import numpy as np
from typing import Dict, List, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyEvaluationSystem:
    """
    模糊逻辑评价系统
    """
    def __init__(self):
        self.setup_fuzzy_system()
        
    def setup_fuzzy_system(self):
        """
        设置模糊逻辑系统
        """
        # 创建模糊变量
        self.short_trend = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'short_trend')
        self.long_trend = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'long_trend')
        self.volatility = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'volatility')
        self.consistency = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'consistency')
        
        self.score = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'score')
        
        # 定义模糊集
        self.short_trend['negative'] = fuzz.trimf(self.short_trend.universe, [-1, -1, 0])
        self.short_trend['neutral'] = fuzz.trimf(self.short_trend.universe, [-0.5, 0, 0.5])
        self.short_trend['positive'] = fuzz.trimf(self.short_trend.universe, [0, 1, 1])
        
        self.long_trend['negative'] = fuzz.trimf(self.long_trend.universe, [-1, -1, 0])
        self.long_trend['neutral'] = fuzz.trimf(self.long_trend.universe, [-0.5, 0, 0.5])
        self.long_trend['positive'] = fuzz.trimf(self.long_trend.universe, [0, 1, 1])
        
        self.volatility['low'] = fuzz.trimf(self.volatility.universe, [0, 0, 0.5])
        self.volatility['medium'] = fuzz.trimf(self.volatility.universe, [0.2, 0.5, 0.8])
        self.volatility['high'] = fuzz.trimf(self.volatility.universe, [0.5, 1, 1])
        
        self.consistency['low'] = fuzz.trimf(self.consistency.universe, [0, 0, 0.5])
        self.consistency['medium'] = fuzz.trimf(self.consistency.universe, [0.2, 0.5, 0.8])
        self.consistency['high'] = fuzz.trimf(self.consistency.universe, [0.5, 1, 1])
        
        self.score['low'] = fuzz.trimf(self.score.universe, [0, 0, 0.5])
        self.score['medium'] = fuzz.trimf(self.score.universe, [0.2, 0.5, 0.8])
        self.score['high'] = fuzz.trimf(self.score.universe, [0.5, 1, 1])
        
        # 定义规则
        self.rules = [
            ctrl.Rule(
                self.short_trend['positive'] & self.long_trend['positive'] & 
                self.volatility['low'] & self.consistency['high'],
                self.score['high']
            ),
            ctrl.Rule(
                self.short_trend['negative'] & self.long_trend['negative'] & 
                self.volatility['low'] & self.consistency['high'],
                self.score['low']
            ),
            # 添加更多规则...
        ]
        
        # 创建控制系统
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
    def evaluate(self, market_state: Dict[str, float]) -> float:
        """
        评估市场状态
        """
        try:
            self.simulation.input['short_trend'] = market_state['short_trend']
            self.simulation.input['long_trend'] = market_state['long_trend']
            self.simulation.input['volatility'] = market_state['volatility']
            self.simulation.input['consistency'] = market_state['consistency']
            
            self.simulation.compute()
            return self.simulation.output['score']
        except:
            return 0.5  # 默认中性评分

class MultiScaleEvaluator:
    """
    多尺度加权评价系统
    """
    def __init__(self, periods: List[str]):
        self.periods = periods
        self.fuzzy_system = FuzzyEvaluationSystem()
        self.weights = self._initialize_weights()
        
    def _initialize_weights(self) -> Dict[str, float]:
        """
        初始化各周期权重
        """
        weights = {
            '5min': 0.15,
            '15min': 0.15,
            '1h': 0.15,
            '1d': 0.25,
            '1w': 0.20,
            '1M': 0.10
        }
        return weights
    
    def evaluate_market_state(self, 
                            features_dict: Dict[str, np.ndarray],
                            market_conditions: Dict[str, float]) -> Dict[str, float]:
        """
        多尺度评价市场状态
        """
        # 计算各周期的评分
        period_scores = {}
        for period in self.periods:
            if period in features_dict:
                market_state = self._extract_market_state(features_dict[period], period)
                score = self.fuzzy_system.evaluate(market_state)
                period_scores[period] = score
        
        # 根据市场条件调整权重
        adjusted_weights = self._adjust_weights(market_conditions)
        
        # 计算加权总分
        total_score = 0
        weight_sum = 0
        for period, score in period_scores.items():
            total_score += score * adjusted_weights[period]
            weight_sum += adjusted_weights[period]
        
        final_score = total_score / weight_sum if weight_sum > 0 else 0.5
        
        return {
            'total_score': final_score,
            'period_scores': period_scores,
            'weights': adjusted_weights
        }
    
    def _extract_market_state(self, 
                            features: np.ndarray, 
                            period: str) -> Dict[str, float]:
        """
        从特征中提取市场状态
        """
        # 计算短期趋势
        short_trend = np.mean(np.diff(features[-10:], axis=0))
        
        # 计算长期趋势
        long_trend = (features[-1] - features[0]) / (features[0] + 1e-8)
        
        # 计算波动率
        volatility = np.std(features) / (np.mean(features) + 1e-8)
        
        # 计算一致性
        consistency = np.corrcoef(features[:-1], features[1:])[0, 1]
        
        return {
            'short_trend': float(short_trend),
            'long_trend': float(long_trend),
            'volatility': float(volatility),
            'consistency': float(consistency)
        }
    
# evaluation/fuzzy_evaluation.py (continued)

    def _adjust_weights(self, market_conditions: Dict[str, float]) -> Dict[str, float]:
        """
        根据市场条件调整权重
        """
        adjusted_weights = self.weights.copy()
        
        # 根据波动率调整权重
        volatility = market_conditions.get('volatility', 0)
        if volatility > 0.5:
            # 高波动率时增加短周期权重
            scale_factor = min(2.0, 1.0 + volatility)
            for period in ['5min', '15min', '1h']:
                if period in adjusted_weights:
                    adjusted_weights[period] *= scale_factor
            # 相应减少长周期权重
            for period in ['1d', '1w', '1M']:
                if period in adjusted_weights:
                    adjusted_weights[period] /= scale_factor
                    
        # 根据趋势强度调整权重
        trend_strength = market_conditions.get('trend_strength', 0)
        if trend_strength > 0.7:
            # 强趋势时增加长周期权重
            scale_factor = min(2.0, 1.0 + trend_strength)
            for period in ['1d', '1w', '1M']:
                if period in adjusted_weights:
                    adjusted_weights[period] *= scale_factor
            # 相应减少短周期权重
            for period in ['5min', '15min', '1h']:
                if period in adjusted_weights:
                    adjusted_weights[period] /= scale_factor
        
        # 归一化权重
        weight_sum = sum(adjusted_weights.values())
        if weight_sum > 0:
            for period in adjusted_weights:
                adjusted_weights[period] /= weight_sum
                
        return adjusted_weights

    def update_weights(self, performance_metrics: Dict[str, float]):
        """
        根据历史表现更新权重
        """
        # 根据各周期的预测准确率调整权重
        for period in self.periods:
            if f'{period}_accuracy' in performance_metrics:
                accuracy = performance_metrics[f'{period}_accuracy']
                # 使用软更新
                self.weights[period] = 0.9 * self.weights[period] + 0.1 * accuracy
        
        # 归一化权重
        weight_sum = sum(self.weights.values())
        for period in self.weights:
            self.weights[period] /= weight_sum

