# evaluation/integrated_evaluation.py

class IntegratedEvaluationSystem:
    """
    综合评价系统：整合模糊逻辑评分和多尺度加权
    """
    def __init__(self, periods: List[str]):
        self.periods = periods
        self.multi_scale_evaluator = MultiScaleEvaluator(periods)
        self.evaluation_history = []
        
    def evaluate(self, 
                features_dict: Dict[str, np.ndarray],
                market_conditions: Dict[str, float],
                position: float,
                trading_signals: Dict[str, float]) -> Dict[str, float]:
        """
        综合评价当前市场状态和交易决策
        """
        # 获取多尺度评价结果
        multi_scale_scores = self.multi_scale_evaluator.evaluate_market_state(
            features_dict, market_conditions)
        
        # 评价仓位合理性
        position_score = self._evaluate_position(
            position, multi_scale_scores['total_score'], market_conditions)
        
        # 评价交易信号质量
        signal_score = self._evaluate_trading_signals(
            trading_signals, multi_scale_scores['period_scores'])
        
        # 计算综合评分
        composite_score = self._calculate_composite_score(
            multi_scale_scores['total_score'],
            position_score,
            signal_score
        )
        
        # 记录评价历史
        self.evaluation_history.append({
            'timestamp': len(self.evaluation_history),
            'market_score': multi_scale_scores['total_score'],
            'position_score': position_score,
            'signal_score': signal_score,
            'composite_score': composite_score
        })
        
        return {
            'market_scores': multi_scale_scores,
            'position_score': position_score,
            'signal_score': signal_score,
            'composite_score': composite_score
        }
    
    def _evaluate_position(self,
                         position: float,
                         market_score: float,
                         market_conditions: Dict[str, float]) -> float:
        """
        评价仓位合理性
        """
        # 市场得分高时，仓位应该较大
        score = 1.0 - abs(market_score - position)
        
        # 考虑市场风险
        risk_penalty = 0.0
        if market_conditions.get('volatility', 0) > 0.5 and position > 0.5:
            risk_penalty = 0.2
        if market_conditions.get('drawdown', 0) > 0.1 and position > 0.3:
            risk_penalty += 0.2
            
        return max(0, score - risk_penalty)
    
    def _evaluate_trading_signals(self,
                                trading_signals: Dict[str, float],
                                period_scores: Dict[str, float]) -> float:
        """
        评价交易信号质量
        """
        signal_scores = []
        for period in self.periods:
            if period in trading_signals and period in period_scores:
                # 信号强度与市场评分的一致性
                signal_strength = abs(trading_signals[period])
                score = 1.0 - abs(signal_strength - period_scores[period])
                signal_scores.append(score)
        
        return np.mean(signal_scores) if signal_scores else 0.5
    
    def _calculate_composite_score(self,
                                 market_score: float,
                                 position_score: float,
                                 signal_score: float) -> float:
        """
        计算综合评分
        """
        weights = {
            'market': 0.4,
            'position': 0.3,
            'signal': 0.3
        }
        
        composite_score = (
            weights['market'] * market_score +
            weights['position'] * position_score +
            weights['signal'] * signal_score
        )
        
        return composite_score
    
    def get_evaluation_statistics(self) -> Dict[str, float]:
        """
        获取评价统计信息
        """
        if not self.evaluation_history:
            return {}
            
        history = pd.DataFrame(self.evaluation_history)
        
        return {
            'avg_market_score': history['market_score'].mean(),
            'avg_position_score': history['position_score'].mean(),
            'avg_signal_score': history['signal_score'].mean(),
            'avg_composite_score': history['composite_score'].mean(),
            'score_volatility': history['composite_score'].std(),
            'score_trend': (history['composite_score'].iloc[-1] - 
                          history['composite_score'].iloc[0]) / len(history)
        }

# 辅助函数
def calculate_trend_strength(data: np.ndarray) -> float:
    """
    计算趋势强度
    """
    # 使用线性回归斜率和R方值计算趋势强度
    x = np.arange(len(data))
    slope, intercept, r_value, _, _ = stats.linregress(x, data)
    
    # 归一化斜率
    normalized_slope = np.arctan(slope) / (np.pi/2)
    
    # 综合考虑斜率和拟合优度
    trend_strength = abs(normalized_slope) * (r_value ** 2)
    
    return float(trend_strength)

def calculate_market_conditions(features_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    计算市场状况
    """
    conditions = {}
    
    # 计算波动率
    short_term_data = features_dict.get('5min', np.array([]))
    if len(short_term_data) > 0:
        conditions['volatility'] = float(np.std(short_term_data[-100:]) / 
                                      (np.mean(short_term_data[-100:]) + 1e-8))
    
    # 计算趋势强度
    long_term_data = features_dict.get('1d', np.array([]))
    if len(long_term_data) > 0:
        conditions['trend_strength'] = calculate_trend_strength(long_term_data[-20:])
    
    # 计算历史最大回撤
    if len(long_term_data) > 0:
        peak = np.maximum.accumulate(long_term_data)
        drawdown = (peak - long_term_data) / peak
        conditions['drawdown'] = float(np.max(drawdown))
    
    return conditions