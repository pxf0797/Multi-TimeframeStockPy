# evaluation_system.py

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import skew, kurtosis

class FuzzyEvaluator:
    """模糊逻辑评分系统"""
    def __init__(self):
        self.membership_functions = self._initialize_membership_functions()
    
    def _initialize_membership_functions(self) -> Dict:
        """初始化隶属度函数"""
        return {
            'trend': {
                'strong_up': lambda x: max(0, min(1, (x - 0.6) / 0.4)) if x > 0 else 0,
                'up': lambda x: max(0, min(1, (x - 0.2) / 0.4, (0.6 - x) / 0.4)) if x > 0 else 0,
                'flat': lambda x: max(0, min(1, (x + 0.2) / 0.2, (0.2 - x) / 0.2)),
                'down': lambda x: max(0, min(1, (-x - 0.2) / 0.4, (0.6 + x) / 0.4)) if x < 0 else 0,
                'strong_down': lambda x: max(0, min(1, (-x - 0.6) / 0.4)) if x < 0 else 0
            },
            'volatility': {
                'low': lambda x: max(0, min(1, (0.02 - x) / 0.02)),
                'medium': lambda x: max(0, min(1, (x - 0.01) / 0.01, (0.03 - x) / 0.01)),
                'high': lambda x: max(0, min(1, (x - 0.02) / 0.02))
            },
            'volume': {
                'low': lambda x: max(0, min(1, (0.8 - x) / 0.3)),
                'normal': lambda x: max(0, min(1, (x - 0.5) / 0.3, (1.1 - x) / 0.3)),
                'high': lambda x: max(0, min(1, (x - 0.8) / 0.3))
            }
        }
    
    def evaluate_market_state(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """评估市场状态"""
        trend_score = self._evaluate_trend(indicators['trend'])
        volatility_score = self._evaluate_volatility(indicators['volatility'])
        volume_score = self._evaluate_volume(indicators['volume'])
        
        consistency_score = self._evaluate_consistency(
            trend_score, volatility_score, volume_score
        )
        
        return {
            'trend_score': trend_score,
            'volatility_score': volatility_score,
            'volume_score': volume_score,
            'consistency_score': consistency_score,
            'final_score': self._calculate_final_score([
                trend_score,])
        }
                
    def _evaluate_trend(self, trend_value: float) -> float:
        """评估趋势"""
        scores = {
            state: func(trend_value)
            for state, func in self.membership_functions['trend'].items()
        }
        
        # 加权计算最终趋势分数
        weights = {
            'strong_up': 1.0,
            'up': 0.7,
            'flat': 0.3,
            'down': -0.7,
            'strong_down': -1.0
        }
        
        return sum(scores[state] * weights[state] for state in scores)
    
    def _evaluate_volatility(self, volatility_value: float) -> float:
        """评估波动率"""
        scores = {
            state: func(volatility_value)
            for state, func in self.membership_functions['volatility'].items()
        }
        
        weights = {
            'low': 0.8,
            'medium': 0.5,
            'high': 0.2
        }
        
        return sum(scores[state] * weights[state] for state in scores)
    
    def _evaluate_volume(self, volume_value: float) -> float:
        """评估成交量"""
        scores = {
            state: func(volume_value)
            for state, func in self.membership_functions['volume'].items()
        }
        
        weights = {
            'low': 0.3,
            'normal': 0.8,
            'high': 0.5
        }
        
        return sum(scores[state] * weights[state] for state in scores)
    
    def _evaluate_consistency(self,
                            trend_score: float,
                            volatility_score: float,
                            volume_score: float) -> float:
        """评估一致性"""
        # 趋势和成交量的一致性
        trend_volume_consistency = 1.0 if (
            (trend_score > 0 and volume_score > 0.5) or
            (trend_score < 0 and volume_score < 0.5)
        ) else 0.0
        
        # 趋势和波动率的一致性
        trend_volatility_consistency = 1.0 if (
            (abs(trend_score) > 0.7 and volatility_score < 0.5) or
            (abs(trend_score) < 0.3 and volatility_score > 0.7)
        ) else 0.0
        
        return 0.6 * trend_volume_consistency + 0.4 * trend_volatility_consistency
    
    def _calculate_final_score(self, scores: List[float]) -> float:
        """计算最终评分"""
        weights = [0.4, 0.2, 0.2, 0.2]  # 各指标权重
        return sum(score * weight for score, weight in zip(scores, weights))

class MultiScaleEvaluator:
    """多尺度加权评价系统"""
    def __init__(self, time_scales: List[str]):
        self.time_scales = time_scales  # ['5min', '15min', '1h', '1d', '1w']
        self.fuzzy_evaluator = FuzzyEvaluator()
        
    def evaluate(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """多尺度评价"""
        scale_weights = self._calculate_scale_weights(data)
        scale_scores = {}
        
        for scale in self.time_scales:
            indicators = self._calculate_indicators(data[scale])
            scale_scores[scale] = self.fuzzy_evaluator.evaluate_market_state(indicators)
        
        return self._combine_scale_scores(scale_scores, scale_weights)
    
    def _calculate_scale_weights(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """计算不同时间尺度的权重"""
        volatilities = {}
        
        for scale in self.time_scales:
            df = data[scale]
            returns = df['close'].pct_change()
            volatilities[scale] = returns.std()
        
        total_volatility = sum(volatilities.values())
        weights = {
            scale: vol / total_volatility
            for scale, vol in volatilities.items()
        }
        
        # 调整短期和长期权重
        for scale in weights:
            if scale in ['5min', '15min']:
                weights[scale] *= 1.2  # 增加短期权重
            elif scale in ['1d', '1w']:
                weights[scale] *= 0.8  # 减少长期权重
        
        # 归一化
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算技术指标"""
        returns = df['close'].pct_change()
        
        trend = self._calculate_trend_indicator(df)
        volatility = returns.std()
        volume = df['volume'] / df['volume'].rolling(20).mean()
        
        return {
            'trend': trend,
            'volatility': volatility,
            'volume': volume.iloc[-1]
        }
    
    def _calculate_trend_indicator(self, df: pd.DataFrame) -> float:
        """计算趋势指标"""
        # 使用多个指标综合判断趋势
        ma_trend = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).mean()
        momentum = df['close'].pct_change(5)
        macd = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        
        # 综合评分
        trend_score = (
            0.4 * ma_trend.iloc[-1] +
            0.3 * momentum.iloc[-1] +
            0.3 * np.sign(macd.iloc[-1])
        )
        
        return np.clip(trend_score, -1, 1)
    
    def _combine_scale_scores(self,
                            scale_scores: Dict[str, Dict[str, float]],
                            scale_weights: Dict[str, float]) -> Dict[str, float]:
        """合并不同时间尺度的评分"""
        combined_scores = {}
        
        # 对每个评分指标进行加权组合
        for metric in ['trend_score', 'volatility_score', 'volume_score', 'consistency_score', 'final_score']:
            combined_scores[metric] = sum(
                scale_scores[scale][metric] * scale_weights[scale]
                for scale in self.time_scales
            )
        
        # 添加跨期一致性评分
        combined_scores['cross_scale_consistency'] = self._calculate_cross_scale_consistency(scale_scores)
        
        return combined_scores
    
    def _calculate_cross_scale_consistency(self, scale_scores: Dict[str, Dict[str, float]]) -> float:
        """计算跨期一致性"""
        trend_scores = [scores['trend_score'] for scores in scale_scores.values()]
        
        # 计算趋势方向的一致性
        trend_signs = np.sign(trend_scores)
        sign_consistency = np.mean(trend_signs == trend_signs[0])
        
        # 计算趋势强度的相关性
        trend_correlation = np.corrcoef(trend_scores)[0, 1]
        
        return 0.6 * sign_consistency + 0.4 * trend_correlation

class StockStateEvaluator:
    """股票状态综合评价"""
    def __init__(self, time_scales: List[str]):
        self.multi_scale_evaluator = MultiScaleEvaluator(time_scales)
    
    def evaluate_stock_state(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """评估股票状态"""
        # 获取多尺度评分
        scale_scores = self.multi_scale_evaluator.evaluate(data)
        
        # 计算市场情绪指标
        sentiment = self._calculate_market_sentiment(data)
        
        # 计算技术健康度
        technical_health = self._calculate_technical_health(data)
        
        # 综合评分
        final_state = {
            'market_scores': scale_scores,
            'market_sentiment': sentiment,
            'technical_health': technical_health,
            'overall_score': self._calculate_overall_score(scale_scores, sentiment, technical_health)
        }
        
        return final_state
    
    def _calculate_market_sentiment(self, data: Dict[str, pd.DataFrame]) -> float:
        """计算市场情绪"""
        sentiment_indicators = {}
        
        # 使用最短周期数据计算
        short_term_data = data[min(data.keys())]
        
        # 计算RSI
        delta = short_term_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        sentiment_indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算成交量变化
        sentiment_indicators['volume_change'] = short_term_data['volume'].pct_change()
        
        # 计算价格动量
        sentiment_indicators['momentum'] = short_term_data['close'].pct_change(5)
        
        # 综合情绪分数
        sentiment_score = (
            0.4 * (sentiment_indicators['rsi'].iloc[-1] / 100) +
            0.3 * np.sign(sentiment_indicators['volume_change'].iloc[-1]) +
            0.3 * np.sign(sentiment_indicators['momentum'].iloc[-1])
        )
        
        return np.clip(sentiment_score, 0, 1)
    
    def _calculate_technical_health(self, data: Dict[str, pd.DataFrame]) -> float:
        """计算技术指标健康度"""
        health_scores = {}
        
        # 使用日线数据计算
        daily_data = data['1d']
        
        # 计算均线系统健康度
        ma_periods = [5, 10, 20, 50]
        ma_health = self._calculate_ma_health(daily_data, ma_periods)
        
        # 计算MACD健康度
        macd_health = self._calculate_macd_health(daily_data)
        
        # 计算布林带健康度
        bollinger_health = self._calculate_bollinger_health(daily_data)
        
        # 综合健康度评分
        health_score = (
            0.4 * ma_health +
            0.3 * macd_health +
            0.3 * bollinger_health
        )
        
        return np.clip(health_score, 0, 1)
    
    def _calculate_ma_health(self, df: pd.DataFrame, periods: List[int]) -> float:
        """计算均线系统健康度"""
        ma_scores = []
        current_price = df['close'].iloc[-1]
        
        # 计算各周期均线
        mas = {
            period: df['close'].rolling(period).mean().iloc[-1]
            for period in periods
        }
        
        # 计算均线排列得分
        for i in range(len(periods)-1):
            if mas[periods[i]] > mas[periods[i+1]]:
                ma_scores.append(1)
            else:
                ma_scores.append(-1)
        
        # 计算价格相对均线位置得分
        price_ma_scores = [
            1 if current_price > ma else -1
            for ma in mas.values()
        ]
        
        # 综合得分
        return np.mean(ma_scores + price_ma_scores)
    
    def _calculate_macd_health(self, df: pd.DataFrame) -> float:
        """计算MACD健康度"""
        # 计算MACD
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        
        # 计算MACD柱状图趋势
        hist_trend = np.sign(hist.diff())
        
        # 计算MACD线和信号线的距离
        macd_distance = abs(macd - signal) / df['close'].std()
        
        # 综合评分
        return np.clip(
            0.6 * np.sign(hist.iloc[-1]) +
            0.4 * (1 - np.clip(macd_distance.iloc[-1], 0, 1)),
            0, 1
        )
    
    def _calculate_bollinger_health(self, df: pd.DataFrame) -> float:
        """计算布林带健康度"""
        # 计算布林带
        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        
        current_price = df['close'].iloc[-1]
        
        # 计算价格在带中的位置
        position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        
        # 计算带宽
        bandwidth = (upper - lower) / ma20
        
        # 综合评分
        position_score = 1 - 2 * abs(position - 0.5)  # 越接近中轨分数越高
        bandwidth_score = 1 - np.clip(bandwidth.iloc[-1], 0, 1)  # 带宽越窄分数越高
        
        return 0.7 * position_score + 0.3 * bandwidth_score
    
    def _calculate_overall_score(self,
                            scale_scores: Dict[str, float],
                            sentiment: float,
                            technical_health: float) -> float:
        """计算总体评分"""
        weights = {
            'scale_scores': 0.4,
            'sentiment': 0.3,
            'technical_health': 0.3
        }
        
        return (
            weights['scale_scores'] * scale_scores['final_score'] +
            weights['sentiment'] * sentiment +
            weights['technical_health'] * technical_health
        )

# evaluation_system.py (continued)

class EvaluationReport:
    """评估报告生成器"""
    def __init__(self):
        self.timestamp = datetime.now()
        
    def generate_report(self,
                       evaluation_results: Dict[str, float],
                       market_data: Dict[str, pd.DataFrame]) -> Dict:
        """生成评估报告"""
        report = {
            'timestamp': self.timestamp,
            'summary': self._generate_summary(evaluation_results),
            'detailed_analysis': self._generate_detailed_analysis(evaluation_results),
            'market_conditions': self._analyze_market_conditions(market_data),
            'risk_assessment': self._assess_risks(evaluation_results, market_data),
            'recommendations': self._generate_recommendations(evaluation_results)
        }
        return report
    
    def _generate_summary(self, results: Dict[str, float]) -> Dict:
        """生成总结"""
        return {
            'overall_score': results['overall_score'],
            'market_sentiment': results['market_sentiment'],
            'technical_health': results['technical_health'],
            'trend_strength': results['market_scores']['trend_score'],
            'risk_level': self._calculate_risk_level(results),
            'trading_suggestion': self._generate_trading_suggestion(results)
        }
    
    def _generate_detailed_analysis(self, results: Dict[str, float]) -> Dict:
        """生成详细分析"""
        return {
            'technical_indicators': {
                'trend_analysis': self._analyze_trend(results),
                'volume_analysis': self._analyze_volume(results),
                'momentum_analysis': self._analyze_momentum(results)
            },
            'scale_analysis': {
                scale: self._analyze_scale(results['market_scores'], scale)
                for scale in ['short_term', 'medium_term', 'long_term']
            }
        }
    
    def _analyze_market_conditions(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """分析市场状况"""
        conditions = {}
        
        # 分析最短周期数据
        short_term_data = market_data[min(market_data.keys())]
        
        # 价格趋势分析
        price_trend = self._analyze_price_trend(short_term_data)
        
        # 波动性分析
        volatility = self._analyze_volatility(short_term_data)
        
        # 成交量分析
        volume_trend = self._analyze_volume_trend(short_term_data)
        
        conditions.update({
            'price_trend': price_trend,
            'volatility': volatility,
            'volume_trend': volume_trend,
            'market_phase': self._determine_market_phase(price_trend, volume_trend)
        })
        
        return conditions
    
    def _assess_risks(self,
                    results: Dict[str, float],
                    market_data: Dict[str, pd.DataFrame]) -> Dict:
        """评估风险"""
        return {
            'systematic_risk': self._calculate_systematic_risk(market_data),
            'volatility_risk': self._calculate_volatility_risk(market_data),
            'liquidity_risk': self._calculate_liquidity_risk(market_data),
            'trend_reversal_risk': self._calculate_trend_reversal_risk(results),
            'overall_risk_score': self._calculate_overall_risk(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, float]) -> Dict:
        """生成建议"""
        overall_score = results['overall_score']
        market_scores = results['market_scores']
        
        # 基于不同分数生成建议
        if overall_score > 0.8:
            position_advice = "可以考虑增加仓位"
            trading_advice = "适合做多"
        elif overall_score > 0.6:
            position_advice = "保持现有仓位"
            trading_advice = "可以小仓位交易"
        elif overall_score > 0.4:
            position_advice = "谨慎持仓"
            trading_advice = "建议观望"
        else:
            position_advice = "建议减仓"
            trading_advice = "不建议交易"
        
        return {
            'position_recommendation': position_advice,
            'trading_recommendation': trading_advice,
            'risk_management_suggestion': self._generate_risk_management_advice(results),
            'timeframe_suggestion': self._suggest_trading_timeframe(market_scores)
        }
    
    def _calculate_risk_level(self, results: Dict[str, float]) -> str:
        """计算风险等级"""
        risk_score = (
            0.4 * (1 - results['technical_health']) +
            0.3 * (1 - results['market_sentiment']) +
            0.3 * abs(results['market_scores']['trend_score'])
        )
        
        if risk_score < 0.2:
            return "低风险"
        elif risk_score < 0.4:
            return "中低风险"
        elif risk_score < 0.6:
            return "中等风险"
        elif risk_score < 0.8:
            return "中高风险"
        else:
            return "高风险"
    
    def _generate_trading_suggestion(self, results: Dict[str, float]) -> str:
        """生成交易建议"""
        trend_score = results['market_scores']['trend_score']
        sentiment = results['market_sentiment']
        health = results['technical_health']
        
        if trend_score > 0.5 and sentiment > 0.6 and health > 0.7:
            return "强势做多"
        elif trend_score > 0.3 and sentiment > 0.5 and health > 0.6:
            return "谨慎做多"
        elif trend_score < -0.5 and sentiment < 0.4 and health < 0.3:
            return "强势做空"
        elif trend_score < -0.3 and sentiment < 0.5 and health < 0.4:
            return "谨慎做空"
        else:
            return "观望"
    
    def _analyze_trend(self, results: Dict[str, float]) -> Dict:
        """分析趋势"""
        trend_score = results['market_scores']['trend_score']
        
        return {
            'trend_direction': 'uptrend' if trend_score > 0 else 'downtrend',
            'trend_strength': abs(trend_score),
            'trend_reliability': results['market_scores']['consistency_score'],
            'trend_momentum': results['market_sentiment']
        }
    
    def _analyze_volume(self, results: Dict[str, float]) -> Dict:
        """分析成交量"""
        volume_score = results['market_scores']['volume_score']
        
        return {
            'volume_trend': 'increasing' if volume_score > 0.5 else 'decreasing',
            'volume_strength': volume_score,
            'volume_consistency': results['market_scores']['consistency_score']
        }
    
    def _analyze_momentum(self, results: Dict[str, float]) -> Dict:
        """分析动量"""
        trend_score = results['market_scores']['trend_score']
        sentiment = results['market_sentiment']
        
        return {
            'momentum_direction': 'positive' if trend_score * sentiment > 0 else 'negative',
            'momentum_strength': abs(trend_score * sentiment),
            'momentum_stability': results['technical_health']
        }
    
    def _analyze_price_trend(self, df: pd.DataFrame) -> Dict:
        """分析价格趋势"""
        returns = df['close'].pct_change()
        
        return {
            'direction': 'up' if returns.mean() > 0 else 'down',
            'strength': abs(returns.mean()),
            'stability': 1 - returns.std()
        }
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """分析波动性"""
        returns = df['close'].pct_change()
        
        return {
            'current_volatility': returns.std(),
            'volatility_trend': 'increasing' if returns.std() > returns.shift(20).std() else 'decreasing',
            'volatility_level': self._categorize_volatility(returns.std())
        }
    
    def _categorize_volatility(self, volatility: float) -> str:
        """对波动率进行分类"""
        if volatility < 0.01:
            return "低波动"
        elif volatility < 0.02:
            return "中等波动"
        else:
            return "高波动"
    
    def _determine_market_phase(self,
                            price_trend: Dict,
                            volume_trend: Dict) -> str:
        """判断市场阶段"""
        if price_trend['direction'] == 'up' and volume_trend['strength'] > 0.7:
            return "上升趋势"
        elif price_trend['direction'] == 'down' and volume_trend['strength'] > 0.7:
            return "下降趋势"
        elif price_trend['stability'] < 0.3:
            return "震荡整理"
        else:
            return "盘整"
    
    def _generate_risk_management_advice(self, results: Dict[str, float]) -> Dict:
        """生成风险管理建议"""
        risk_level = self._calculate_risk_level(results)
        
        advice = {
            "止损建议": self._get_stop_loss_advice(risk_level),
            "仓位建议": self._get_position_size_advice(risk_level),
            "交易频率建议": self._get_trading_frequency_advice(results),
            "风险提示": self._get_risk_warnings(results)
        }
        
        return advice
    
    def _get_stop_loss_advice(self, risk_level: str) -> str:
        """获取止损建议"""
        stop_loss_levels = {
            "低风险": "2%",
            "中低风险": "3%",
            "中等风险": "4%",
            "中高风险": "5%",
            "高风险": "6%"
        }
        return f"建议将止损设置在{stop_loss_levels[risk_level]}"
    
    def _get_position_size_advice(self, risk_level: str) -> str:
        """获取仓位建议"""
        position_sizes = {
            "低风险": "可以考虑满仓",
            "中低风险": "建议仓位70%-80%",
            "中等风险": "建议仓位50%-60%",
            "中高风险": "建议仓位30%-40%",
            "高风险": "建议仓位20%以下"
        }
        return position_sizes[risk_level]

def main():
    # 示例使用
    time_scales = ['5min', '15min', '1h', '1d', '1w']
    stock_evaluator = StockStateEvaluator(time_scales)
    report_generator = EvaluationReport()
    
    # 加载市场数据
    market_data = {}  # 加载实际数据
    
    # 评估股票状态
    evaluation_results = stock_evaluator.evaluate_stock_state(market_data)
    
    # 生成评估报告
    report = report_generator.generate_report(evaluation_results, market_data)
    
    # 打印报告
    print("股票评估报告")
    print("=" * 50)
    print(f"评估时间: {report['timestamp']}")
    print("\n总体评分:")
    print(f"整体得分: {report['summary']['overall_score']:.2f}")
    print(f"市场情绪: {report['summary']['market_sentiment']:.2f}")
    print(f"技术健康度: {report['summary']['technical_health']:.2f}")
    print(f"风险等级: {report['summary']['risk_level']}")
    print(f"交易建议: {report['summary']['trading_suggestion']}")
    print("\n详细分析:")
    print("技术指标分析:")
    print(report['detailed_analysis']['technical_indicators'])
    print("\n市场状况:")
    print(report['market_conditions'])
    print("\n风险评估:")
    print(report['risk_assessment'])
    print("\n建议:")
    print(report['recommendations'])

if __name__ == "__main__":
    main()