import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from datetime import datetime

class ModelEvaluator:
    """模型评估器"""
    def __init__(self, model, environment: TradingEnvironment):
        self.model = model
        self.env = environment
        
    def evaluate(self, test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """评估模型性能"""
        predictions = self.model.predict(test_data)
        
        # 回测
        self.env.reset()
        returns = []
        positions = []
        
        for i in range(len(test_data['price'])):
            # 获取模型预测
            signal_probs = predictions['signal'][i]
            price = test_data['price'][i]
            
            # 执行交易
            action = self._get_action(signal_probs)
            reward, done = self.env.step(action, price)
            
            returns.append(self.env.calculate_total_value(price))
            positions.append(self.env.position)
            
            if done:
                break
        
        # 计算评估指标
        metrics = self._calculate_metrics(returns, positions, test_data)
        
        return metrics
    
    def _get_action(self, signal_probs: np.ndarray) -> Dict[str, float]:
        """根据模型预测获取交易动作"""
        action_idx = np.argmax(signal_probs)
        
        if action_idx == 0:  # 买入
            return {'type': 'buy', 'amount': 0.1}
        elif action_idx == 1:  # 卖出
            return {'type': 'sell', 'amount': 0.1}
        else:  # 持有
            return {'type': 'hold', 'amount': 0.0}
    
    def _calculate_metrics(self,
                        returns: List[float],
                        positions: List[float],
                        test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算评估指标"""
        returns = np.array(returns)
        daily_returns = (returns[1:] - returns[:-1]) / returns[:-1]
        
        metrics = {
            'total_return': (returns[-1] - returns[0]) / returns[0],
            'annualized_return': self._calculate_annualized_return(daily_returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(daily_returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': self._calculate_win_rate(daily_returns),
            'profit_loss_ratio': self._calculate_profit_loss_ratio(daily_returns),
            'total_trades': self.env.total_trades,
            'total_fees': self.env.total_fees
        }
        
        return metrics
    
    def _calculate_annualized_return(self, daily_returns: np.ndarray) -> float:
        """计算年化收益率"""
        total_days = len(daily_returns)
        cumulative_return = np.prod(1 + daily_returns) - 1
        annual_return = (1 + cumulative_return) ** (252 / total_days) - 1
        return annual_return
    
    def _calculate_sharpe_ratio(self, daily_returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        excess_returns = daily_returns - risk_free_rate/252
        if len(excess_returns) == 0:
            return 0.0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative = np.maximum.accumulate(returns)
        drawdowns = (cumulative - returns) / cumulative
        return np.max(drawdowns)
    
    def _calculate_win_rate(self, daily_returns: np.ndarray) -> float:
        """计算胜率"""
        winning_days = np.sum(daily_returns > 0)
        return winning_days / len(daily_returns)
    
    def _calculate_profit_loss_ratio(self, daily_returns: np.ndarray) -> float:
        """计算盈亏比"""
        gains = daily_returns[daily_returns > 0]
        losses = abs(daily_returns[daily_returns < 0])
        
        if len(losses) == 0 or np.mean(losses) == 0:
            return float('inf')
        return np.mean(gains) / np.mean(losses)



class BackTester:
    """回测系统"""
    def __init__(self,
                model,
                environment: TradingEnvironment,
                risk_manager: Optional['RiskManager'] = None):
        self.model = model
        self.env = environment
        self.risk_manager = risk_manager or RiskManager()
        self.evaluator = ModelEvaluator(model, environment)
        
    def run_backtest(self,
                    test_data: Dict[str, np.ndarray],
                    price_data: pd.DataFrame) -> Dict[str, Any]:
        """运行回测"""
        self.env.reset()
        trades = []
        positions = []
        returns = []
        equity_curve = []
        
        initial_value = self.env.calculate_total_value(price_data.iloc[0]['close'])
        
        for i in range(len(test_data['price'])):
            current_price = price_data.iloc[i]
            state = {k: v[i:i+1] for k, v in test_data.items()}
            
            # 获取模型预测
            predictions = self.model.predict(state)
            
            # 风险检查
            if self.risk_manager.check_risk(current_price, self.env):
                action = {'type': 'sell', 'amount': self.env.position}
            else:
                action = self._get_action(predictions['signal'][0], current_price)
            
            # 执行交易
            reward, done = self.env.step(action, current_price['close'])
            
            # 记录结果
            trades.append(self._record_trade(action, current_price, reward))
            positions.append(self.env.position)
            current_value = self.env.calculate_total_value(current_price['close'])
            returns.append((current_value - initial_value) / initial_value)
            equity_curve.append(current_value)
            
            if done:
                break
        
        # 计算回测指标
        metrics = self._calculate_backtest_metrics(trades, returns, equity_curve)
        
        return {
            'trades': pd.DataFrame(trades),
            'positions': np.array(positions),
            'returns': np.array(returns),
            'equity_curve': np.array(equity_curve),
            'metrics': metrics
        }
    
    def _get_action(self, signal_probs: np.ndarray, current_price: pd.Series) -> Dict[str, float]:
        """获取交易动作"""
        action_idx = np.argmax(signal_probs)
        position_value = self.env.position * current_price['close']
        
        if action_idx == 0:  # 买入信号
            # 计算可买入金额
            available_cash = self.env.balance * 0.95  # 留5%作为缓冲
            max_buyable = min(available_cash / current_price['close'],
                            (self.env.max_position - self.env.position))
            
            if max_buyable > 0:
                return {'type': 'buy', 'amount': max_buyable}
        
        elif action_idx == 1:  # 卖出信号
            if self.env.position > 0:
                return {'type': 'sell', 'amount': self.env.position}
        
        return {'type': 'hold', 'amount': 0.0}
    
    def _record_trade(self, action: Dict[str, float], price: pd.Series, reward: float) -> Dict:
        """记录交易"""
        return {
            'timestamp': price.name,
            'action': action['type'],
            'amount': action['amount'],
            'price': price['close'],
            'position': self.env.position,
            'balance': self.env.balance,
            'total_value': self.env.calculate_total_value(price['close']),
            'reward': reward
        }
    
    def _calculate_backtest_metrics(self,
                                trades: List[Dict],
                                returns: np.ndarray,
                                equity_curve: np.ndarray) -> Dict[str, float]:
        """计算回测指标"""
        daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        
        metrics = {
            'total_return': returns[-1],
            'annualized_return': self.evaluator._calculate_annualized_return(daily_returns),
            'sharpe_ratio': self.evaluator._calculate_sharpe_ratio(daily_returns),
            'max_drawdown': self.evaluator._calculate_max_drawdown(equity_curve),
            'win_rate': self.evaluator._calculate_win_rate(daily_returns),
            'profit_loss_ratio': self.evaluator._calculate_profit_loss_ratio(daily_returns),
            'total_trades': len([t for t in trades if t['action'] != 'hold']),
            'trade_frequency': len(trades) / len(returns),
            'avg_trade_duration': self._calculate_avg_trade_duration(trades),
            'avg_profit_per_trade': np.mean([t['reward'] for t in trades if t['action'] != 'hold'])
        }
        
        return metrics
    
    def _calculate_avg_trade_duration(self, trades: List[Dict]) -> float:
        """计算平均持仓时间"""
        durations = []
        current_trade_start = None
        
        for trade in trades:
            if trade['action'] == 'buy' and current_trade_start is None:
                current_trade_start = trade['timestamp']
            elif trade['action'] == 'sell' and current_trade_start is not None:
                duration = (trade['timestamp'] - current_trade_start).total_seconds() / 86400  # 转换为天
                durations.append(duration)
                current_trade_start = None
        
        return np.mean(durations) if durations else 0.0

class RiskManager:
    """风险管理器"""
    def __init__(self,
                max_position_size: float = 1.0,
                stop_loss_threshold: float = 0.02,
                max_drawdown_threshold: float = 0.1,
                volatility_threshold: float = 0.02):
        self.max_position_size = max_position_size
        self.stop_loss_threshold = stop_loss_threshold
        self.max_drawdown_threshold = max_drawdown_threshold
        self.volatility_threshold = volatility_threshold
        
        self.position_history = []
        self.price_history = []
    
    def check_risk(self, current_price: pd.Series, environment: TradingEnvironment) -> bool:
        """检查风险状态"""
        self.update_history(current_price, environment)
        
        # 检查止损
        if self._check_stop_loss(current_price['close']):
            return True
        
        # 检查回撤
        if self._check_drawdown():
            return True
        
        # 检查波动率
        if self._check_volatility():
            return True
        
        return False
    
    def update_history(self, current_price: pd.Series, environment: TradingEnvironment):
        """更新历史数据"""
        self.position_history.append(environment.position)
        self.price_history.append(current_price['close'])
        
        # 保留最近的1000个数据点
        if len(self.position_history) > 1000:
            self.position_history.pop(0)
            self.price_history.pop(0)
    
    def _check_stop_loss(self, current_price: float) -> bool:
        """检查止损"""
        if len(self.price_history) < 2:
            return False
        
        price_change = (current_price - self.price_history[-2]) / self.price_history[-2]
        return price_change < -self.stop_loss_threshold
    
    def _check_drawdown(self) -> bool:
        """检查回撤"""
        if len(self.price_history) < 2:
            return False
        
        peak = np.maximum.accumulate(self.price_history)
        drawdown = (peak[-1] - self.price_history[-1]) / peak[-1]
        return drawdown > self.max_drawdown_threshold
    
    def _check_volatility(self) -> bool:
        """检查波动率"""
        if len(self.price_history) < 20:
            return False
        
        returns = np.diff(self.price_history[-20:]) / self.price_history[-21:-1]
        volatility = np.std(returns)
        return volatility > self.volatility_threshold

def main():
    # 示例使用
    model = None  # 载入训练好的模型
    environment = TradingEnvironment()
    risk_manager = RiskManager()
    backtester = BackTester(model, environment, risk_manager)
    
    # 加载测试数据
    test_data = {}  # 加载测试数据
    price_data = pd.DataFrame()  # 加载价格数据
    
    # 运行回测
    results = backtester.run_backtest(test_data, price_data)
    
    # 打印回测结果
    print("回测指标:")
    for metric, value in results['metrics'].items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()