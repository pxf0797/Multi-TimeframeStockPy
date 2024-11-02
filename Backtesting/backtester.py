# backtesting/backtester.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import defaultdict

@dataclass
class BacktestConfig:
    """回测配置"""
    # 资金配置
    initial_capital: float = 1000000
    max_position_size: float = 0.2
    
    # 交易成本
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    
    # 风险控制
    max_drawdown: float = 0.2
    stop_loss_rate: float = 0.05
    
    # 回测参数
    warmup_period: int = 100
    rebalance_interval: int = 20

class BacktestSystem:
    """
    回测系统
    """
    def __init__(self,
                 model: nn.Module,
                 config: BacktestConfig,
                 evaluation_manager: EvaluationManager):
        self.model = model
        self.config = config
        self.evaluation_manager = evaluation_manager
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 初始化回测状态
        self.portfolio = PortfolioState(config.initial_capital)
        self.trade_history = []
        self.position_history = []
        self.performance_metrics = defaultdict(list)
        
    def _setup_logging(self):
        """
        设置日志记录
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def run_backtest(self,
                    data: Dict[str, pd.DataFrame],
                    start_date: str,
                    end_date: str) -> Dict[str, Any]:
        """
        运行回测
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # 初始化数据
        self.data = self._prepare_data(data, start_date, end_date)
        dates = self.data[list(self.data.keys())[0]].index
        
        # 主回测循环
        for current_date in tqdm(dates[self.config.warmup_period:]):
            # 获取当前市场数据
            current_data = self._get_current_data(current_date)
            
            # 评估市场状态
            evaluation = self.evaluation_manager.update(current_data)
            
            # 模型预测
            predictions = self._get_model_predictions(current_data)
            
            # 生成交易信号
            signals = self._generate_trading_signals(predictions, evaluation)
            
            # 执行交易
            self._execute_trades(signals, current_date)
            
            # 更新投资组合状态
            self._update_portfolio_state(current_date)
            
            # 检查风险控制
            if self._check_risk_controls():
                self.logger.warning(f"Risk control triggered at {current_date}")
                self._handle_risk_event()
            
            # 记录性能指标
            self._record_performance_metrics(current_date)
        
        # 生成回测报告
        return self._generate_backtest_report()
    
    def _prepare_data(self,
                     data: Dict[str, pd.DataFrame],
                     start_date: str,
                     end_date: str) -> Dict[str, pd.DataFrame]:
        """
        准备回测数据
        """
        prepared_data = {}
        for period, df in data.items():
            # 截取时间范围
            mask = (df.index >= start_date) & (df.index <= end_date)
            prepared_data[period] = df[mask].copy()
            
            # 添加计算所需的指标
            prepared_data[period] = self._add_technical_indicators(prepared_data[period])
            
        return prepared_data
    
    def _get_current_data(self, current_date: pd.Timestamp) -> Dict[str, Dict[str, float]]:
        """
        获取当前市场数据
        """
        current_data = {}
        for period, df in self.data.items():
            # 获取当前日期之前的数据
            historical_data = df[df.index <= current_date]
            if len(historical_data) > 0:
                current_data[period] = historical_data.iloc[-1].to_dict()
            
        return current_data
    
# backtesting/backtester.py (continued)

    def _get_model_predictions(self,
                             current_data: Dict[str, Dict[str, float]]) -> Dict[str, torch.Tensor]:
        """
        获取模型预测
        """
        self.model.eval()
        with torch.no_grad():
            # 准备模型输入
            model_input = self._prepare_model_input(current_data)
            
            # 获取预测
            predictions = self.model(model_input)
            
            # 转换预测结果为CPU numpy数组
            predictions = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in predictions.items()
            }
            
        return predictions
    
    def _generate_trading_signals(self,
                                predictions: Dict[str, np.ndarray],
                                evaluation: Dict[str, float]) -> Dict[str, float]:
        """
        生成交易信号
        """
        # 提取各类预测结果
        price_predictions = predictions['price']
        trend_predictions = predictions['trend']
        signal_predictions = predictions['signal']
        
        # 评分加权
        evaluation_score = evaluation['total_score']
        
        # 生成仓位信号
        position_signal = self._calculate_position_signal(
            trend_predictions, evaluation_score)
        
        # 生成交易信号
        trading_signal = self._calculate_trading_signal(
            signal_predictions, price_predictions, evaluation_score)
        
        return {
            'position': position_signal,
            'trading': trading_signal
        }
    
    def _calculate_position_signal(self,
                                 trend_predictions: np.ndarray,
                                 evaluation_score: float) -> float:
        """
        计算仓位信号
        """
        # 提取趋势方向和强度
        trend_direction = trend_predictions['direction_probs'].argmax()
        trend_strength = trend_predictions['strength']
        
        # 基础仓位
        if trend_direction == 2:  # 上涨
            base_position = trend_strength
        elif trend_direction == 0:  # 下跌
            base_position = -trend_strength
        else:  # 震荡
            base_position = 0
            
        # 根据评分调整仓位
        adjusted_position = base_position * evaluation_score
        
        # 限制仓位范围
        return np.clip(adjusted_position, -self.config.max_position_size, 
                      self.config.max_position_size)
    
    def _calculate_trading_signal(self,
                                signal_predictions: np.ndarray,
                                price_predictions: np.ndarray,
                                evaluation_score: float) -> Dict[str, float]:
        """
        计算交易信号
        """
        # 提取信号概率和强度
        signal_probs = signal_predictions['signal_probs']
        signal_strength = signal_predictions['strength']
        
        # 价格预测
        predicted_return = (price_predictions['5min_mean'] - 1.0)
        
        # 综合交易信号
        trading_signal = {
            'action': signal_probs.argmax(),  # 0: buy, 1: sell, 2: hold
            'confidence': signal_probs.max() * evaluation_score,
            'strength': signal_strength * evaluation_score,
            'predicted_return': predicted_return
        }
        
        return trading_signal
    
    def _execute_trades(self,
                       signals: Dict[str, float],
                       current_date: pd.Timestamp):
        """
        执行交易
        """
        # 获取当前市场价格
        current_price = self._get_current_price(current_date)
        
        # 处理仓位调整
        if signals['position'] != self.portfolio.current_position:
            self._adjust_position(signals['position'], current_price, current_date)
        
        # 处理交易信号
        if self._should_execute_trade(signals['trading']):
            self._execute_trade(signals['trading'], current_price, current_date)
    
    def _adjust_position(self,
                        target_position: float,
                        current_price: float,
                        current_date: pd.Timestamp):
        """
        调整仓位
        """
        position_change = target_position - self.portfolio.current_position
        
        if abs(position_change) > 0:
            # 计算交易成本
            transaction_cost = abs(position_change * current_price * 
                                 (self.config.commission_rate + self.config.slippage_rate))
            
            # 执行交易
            self.portfolio.cash -= transaction_cost
            self.portfolio.current_position = target_position
            self.portfolio.position_value = target_position * current_price
            
            # 记录交易
            self.trade_history.append({
                'date': current_date,
                'type': 'position_adjustment',
                'price': current_price,
                'size': position_change,
                'cost': transaction_cost
            })
    
    def _execute_trade(self,
                      trading_signal: Dict[str, float],
                      current_price: float,
                      current_date: pd.Timestamp):
        """
        执行交易操作
        """
        action = trading_signal['action']
        confidence = trading_signal['confidence']
        
        # 计算交易规模
        trade_size = self._calculate_trade_size(
            action, confidence, current_price)
        
        if trade_size > 0:
            # 计算交易成本
            transaction_cost = trade_size * current_price * \
                             (self.config.commission_rate + self.config.slippage_rate)
            
            # 执行交易
            if action == 0:  # 买入
                self.portfolio.cash -= (trade_size * current_price + transaction_cost)
                self.portfolio.current_position += trade_size
            else:  # 卖出
                self.portfolio.cash += (trade_size * current_price - transaction_cost)
                self.portfolio.current_position -= trade_size
            
            # 更新持仓价值
            self.portfolio.position_value = self.portfolio.current_position * current_price
            
            # 记录交易
            self.trade_history.append({
                'date': current_date,
                'type': 'trade',
                'action': ['buy', 'sell', 'hold'][action],
                'price': current_price,
                'size': trade_size,
                'cost': transaction_cost,
                'confidence': confidence
            })
    
    def _calculate_trade_size(self,
                            action: int,
                            confidence: float,
                            current_price: float) -> float:
        """
        计算交易规模
        """
        # 基于可用资金和持仓计算最大交易规模
        if action == 0:  # 买入
            max_size = self.portfolio.cash / current_price / \
                      (1 + self.config.commission_rate + self.config.slippage_rate)
        else:  # 卖出
            max_size = self.portfolio.current_position
        
        # 根据置信度调整交易规模
        trade_size = max_size * confidence
        
        return min(trade_size, self.config.max_position_size * 
                  self.portfolio.initial_capital / current_price)
    
    def _update_portfolio_state(self, current_date: pd.Timestamp):
        """
        更新投资组合状态
        """
        current_price = self._get_current_price(current_date)
        
        # 更新持仓价值
        self.portfolio.position_value = self.portfolio.current_position * current_price
        
        # 计算总价值
        total_value = self.portfolio.cash + self.portfolio.position_value
        
        # 更新历史最高价值
        self.portfolio.high_value = max(self.portfolio.high_value, total_value)
        
        # 计算回撤
        current_drawdown = (self.portfolio.high_value - total_value) / \
                          self.portfolio.high_value
        
        # 记录状态
        self.position_history.append({
            'date': current_date,
            'position': self.portfolio.current_position,
            'cash': self.portfolio.cash,
            'position_value': self.portfolio.position_value,
            'total_value': total_value,
            'drawdown': current_drawdown
        })
    
    def _check_risk_controls(self) -> bool:
        """
        检查风险控制
        """
        current_state = self.position_history[-1]
        
        # 检查最大回撤
        if current_state['drawdown'] > self.config.max_drawdown:
            return True
            
        # 检查止损
        if self.portfolio.current_position > 0:
            position_return = (current_state['position_value'] / 
                             self.portfolio.position_cost - 1)
            if position_return < -self.config.stop_loss_rate:
                return True
        
        return False
    
    def _handle_risk_event(self):
        """
        处理风险事件
        """
        # 强制清仓
        current_date = self.position_history[-1]['date']
        current_price = self._get_current_price(current_date)
        
        self._adjust_position(0, current_price, current_date)
        
        self.logger.warning("Risk control triggered - Position cleared")
    
    def _generate_backtest_report(self) -> Dict[str, Any]:
        """
        生成回测报告
        """
        performance = self._calculate_performance_metrics()
        
        return {
            'performance_metrics': performance,
            'trade_history': pd.DataFrame(self.trade_history),
            'position_history': pd.DataFrame(self.position_history),
            'final_portfolio_state': self.portfolio.__dict__,
            'config': self.config.__dict__
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """
        计算性能指标
        """
        position_df = pd.DataFrame(self.position_history)
        trade_df = pd.DataFrame(self.trade_history)
        
        # 计算收益率
        total_return = (position_df['total_value'].iloc[-1] / 
                       self.config.initial_capital - 1)
        
        # 计算年化收益率
        days = (position_df['date'].iloc[-1] - position_df['date'].iloc[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # 计算夏普比率
        returns = position_df['total_value'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # 计算最大回撤
        max_drawdown = position_df['drawdown'].max()
        
        # 计算胜率
        winning_trades = trade_df[trade_df['type'] == 'trade']['price'].diff() > 0
        win_rate = winning_trades.mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trade_count': len(trade_df),
            'avg_trade_size': trade_df['size'].mean(),
            'avg_trade_cost': trade_df['cost'].mean()
        }

class PortfolioState:
    """
    投资组合状态
    """
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.current_position = 0
        self.position_value = 0
        self.position_cost = 0
        self.high_value = initial_capital

def build_backtest_system(model: nn.Module,
                         config: BacktestConfig,
                         evaluation_manager: EvaluationManager) -> BacktestSystem:
    """
    构建回测系统
    """
    return BacktestSystem(model, config, evaluation_manager)