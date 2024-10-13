import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, config):
        self.config = config

    def run_backtest(self, signals, data, dynamic_weights):
        results = {}
        for tf in signals.keys():
            if tf in data:
                try:
                    pnl = self.calculate_pnl(signals[tf], data[tf])
                    logger.info(f"PnL for {tf}: {pnl}")
                    sharpe_ratio = self.calculate_sharpe_ratio(pnl)
                    max_drawdown = self.calculate_max_drawdown(pnl)
                    total_return = self.calculate_total_return(pnl)
                    win_rate = self.calculate_win_rate(pnl)
                    
                    results[tf] = {
                        'PnL': pnl.sum(),
                        'Sharpe Ratio': sharpe_ratio,
                        'Max Drawdown': max_drawdown,
                        'Total Return': total_return,
                        'Win Rate': win_rate
                    }
                    logger.info(f"Backtest results for {tf}: {results[tf]}")
                except Exception as e:
                    logger.error(f"Error in backtesting for {tf}: {str(e)}", exc_info=True)
        return results

    def calculate_pnl(self, signal, data):
        try:
            # Ensure the signal and data have the same length
            min_length = min(len(signal['position_size']), len(data))
            
            # Use the last min_length elements
            position = signal['position_size'][-min_length:]
            returns = data['returns'].iloc[-min_length:].values
            
            if len(position) != len(returns):
                raise ValueError(f"Mismatch in lengths: position ({len(position)}) and returns ({len(returns)})")
            
            pnl = pd.Series(position * returns).fillna(0)
            logger.info(f"PnL calculation: position range [{position.min()}, {position.max()}], returns range [{returns.min()}, {returns.max()}]")
            return pnl
        except Exception as e:
            logger.error(f"Error in PnL calculation: {str(e)}", exc_info=True)
            return pd.Series()

    def calculate_sharpe_ratio(self, pnl, risk_free_rate=0.02):
        try:
            returns = pnl.pct_change().dropna()
            excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
            mean_excess_return = excess_returns.mean()
            std_excess_return = excess_returns.std()
            
            logger.info(f"Sharpe ratio calculation: mean excess return = {mean_excess_return}, std excess return = {std_excess_return}")
            
            if std_excess_return == 0:
                logger.warning("Standard deviation of excess returns is zero. Setting Sharpe ratio to 0.")
                return 0
            
            sharpe_ratio = np.sqrt(252) * mean_excess_return / std_excess_return
            
            if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
                logger.warning(f"Sharpe ratio calculation resulted in {sharpe_ratio}. Setting to 0.")
                return 0
            
            return sharpe_ratio
        except Exception as e:
            logger.error(f"Error in Sharpe ratio calculation: {str(e)}", exc_info=True)
            return 0

    def calculate_max_drawdown(self, pnl):
        try:
            cumulative_returns = (1 + pnl).cumprod()
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns / peak) - 1
            max_drawdown = drawdown.min()
            
            logger.info(f"Max drawdown calculation: cumulative returns range [{cumulative_returns.min()}, {cumulative_returns.max()}], max drawdown = {max_drawdown}")
            
            if np.isnan(max_drawdown) or np.isinf(max_drawdown):
                logger.warning(f"Max drawdown calculation resulted in {max_drawdown}. Setting to 0.")
                return 0
            
            return max_drawdown
        except Exception as e:
            logger.error(f"Error in max drawdown calculation: {str(e)}", exc_info=True)
            return 0

    def calculate_total_return(self, pnl):
        try:
            total_return = (1 + pnl).prod() - 1
            logger.info(f"Total return calculation: {total_return}")
            return total_return
        except Exception as e:
            logger.error(f"Error in total return calculation: {str(e)}", exc_info=True)
            return 0

    def calculate_win_rate(self, pnl):
        try:
            wins = (pnl > 0).sum()
            total_trades = len(pnl)
            win_rate = wins / total_trades if total_trades > 0 else 0
            logger.info(f"Win rate calculation: wins = {wins}, total trades = {total_trades}, win rate = {win_rate}")
            return win_rate
        except Exception as e:
            logger.error(f"Error in win rate calculation: {str(e)}", exc_info=True)
            return 0
