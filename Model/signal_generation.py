import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self, config):
        self.config = config

    def generate_signals(self, model, featured_data):
        """
        Generate trading signals for each timeframe using the trained model.

        Args:
            model (torch.nn.Module): Trained model.
            featured_data (dict): Dictionary of DataFrames with engineered features for each timeframe.

        Returns:
            tuple: Dictionaries of signals and dynamic weights for each timeframe.
        """
        model.eval()
        signals = {}
        dynamic_weights = {}
        with torch.no_grad():
            for tf, df in featured_data.items():
                try:
                    X, volatility, accuracy, trend_strength = self.prepare_data(df)
                    
                    predictions, weights = model(X.unsqueeze(0), volatility.unsqueeze(0), accuracy.unsqueeze(0), trend_strength.unsqueeze(0))
                    signals[tf] = self.process_predictions(predictions.squeeze().cpu().numpy(), df)
                    dynamic_weights[tf] = weights.squeeze().cpu().numpy()
                    logger.info(f"Generated signal for timeframe {tf}")
                except Exception as e:
                    logger.error(f"Error generating signal for timeframe {tf}: {e}")
        return signals, dynamic_weights

    def prepare_data(self, df):
        """
        Prepare data for model input.

        Args:
            df (pd.DataFrame): DataFrame with features for a single timeframe.

        Returns:
            tuple: Tensors for model input (X, volatility, accuracy, trend_strength).
        """
        required_columns = ['Volatility', 'Accuracy', 'Trend_Strength', 'returns', 'log_returns', 'ATR']
        if not all(col in df.columns for col in required_columns):
            missing_columns = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_columns}")

        X = torch.FloatTensor(df.drop(['returns', 'log_returns'], axis=1).values).to(self.config['device'])
        volatility = torch.FloatTensor(df['Volatility'].values).to(self.config['device'])
        accuracy = torch.FloatTensor(df['Accuracy'].values).to(self.config['device'])
        trend_strength = torch.FloatTensor(df['Trend_Strength'].values).to(self.config['device'])

        return X, volatility, accuracy, trend_strength

    def process_predictions(self, predictions, df):
        """
        Process model predictions to generate trading signals.

        Args:
            predictions (np.array): Model predictions.
            df (pd.DataFrame): DataFrame with features for a single timeframe.

        Returns:
            dict: Processed trading signals.
        """
        signal_strength = predictions[:, 0]
        entry_level = self.determine_entry_level(predictions[:, 1], df['Trend_Strength'].values)
        stop_loss = predictions[:, 2] * df['ATR'].values
        
        # Calculate position size based on signal strength
        position_size = np.abs(signal_strength)
        
        return {
            'signal_strength': signal_strength,
            'entry_level': entry_level,
            'stop_loss': stop_loss,
            'position_size': position_size
        }

    def determine_entry_level(self, raw_level, trend_strength):
        """
        Determine entry levels based on raw model output and trend strength.

        Args:
            raw_level (np.array): Raw entry level predictions from the model.
            trend_strength (np.array): Trend strength values.

        Returns:
            np.array: Determined entry levels.
        """
        entry_levels = np.zeros_like(raw_level)
        entry_levels[np.logical_and(raw_level > 0.7, trend_strength > 0.05)] = 2  # Strong entry
        entry_levels[np.logical_and(np.logical_and(raw_level > 0.5, raw_level <= 0.7), trend_strength > 0.03)] = 1  # Medium entry
        entry_levels[np.logical_and(np.logical_and(raw_level > 0.3, raw_level <= 0.5), trend_strength > 0.01)] = 0.5  # Weak entry
        return entry_levels

    def generate_comprehensive_signal(self, signals, dynamic_weights, trend_cons):
        """
        Generate a comprehensive signal by combining signals from all timeframes.

        Args:
            signals (dict): Dictionary of signals for each timeframe.
            dynamic_weights (dict): Dictionary of dynamic weights for each timeframe.
            trend_cons (float): Trend consistency value.

        Returns:
            np.array: Comprehensive signal.
        """
        # Find the minimum length among all signals
        min_length = min(len(signals[tf]['signal_strength']) for tf in signals)
        
        s_final = np.zeros(min_length)
        total_weight = np.zeros(min_length)
        
        for tf in signals.keys():
            weight = dynamic_weights[tf][-min_length:]  # Use the last min_length weights
            signal_strength = signals[tf]['signal_strength'][-min_length:]  # Use the last min_length signal strengths
            
            s_final += weight * signal_strength
            total_weight += weight
        
        # Avoid division by zero
        total_weight = np.where(total_weight == 0, 1, total_weight)
        s_final /= total_weight
        
        # Apply trend_cons as a scalar
        s_comprehensive = s_final * trend_cons
        logger.info(f"Comprehensive signal shape: {s_comprehensive.shape}")
        return s_comprehensive

    def determine_entry_strategy(self, s_comprehensive, entry_level, tc, risk_total):
        """
        Determine the entry strategy based on comprehensive signal, entry level, trend consistency, and total risk.

        Args:
            s_comprehensive (np.array): Comprehensive signal.
            entry_level (np.array): Entry levels.
            tc (np.array): Trend consistency.
            risk_total (np.array): Total risk.

        Returns:
            np.array: Array of determined strategies.
        """
        th1, th2, th3 = 0.7, 0.5, 0.3
        r1, r2, r3 = 0.1, 0.2, 0.3

        # Ensure all inputs are numpy arrays
        s_comprehensive = np.atleast_1d(s_comprehensive)
        entry_level = np.atleast_1d(entry_level)
        tc = np.atleast_1d(tc)
        risk_total = np.atleast_1d(risk_total)

        # Broadcast to the same shape
        max_len = max(len(s_comprehensive), len(entry_level), len(tc), len(risk_total))
        s_comprehensive = np.broadcast_to(s_comprehensive, (max_len,))
        entry_level = np.broadcast_to(entry_level, (max_len,))
        tc = np.broadcast_to(tc, (max_len,))
        risk_total = np.broadcast_to(risk_total, (max_len,))

        strategies = np.full(max_len, "Neutral", dtype=object)

        strong_long = np.logical_and.reduce((s_comprehensive > th1, entry_level > 1.5, tc > 0, risk_total < r1))
        medium_long = np.logical_and.reduce((np.logical_and(th2 < s_comprehensive, s_comprehensive <= th1), np.logical_and(1 < entry_level, entry_level <= 1.5), tc > 0, risk_total < r2))
        weak_long = np.logical_and.reduce((np.logical_and(th3 < s_comprehensive, s_comprehensive <= th2), np.logical_and(0.5 < entry_level, entry_level <= 1), tc > 0, risk_total < r3))
        weak_short = np.logical_and.reduce((np.logical_and(-th2 < s_comprehensive, s_comprehensive <= -th3), np.logical_and(0.5 < entry_level, entry_level <= 1), tc < 0, risk_total < r3))
        medium_short = np.logical_and.reduce((np.logical_and(-th1 < s_comprehensive, s_comprehensive <= -th2), np.logical_and(1 < entry_level, entry_level <= 1.5), tc < 0, risk_total < r2))
        strong_short = np.logical_and.reduce((s_comprehensive <= -th1, entry_level > 1.5, tc < 0, risk_total < r1))

        strategies[strong_long] = "Strong_Long"
        strategies[medium_long] = "Medium_Long"
        strategies[weak_long] = "Weak_Long"
        strategies[weak_short] = "Weak_Short"
        strategies[medium_short] = "Medium_Short"
        strategies[strong_short] = "Strong_Short"

        return strategies

    def combine_timeframe_signals(self, signals):
        """
        Combine signals from different timeframes into a single signal.

        Args:
            signals (dict): Dictionary of signals for each timeframe.

        Returns:
            dict: Combined signal.
        """
        combined_signal = {
            'signal_strength': 0,
            'entry_level': 0,
            'stop_loss': 0,
            'position_size': 0
        }
        weights = {
            '1m': 0.05, '5m': 0.1, '15m': 0.15, '1h': 0.2, '4h': 0.2, '1d': 0.2, '1w': 0.1
        }
        total_weight = 0
        
        for tf, signal in signals.items():
            if tf in weights:
                weight = weights[tf]
                combined_signal['signal_strength'] += weight * np.mean(signal['signal_strength'])
                combined_signal['entry_level'] += weight * np.mean(signal['entry_level'])
                combined_signal['stop_loss'] += weight * np.mean(signal['stop_loss'])
                combined_signal['position_size'] += weight * np.mean(signal['position_size'])
                total_weight += weight
        
        if total_weight > 0:
            combined_signal['signal_strength'] /= total_weight
            combined_signal['entry_level'] /= total_weight
            combined_signal['stop_loss'] /= total_weight
            combined_signal['position_size'] /= total_weight
        
        logger.info(f"Combined signal: {combined_signal}")
        return combined_signal
