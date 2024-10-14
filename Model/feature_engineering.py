import pandas as pd
import numpy as np
from ta_wrapper import ta
from Utils.utils import handle_nan_inf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config):
        self.config = config

    def engineer_features(self, data):
        """
        Main method to engineer features for all timeframes.
        
        Args:
            data (dict): Dictionary of DataFrames for each timeframe.
        
        Returns:
            dict: Dictionary of DataFrames with engineered features for each timeframe.
        """
        featured_data = {}
        for tf, df in data.items():
            logger.info(f"Processing timeframe: {tf}")
            logger.info(f"DataFrame shape: {df.shape}")
            if df.empty:
                logger.warning(f"Empty DataFrame for timeframe {tf}")
                continue
            try:
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_columns):
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    raise ValueError(f"Missing required columns: {missing_columns}")

                df = self.calculate_technical_indicators(df)
                df = self.calculate_volatility(df)
                df = self.calculate_trend_strength(df)
                df = self.calculate_volume_indicators(df)
                df = self.calculate_wave_trend(df)
                df = self.calculate_accuracy(df)
                df = handle_nan_inf(df)  # Handle NaN and Inf values
                df = self.normalize_features(df)  # Normalize features
                df = self.maintain_sequence_length(df)
                
                # Ensure required columns are present with correct capitalization
                required_features = ['Volatility', 'Accuracy', 'trend_strength', 'ATR']
                for feature in required_features:
                    if feature not in df.columns:
                        df[feature] = 0  # Set to 0 if not calculated
                
                featured_data[tf] = df
                logger.info(f"Engineered features shape: {df.shape}")
                logger.info(f"Columns: {df.columns}")
            except Exception as e:
                logger.error(f"Error processing timeframe {tf}: {e}", exc_info=True)
        return featured_data

    def calculate_technical_indicators(self, df):
        """
        Calculate various technical indicators.
        
        Args:
            df (pd.DataFrame): Input DataFrame with OHLCV data.
        
        Returns:
            pd.DataFrame: DataFrame with added technical indicators.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to calculate_technical_indicators")
            return df

        # Calculate Moving Averages
        for period in self.config['ma_periods']:
            df[f'ma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
        
        # Calculate MACD
        fast, slow, signal = self.config['macd_params']
        try:
            df['ema_fast'] = df['close'].ewm(span=fast, adjust=False, min_periods=1).mean()
            df['ema_slow'] = df['close'].ewm(span=slow, adjust=False, min_periods=1).mean()
            df['macd'] = df['ema_fast'] - df['ema_slow']
            df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False, min_periods=1).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}", exc_info=True)
        
        # Calculate RSI
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
            df['rsi'] = 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}", exc_info=True)

        # Calculate ATR
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}", exc_info=True)
        
        # Calculate other indicators
        try:
            df['ema_12'] = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
            df['diff'] = df['ema_12'] - df['ema_26']
            df['dea'] = df['diff'].ewm(span=9, adjust=False, min_periods=1).mean()
            df['macd'] = 2 * (df['diff'] - df['dea'])
            
            df['mom'] = df['close'].diff(10)
            
            df['tsi'] = self.calculate_tsi(df['close'])
        except Exception as e:
            logger.error(f"Error calculating other indicators: {e}", exc_info=True)
        
        return df

    def calculate_volatility(self, df):
        """
        Calculate volatility using standard deviation of percentage changes.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with added volatility.
        """
        try:
            df['Volatility'] = df['close'].pct_change().rolling(window=20, min_periods=1).std() * np.sqrt(252)
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}", exc_info=True)
        return df

    def calculate_trend_strength(self, df):
        """
        Calculate trend strength using the difference between short-term and long-term moving averages.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with added trend strength.
        """
        try:
            df['trend_strength'] = (df['ma_5'] - df['ma_20']) / df['ma_20'].replace(0, np.finfo(float).eps)
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}", exc_info=True)
        return df

    def calculate_volume_indicators(self, df):
        """
        Calculate volume-based indicators.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with added volume indicators.
        """
        try:
            df['vol_ma_5'] = df['volume'].rolling(window=5, min_periods=1).mean()
            df['vol_rate'] = (df['volume'] - df['vol_ma_5']) / df['vol_ma_5'].replace(0, np.finfo(float).eps)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}", exc_info=True)
        return df

    def calculate_tsi(self, close, r=25, s=13):
        """
        Calculate True Strength Index (TSI).
        
        Args:
            close (pd.Series): Close price series.
            r (int): First smoothing period.
            s (int): Second smoothing period.
        
        Returns:
            pd.Series: TSI values.
        """
        try:
            diff = close - close.shift(1)
            abs_diff = abs(diff)
            
            smooth_diff = diff.ewm(span=r, adjust=False, min_periods=1).mean().ewm(span=s, adjust=False, min_periods=1).mean()
            smooth_abs_diff = abs_diff.ewm(span=r, adjust=False, min_periods=1).mean().ewm(span=s, adjust=False, min_periods=1).mean()
            
            tsi = 100 * smooth_diff / smooth_abs_diff.replace(0, np.finfo(float).eps)
            return tsi
        except Exception as e:
            logger.error(f"Error calculating TSI: {e}", exc_info=True)
            return pd.Series(index=close.index)

    def calculate_wave_trend(self, df, n1=10, n2=21):
        """
        Calculate WaveTrend indicator.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            n1 (int): First period.
            n2 (int): Second period.
        
        Returns:
            pd.DataFrame: DataFrame with added WaveTrend indicators.
        """
        try:
            ap = (df['high'] + df['low'] + df['close']) / 3
            esa = ap.ewm(span=n1, adjust=False, min_periods=1).mean()
            d = (ap - esa).abs().ewm(span=n1, adjust=False, min_periods=1).mean()
            ci = (ap - esa) / (0.015 * d.replace(0, np.finfo(float).eps))
            wt1 = ci.ewm(span=n2, adjust=False, min_periods=1).mean()
            wt2 = wt1.rolling(window=4, min_periods=1).mean()
            df['wavetrend'] = wt1
            df['wavetrend_signal'] = wt2
        except Exception as e:
            logger.error(f"Error calculating WaveTrend: {e}", exc_info=True)
        return df

    def calculate_accuracy(self, df):
        """
        Calculate accuracy based on MACD crossovers.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with added accuracy indicator.
        """
        try:
            df['macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
            df['price_direction'] = np.where(df['close'].diff() > 0, 1, -1)
            df['correct_signal'] = np.where(df['macd_signal'] == df['price_direction'], 1, 0)
            df['Accuracy'] = df['correct_signal'].rolling(window=20, min_periods=1).mean()
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}", exc_info=True)
        return df

    def maintain_sequence_length(self, df):
        """
        Maintain a consistent sequence length for all DataFrames.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with adjusted sequence length.
        """
        seq_length = self.config['sequence_length']
        if len(df) > seq_length:
            return df.iloc[-seq_length:]
        elif len(df) < seq_length:
            pad_length = seq_length - len(df)
            pad_df = pd.DataFrame(index=range(pad_length), columns=df.columns)
            return pd.concat([pad_df, df]).reset_index(drop=True)
        else:
            return df

    def normalize_features(self, df):
        """
        Normalize features using min-max scaling.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with normalized features.
        """
        try:
            for column in df.columns:
                if column not in ['open', 'high', 'low', 'close', 'volume']:
                    min_val = df[column].min()
                    max_val = df[column].max()
                    if min_val != max_val:
                        df[column] = (df[column] - min_val) / (max_val - min_val)
                    else:
                        df[column] = 0  # or another appropriate value for constant features
        except Exception as e:
            logger.error(f"Error normalizing features: {e}", exc_info=True)
        return df
