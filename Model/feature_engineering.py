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
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
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
                featured_data[tf] = df
                logger.info(f"Engineered features shape: {df.shape}")
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
            df[f'MA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
        
        # Calculate MACD
        fast, slow, signal = self.config['macd_params']
        try:
            df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False, min_periods=1).mean()
            df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False, min_periods=1).mean()
            df['MACD'] = df['EMA_fast'] - df['EMA_slow']
            df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False, min_periods=1).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}", exc_info=True)
        
        # Calculate RSI
        try:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
            df['RSI'] = 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}", exc_info=True)

        # Calculate ATR
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['ATR'] = true_range.rolling(window=14, min_periods=1).mean()
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}", exc_info=True)
        
        # Calculate other indicators
        try:
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            df['DIFF'] = df['EMA_12'] - df['EMA_26']
            df['DEA'] = df['DIFF'].ewm(span=9, adjust=False, min_periods=1).mean()
            df['MACD'] = 2 * (df['DIFF'] - df['DEA'])
            
            df['MOM'] = df['Close'].diff(10)
            
            df['TSI'] = self.calculate_tsi(df['Close'])
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
            df['Volatility'] = df['Close'].pct_change().rolling(window=20, min_periods=1).std() * np.sqrt(252)
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
            df['Trend_Strength'] = (df['MA_5'] - df['MA_20']) / df['MA_20'].replace(0, np.finfo(float).eps)
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
            df['VOL_MA_5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
            df['VOL_RATE'] = (df['Volume'] - df['VOL_MA_5']) / df['VOL_MA_5'].replace(0, np.finfo(float).eps)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
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
            ap = (df['High'] + df['Low'] + df['Close']) / 3
            esa = ap.ewm(span=n1, adjust=False, min_periods=1).mean()
            d = (ap - esa).abs().ewm(span=n1, adjust=False, min_periods=1).mean()
            ci = (ap - esa) / (0.015 * d.replace(0, np.finfo(float).eps))
            wt1 = ci.ewm(span=n2, adjust=False, min_periods=1).mean()
            wt2 = wt1.rolling(window=4, min_periods=1).mean()
            df['WaveTrend'] = wt1
            df['WaveTrend_Signal'] = wt2
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
            df['MACD_Signal'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
            df['Price_Direction'] = np.where(df['Close'].diff() > 0, 1, -1)
            df['Correct_Signal'] = np.where(df['MACD_Signal'] == df['Price_Direction'], 1, 0)
            df['Accuracy'] = df['Correct_Signal'].rolling(window=20, min_periods=1).mean()
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
                if column not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    min_val = df[column].min()
                    max_val = df[column].max()
                    if min_val != max_val:
                        df[column] = (df[column] - min_val) / (max_val - min_val)
                    else:
                        df[column] = 0  # or another appropriate value for constant features
        except Exception as e:
            logger.error(f"Error normalizing features: {e}", exc_info=True)
        return df
