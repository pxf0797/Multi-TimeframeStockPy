import numpy as np
import torch

class SignalGenerator:
    def __init__(self, config):
        self.config = config

    def generate_signals(self, model, featured_data):
        signals = {}
        dynamic_weights = {}

        for tf, df in featured_data.items():
            print(f"\nDebugging signal generation for timeframe {tf}:")
            
            # Only use the last sequence_length rows
            df = df.iloc[-self.config['sequence_length']:]
            
            # Only drop 'returns' and 'log_returns' if they exist
            columns_to_drop = [col for col in ['returns', 'log_returns'] if col in df.columns]
            features = df.drop(columns_to_drop, axis=1, errors='ignore')
            
            print(f"Features included: {features.columns.tolist()}")
            print(f"Number of features: {len(features.columns)}")
            print(f"Number of rows: {len(features)}")
            
            # Convert to tensor and only use the first 39 features
            features_tensor = torch.FloatTensor(features.values[:, :39]).to(self.config['device'])
            
            # Extract required additional features (now the last 3 columns)
            volatility = features_tensor[:, -3]
            accuracy = features_tensor[:, -2]
            trend_strength = features_tensor[:, -1]
            
            with torch.no_grad():
                output = model(features_tensor.unsqueeze(0), volatility.unsqueeze(0), accuracy.unsqueeze(0), trend_strength.unsqueeze(0))
                signal = output.squeeze().cpu().numpy()

            print(f"Raw model output shape: {signal.shape}")
            print(f"Raw model output range: {np.min(signal)} to {np.max(signal)}")
            print(f"Number of NaN in raw output: {np.isnan(signal).sum()}")
            print(f"Number of inf in raw output: {np.isinf(signal).sum()}")

            # Apply activation function (e.g., tanh) to bound the signal
            signal = np.tanh(signal)

            # Ensure signal is an array
            if np.isscalar(signal):
                signal = np.full(len(df), signal)

            print(f"Processed signal shape: {signal.shape}")
            print(f"Processed signal range: {np.min(signal)} to {np.max(signal)}")
            print(f"Number of NaN in processed signal: {np.isnan(signal).sum()}")
            print(f"Number of inf in processed signal: {np.isinf(signal).sum()}")

            signals[tf] = signal
            dynamic_weights[tf] = self.calculate_dynamic_weight(features)

        return signals, dynamic_weights

    def calculate_dynamic_weight(self, df):
        volatility = df['Volatility'].values
        trend_strength = df['Trend_Strength'].values
        
        print(f"Volatility shape: {volatility.shape}")
        print(f"Trend_Strength shape: {trend_strength.shape}")
        
        # If volatility is 2D, use only the first column
        if volatility.ndim > 1:
            volatility = volatility[:, 0]
        
        # Avoid division by zero
        volatility = np.where(volatility == 0, 1e-8, volatility)
        
        # Normalize weights
        weight = (1 / volatility) * np.abs(trend_strength)
        weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight) + 1e-8)
        
        print(f"Calculated weight shape: {weight.shape}")
        
        return weight

    def generate_comprehensive_signal(self, signals, dynamic_weights, trend_consistency):
        weighted_signals = []
        for tf in self.config['timeframes']:
            if tf in signals and tf in dynamic_weights:
                weighted_signals.append(signals[tf] * dynamic_weights[tf])
        
        if not weighted_signals:
            return np.zeros(self.config['sequence_length'])
        
        comprehensive_signal = np.mean(weighted_signals, axis=0)
        comprehensive_signal *= trend_consistency
        
        return comprehensive_signal

    def determine_entry_strategy(self, comprehensive_signal, entry_level, trend_consistency, risk_total):
        signal_strength = np.abs(comprehensive_signal)
        if signal_strength > entry_level and trend_consistency > 0:
            return 'long' if comprehensive_signal > 0 else 'short'
        elif signal_strength > entry_level and trend_consistency < 0:
            return 'short' if comprehensive_signal > 0 else 'long'
        else:
            return 'hold'
