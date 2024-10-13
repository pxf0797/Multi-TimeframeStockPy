import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicWeightModule(nn.Module):
    def __init__(self, lstm_output_size, num_timeframes, hidden_size):
        super(DynamicWeightModule, self).__init__()
        self.lstm_output_size = lstm_output_size
        self.num_timeframes = num_timeframes
        total_input_size = lstm_output_size + 3 * num_timeframes
        self.fc1 = nn.Linear(total_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_timeframes)
        
    def forward(self, lstm_out, accuracy, trend_strength, volatility):
        # Ensure all inputs have the same dtype and are 1D
        lstm_out = lstm_out.float().view(-1)
        accuracy = accuracy.float()
        trend_strength = trend_strength.float()
        volatility = volatility.float()
        
        # Pad or truncate inputs to match the expected number of timeframes
        lstm_out = self._adjust_tensor(lstm_out, self.lstm_output_size)
        accuracy = self._adjust_tensor(accuracy, self.num_timeframes)
        trend_strength = self._adjust_tensor(trend_strength, self.num_timeframes)
        volatility = self._adjust_tensor(volatility, self.num_timeframes)
        
        x = torch.cat([lstm_out, accuracy, trend_strength, volatility])
        x = F.relu(self.fc1(x.unsqueeze(0)))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
    
    def _adjust_tensor(self, tensor, target_size):
        current_size = tensor.size(0)
        if current_size < target_size:
            return F.pad(tensor, (0, target_size - current_size))
        elif current_size > target_size:
            return tensor[:target_size]
        return tensor

def calculate_dynamic_weight(lstm_out, accuracy, trend_strength, volatility, alpha=1.0):
    lstm_out = lstm_out.view(-1)
    weights = torch.zeros(len(lstm_out), dtype=torch.float32)
    
    for i in range(len(lstm_out)):
        nn_input = lstm_out[i]
        acc = accuracy[i] if i < len(accuracy) else accuracy[-1]
        ts = trend_strength[i] if i < len(trend_strength) else trend_strength[-1]
        vol = volatility[i] if i < len(volatility) else volatility[-1]
        
        weight = nn_input * (1 + acc) * (1 + ts) * (vol ** alpha)
        weights[i] = weight
    
    return F.softmax(weights, dim=0)

def generate_signal(weighted_lstm_output, threshold=0.5):
    signal_strength = torch.tanh(weighted_lstm_output)
    
    buy_signal = (signal_strength > threshold).float()
    sell_signal = (signal_strength < -threshold).float()
    
    return signal_strength, buy_signal, sell_signal

if __name__ == "__main__":
    # Example usage
    lstm_output_size = 1
    num_timeframes = 3
    lstm_out = torch.randn(1, lstm_output_size, dtype=torch.float32)
    accuracy = torch.rand(num_timeframes, dtype=torch.float32)
    trend_strength = torch.rand(num_timeframes, dtype=torch.float32) * 2 - 1  # Range: [-1, 1]
    volatility = torch.rand(num_timeframes, dtype=torch.float32)
    
    dynamic_weight_module = DynamicWeightModule(lstm_output_size, num_timeframes, 32)
    weights = dynamic_weight_module(lstm_out, accuracy, trend_strength, volatility)
    
    weighted_output = (weights * lstm_out).sum()
    signal_strength, buy_signal, sell_signal = generate_signal(weighted_output)
    
    print("Weights:", weights)
    print("Signal Strength:", signal_strength)
    print("Buy Signal:", buy_signal)
    print("Sell Signal:", sell_signal)
