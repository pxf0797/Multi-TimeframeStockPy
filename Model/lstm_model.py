import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.float()  # Ensure input is float32
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device).float()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device).float()
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class MultiTimeframeLSTM(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_layers, output_size, num_timeframes):
        super(MultiTimeframeLSTM, self).__init__()
        self.models = nn.ModuleList([LSTMModel(input_size, hidden_size, num_layers, output_size) 
                                     for input_size in input_sizes])
        self.attention = nn.MultiheadAttention(embed_dim=output_size, num_heads=1)
        self.fc = nn.Linear(output_size, output_size)
        
    def forward(self, x_list):
        outputs = [model(x.float()) for model, x in zip(self.models, x_list)]  # Ensure each input is float32
        outputs = torch.stack(outputs, dim=0)
        
        attn_output, _ = self.attention(outputs, outputs, outputs)
        attn_output = attn_output.mean(dim=0)
        
        final_output = self.fc(attn_output)
        return final_output

if __name__ == "__main__":
    # Example usage
    input_sizes = [10, 10, 10]  # For 3 timeframes
    hidden_size = 64
    num_layers = 2
    output_size = 1
    num_timeframes = 3
    
    model = MultiTimeframeLSTM(input_sizes, hidden_size, num_layers, output_size, num_timeframes)
    print(model)
    
    # Test with random input
    x_list = [torch.randn(1, 100, input_size).float() for input_size in input_sizes]
    output = model(x_list)
    print("Output shape:", output.shape)
