import torch
import torch.nn as nn
import torch.optim as optim

class DynamicWeightModule(nn.Module):
    def __init__(self, num_timeframes, hidden_size):
        super(DynamicWeightModule, self).__init__()
        self.weight_calc = nn.Sequential(
            nn.Linear(num_timeframes * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_timeframes)
        )
    
    def forward(self, lstm_outputs, volatility, accuracy, trend_strength):
        batch_size = lstm_outputs.size(1)
        flattened = lstm_outputs.transpose(0, 1).reshape(batch_size, -1)
        raw_weights = self.weight_calc(flattened)
        adjusted_weights = raw_weights * (1 + accuracy) * (1 + trend_strength) * volatility
        return torch.softmax(adjusted_weights, dim=1)

class MultiTimeframeLSTM(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_layers, num_heads, output_size):
        super(MultiTimeframeLSTM, self).__init__()
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            for input_size in input_sizes
        ])
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.dynamic_weight = DynamicWeightModule(len(input_sizes), hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, volatility, accuracy, trend_strength):
        lstm_outs = [lstm(x_i)[0][:, -1, :] for lstm, x_i in zip(self.lstms, x)]
        lstm_outs = torch.stack(lstm_outs, dim=0)
        
        dynamic_weights = self.dynamic_weight(lstm_outs, volatility, accuracy, trend_strength)
        weighted_lstm_outs = lstm_outs * dynamic_weights.unsqueeze(2)
        
        attended, _ = self.attention(weighted_lstm_outs, weighted_lstm_outs, weighted_lstm_outs)
        
        attended_mean = attended.mean(dim=0)
        attended_mean = self.dropout(attended_mean)
        return self.fc(attended_mean), dynamic_weights

class ModelBuilder:
    def __init__(self, config):
        self.config = config

    def build_model(self, featured_data):
        input_sizes = [df.shape[1] - 2 for df in featured_data.values()]  # -2 for 'returns' and 'log_returns'
        model = MultiTimeframeLSTM(
            input_sizes,
            self.config['hidden_size'],
            self.config['num_layers'],
            self.config['num_heads'],
            output_size=3  # signal strength, entry level, stop loss point
        ).to(self.config['device'])
        return model

    def train_model(self, model, featured_data):
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        for epoch in range(self.config['epochs']):
            model.train()
            total_loss = 0
            for tf, df in featured_data.items():
                X = torch.FloatTensor(df.drop(['returns', 'log_returns'], axis=1).values).to(self.config['device'])
                y = torch.FloatTensor(df[['returns', 'Trend_Strength', 'ATR']].values).to(self.config['device'])
                
                volatility = torch.FloatTensor(df['Volatility'].values).to(self.config['device'])
                accuracy = torch.FloatTensor(df['Accuracy'].values).to(self.config['device'])
                trend_strength = torch.FloatTensor(df['Trend_Strength'].values).to(self.config['device'])
                
                optimizer.zero_grad()
                outputs, _ = model([X.unsqueeze(0)], volatility, accuracy, trend_strength)
                loss = criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step(total_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config["epochs"]}], Loss: {total_loss:.4f}')
        
        return model
