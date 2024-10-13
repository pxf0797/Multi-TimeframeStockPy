import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DynamicWeightModule(nn.Module):
    """
    Module for calculating dynamic weights based on LSTM outputs and additional features.
    """
    def __init__(self, hidden_size):
        super(DynamicWeightModule, self).__init__()
        self.weight_calc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_outputs, volatility, accuracy, trend_strength):
        """
        Calculate dynamic weights.
        
        Args:
            lstm_outputs (torch.Tensor): Output from LSTM layer.
            volatility (torch.Tensor): Volatility values.
            accuracy (torch.Tensor): Accuracy values.
            trend_strength (torch.Tensor): Trend strength values.
        
        Returns:
            torch.Tensor: Calculated dynamic weights.
        """
        raw_weights = self.weight_calc(lstm_outputs)
        adjusted_weights = raw_weights * (1 + accuracy) * (1 + trend_strength) * volatility
        return torch.sigmoid(adjusted_weights)

class MultiTimeframeLSTM(nn.Module):
    """
    Multi-timeframe LSTM model with attention and dynamic weighting.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(MultiTimeframeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.dynamic_weight = DynamicWeightModule(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, volatility, accuracy, trend_strength):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
            volatility (torch.Tensor): Volatility values.
            accuracy (torch.Tensor): Accuracy values.
            trend_strength (torch.Tensor): Trend strength values.
        
        Returns:
            tuple: Output tensor and dynamic weights.
        """
        lstm_out, _ = self.lstm(x)
        
        dynamic_weights = self.dynamic_weight(lstm_out, volatility.unsqueeze(-1), accuracy.unsqueeze(-1), trend_strength.unsqueeze(-1))
        weighted_lstm_out = lstm_out * dynamic_weights
        
        # Reshape for attention
        weighted_lstm_out = weighted_lstm_out.transpose(0, 1)
        attended, _ = self.attention(weighted_lstm_out, weighted_lstm_out, weighted_lstm_out)
        attended = attended.transpose(0, 1)
        
        attended = self.dropout(attended)
        output = self.fc(attended)
        return output, dynamic_weights

class ModelBuilder:
    """
    Class for building and training the multi-timeframe LSTM model.
    """
    def __init__(self, config):
        self.config = config

    def build_model(self, featured_data):
        """
        Build the model based on the input data and configuration.
        
        Args:
            featured_data (dict): Dictionary of DataFrames with engineered features.
        
        Returns:
            MultiTimeframeLSTM: Instantiated model.
        """
        # Calculate input size based on actual feature count
        sample_df = next(iter(featured_data.values()))
        input_size = len(sample_df.columns) - 2  # -2 for 'returns' and 'log_returns'
        print(f"Building model with input_size: {input_size}")
        model = MultiTimeframeLSTM(
            input_size,
            self.config['hidden_size'],
            self.config['num_layers'],
            self.config['num_heads'],
            output_size=3  # signal strength, entry level, stop loss point
        ).to(self.config['device'])
        return model

    def train_model(self, model, featured_data):
        """
        Train the model using the provided data.
        
        Args:
            model (MultiTimeframeLSTM): The model to train.
            featured_data (dict): Dictionary of DataFrames with engineered features.
        
        Returns:
            MultiTimeframeLSTM: Trained model.
        """
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        # Split data into train and validation sets
        train_data, val_data = self.split_data(featured_data)

        for epoch in range(self.config['epochs']):
            model.train()
            total_loss = 0
            for tf, df in train_data.items():
                try:
                    X, y, volatility, accuracy, trend_strength = self.prepare_data(df)
                    
                    optimizer.zero_grad()
                    outputs, _ = model(X.unsqueeze(0), volatility.unsqueeze(0), accuracy.unsqueeze(0), trend_strength.unsqueeze(0))
                    loss = criterion(outputs.squeeze(0), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                except Exception as e:
                    print(f"Error processing timeframe {tf}: {e}")

            # Validation step
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for tf, df in val_data.items():
                    try:
                        X, y, volatility, accuracy, trend_strength = self.prepare_data(df)
                        outputs, _ = model(X.unsqueeze(0), volatility.unsqueeze(0), accuracy.unsqueeze(0), trend_strength.unsqueeze(0))
                        val_loss += criterion(outputs.squeeze(0), y).item()
                    except Exception as e:
                        print(f"Error processing validation data for timeframe {tf}: {e}")
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config["epochs"]}], Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        return model

    def prepare_data(self, df):
        """
        Prepare data for model input.
        
        Args:
            df (pd.DataFrame): DataFrame with features.
        
        Returns:
            tuple: Tensors for model input (X, y, volatility, accuracy, trend_strength).
        """
        required_columns = ['Volatility', 'Accuracy', 'Trend_Strength', 'returns', 'ATR']
        if not all(col in df.columns for col in required_columns):
            missing_columns = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_columns}")

        X = torch.FloatTensor(df.drop(['returns', 'log_returns'], axis=1).values).to(self.config['device'])
        y = torch.FloatTensor(df[['returns', 'Trend_Strength', 'ATR']].values).to(self.config['device'])
        volatility = torch.FloatTensor(df['Volatility'].values).to(self.config['device'])
        accuracy = torch.FloatTensor(df['Accuracy'].values).to(self.config['device'])
        trend_strength = torch.FloatTensor(df['Trend_Strength'].values).to(self.config['device'])

        return X, y, volatility, accuracy, trend_strength

    def split_data(self, data, train_ratio=0.8):
        """
        Split data into training and validation sets.
        
        Args:
            data (dict): Dictionary of DataFrames with features.
            train_ratio (float): Ratio of data to use for training.
        
        Returns:
            tuple: Dictionaries of training and validation data.
        """
        train_data = {}
        val_data = {}
        for tf, df in data.items():
            split_idx = int(len(df) * train_ratio)
            train_data[tf] = df.iloc[:split_idx]
            val_data[tf] = df.iloc[split_idx:]
        return train_data, val_data

def print_model_summary(model, config):
    """
    Print a summary of the model architecture and output shapes.
    
    Args:
        model (MultiTimeframeLSTM): The model to summarize.
        config (dict): Configuration dictionary.
    """
    print(model)
    print(f"\nModel Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create a dummy input
    input_size = config['input_size']
    sequence_length = config['sequence_length']
    dummy_input = torch.randn(1, sequence_length, input_size)
    dummy_volatility = torch.randn(1, sequence_length)
    dummy_accuracy = torch.randn(1, sequence_length)
    dummy_trend_strength = torch.randn(1, sequence_length)
    
    # Pass the dummy input through the model
    with torch.no_grad():
        output, _ = model(dummy_input, dummy_volatility, dummy_accuracy, dummy_trend_strength)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
