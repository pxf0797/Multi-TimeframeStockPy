import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from torchviz import make_dot
from Visualization.pytorch_visual import torch_model_visualize

logger = logging.getLogger(__name__)

class MultiPeriodConvBlock(nn.Module):
    """多周期卷积块，类似于处理RGB通道"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class PeriodAttention(nn.Module):
    """周期间的注意力机制"""
    def __init__(self, hidden_size, num_periods):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.period_weights = nn.Parameter(torch.ones(num_periods) / num_periods)
        
    def forward(self, period_features):
        batch_size, num_periods, seq_len, hidden_size = period_features.shape
        features = period_features.transpose(1, 2).reshape(batch_size * seq_len, num_periods, hidden_size)
        attn_output, _ = self.attention(features, features, features)
        weighted_output = attn_output * F.softmax(self.period_weights, dim=0).unsqueeze(0).unsqueeze(-1)
        return weighted_output.reshape(batch_size, seq_len, num_periods, hidden_size)

class DynamicWeightModule(nn.Module):
    def __init__(self, hidden_size):
        super(DynamicWeightModule, self).__init__()
        self.weight_calc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_outputs, volatility, accuracy, trend_strength):
        raw_weights = self.weight_calc(lstm_outputs)
        adjusted_weights = raw_weights * (1 + accuracy) * (1 + trend_strength) * volatility
        return torch.sigmoid(adjusted_weights)

class MultiTimeframeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size=3):
        super(MultiTimeframeLSTM, self).__init__()
        self.periods = ['5m', '15m', '60m', '1d', '1m', '1q']
        self.num_periods = len(self.periods)
        
        # Feature extractors for each period
        self.feature_extractors = nn.ModuleDict({
            period: MultiPeriodConvBlock(input_size, hidden_size//2) 
            for period in self.periods
        })
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size//2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Period attention
        self.period_attention = PeriodAttention(
            hidden_size=hidden_size * 2,  # bidirectional
            num_periods=self.num_periods
        )
        
        # Original components from previous implementation
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads)
        self.dynamic_weight = DynamicWeightModule(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x_dict, volatility, accuracy, trend_strength):
        batch_size = x_dict[self.periods[0]].shape[0]
        period_features = []
        
        # Process each period
        for period in self.periods:
            x = x_dict[period].transpose(1, 2)
            features = self.feature_extractors[period](x)
            period_features.append(features.transpose(1, 2))
        
        # LSTM processing for each period
        lstm_outputs = []
        for period_feat in period_features:
            lstm_out, _ = self.lstm(period_feat)
            lstm_outputs.append(lstm_out)
        
        # Stack period outputs
        stacked_outputs = torch.stack(lstm_outputs, dim=1)
        
        # Period attention
        attended_features = self.period_attention(stacked_outputs)
        
        # Flatten features
        flat_features = attended_features.reshape(
            batch_size, -1, self.lstm.hidden_size * 2
        )
        
        # Dynamic weighting
        dynamic_weights = self.dynamic_weight(
            flat_features[:, -1, :],
            volatility,
            accuracy,
            trend_strength
        )
        
        weighted_features = flat_features * dynamic_weights.unsqueeze(1)
        
        # Multi-head attention
        weighted_features = weighted_features.transpose(0, 1)
        attended_output, _ = self.attention(
            weighted_features,
            weighted_features,
            weighted_features
        )
        attended_output = attended_output.transpose(0, 1)
        
        # Final output
        attended_output = self.dropout(attended_output)
        output = self.fc(attended_output[:, -1, :])
        
        return output, dynamic_weights

class ModelBuilder:
    def __init__(self, config):
        self.config = config

    def build_model(self, featured_data):
        if not featured_data:
            logger.error("No data available to build the model")
            return None

        try:
            if 'input_size' in self.config:
                input_size = self.config['input_size']
            else:
                sample_df = next(iter(featured_data.values()))
                if sample_df.empty:
                    logger.error("Sample DataFrame is empty")
                    return None
                input_size = len(sample_df.columns) - 2  # -2 for 'returns' and 'log_returns'

            logger.info(f"Building model with input_size: {input_size}")
            
            model = MultiTimeframeLSTM(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                num_heads=self.config['num_heads']
            ).to(self.config['device'])
            
            return model
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return None

    def train_model(self, model, featured_data):
        if model is None:
            logger.error("No model to train")
            return None

        if not featured_data:
            logger.error("No data available for training")
            return model

        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        train_data, val_data = self.split_data(featured_data)

        for epoch in range(self.config['epochs']):
            model.train()
            total_loss = 0
            batch_count = 0
            
            try:
                for tf, df in train_data.items():
                    X, y, volatility, accuracy, trend_strength = self.prepare_data(df)
                    
                    optimizer.zero_grad()
                    outputs, _ = model(X.unsqueeze(0), volatility, accuracy, trend_strength)
                    loss = criterion(outputs.squeeze(0), y)
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        total_loss += loss.item()
                        batch_count += 1
                    else:
                        logger.warning(f"NaN or Inf loss encountered in training")
            except Exception as e:
                logger.error(f"Error in training: {str(e)}")

            model.eval()
            val_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for tf, df in val_data.items():
                    try:
                        X, y, volatility, accuracy, trend_strength = self.prepare_data(df)
                        outputs, _ = model(X.unsqueeze(0), volatility, accuracy, trend_strength)
                        loss = criterion(outputs.squeeze(0), y)
                        
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            val_loss += loss.item()
                            val_batch_count += 1
                        else:
                            logger.warning(f"NaN or Inf loss encountered in validation")
                    except Exception as e:
                        logger.error(f"Error in validation: {str(e)}")

            if batch_count > 0:
                avg_train_loss = total_loss / batch_count
            else:
                avg_train_loss = float('nan')

            if val_batch_count > 0:
                avg_val_loss = val_loss / val_batch_count
                scheduler.step(avg_val_loss)
            else:
                avg_val_loss = float('nan')

            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                          f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        return model

    def prepare_data(self, df):
        required_columns = ['Volatility', 'Accuracy', 'Trend_Strength', 'returns', 'ATR']
        if not all(col in df.columns for col in required_columns):
            missing_columns = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_columns}")

        X = torch.FloatTensor(df.drop(['returns', 'log_returns'], axis=1).values).to(self.config['device'])
        y = torch.FloatTensor(df[['returns', 'Trend_Strength', 'ATR']].values).to(self.config['device'])
        volatility = torch.FloatTensor(df['Volatility'].values).to(self.config['device'])
        accuracy = torch.FloatTensor(df['Accuracy'].values).to(self.config['device'])
        trend_strength = torch.FloatTensor(df['Trend_Strength'].values).to(self.config['device'])

        if torch.isnan(X).any() or torch.isinf(X).any():
            raise ValueError("NaN or Inf values found in input data")
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError("NaN or Inf values found in target data")

        return X, y, volatility, accuracy, trend_strength

    def split_data(self, data, train_ratio=0.8):
        train_data = {}
        val_data = {}
        for tf, df in data.items():
            split_idx = int(len(df) * train_ratio)
            train_data[tf] = df.iloc[:split_idx]
            val_data[tf] = df.iloc[split_idx:]
        return train_data, val_data

    def visualize_model(self, model, config):
        if model is None:
            logger.error("No model to visualize")
            return
            
        try:
            torch_model_visualize(model)
            logger.info("Model visualization completed")
        except Exception as e:
            logger.error(f"Error in model visualization: {str(e)}")

def print_model_summary(model, config):
    if model is None:
        logger.error("No model to summarize")
        return

    logger.info(str(model))
    logger.info(f"\nModel Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create example inputs
    batch_size = 1
    sequence_length = config['sequence_length']
    x_dict = {
        period: torch.randn(batch_size, sequence_length, config['input_size'])
        for period in ['5m', '15m', '60m', '1d', '1m', '1q']
    }
    volatility = torch.randn(batch_size, sequence_length)
    accuracy = torch.randn(batch_size, sequence_length)
    trend_strength = torch.randn(batch_size, sequence_length)
    
    with torch.no_grad():
        output, _ = model(x_dict, volatility, accuracy, trend_strength)
    
    logger.info(f"\nInput shapes:")
    for period, tensor in x_dict.items():
        logger.info(f"{period}: {tensor.shape}")
    logger.info(f"Output shape: {output.shape}")