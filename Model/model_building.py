import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from torchviz import make_dot
from Visualization.pytorch_visual import torch_model_visualize

logger = logging.getLogger(__name__)

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
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size):
        super(MultiTimeframeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.dynamic_weight = DynamicWeightModule(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, volatility, accuracy, trend_strength):
        lstm_out, _ = self.lstm(x)
        
        dynamic_weights = self.dynamic_weight(lstm_out, volatility.unsqueeze(-1), accuracy.unsqueeze(-1), trend_strength.unsqueeze(-1))
        weighted_lstm_out = lstm_out * dynamic_weights
        
        weighted_lstm_out = weighted_lstm_out.transpose(0, 1)
        attended, _ = self.attention(weighted_lstm_out, weighted_lstm_out, weighted_lstm_out)
        attended = attended.transpose(0, 1)
        
        attended = self.dropout(attended)
        output = self.fc(attended)
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
                input_size,
                self.config['hidden_size'],
                self.config['num_layers'],
                self.config['num_heads'],
                output_size=3  # signal strength, entry level, stop loss point
            ).to(self.config['device'])
            return model
        except StopIteration:
            logger.error("No data available in featured_data")
            return None
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
            for tf, df in train_data.items():
                try:
                    X, y, volatility, accuracy, trend_strength = self.prepare_data(df)
                    
                    optimizer.zero_grad()
                    outputs, _ = model(X.unsqueeze(0), volatility.unsqueeze(0), accuracy.unsqueeze(0), trend_strength.unsqueeze(0))
                    loss = criterion(outputs.squeeze(0), y)
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        total_loss += loss.item()
                        batch_count += 1
                    else:
                        logger.warning(f"NaN or Inf loss encountered in training for timeframe {tf}")
                except Exception as e:
                    logger.error(f"Error processing timeframe {tf}: {str(e)}")

            model.eval()
            val_loss = 0
            val_batch_count = 0
            with torch.no_grad():
                for tf, df in val_data.items():
                    try:
                        X, y, volatility, accuracy, trend_strength = self.prepare_data(df)
                        outputs, _ = model(X.unsqueeze(0), volatility.unsqueeze(0), accuracy.unsqueeze(0), trend_strength.unsqueeze(0))
                        loss = criterion(outputs.squeeze(0), y)
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            val_loss += loss.item()
                            val_batch_count += 1
                        else:
                            logger.warning(f"NaN or Inf loss encountered in validation for timeframe {tf}")
                    except Exception as e:
                        logger.error(f"Error processing validation data for timeframe {tf}: {str(e)}")
            
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
                logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
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

        # Check for NaN or Inf values
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

        #input_size = config['input_size']
        #sequence_length = config['sequence_length']
        #dummy_input = torch.randn(1, sequence_length, input_size)
        #dummy_volatility = torch.randn(1, sequence_length)
        #dummy_accuracy = torch.randn(1, sequence_length)
        #dummy_trend_strength = torch.randn(1, sequence_length)

        #output, _ = model(dummy_input, dummy_volatility, dummy_accuracy, dummy_trend_strength)
        #dot = make_dot(output, params=dict(model.named_parameters()))
        #dot.render("model_visualization", format="png", cleanup=True)
        #logger.info("Model visualization saved as 'model_visualization.png'")
        torch_model_visualize(model)

def print_model_summary(model, config):
    if model is None:
        logger.error("No model to summarize")
        return

    logger.info(str(model))
    logger.info(f"\nModel Parameter Count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    input_size = config['input_size']
    sequence_length = config['sequence_length']
    dummy_input = torch.randn(1, sequence_length, input_size)
    dummy_volatility = torch.randn(1, sequence_length)
    dummy_accuracy = torch.randn(1, sequence_length)
    dummy_trend_strength = torch.randn(1, sequence_length)
    
    with torch.no_grad():
        output, _ = model(dummy_input, dummy_volatility, dummy_accuracy, dummy_trend_strength)
    
    logger.info(f"\nInput shape: {dummy_input.shape}")
    logger.info(f"Output shape: {output.shape}")

