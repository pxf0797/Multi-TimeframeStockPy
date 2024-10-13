import torch
import torch.nn as nn

class MultiTimeframeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiTimeframeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size + 3, output_size)  # +3 for volatility, accuracy, trend_strength
        
    def forward(self, x, volatility, accuracy, trend_strength):
        # Print shapes for debugging
        print(f"x shape: {x.shape}")
        print(f"volatility shape: {volatility.shape}")
        print(f"accuracy shape: {accuracy.shape}")
        print(f"trend_strength shape: {trend_strength.shape}")

        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Ensure all tensors have the same batch size and are 2D
        batch_size = lstm_out.size(0)
        if volatility.dim() == 1:
            volatility = volatility.view(batch_size, 1)
        if accuracy.dim() == 1:
            accuracy = accuracy.view(batch_size, 1)
        if trend_strength.dim() == 1:
            trend_strength = trend_strength.view(batch_size, 1)
        
        # Print shapes after reshaping
        print(f"lstm_out shape: {lstm_out.shape}")
        print(f"volatility shape after reshape: {volatility.shape}")
        print(f"accuracy shape after reshape: {accuracy.shape}")
        print(f"trend_strength shape after reshape: {trend_strength.shape}")

        # Combine LSTM output with additional features
        combined = torch.cat([lstm_out, volatility, accuracy, trend_strength], dim=1)
        
        # Print combined shape
        print(f"combined shape: {combined.shape}")

        # Final fully connected layer
        out = self.fc(combined)
        return out

def build_model(config):
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    output_size = config['output_size']
    
    model = MultiTimeframeLSTM(input_size, hidden_size, num_layers, output_size)
    print(f"Model structure: {model}")
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs, volatility, accuracy, trend_strength, targets = [b.to(device) for b in batch]
            
            # Print shapes of input tensors
            print(f"inputs shape: {inputs.shape}")
            print(f"volatility shape: {volatility.shape}")
            print(f"accuracy shape: {accuracy.shape}")
            print(f"trend_strength shape: {trend_strength.shape}")
            print(f"targets shape: {targets.shape}")

            optimizer.zero_grad()
            outputs = model(inputs, volatility, accuracy, trend_strength)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, volatility, accuracy, trend_strength, targets = [b.to(device) for b in batch]
                outputs = model(inputs, volatility, accuracy, trend_strength)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}')
    
    return model
