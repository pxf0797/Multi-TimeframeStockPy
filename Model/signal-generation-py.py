import numpy as np
import torch

class SignalGenerator:
    def __init__(self, config):
        self.config = config

    def generate_signals(self, model, featured_data):
        model.eval()
        signals = {}
        dynamic_weights = {}
        with torch.no_grad():
            for tf, df in featured_data.items():
                X = torch.FloatTensor(df.drop(['