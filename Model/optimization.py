import numpy as np
from scipy.optimize import minimize
from model_building import ModelBuilder
from backtesting import Backtester

class Optimizer:
    def __init__(self, config):
        self.config = config
        self.model_builder = ModelBuilder(config)
        self.backtester = Backtester(config)

    def optimize(self, model, featured_data, processed_data):
        # Define the parameters to optimize
        initial_params = np.array([
            self.config['learning_rate'],
            self.config['hidden_size'],
            self.config['num_layers'],
            self.config['num_heads']
        ])

        # Define the bounds for each parameter
        bounds = [
            (1e-5, 1e-1),  # learning_rate
            (32, 256),     # hidden_size
            (1, 5),        # num_layers
            (1, 8)         # num_heads
        ]

        # Define the objective function
        def objective(params):
            # Check for NaN values in params
            if np.isnan(params).any():
                return 1e10  # Return a large error value

            try:
                # Update the model with new parameters
                self.config['learning_rate'] = params[0]
                self.config['num_layers'] = max(1, min(5, int(params[2])))  # Ensure it's an integer between 1 and 5
                self.config['num_heads'] = max(1, min(8, int(params[3])))   # Ensure it's an integer between 1 and 8
                
                # Ensure hidden_size is divisible by num_heads
                self.config['hidden_size'] = max(32, int(params[1] - (params[1] % self.config['num_heads'])))
                
                # Ensure hidden_size is at least num_heads
                self.config['hidden_size'] = max(self.config['hidden_size'], self.config['num_heads'])

                # Rebuild and retrain the model with new parameters
                new_model = self.model_builder.build_model(featured_data)
                trained_model = self.model_builder.train_model(new_model, featured_data)

                # Generate signals using the trained model
                signal_generator = self.config['signal_generator']
                signals, dynamic_weights = signal_generator.generate_signals(trained_model, featured_data)

                # Run backtesting
                backtest_results = self.backtester.run_backtest(signals, processed_data, dynamic_weights)

                # Use the average Sharpe ratio across all timeframes as the evaluation metric
                evaluation_metric = np.mean([result['Sharpe Ratio'] for result in backtest_results.values()])

                # Return the negative of the evaluation metric (we want to maximize it)
                return -evaluation_metric if not np.isnan(evaluation_metric) else 1e10
            except Exception as e:
                print(f"Error in objective function: {e}")
                return 1e10  # Return a large error value

        # Run the optimization
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)

        # Return the optimized parameters
        optimized_params = {
            'learning_rate': result.x[0],
            'hidden_size': max(32, int(result.x[1] - (result.x[1] % max(1, min(8, int(result.x[3])))))),
            'num_layers': max(1, min(5, int(result.x[2]))),
            'num_heads': max(1, min(8, int(result.x[3])))
        }

        # Ensure hidden_size is at least num_heads
        optimized_params['hidden_size'] = max(optimized_params['hidden_size'], optimized_params['num_heads'])

        return optimized_params
