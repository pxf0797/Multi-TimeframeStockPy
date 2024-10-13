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
        """
        Optimize the model's hyperparameters using scipy's minimize function.

        Args:
            model (torch.nn.Module): The initial model.
            featured_data (dict): Dictionary of DataFrames with engineered features for each timeframe.
            processed_data (dict): Dictionary of processed data for each timeframe.

        Returns:
            dict: Optimized hyperparameters.
        """
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
            """
            Objective function to minimize. It rebuilds and retrains the model with new parameters,
            then evaluates its performance using backtesting.

            Args:
                params (np.array): Array of hyperparameters to evaluate.

            Returns:
                float: Negative of the average Sharpe ratio across all timeframes.
            """
            # Update the model with new parameters
            self.config['learning_rate'] = params[0]
            self.config['num_layers'] = int(params[2])
            self.config['num_heads'] = int(params[3])
            
            # Ensure hidden_size is divisible by num_heads
            self.config['hidden_size'] = int(params[1] - (params[1] % self.config['num_heads']))
            
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
            return -evaluation_metric

        # Run the optimization
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)

        # Return the optimized parameters
        optimized_params = {
            'learning_rate': result.x[0],
            'hidden_size': int(result.x[1] - (result.x[1] % int(result.x[3]))),  # Ensure divisibility
            'num_layers': int(result.x[2]),
            'num_heads': int(result.x[3])
        }

        # Ensure hidden_size is at least num_heads
        optimized_params['hidden_size'] = max(optimized_params['hidden_size'], optimized_params['num_heads'])

        return optimized_params
