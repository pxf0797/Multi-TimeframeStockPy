# Multi-TimeframeStockPy

Multi-TimeframeStockPy is a sophisticated multi-timeframe stock trading system that uses machine learning to analyze market data across various timeframes and generate trading signals.

## Recent Updates

The data acquisition process has been optimized to use local CSV files instead of fetching data online. This change allows for faster data loading and the ability to use custom datasets.

## Data Acquisition

The system now reads data from CSV files for different time periods. Here's how it works:

1. CSV files should be placed in the `Data/csv_files` directory (configurable in `main.py`).
2. Files should be named in the format: `{symbol}_{timeframe}.csv` (e.g., `BTC-USD_1d.csv`).
3. The DataAcquisition class in `Data/data_acquisition.py` handles reading these files.
4. The DataProcessor class in `Model/data_processing.py` uses DataAcquisition to load and process the data.

## Usage

To run the system:

1. Ensure your CSV files are in the correct location (`Data/csv_files` by default).
2. Run the main script with the desired mode:

```
python main.py --mode [train|backtest|optimize|live]
```

Use the `--continue_training` flag to resume training from a saved model.

## Setup

1. Clone this repository.
2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Place your CSV data files in the `Data/csv_files` directory.

## Configuration

You can modify the configuration in the `load_config()` function in `main.py`. Key parameters include:

- `asset`: The trading asset (e.g., 'BTC-USD')
- `timeframes`: List of timeframes to analyze
- `data_dir`: Directory containing the CSV files

## Example

Here's a simple example of how to use the new data acquisition process:

```python
from Data.data_acquisition import DataAcquisition
from datetime import datetime, timedelta

# Initialize DataAcquisition
data_dir = 'Data/csv_files'
symbol = 'BTC-USD'
timeframes = ['1d', '1h', '15m']
data_acq = DataAcquisition(symbol, timeframes, data_dir)

# Set date range
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

# Fetch data
data_acq.fetch_data(start_date, end_date)

# Get data for a specific timeframe
daily_data = data_acq.get_data('1d')
print(daily_data.head())
```

This example demonstrates how to initialize the DataAcquisition class, fetch data for the last 30 days, and print the first few rows of the daily data.

## Contributing

Contributions to Multi-TimeframeStockPy are welcome. Please ensure that your code adheres to the project's coding standards and include tests for new functionality.

## License

[Insert your chosen license here]
