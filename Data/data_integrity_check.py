import pandas as pd
import numpy as np
import os
import glob
import logging
from datetime import datetime, timedelta, date
from typing import Set, List, Tuple, Optional, Union

def load_holidays(holiday_file: str) -> Set[date]:
    """
    Load holiday dates from a CSV file.
    
    Args:
    holiday_file (str): Path to the CSV file containing holiday dates.
    
    Returns:
    Set[date]: A set of holiday dates.
    """
    holidays_df = pd.read_csv(holiday_file, parse_dates=['date'])
    return set(holidays_df['date'].dt.date)

def check_data_content(df: pd.DataFrame) -> List[str]:
    """
    Check the content of the dataframe for various data integrity issues.
    
    Checks include:
    1. Missing columns
    2. Invalid numeric data
    3. Suspicious price data (<=0 or >1000)
    4. Negative trading volume
    5. Inconsistent price data (e.g., low price higher than high price)
    
    Args:
    df (pd.DataFrame): The dataframe to check.
    
    Returns:
    List[str]: A list of issues found.
    """
    issues_found = []
    expected_columns = ['day', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = set(expected_columns) - set(df.columns)
    
    # Check for missing columns
    if missing_columns:
        issues_found.append(f"Missing {len(missing_columns)} columns: {', '.join(missing_columns)}")
    
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            # Check for invalid numeric data
            invalid_rows = df[~pd.to_numeric(df[col], errors='coerce').notna()]
            if not invalid_rows.empty:
                issues_found.append(f"'{col}' column has {len(invalid_rows)} rows with invalid data:")
                for idx, row in invalid_rows.iterrows():
                    issues_found.append(f"  Date: {row['day']}, {col}: {row[col]} - Invalid numeric value")
            
            # Check for suspicious price data
            if col != 'volume':
                unreasonable_prices = df[(df[col] <= 0) | (df[col] > 1000)]
                if not unreasonable_prices.empty:
                    issues_found.append(f"'{col}' column has {len(unreasonable_prices)} rows with suspicious prices:")
                    for idx, row in unreasonable_prices.iterrows():
                        reason = "Price <= 0" if row[col] <= 0 else "Price > 1000"
                        issues_found.append(f"  Date: {row['day']}, {col}: {row[col]} - {reason}")
    
    # Check for negative trading volume
    if 'volume' in df.columns:
        unreasonable_volume = df[df['volume'] < 0]
        if not unreasonable_volume.empty:
            issues_found.append(f"'volume' column has {len(unreasonable_volume)} rows with suspicious volume:")
            for idx, row in unreasonable_volume.iterrows():
                issues_found.append(f"  Date: {row['day']}, Volume: {row['volume']} - Negative volume")
    
    # Check for inconsistent price data
    price_issues = df[(df['low'] > df['high']) | (df['open'] > df['high']) | (df['open'] < df['low']) |
                      (df['close'] > df['high']) | (df['close'] < df['low'])]
    if not price_issues.empty:
        issues_found.append(f"{len(price_issues)} rows with inconsistent price data:")
        for idx, row in price_issues.iterrows():
            reasons = []
            if row['low'] > row['high']:
                reasons.append("Low > High")
            if row['open'] > row['high']:
                reasons.append("Open > High")
            if row['open'] < row['low']:
                reasons.append("Open < Low")
            if row['close'] > row['high']:
                reasons.append("Close > High")
            if row['close'] < row['low']:
                reasons.append("Close < Low")
            issues_found.append(f"  Date: {row['day']}, Open: {row['open']}, High: {row['high']}, Low: {row['low']}, Close: {row['close']} - Reasons: {', '.join(reasons)}")
    
    return issues_found

def get_last_trading_day(date_input: Union[datetime, date], holidays: Set[date], period: str) -> Optional[date]:
    """
    Get the last trading day of the period for a given date.

    Args:
    date_input (Union[datetime, date]): The input date.
    holidays (Set[date]): Set of holiday dates.
    period (str): The period type ('week', 'month', or 'quarter').

    Returns:
    Optional[date]: The last trading day of the period, or None if not found.
    """
    if isinstance(date_input, datetime):
        date_input = date_input.date()
    
    if period == 'week':
        # Find the last weekday (Friday) of the week
        end_date = date_input + timedelta(days=(4 - date_input.weekday() + 7) % 7)
        
        # If Friday is a holiday, move backwards to find the last trading day
        while end_date.weekday() >= 5 or end_date in holidays:
            end_date -= timedelta(days=1)
            
        # Ensure we're still in the same week
        if end_date < date_input:
            return None
        
        return end_date
    
    elif period == 'month':
        # Find the last day of the month
        next_month = date_input.replace(day=28) + timedelta(days=4)
        end_date = next_month - timedelta(days=next_month.day)
        
        # Move backwards to find the last trading day of the month
        while end_date.weekday() >= 5 or end_date in holidays:
            end_date -= timedelta(days=1)
        
        return end_date
    
    elif period == 'quarter':
        # Find the last day of the quarter
        quarter_end = pd.Timestamp(date_input).to_period('Q').end_time.date()
        
        # Move backwards to find the last trading day of the quarter
        while quarter_end.weekday() >= 5 or quarter_end in holidays:
            quarter_end -= timedelta(days=1)
        
        return quarter_end
    
    else:
        raise ValueError(f"Invalid period: {period}")

def generate_trading_hours(period: str) -> List[str]:
    """
    Generate trading hours based on the given period for Chinese stock market.
    Starts from 09:30 and 13:00 but excludes these times from the final result.
    
    Args:
    period (str): The trading period ('5min', '15min', or '60min').
    
    Returns:
    List[str]: A list of trading times in 'HH:MM' format.
    """
    trading_sessions = [
        ('09:30', '11:30'),
        ('13:00', '15:00')
    ]
    
    if period == '5min':
        interval = timedelta(minutes=5)
    elif period == '15min':
        interval = timedelta(minutes=15)
    elif period == '60min':
        interval = timedelta(minutes=60)
    else:
        raise ValueError(f"Unsupported period: {period}")
    
    trading_hours = []
    for start, end in trading_sessions:
        current_time = datetime.strptime(start, '%H:%M')
        end_time = datetime.strptime(end, '%H:%M')
        while current_time <= end_time:
            trading_hours.append(current_time.strftime('%H:%M'))
            current_time += interval
    
    # Remove 09:30 and 13:00 from the generated times
    trading_hours = [time for time in trading_hours if time not in ['09:30', '13:00']]
    
    return trading_hours

def log_summary(df: pd.DataFrame, issues_found: List[str], period_name: str, logger: logging.Logger) -> None:
    """
    Log a summary of the data integrity check.

    Args:
    df (pd.DataFrame): The dataframe that was checked.
    issues_found (List[str]): List of issues found during the check.
    period_name (str): Name of the period (e.g., "Daily", "Weekly", "Quarterly").
    logger (logging.Logger): Logger object for writing to log file.
    """
    logger.info("\nCheck Status Summary:")
    if issues_found:
        logger.info("The following issues were found:")
        for issue in issues_found:
            logger.info(f"- {issue}")
    else:
        logger.info("No issues found. Data integrity is good.")
    
    logger.info("\nCheck Content Summary:")
    logger.info(f"- Total rows: {len(df)}")
    logger.info(f"- Date range: from {df['day'].min()} to {df['day'].max()}")
    logger.info(f"- Columns checked: {', '.join(df.columns)}")
    if period_name in ['Daily', 'Weekly', 'Monthly', 'Quarterly']:
        logger.info(f"- Number of {period_name.lower()} periods: {len(df)}")
    else:
        logger.info(f"- Number of trading days: {len(df['day'].dt.date.unique())}")
    
    logger.info(f"\n{period_name} data integrity check completed.")

def check_period_data_integrity(file_path: str, holiday_file: str, period: str, logger: logging.Logger) -> None:
    """
    Check the integrity of period (daily, weekly, monthly, quarterly) stock data.

    Args:
    file_path (str): Path to the CSV file containing stock data.
    holiday_file (str): Path to the CSV file containing holiday dates.
    period (str): The period type ('daily', 'weekly', 'monthly', or 'quarterly').
    logger (logging.Logger): Logger object for writing to log file.
    """
    logger.info(f"Checking file: {os.path.basename(file_path)}")
    print(f"Checking file: {os.path.basename(file_path)}")

    df = pd.read_csv(file_path, parse_dates=['day'])
    holidays = load_holidays(holiday_file)
    df = df.sort_values('day')
    
    logger.info(f"Checking {period} data integrity:")
    issues_found = []
    
    start_date = df['day'].min().date()
    end_date = df['day'].max().date()
    
    if period == 'daily':
        current_period = start_date
        delta = timedelta(days=1)
    elif period == 'weekly':
        current_period = start_date - timedelta(days=start_date.weekday())
        delta = timedelta(days=7)
    elif period == 'monthly':
        current_period = start_date.replace(day=1)
        delta = pd.DateOffset(months=1)
    elif period == 'quarterly':
        current_period = pd.Timestamp(start_date).to_period('Q').start_time.date()
        delta = pd.DateOffset(months=3)
    
    expected_periods = []
    while current_period <= end_date:
        if period == 'daily':
            if current_period.weekday() < 5 and current_period not in holidays:
                expected_periods.append(current_period)
        else:
            last_trading_day = get_last_trading_day(current_period, holidays, period[:-2])
            if last_trading_day:
                expected_periods.append(last_trading_day)
        current_period += delta
        if isinstance(current_period, pd.Timestamp):
            current_period = current_period.date()
    
    actual_periods = df['day'].dt.date.tolist()
    
    missing_periods = set(expected_periods) - set(actual_periods)
    extra_periods = set(actual_periods) - set(expected_periods)
    
    if missing_periods:
        issues_found.append(f"Missing {len(missing_periods)} {period} of data:")
        for missing_date in sorted(missing_periods):
            issues_found.append(f"  Missing date: {missing_date}")
    
    if extra_periods:
        issues_found.append(f"{len(extra_periods)} extra {period} present:")
        for extra_date in sorted(extra_periods):
            issues_found.append(f"  Extra date: {extra_date}")
    
    if period != 'daily':
        incorrect_last_trading_days = []
        for _, row in df.iterrows():
            expected_last_trading_day = get_last_trading_day(row['day'].date(), holidays, period[:-2])
            if row['day'].date() != expected_last_trading_day:
                incorrect_last_trading_days.append((row['day'].date(), expected_last_trading_day))
        
        if incorrect_last_trading_days:
            issues_found.append(f"{len(incorrect_last_trading_days)} {period} have incorrect last trading days:")
            for actual, expected in incorrect_last_trading_days:
                issues_found.append(f"  Actual: {actual}, Expected: {expected}")

    issues_found.extend(check_data_content(df))
    
    log_summary(df, issues_found, period.capitalize(), logger)

def check_intraday_data_integrity(file_path: str, holiday_file: str, period: str, logger: logging.Logger) -> None:
    """
    Check the integrity of intraday stock data for Chinese stock market.
    
    Args:
    file_path (str): Path to the CSV file containing intraday stock data.
    holiday_file (str): Path to the CSV file containing holiday dates.
    period (str): The trading period ('5min', '15min', or '60min').
    logger (logging.Logger): Logger object for writing to log file.
    """
    logger.info(f"Checking file: {os.path.basename(file_path)}")
    print(f"Checking file: {os.path.basename(file_path)}")

    df = pd.read_csv(file_path, parse_dates=['day'])
    holidays = load_holidays(holiday_file)
    df = df.sort_values('day')
    
    trading_hours = generate_trading_hours(period)
    
    logger.info(f"Checking {period} data integrity:")
    issues_found = []
    
    for date, group in df.groupby(df['day'].dt.date):
        if date.weekday() < 5 and date not in holidays:
            times = group['day'].dt.strftime('%H:%M').tolist()
            missing_times = set(trading_hours) - set(times)
            if missing_times:
                issues_found.append(f"Date {date} is missing {len(missing_times)} time periods: {', '.join(sorted(missing_times))}")
    
    issues_found.extend(check_data_content(df))
    
    log_summary(df, issues_found, f"{period} intraday", logger)

def test_data_integrity():
    """
    Test function to run integrity checks on various types of stock data.
    """
    file_path = 'csv_files/'
    holiday_file = 'chinese_holidays.csv'
    symbol = "sz000001"
    timeframes = ["5m", "15m", "60m", "1d", "1w", "1m", "1q"]
    log_dir = 'integrity_check_logs'
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    log_filename = f"{symbol}_integrity_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    logging.basicConfig(filename=log_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    data_types = {
        '1d': ('daily', check_period_data_integrity, 'daily'),
        '1w': ('weekly', check_period_data_integrity, 'weekly'),
        '1m': ('monthly', check_period_data_integrity, 'monthly'),
        '1q': ('quarterly', check_period_data_integrity, 'quarterly'),
        '60m': ('60min', check_intraday_data_integrity, '60min'),
        '15m': ('15min', check_intraday_data_integrity, '15min'),
        '5m': ('5min', check_intraday_data_integrity, '5min')
    }

    def glob_file_match(symbol, timeframe):
        pattern = os.path.join(file_path, f"{symbol}*{timeframe}*.csv")
        matching_files = glob.glob(pattern)
        return matching_files[0] if matching_files else None

    summary = []

    for timeframe in timeframes:
        if timeframe in data_types:
            print(f"\nTesting {timeframe} data integrity check for {symbol}:")
            data_type, check_function, period = data_types[timeframe]
            filename = glob_file_match(symbol, timeframe)
            if filename:
                try:
                    check_function(filename, holiday_file, period, logger)
                    summary.append(f"{symbol} {timeframe}: Check completed.")
                except Exception as e:
                    logger.error(f"Error occurred while checking {timeframe} data: {str(e)}")
                    summary.append(f"{symbol} {timeframe}: Error during check.")
            else:
                print(f"No matching file found for {symbol} with timeframe {timeframe}")
                summary.append(f"{symbol} {timeframe}: No matching file found.")
            print("="*50)
        else:
            print(f"No check defined for timeframe: {timeframe}")
            summary.append(f"{symbol} {timeframe}: No check defined.")

    print("\nSummary of Integrity Checks:")
    for item in summary:
        print(item)
    print(f"\nDetailed log can be found in {log_filename}")

if __name__ == "__main__":
    test_data_integrity()
    