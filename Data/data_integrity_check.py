import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Set, List, Tuple, Optional, Union

def load_holidays(holiday_file: str) -> Set[date]:
    """Load holiday dates from a CSV file."""
    holidays_df = pd.read_csv(holiday_file, parse_dates=['date'])
    return set(holidays_df['date'].dt.date)

def check_data_content(df: pd.DataFrame) -> List[str]:
    """Check the content of the dataframe for various data integrity issues."""
    issues_found = []
    expected_columns = ['day', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = set(expected_columns) - set(df.columns)
    
    if missing_columns:
        issues_found.append(f"Missing {len(missing_columns)} columns: {', '.join(missing_columns)}")
    
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            invalid_rows = df[~pd.to_numeric(df[col], errors='coerce').notna()]
            if not invalid_rows.empty:
                issues_found.append(f"'{col}' column has {len(invalid_rows)} rows with invalid data:")
                for idx, row in invalid_rows.iterrows():
                    issues_found.append(f"  Date: {row['day']}, {col}: {row[col]} - Invalid numeric value")
            
            if col != 'volume':
                unreasonable_prices = df[(df[col] <= 0) | (df[col] > 1000)]
                if not unreasonable_prices.empty:
                    issues_found.append(f"'{col}' column has {len(unreasonable_prices)} rows with suspicious prices:")
                    for idx, row in unreasonable_prices.iterrows():
                        reason = "Price <= 0" if row[col] <= 0 else "Price > 1000"
                        issues_found.append(f"  Date: {row['day']}, {col}: {row[col]} - {reason}")
    
    if 'volume' in df.columns:
        unreasonable_volume = df[df['volume'] < 0]
        if not unreasonable_volume.empty:
            issues_found.append(f"'volume' column has {len(unreasonable_volume)} rows with suspicious volume:")
            for idx, row in unreasonable_volume.iterrows():
                issues_found.append(f"  Date: {row['day']}, Volume: {row['volume']} - Negative volume")
    
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

def print_summary(df: pd.DataFrame, issues_found: List[str], period_name: str) -> None:
    """Print a summary of the data integrity check."""
    print("\nCheck Status Summary:")
    if issues_found:
        print("The following issues were found:")
        for issue in issues_found:
            print(f"- {issue}")
    else:
        print("No issues found. Data integrity is good.")
    
    print("\nCheck Content Summary:")
    print(f"- Total rows: {len(df)}")
    print(f"- Date range: from {df['day'].min()} to {df['day'].max()}")
    print(f"- Columns checked: {', '.join(df.columns)}")
    if period_name in ['Daily', 'Weekly', 'Monthly', 'Quarterly']:
        print(f"- Number of {period_name.lower()} periods: {len(df)}")
    else:
        print(f"- Number of trading days: {len(df['day'].dt.date.unique())}")
    
    print(f"\n{period_name} data integrity check completed.")

def check_period_data_integrity(file_path: str, holiday_file: str, period: str) -> None:
    """Check the integrity of period (daily, weekly, monthly, quarterly) stock data."""
    df = pd.read_csv(file_path, parse_dates=['day'])
    holidays = load_holidays(holiday_file)
    df = df.sort_values('day')
    
    print(f"Checking {period} data integrity:")
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
    
    print_summary(df, issues_found, period.capitalize())

def get_last_trading_day(date_input: Union[datetime, date], holidays: Set[date], period: str) -> Optional[date]:
    """Get the last trading day of the period for a given date."""
    if isinstance(date_input, datetime):
        date_input = date_input.date()
    
    if period == 'week':
        end_date = date_input + timedelta(days=6 - date_input.weekday())
    elif period == 'month':
        next_month = date_input.replace(day=28) + timedelta(days=4)
        end_date = next_month - timedelta(days=next_month.day)
    elif period == 'quarter':
        quarter_end = pd.Timestamp(date_input).to_period('Q').end_time.date()
        end_date = quarter_end
    else:
        raise ValueError(f"Invalid period: {period}")
    
    for i in range(10):
        day = end_date - timedelta(days=i)
        if day.weekday() < 5 and day not in holidays:
            return day
    return None

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

def check_intraday_data_integrity(file_path: str, holiday_file: str, period: str) -> None:
    """
    Check the integrity of intraday stock data for Chinese stock market.
    
    Args:
    file_path (str): Path to the CSV file containing intraday stock data.
    holiday_file (str): Path to the CSV file containing holiday dates.
    period (str): The trading period ('5min', '15min', or '60min').
    """
    df = pd.read_csv(file_path, parse_dates=['day'])
    holidays = load_holidays(holiday_file)
    df = df.sort_values('day')
    
    trading_hours = generate_trading_hours(period)
    
    print(f"Checking {period} data integrity:")
    issues_found = []
    
    for date, group in df.groupby(df['day'].dt.date):
        if date.weekday() < 5 and date not in holidays:
            times = group['day'].dt.strftime('%H:%M').tolist()
            missing_times = set(trading_hours) - set(times)
            if missing_times:
                issues_found.append(f"Date {date} is missing {len(missing_times)} time periods: {', '.join(sorted(missing_times))}")
    
    issues_found.extend(check_data_content(df))
    
    print_summary(df, issues_found, f"{period} intraday")

def test_data_integrity():
    """Test function to run integrity checks on various types of stock data."""
    file_path = 'csv_files/'
    holiday_file = 'chinese_holidays.csv'

    data_types = {
        'daily': ('sz000001_1d_1983-09-24_2024-10-18_2.csv', check_period_data_integrity, 'daily'),
        'weekly': ('sz000001_1w_1909-10-22_2024-10-18.csv', check_period_data_integrity, 'weekly'),
        'monthly': ('sz000001_1m_1860-07-10_2024-10-18.csv', check_period_data_integrity, 'monthly'),
        'quarterly': ('sz000001_1q_1901-08-05_2024-10-18.csv', check_period_data_integrity, 'quarterly'),
        '60min': ('sz000001_60m_1983-09-24_2024-10-18.csv', check_intraday_data_integrity, '60min'),
        '15min': ('sz000001_15m_1983-09-24_2024-10-18_2.csv', check_intraday_data_integrity, '15min'),
        '5min': ('sz000001_5m_1983-09-24_2024-10-18_2.csv', check_intraday_data_integrity, '5min')
    }

    for data_type, (filename, check_function, period) in data_types.items():
        print(f"\nTesting {data_type} data integrity check:")
        try:
            check_function(file_path + filename, holiday_file, period)
        except Exception as e:
            print(f"Error occurred while checking {data_type} data: {str(e)}")
        print("="*50)

if __name__ == "__main__":
    test_data_integrity()
