import pandas as pd
from datetime import datetime, timedelta

def load_holidays(holiday_file):
    """
    Load holiday dates from a CSV file.
    
    Args:
    holiday_file (str): Path to the CSV file containing holiday dates.
    
    Returns:
    set: A set of holiday dates.
    """
    holidays_df = pd.read_csv(holiday_file, parse_dates=['date'])
    return set(holidays_df['date'].dt.date)

def check_data_content(df):
    """
    Check the content of the dataframe for various data integrity issues.
    
    This function checks for:
    1. Missing columns
    2. Invalid data in numeric columns
    3. Unreasonable price data
    4. Negative volume data
    5. Inconsistent price data (e.g., low price higher than high price)
    
    Args:
    df (pd.DataFrame): The dataframe to check.
    
    Returns:
    list: A list of issues found during the check.
    """
    issues_found = []
    
    # Check if all expected columns are present
    expected_columns = ['day', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        print(f"  Missing columns: {', '.join(missing_columns)}")
        issues_found.append(f"Missing {len(missing_columns)} columns")
    
    # Check numeric columns for validity and reasonableness
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            # Check for non-numeric data
            invalid_rows = df[~df[col].apply(lambda x: isinstance(x, (int, float)))].index
            if not invalid_rows.empty:
                print(f"  Invalid data in '{col}' column, number of rows: {len(invalid_rows)}")
                issues_found.append(f"'{col}' column has {len(invalid_rows)} rows with invalid data")
            
            # Check for unreasonable price data (excluding volume)
            if col != 'volume':
                unreasonable_prices = df[(df[col] <= 0) | (df[col] > 1000)].index
                if not unreasonable_prices.empty:
                    print(f"  Suspicious price data in '{col}' column, number of rows: {len(unreasonable_prices)}")
                    issues_found.append(f"'{col}' column has {len(unreasonable_prices)} rows with suspicious prices")
    
    # Check for negative volume
    if 'volume' in df.columns:
        unreasonable_volume = df[df['volume'] < 0].index
        if not unreasonable_volume.empty:
            print(f"  Suspicious volume data in 'volume' column, number of rows: {len(unreasonable_volume)}")
            issues_found.append(f"'volume' column has {len(unreasonable_volume)} rows with suspicious volume")
    
    # Check for price inconsistencies
    price_issues = df[(df['low'] > df['high']) | (df['open'] > df['high']) | (df['open'] < df['low']) |
                      (df['close'] > df['high']) | (df['close'] < df['low'])]
    if not price_issues.empty:
        print(f"  Inconsistent price data, number of rows: {len(price_issues)}")
        for index, row in price_issues.iterrows():
            print(f"    Date/Time: {row['day']}, Open: {row['open']}, High: {row['high']}, Low: {row['low']}, Close: {row['close']}")
        issues_found.append(f"{len(price_issues)} rows with inconsistent price data")
    
    return issues_found

def print_summary(df, issues_found, period_name):
    """
    Print a summary of the data integrity check.
    
    Args:
    df (pd.DataFrame): The dataframe that was checked.
    issues_found (list): List of issues found during the check.
    period_name (str): Name of the period (e.g., "Daily", "Weekly", "Quarterly").
    """
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
    if period_name in ['Daily', 'Weekly', 'Quarterly']:
        print(f"- Number of {period_name.lower()} periods: {len(df)}")
    else:
        print(f"- Number of trading days: {len(df['day'].dt.date.unique())}")
    
    print(f"\n{period_name} data integrity check completed.")

def check_daily_data_integrity(file_path, holiday_file):
    """
    Check the integrity of daily stock data.
    
    This function checks for:
    1. Missing workdays (excluding weekends and holidays)
    2. Data content issues
    
    Args:
    file_path (str): Path to the CSV file containing daily stock data.
    holiday_file (str): Path to the CSV file containing holiday dates.
    """
    df = pd.read_csv(file_path, parse_dates=['day'])
    holidays = load_holidays(holiday_file)
    df = df.sort_values('day')
    
    print("Checking daily data integrity:")
    issues_found = []
    
    # Check for missing workdays
    date_range = pd.date_range(start=df['day'].min(), end=df['day'].max())
    missing_dates = set(date_range) - set(df['day'])
    
    work_days_missing = [date for date in sorted(missing_dates) 
                         if date.weekday() < 5 and date.date() not in holidays]
    
    if work_days_missing:
        print("\n1. The following workdays are missing (excluding weekends and holidays):")
        for date in work_days_missing:
            print(f"   {date.date()}")
        issues_found.append(f"Missing {len(work_days_missing)} non-holiday workdays")
    else:
        print("\n1. All non-holiday workdays are present.")
    
    # Check data content
    print("\n2. Checking data content:")
    issues_found.extend(check_data_content(df))
    
    print_summary(df, issues_found, "Daily")

def get_last_trading_day_of_week(date, holidays):
    """
    Get the last trading day of the week for a given date.
    
    Args:
    date (datetime): The date to check.
    holidays (set): Set of holiday dates.
    
    Returns:
    date: The last trading day of the week, or None if no trading day found.
    """
    for i in range(4, -1, -1):
        day = date - timedelta(days=date.weekday() - i)
        if day.date() not in holidays:
            return day.date()
    return None

def check_weekly_data_integrity(file_path, holiday_file):
    """
    Check the integrity of weekly stock data.
    
    This function checks for:
    1. Missing or extra weeks
    2. Incorrect last trading day of each week
    3. Data content issues
    
    Args:
    file_path (str): Path to the CSV file containing weekly stock data.
    holiday_file (str): Path to the CSV file containing holiday dates.
    """
    df = pd.read_csv(file_path, parse_dates=['day'])
    holidays = load_holidays(holiday_file)
    df = df.sort_values('day')
    
    print("Checking weekly data integrity:")
    issues_found = []
    
    # Check for missing or extra weeks
    start_date = df['day'].min()
    end_date = df['day'].max()
    
    current_week = start_date
    expected_weeks = []
    while current_week <= end_date:
        last_trading_day = get_last_trading_day_of_week(current_week, holidays)
        if last_trading_day:
            expected_weeks.append(last_trading_day)
        current_week += timedelta(days=7)
    
    actual_weeks = df['day'].dt.date.tolist()
    
    missing_weeks = set(expected_weeks) - set(actual_weeks)
    extra_weeks = set(actual_weeks) - set(expected_weeks)
    
    if missing_weeks:
        print("\n1. The following weeks are missing data:")
        for week in sorted(missing_weeks):
            print(f"   - {week}")
        issues_found.append(f"Missing {len(missing_weeks)} weeks of data")
    
    if extra_weeks:
        print("\n  The following extra weeks are present in the data:")
        for week in sorted(extra_weeks):
            print(f"   - {week}")
        issues_found.append(f"{len(extra_weeks)} extra weeks present")
    
    # Check if each week's data is on the last trading day
    incorrect_last_trading_days = []
    for _, row in df.iterrows():
        expected_last_trading_day = get_last_trading_day_of_week(row['day'], holidays)
        if row['day'].date() != expected_last_trading_day:
            incorrect_last_trading_days.append((row['day'].date(), expected_last_trading_day))
    
    if incorrect_last_trading_days:
        print("\n2. The following weeks have incorrect last trading days:")
        for actual, expected in incorrect_last_trading_days:
            print(f"   - Actual: {actual}, Expected: {expected or 'None'}")
        issues_found.append(f"{len(incorrect_last_trading_days)} weeks have incorrect last trading days")
    
    # Check data content
    print("\n3. Checking data content:")
    issues_found.extend(check_data_content(df))
    
    print_summary(df, issues_found, "Weekly")

def get_last_trading_day_of_quarter(date, holidays):
    """
    Get the last trading day of the quarter for a given date.
    
    Args:
    date (datetime): The date to check.
    holidays (set): Set of holiday dates.
    
    Returns:
    date: The last trading day of the quarter, or None if no trading day found.
    """
    quarter_end = pd.Timestamp(date).to_period('Q').end_time.date()
    for i in range(7):  # Check up to 7 days before quarter end
        day = quarter_end - timedelta(days=i)
        if day.weekday() < 5 and day not in holidays:
            return day
    return None

def check_quarterly_data_integrity(file_path, holiday_file):
    """
    Check the integrity of quarterly stock data.
    
    This function checks for:
    1. Missing or extra quarters
    2. Incorrect last trading day of each quarter
    3. Data content issues
    
    Args:
    file_path (str): Path to the CSV file containing quarterly stock data.
    holiday_file (str): Path to the CSV file containing holiday dates.
    """
    df = pd.read_csv(file_path, parse_dates=['day'])
    holidays = load_holidays(holiday_file)
    df = df.sort_values('day')
    
    print("Checking quarterly data integrity:")
    issues_found = []
    
    # Check for missing or extra quarters
    start_date = df['day'].min()
    end_date = df['day'].max()
    
    current_quarter = pd.Timestamp(start_date).to_period('Q')
    end_quarter = pd.Timestamp(end_date).to_period('Q')
    expected_quarters = []
    
    while current_quarter <= end_quarter:
        last_trading_day = get_last_trading_day_of_quarter(current_quarter.end_time, holidays)
        if last_trading_day:
            expected_quarters.append(last_trading_day)
        current_quarter += 1
    
    actual_quarters = df['day'].dt.date.tolist()
    
    missing_quarters = set(expected_quarters) - set(actual_quarters)
    extra_quarters = set(actual_quarters) - set(expected_quarters)
    
    if missing_quarters:
        print("\n1. The following quarters are missing data:")
        for quarter in sorted(missing_quarters):
            print(f"   - {quarter}")
        issues_found.append(f"Missing {len(missing_quarters)} quarters of data")
    
    if extra_quarters:
        print("\n  The following extra quarters are present in the data:")
        for quarter in sorted(extra_quarters):
            print(f"   - {quarter}")
        issues_found.append(f"{len(extra_quarters)} extra quarters present")
    
    # Check if each quarter's data is on the last trading day
    incorrect_last_trading_days = []
    for _, row in df.iterrows():
        expected_last_trading_day = get_last_trading_day_of_quarter(row['day'], holidays)
        if row['day'].date() != expected_last_trading_day:
            incorrect_last_trading_days.append((row['day'].date(), expected_last_trading_day))
    
    if incorrect_last_trading_days:
        print("\n2. The following quarters have incorrect last trading days:")
        for actual, expected in incorrect_last_trading_days:
            print(f"   - Actual: {actual}, Expected: {expected or 'None'}")
        issues_found.append(f"{len(incorrect_last_trading_days)} quarters have incorrect last trading days")

    # Check data content
    print("\n3. Checking data content:")
    issues_found.extend(check_data_content(df))
    
    print_summary(df, issues_found, "Quarterly")

def check_intraday_data_integrity(file_path, holiday_file, period_name, trading_hours):
    """
    Check the integrity of intraday stock data.
    
    This function checks for:
    1. Missing time periods on trading days
    2. Data content issues
    
    Args:
    file_path (str): Path to the CSV file containing intraday stock data.
    holiday_file (str): Path to the CSV file containing holiday dates.
    period_name (str): Name of the intraday period (e.g., "60-minute", "15-minute", "5-minute").
    trading_hours (list): List of expected trading times in 'HH:MM' format.
    """
    df = pd.read_csv(file_path, parse_dates=['day'])
    holidays = load_holidays(holiday_file)
    df = df.sort_values('day')
    
    print(f"Checking {period_name} data integrity:")
    issues_found = []
    
    # Check for missing time periods on trading days
    for date, group in df.groupby(df['day'].dt.date):
        if date.weekday() < 5 and date not in holidays:  # Workday and not a holiday
            times = group['day'].dt.strftime('%H:%M').tolist()
            missing_times = set(trading_hours) - set(times)
            if missing_times:
                print(f"Date {date} is missing the following time periods: {','.join(missing_times)}")
                issues_found.append(f"Date {date} is missing {len(missing_times)} time periods")
    
    # Check data content
    print("\n2. Checking data content:")
    issues_found.extend(check_data_content(df))
    
    print_summary(df, issues_found, period_name)

def check_60min_data_integrity(file_path, holiday_file):
    """
    Check the integrity of 60-minute interval stock data.
    
    This function is a wrapper for check_intraday_data_integrity with 60-minute specific trading hours.
    
    Args:
    file_path (str): Path to the CSV file containing 60-minute interval stock data.
    holiday_file (str): Path to the CSV file containing holiday dates.
    """
    trading_hours = ['10:30', '11:30', '14:00', '15:00']
    check_intraday_data_integrity(file_path, holiday_file, "60-minute", trading_hours)

def check_15min_data_integrity(file_path, holiday_file):
    """
    Check the integrity of 15-minute interval stock data.
    
    This function is a wrapper for check_intraday_data_integrity with 15-minute specific trading hours.
    
    Args:
    file_path (str): Path to the CSV file containing 15-minute interval stock data.
    holiday_file (str): Path to the CSV file containing holiday dates.
    """
    trading_hours = [
        '09:45', '10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30',
        '13:15', '13:30', '13:45', '14:00', '14:15', '14:30', '14:45', '15:00'
    ]
    check_intraday_data_integrity(file_path, holiday_file, "15-minute", trading_hours)

def check_5min_data_integrity(file_path, holiday_file):
    """
    Check the integrity of 5-minute interval stock data.
    
    This function is a wrapper for check_intraday_data_integrity with 5-minute specific trading hours.
    
    Args:
    file_path (str): Path to the CSV file containing 5-minute interval stock data.
    holiday_file (str): Path to the CSV file containing holiday dates.
    """
    trading_hours = [
        '09:35', '09:40', '09:45', '09:50', '09:55',
        '10:00', '10:05', '10:10', '10:15', '10:20', '10:25', '10:30', '10:35', '10:40', '10:45', '10:50', '10:55',
        '11:00', '11:05', '11:10', '11:15', '11:20', '11:25', '11:30',
        '13:05', '13:10', '13:15', '13:20', '13:25', '13:30', '13:35', '13:40', '13:45', '13:50', '13:55',
        '14:00', '14:05', '14:10', '14:15', '14:20', '14:25', '14:30', '14:35', '14:40', '14:45', '14:50', '14:55',
        '15:00'
    ]
    check_intraday_data_integrity(file_path, holiday_file, "5-minute", trading_hours)

def test_data_integrity():
    """
    Test function to run integrity checks on various types of stock data.
    
    This function demonstrates how to use the different data integrity check functions
    for daily, weekly, quarterly, and intraday (60-minute, 15-minute, 5-minute) data.
    
    Note: Ensure that the CSV files mentioned in this function exist in the same directory,
    or provide the full path to these files.
    """
    # Test daily data integrity check
    print("Testing daily data integrity check:")
    check_daily_data_integrity('sz000001_1d_1983-09-24_2024-10-18_2.csv', 'chinese_holidays.csv')

    print("\n" + "="*50 + "\n")

    # Test weekly data integrity check
    print("Testing weekly data integrity check:")
    check_weekly_data_integrity('sz000001_1w_1909-10-22_2024-10-18.csv', 'chinese_holidays.csv')

    print("\n" + "="*50 + "\n")

    # Test quarterly data integrity check
    print("Testing quarterly data integrity check:")
    check_quarterly_data_integrity('sz000001_1q_1901-08-05_2024-10-18.csv', 'chinese_holidays.csv')

    print("\n" + "="*50 + "\n")

    # Test 60-minute data integrity check
    print("Testing 60-minute data integrity check:")
    check_60min_data_integrity('sz000001_60m_1983-09-24_2024-10-18.csv', 'chinese_holidays.csv')
    
    print("\n" + "="*50 + "\n")

    # Test 15-minute data integrity check
    print("Testing 15-minute data integrity check:")
    check_15min_data_integrity('sz000001_15m_1983-09-24_2024-10-18_2.csv', 'chinese_holidays.csv')

    print("\n" + "="*50 + "\n")

    # Test 5-minute data integrity check
    print("Testing 5-minute data integrity check:")
    check_5min_data_integrity('sz000001_5m_1983-09-24_2024-10-18_2.csv', 'chinese_holidays.csv')

# Run the test function if this script is executed directly
if __name__ == "__main__":
    test_data_integrity()