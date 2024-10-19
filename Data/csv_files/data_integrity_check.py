import pandas as pd
from datetime import datetime, timedelta

def load_holidays(holiday_file):
    holidays_df = pd.read_csv(holiday_file, parse_dates=['date'])
    return set(holidays_df['date'].dt.date)

def check_data_integrity(file_path, holiday_file):
    # 读取CSV文件
    df = pd.read_csv(file_path, parse_dates=['day'])
    
    # 加载假期数据
    holidays = load_holidays(holiday_file)
    
    # 按日期排序
    df = df.sort_values('day')
    
    print("检查数据完整性:")
    issues_found = []
    
    # 1. 检查连续的周期是否完整
    date_range = pd.date_range(start=df['day'].min(), end=df['day'].max())
    missing_dates = set(date_range) - set(df['day'])
    existing_dates = set(df['day'].dt.date)
    
    work_days_missing = [date for date in sorted(missing_dates) 
                         if date.weekday() < 5 and date.date() not in holidays]
    
    extra_days = existing_dates - set(date.date() for date in date_range if date.weekday() < 5 and date.date() not in holidays)
    
    print("\n1. 检查日期完整性:")
    if work_days_missing:
        print("  以下工作日缺失（不包括周末和法定假期）:")
        for date in work_days_missing:
            print(f"   - {date.date()}")
        issues_found.append(f"缺失 {len(work_days_missing)} 个非假期工作日")
    else:
        print("  所有非假期工作日的数据都存在。")
    
    if extra_days:
        print("\n  数据中存在以下多余的日期（周末或法定假期）:")
        for date in sorted(extra_days):
            print(f"   - {date}")
        issues_found.append(f"存在 {len(extra_days)} 个多余的日期（周末或法定假期）")
    else:
        print("  数据中不存在多余的日期。")
    
    # 2. 检查每个数据内容是否完整以及内容是否正确
    print("\n2. 检查数据内容:")
    
    # 检查列是否完整
    expected_columns = ['day', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        print(f"  缺失列: {', '.join(missing_columns)}")
        issues_found.append(f"缺失 {len(missing_columns)} 列")
    
    # 检查数值列的有效性
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            invalid_rows = df[~df[col].apply(lambda x: isinstance(x, (int, float)))].index
            if not invalid_rows.empty:
                print(f"  '{col}'列中存在无效数据，行数: {len(invalid_rows)}")
                issues_found.append(f"'{col}'列有 {len(invalid_rows)} 行无效数据")
            
            # 检查价格数据的合理性
            if col != 'volume':
                unreasonable_prices = df[(df[col] <= 0) | (df[col] > 1000)].index
                if not unreasonable_prices.empty:
                    print(f"  '{col}'列中存在可疑的价格数据，行数: {len(unreasonable_prices)}")
                    issues_found.append(f"'{col}'列有 {len(unreasonable_prices)} 行可疑价格")
    
    # 检查成交量的合理性
    if 'volume' in df.columns:
        unreasonable_volume = df[df['volume'] <= 0].index
        if not unreasonable_volume.empty:
            print(f"  'volume'列中存在可疑的成交量数据，行数: {len(unreasonable_volume)}")
            issues_found.append(f"'volume'列有 {len(unreasonable_volume)} 行可疑成交量")
    
    # 检查高低价的合理性
    price_issues = df[(df['low'] > df['high']) | (df['open'] > df['high']) | (df['open'] < df['low']) |
                      (df['close'] > df['high']) | (df['close'] < df['low'])]
    if not price_issues.empty:
        print(f"  存在价格数据不一致的行，行数: {len(price_issues)}")
        issues_found.append(f"有 {len(price_issues)} 行价格数据不一致")
    
    print("\n检查状态汇总:")
    if issues_found:
        print("发现以下问题:")
        for issue in issues_found:
            print(f"- {issue}")
    else:
        print("未发现任何问题，数据完整性良好。")
    
    print("\n检查内容汇总:")
    print(f"- 总行数: {len(df)}")
    print(f"- 日期范围: 从 {df['day'].min().date()} 到 {df['day'].max().date()}")
    print(f"- 检查的列: {', '.join(df.columns)}")
    print(f"- 已考虑的法定假期数: {len(holidays)}")
    print(f"- 缺失的工作日数: {len(work_days_missing)}")
    print(f"- 多余的日期数（周末或法定假期）: {len(extra_days)}")
    
    print("\n数据完整性检查完成。")

# 使用函数
check_data_integrity('sz000001_1d_1983-09-24_2024-10-18.csv', 'chinese_holidays.csv')