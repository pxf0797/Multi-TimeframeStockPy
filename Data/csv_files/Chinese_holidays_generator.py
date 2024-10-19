import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta
import time
import re

def parse_date(date_str, year):
    date_formats = [
        "%d %b",      # 例如 "01 Jan"
        "%b %d",      # 例如 "Jan 01"
        "%d %B",      # 例如 "01 January"
        "%B %d",      # 例如 "January 01"
        "%d.%m",      # 例如 "01.01"
        "%m.%d"       # 例如 "01.01"
    ]
    
    for date_format in date_formats:
        try:
            date = datetime.strptime(f"{year} {date_str}", f"%Y {date_format}").date()
            return date
        except ValueError:
            continue
    
    # 如果上面的格式都不匹配，尝试直接解析完整的日期字符串
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"  无法解析日期: {date_str}")
        return None

def fetch_holidays_from_web(year):
    print(f"正在从网络获取 {year} 年的假期数据...")
    url = f"https://www.officeholidays.com/countries/china/{year}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        holidays = []
        table = soup.find('table', class_='country-table')
        if table:
            rows = table.find_all('tr')
            for row in rows[1:]:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    date_str = cells[1].text.strip()
                    date = parse_date(date_str, year)
                    if date:
                        holidays.append(date)
        
        if holidays:
            print(f"  成功获取到 {len(holidays)} 个假期日期")
        else:
            print("  未能从网络获取到假期数据")
        return holidays
    except requests.RequestException as e:
        print(f"  获取数据时出错: {e}")
        return []

def generate_default_holidays(year):
    print(f"正在生成 {year} 年的默认假期数据...")
    holidays = []
    
    # 新年
    holidays.append(datetime(year, 1, 1).date())
    
    # 春节（简化为固定日期，实际上是农历）
    for i in range(1, 8):
        holidays.append(datetime(year, 2, i).date())
    
    # 清明节（简化为固定日期）
    holidays.append(datetime(year, 4, 5).date())
    
    # 劳动节
    holidays.append(datetime(year, 5, 1).date())
    
    # 端午节（简化为固定日期）
    holidays.append(datetime(year, 6, 5).date())
    
    # 中秋节（简化为固定日期）
    holidays.append(datetime(year, 9, 15).date())
    
    # 国庆节
    for i in range(1, 8):
        holidays.append(datetime(year, 10, i).date())
    
    print(f"  已生成 {len(holidays)} 个默认假期日期")
    return holidays

def generate_chinese_holidays(start_year=1991):
    end_year = datetime.now().year
    print(f"开始生成从 {start_year} 年到 {end_year} 年的中国法定假期数据")
    
    all_holidays = []
    
    for year in range(start_year, end_year + 1):
        print(f"\n处理 {year} 年的数据:")
        holidays = fetch_holidays_from_web(year)
        
        if not holidays:
            print("  使用默认生成方法")
            holidays = generate_default_holidays(year)
        
        all_holidays.extend(holidays)
        time.sleep(1)  # 添加延迟以避免对服务器造成压力
    
    # 排序并去重
    all_holidays = sorted(list(set(all_holidays)))
    
    # 写入CSV文件
    csv_filename = 'chinese_holidays.csv'
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['date'])  # 写入表头
        for holiday in all_holidays:
            writer.writerow([holiday.strftime("%Y-%m-%d")])
    
    print(f"\n已生成中国法定假期CSV文件: {csv_filename}")
    print(f"总假期数: {len(all_holidays)}")
    print(f"数据范围: 从 {min(all_holidays)} 到 {max(all_holidays)}")

# 生成假期文件
generate_chinese_holidays()