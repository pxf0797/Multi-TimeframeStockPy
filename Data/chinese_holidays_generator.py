import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta
import time
import re

def parse_date(date_str, year):
    """
    Parse a date string into a datetime.date object.
    
    This function attempts to parse the date string using various common formats.
    If all attempts fail, it tries to parse the string as a full date (YYYY-MM-DD).
    
    Args:
    date_str (str): The date string to parse.
    year (int): The year to use if the date string doesn't include a year.
    
    Returns:
    datetime.date: The parsed date, or None if parsing fails.
    """
    date_formats = [
        "%d %b",      # e.g., "01 Jan"
        "%b %d",      # e.g., "Jan 01"
        "%d %B",      # e.g., "01 January"
        "%B %d",      # e.g., "January 01"
        "%d.%m",      # e.g., "01.01"
        "%m.%d"       # e.g., "01.01"
    ]
    
    # Try parsing with each format
    for date_format in date_formats:
        try:
            date = datetime.strptime(f"{year} {date_str}", f"%Y {date_format}").date()
            return date
        except ValueError:
            continue
    
    # If none of the above formats work, try parsing as a full date string
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"  Unable to parse date: {date_str}")
        return None

def fetch_holidays_from_web(year):
    """
    Fetch holiday data for a specific year from a web source.
    
    This function attempts to scrape holiday data from officeholidays.com.
    If successful, it returns a list of holiday dates for the given year.
    
    Args:
    year (int): The year for which to fetch holiday data.
    
    Returns:
    list: A list of datetime.date objects representing holidays, or an empty list if fetching fails.
    """
    print(f"Fetching holiday data for {year} from the web...")
    url = f"https://www.officeholidays.com/countries/china/{year}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        holidays = []
        table = soup.find('table', class_='country-table')
        if table:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip the header row
                cells = row.find_all('td')
                if len(cells) >= 2:
                    date_str = cells[1].text.strip()
                    date = parse_date(date_str, year)
                    if date:
                        holidays.append(date)
        
        if holidays:
            print(f"  Successfully retrieved {len(holidays)} holiday dates")
        else:
            print("  Failed to retrieve holiday data from the web")
        return holidays
    except requests.RequestException as e:
        print(f"  Error fetching data: {e}")
        return []

def generate_default_holidays(year):
    """
    Generate a list of default holiday dates for a given year.
    
    This function creates a basic list of Chinese holidays based on fixed dates.
    Note that some holidays (like Spring Festival) are actually based on the lunar calendar,
    so this is a simplified approximation.
    
    Args:
    year (int): The year for which to generate default holidays.
    
    Returns:
    list: A list of datetime.date objects representing default holidays.
    """
    print(f"Generating default holiday data for {year}...")
    holidays = []
    
    # New Year's Day
    holidays.append(datetime(year, 1, 1).date())
    
    # Spring Festival (simplified as fixed dates, actually based on lunar calendar)
    for i in range(1, 8):
        holidays.append(datetime(year, 2, i).date())
    
    # Qingming Festival (simplified as fixed date)
    holidays.append(datetime(year, 4, 5).date())
    
    # Labor Day
    holidays.append(datetime(year, 5, 1).date())
    
    # Dragon Boat Festival (simplified as fixed date)
    holidays.append(datetime(year, 6, 5).date())
    
    # Mid-Autumn Festival (simplified as fixed date)
    holidays.append(datetime(year, 9, 15).date())
    
    # National Day
    for i in range(1, 8):
        holidays.append(datetime(year, 10, i).date())
    
    print(f"  Generated {len(holidays)} default holiday dates")
    return holidays

def generate_chinese_holidays(start_year=1991):
    """
    Generate a CSV file containing Chinese holiday dates from the start year to the current year.
    
    This function attempts to fetch holiday data from the web for each year.
    If web fetching fails, it falls back to generating default holiday dates.
    The resulting holidays are saved to a CSV file named 'chinese_holidays.csv'.
    
    Args:
    start_year (int): The year from which to start generating holiday data. Defaults to 1991.
    """
    end_year = datetime.now().year
    print(f"Starting to generate Chinese holiday data from {start_year} to {end_year}")
    
    all_holidays = []
    
    for year in range(start_year, end_year + 1):
        print(f"\nProcessing data for {year}:")
        holidays = fetch_holidays_from_web(year)
        
        if not holidays:
            print("  Using default generation method")
            holidays = generate_default_holidays(year)
        
        all_holidays.extend(holidays)
        time.sleep(1)  # Add delay to avoid putting too much pressure on the server
    
    # Sort and remove duplicates
    all_holidays = sorted(list(set(all_holidays)))
    
    # Write to CSV file
    csv_filename = 'chinese_holidays.csv'
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['date'])  # Write header
        for holiday in all_holidays:
            writer.writerow([holiday.strftime("%Y-%m-%d")])
    
    print(f"\nChinese holiday CSV file generated: {csv_filename}")
    print(f"Total number of holidays: {len(all_holidays)}")
    print(f"Date range: from {min(all_holidays)} to {max(all_holidays)}")

# Generate the holiday file
generate_chinese_holidays()