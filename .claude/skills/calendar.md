# /calendar

Check market calendar - trading days, holidays, early closes.

## Usage
```
/calendar [--week|--month] [--date YYYY-MM-DD]
```

## What it does
1. Shows if today is a trading day
2. Lists upcoming holidays
3. Identifies early close days
4. Shows market hours

## Commands
```bash
# Is market open today?
python -c "
from datetime import datetime, date
import sys
sys.path.insert(0, '.')

from core.calendar import is_trading_day, get_market_hours, get_next_trading_day

today = date.today()
now = datetime.now()

print(f'=== MARKET CALENDAR ===')
print(f'Date: {today}')
print(f'Day: {today.strftime(\"%A\")}')
print()

if is_trading_day(today):
    hours = get_market_hours(today)
    print(f'✅ TRADING DAY')
    print(f'   Open: {hours[\"open\"]}')
    print(f'   Close: {hours[\"close\"]}')
    if hours.get('early_close'):
        print(f'   ⚠️ EARLY CLOSE: {hours[\"close\"]}')
else:
    next_day = get_next_trading_day(today)
    print(f'❌ MARKET CLOSED')
    print(f'   Next trading day: {next_day}')
"

# This week's schedule
python -c "
from datetime import date, timedelta
import sys
sys.path.insert(0, '.')
from core.calendar import is_trading_day, get_market_hours

today = date.today()
start = today - timedelta(days=today.weekday())  # Monday

print('=== THIS WEEK ===')
for i in range(7):
    d = start + timedelta(days=i)
    marker = ' <-- TODAY' if d == today else ''
    if is_trading_day(d):
        hours = get_market_hours(d)
        early = ' (EARLY)' if hours.get('early_close') else ''
        print(f'{d.strftime(\"%a %m/%d\")}: Open {hours[\"open\"]}-{hours[\"close\"]}{early}{marker}')
    else:
        print(f'{d.strftime(\"%a %m/%d\")}: CLOSED{marker}')
"

# Upcoming holidays
python -c "
from datetime import date, timedelta
import sys
sys.path.insert(0, '.')
from core.calendar import get_holidays

today = date.today()
holidays = get_holidays(today.year)
upcoming = [h for h in holidays if h['date'] >= today][:5]

print('=== UPCOMING HOLIDAYS ===')
for h in upcoming:
    print(f'{h[\"date\"]}: {h[\"name\"]}')
"
```

## Market Hours (NYSE/NASDAQ)
- Regular: 9:30 AM - 4:00 PM ET
- Early close: 9:30 AM - 1:00 PM ET
- Pre-market: 4:00 AM - 9:30 AM ET
- After-hours: 4:00 PM - 8:00 PM ET

## 2025 Holidays
- Jan 1: New Year's Day
- Jan 20: MLK Day
- Feb 17: Presidents Day
- Apr 18: Good Friday
- May 26: Memorial Day
- Jun 19: Juneteenth
- Jul 4: Independence Day
- Sep 1: Labor Day
- Nov 27: Thanksgiving
- Dec 25: Christmas
