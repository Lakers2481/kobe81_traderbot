"""Verify PLTR consecutive down streak pattern - LIVE PROOF."""
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

print('='*80)
print('COMPREHENSIVE PLTR CONSECUTIVE DOWN STREAK ANALYSIS')
print('='*80)
print()

# PLTR IPO'd in September 2020, so we have ~4 years of data
# Load the longest available file
polygon_file = 'data/cache/polygon/polygon/PLTR_2015-01-01_2024-12-31.csv'
if os.path.exists(polygon_file):
    df = pd.read_csv(polygon_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
    print(f'Loaded: {polygon_file}')
else:
    # Fallback to other file
    polygon_file = 'data/cache/polygon/PLTR_2024-01-03_2026-01-02.csv'
    df = pd.read_csv(polygon_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
    print(f'Loaded: {polygon_file}')

start_date = df.iloc[0]['timestamp'].strftime('%Y-%m-%d')
end_date = df.iloc[-1]['timestamp'].strftime('%Y-%m-%d')
print(f'Date range: {start_date} to {end_date}')
print(f'Total trading days: {len(df)}')
print('='*80)

# Calculate daily returns
df['daily_return'] = df['close'].pct_change()
df['is_down'] = df['daily_return'] < 0

# The pre-game said 5+ consecutive down days for PLTR
# Let's find all instances of 5+ consecutive down days
MIN_STREAK = 5

print()
print(f'FINDING ALL INSTANCES OF {MIN_STREAK}+ CONSECUTIVE DOWN DAYS')
print('='*80)

streak = 0
streak_start = None
all_streaks = []

for i in range(len(df)):
    if df.iloc[i]['is_down']:
        if streak == 0:
            streak_start = i
        streak += 1
    else:
        if streak >= MIN_STREAK:
            streak_end = i - 1
            next_day_return = df.iloc[i]['daily_return'] if i < len(df) else None
            all_streaks.append({
                'start_date': df.iloc[streak_start]['timestamp'].strftime('%Y-%m-%d'),
                'end_date': df.iloc[streak_end]['timestamp'].strftime('%Y-%m-%d'),
                'streak_length': streak,
                'next_day_return': next_day_return,
                'next_day_bounced': next_day_return > 0 if next_day_return else None,
                'start_price': df.iloc[streak_start-1]['close'] if streak_start > 0 else df.iloc[streak_start]['open'],
                'end_price': df.iloc[streak_end]['close']
            })
        streak = 0

# Check if ends with a streak
if streak >= MIN_STREAK:
    all_streaks.append({
        'start_date': df.iloc[streak_start]['timestamp'].strftime('%Y-%m-%d'),
        'end_date': df.iloc[-1]['timestamp'].strftime('%Y-%m-%d'),
        'streak_length': streak,
        'next_day_return': None,
        'next_day_bounced': None,
        'start_price': df.iloc[streak_start-1]['close'] if streak_start > 0 else df.iloc[streak_start]['open'],
        'end_price': df.iloc[-1]['close']
    })

print(f'\nFound {len(all_streaks)} instances of {MIN_STREAK}+ consecutive down days\n')

if all_streaks:
    print(f"{'#':>2} | {'Start':>12} | {'End':>12} | {'Days':>4} | {'Drop':>8} | {'Day N+1':>8} | Bounced?")
    print('-'*75)
    for i, s in enumerate(all_streaks, 1):
        drop_pct = (s['end_price'] - s['start_price']) / s['start_price'] * 100
        if s['next_day_return'] is not None:
            ret_str = f"{s['next_day_return']*100:+.2f}%"
            bounce_str = "YES" if s['next_day_bounced'] else "NO"
        else:
            ret_str = "PENDING"
            bounce_str = "CURRENT"
        print(f"{i:>2} | {s['start_date']:>12} | {s['end_date']:>12} | {s['streak_length']:>4} | {drop_pct:>+7.1f}% | {ret_str:>8} | {bounce_str}")

# Calculate reversal stats
bounced = [s for s in all_streaks if s['next_day_bounced']]
failed = [s for s in all_streaks if not s['next_day_bounced']]
pending = [s for s in all_streaks if s['next_day_bounced'] is None]

if bounced:
    avg_bounce = sum(s['next_day_return'] for s in bounced) / len(bounced) * 100
else:
    avg_bounce = 0

print()
print('='*80)
print('SUMMARY:')
print('='*80)
print(f'  Data period: {start_date} to {end_date}')
print(f'  Total trading days analyzed: {len(df)}')
print(f'  Total {MIN_STREAK}+ day down streaks: {len(all_streaks)}')
print(f'  Bounced next day (UP): {len(bounced)}')
print(f'  Continued down: {len(failed)}')
print(f'  Pending (current): {len(pending)}')
if bounced or failed:
    reversal_rate = len(bounced) / (len(bounced) + len(failed)) * 100
    print(f'  HISTORICAL REVERSAL RATE: {reversal_rate:.1f}%')
    print(f'  AVERAGE BOUNCE: +{avg_bounce:.2f}%')
print('='*80)

# Now show the current streak
print()
print('='*80)
print('CURRENT PLTR STREAK (LAST 15 TRADING DAYS)')
print('='*80)
print()

# Fetch latest data
from data.providers.polygon_eod import fetch_daily_bars_polygon
df_latest = fetch_daily_bars_polygon('PLTR', '2025-12-15', '2026-01-02')
if len(df_latest) > 0:
    df_latest['daily_return'] = df_latest['close'].pct_change()
    df_latest['is_down'] = df_latest['daily_return'] < 0

    print(f"{'#':>2} | {'Date':>12} | {'Open':>8} | {'High':>8} | {'Low':>8} | {'Close':>8} | {'Return':>8} | Down?")
    print('-'*82)

    for i, (_, row) in enumerate(df_latest.iterrows(), 1):
        date_str = row['timestamp'].strftime('%Y-%m-%d')
        if pd.notna(row['daily_return']):
            ret_str = f"{row['daily_return']*100:+.2f}%"
        else:
            ret_str = 'N/A'
        if row['is_down']:
            down_str = 'DOWN'
        elif pd.notna(row['daily_return']) and row['daily_return'] > 0:
            down_str = 'UP'
        else:
            down_str = ''
        print(f"{i:>2} | {date_str:>12} | ${row['open']:>7.2f} | ${row['high']:>7.2f} | ${row['low']:>7.2f} | ${row['close']:>7.2f} | {ret_str:>8} | {down_str}")

    # Count current streak
    streak = 0
    for i in range(len(df_latest)-1, -1, -1):
        if df_latest.iloc[i]['is_down']:
            streak += 1
        else:
            break

    print()
    print('='*80)
    print(f'CURRENT CONSECUTIVE DOWN STREAK: {streak} DAYS')
    print('='*80)
