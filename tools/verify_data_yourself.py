"""
SELF-VERIFICATION SCRIPT
========================
Run this yourself to prove the data is real and not fabricated.

This script:
1. Downloads data DIRECTLY from Yahoo Finance (you can see the network calls)
2. Saves raw data to CSV files you can inspect
3. Shows you the EXACT calculations with no black boxes
4. Outputs results you can manually verify

YOU CAN:
- Open the CSV files in Excel to see the raw data
- Verify the calculations manually
- Check the data against Yahoo Finance website
- Re-run this script multiple times to ensure consistency
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

print("="*80)
print("SELF-VERIFICATION: Markov 5-Down-Day Pattern")
print("="*80)
print()
print("This script will:")
print("1. Download SPY data from Yahoo Finance")
print("2. Save it to CSV so you can inspect it")
print("3. Show you exactly how we calculate the pattern")
print("4. Output raw counts you can manually verify")
print()

# Step 1: Download data
print("STEP 1: Downloading SPY data from Yahoo Finance...")
print("  Symbol: SPY")
print("  Start: 2015-01-01")
print("  End: 2025-12-31")
print("  Source: yfinance (Yahoo Finance API)")
print()

spy = yf.Ticker("SPY")
df = spy.history(start="2015-01-01", end="2025-12-31")

# Save raw data
output_dir = Path("data/audit_verification")
output_dir.mkdir(exist_ok=True, parents=True)

raw_file = output_dir / "spy_raw_data.csv"
df.to_csv(raw_file)
print(f"[OK] Raw data saved to: {raw_file}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print()

# Step 2: Calculate returns
print("STEP 2: Calculating daily returns...")
df['Return'] = df['Close'].pct_change()
df['IsDown'] = df['Return'] < 0

returns_file = output_dir / "spy_with_returns.csv"
df[['Close', 'Return', 'IsDown']].to_csv(returns_file)
print(f"[OK] Returns calculated and saved to: {returns_file}")
print(f"  Formula: Return = (Close_t - Close_t-1) / Close_t-1")
print(f"  IsDown = Return < 0")
print()

# Step 3: Find 5-down-day patterns
print("STEP 3: Finding 5 consecutive down days...")
print()

matches = []
for i in range(5, len(df) - 1):
    # Check if previous 5 days were all down
    last_5_down = all(df['IsDown'].iloc[i-5+j] for j in range(5))

    if last_5_down:
        date_last_down = df.index[i-1]
        date_next = df.index[i]
        next_return = df['Return'].iloc[i]
        next_up = next_return >= 0

        matches.append({
            'Pattern_End_Date': date_last_down.strftime('%Y-%m-%d'),
            'Next_Date': date_next.strftime('%Y-%m-%d'),
            'Next_Return': f"{next_return:.4f}",
            'Next_Up': 'YES' if next_up else 'NO',
            # Include the 5 down days for verification
            'Day1_Date': df.index[i-5].strftime('%Y-%m-%d'),
            'Day1_Return': f"{df['Return'].iloc[i-5]:.4f}",
            'Day2_Date': df.index[i-4].strftime('%Y-%m-%d'),
            'Day2_Return': f"{df['Return'].iloc[i-4]:.4f}",
            'Day3_Date': df.index[i-3].strftime('%Y-%m-%d'),
            'Day3_Return': f"{df['Return'].iloc[i-3]:.4f}",
            'Day4_Date': df.index[i-2].strftime('%Y-%m-%d'),
            'Day4_Return': f"{df['Return'].iloc[i-2]:.4f}",
            'Day5_Date': df.index[i-1].strftime('%Y-%m-%d'),
            'Day5_Return': f"{df['Return'].iloc[i-1]:.4f}",
        })

matches_df = pd.DataFrame(matches)
matches_file = output_dir / "spy_5down_patterns.csv"
matches_df.to_csv(matches_file, index=False)

print(f"[OK] Found {len(matches)} instances of 5 consecutive down days")
print(f"[OK] Saved all instances to: {matches_file}")
print()
print("You can now:")
print(f"  1. Open {matches_file} in Excel")
print("  2. Manually verify each pattern by checking the 5 returns are all negative")
print("  3. Check the Next_Up column against Next_Return (>= 0 means UP)")
print()

# Step 4: Calculate statistics
next_up_count = matches_df['Next_Up'].value_counts().get('YES', 0)
total_instances = len(matches_df)
prob_up = next_up_count / total_instances if total_instances > 0 else 0

print("STEP 4: Calculate Statistics")
print("="*80)
print(f"Total Instances: {total_instances}")
print(f"Next Day UP: {next_up_count}")
print(f"Next Day DOWN: {total_instances - next_up_count}")
print(f"Probability of UP: {next_up_count}/{total_instances} = {prob_up:.4f} = {prob_up:.1%}")
print()
print(f"CLAIMED: 66%")
print(f"MEASURED: {prob_up:.1%}")
print(f"DIFFERENCE: {abs(prob_up - 0.66):.1%}")
print()

if abs(prob_up - 0.66) < 0.10:
    print("[OK] VERIFIED: Claim is supported by data (within 10%)")
else:
    print("[FAIL] REJECTED: Claim is NOT supported (outside 10%)")

print()
print("="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print()
print("TO MANUALLY VERIFY:")
print("1. Open spy_5down_patterns.csv in Excel")
print("2. For each row, check that Day1-Day5 returns are all negative")
print("3. Count how many rows have Next_Up = 'YES'")
print("4. Divide by total rows to get probability")
print()
print("TO VERIFY DATA SOURCE:")
print("1. Go to https://finance.yahoo.com/quote/SPY/history")
print("2. Pick any date from the CSV file")
print("3. Compare the Close price in our CSV to Yahoo's website")
print("4. They should match exactly")
print()
print(f"ALL VERIFICATION FILES SAVED TO: {output_dir}")
