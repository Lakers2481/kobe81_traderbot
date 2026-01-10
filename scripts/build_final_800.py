#!/usr/bin/env python3
"""
Build Final 800-Stock Universe
Combine 795 verified + 5 additional = 800 total

Usage:
    python scripts/build_final_800.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


def main():
    # Load 795 verified
    verified_795 = pd.read_csv(ROOT / 'data' / 'universe' / 'optionable_liquid_900_verified.full.csv')
    print(f'Loaded 795 verified stocks')

    # Load 5 additional
    additional_5 = pd.read_csv(ROOT / 'additional_5_stocks.csv')
    print(f'Loaded 5 additional stocks')

    # Combine
    final_800 = pd.concat([verified_795, additional_5], ignore_index=True)

    # Verify count
    assert len(final_800) == 800, f"Expected 800, got {len(final_800)}"

    # Sort by volume (most liquid first)
    final_800 = final_800.sort_values('avg_volume', ascending=False).reset_index(drop=True)

    print()
    print('=' * 70)
    print('FINAL 800-STOCK UNIVERSE')
    print('=' * 70)
    print(f'Total symbols: {len(final_800)}')
    print()

    # Save symbol-only file
    output_file = ROOT / 'data' / 'universe' / 'optionable_liquid_800.csv'
    final_800[['symbol']].to_csv(output_file, index=False)
    print(f'[SAVED] {output_file.name} (symbols only)')

    # Save full metadata
    metadata_file = ROOT / 'data' / 'universe' / 'optionable_liquid_800.full.csv'
    final_800.to_csv(metadata_file, index=False)
    print(f'[SAVED] {metadata_file.name} (with full metadata)')
    print()

    # Show top 20 by volume
    print('Top 20 symbols by volume:')
    for idx, row in final_800.head(20).iterrows():
        print(f"  {row['symbol']:6s} {row['years']:5.1f}y ${row['avg_volume']/1e6:8.1f}M/day")

    print()

    # Show years distribution
    print('Years distribution:')
    for min_years in [10, 9.5, 9.0]:
        count = len(final_800[final_800['years'] >= min_years])
        pct = 100 * count / len(final_800)
        print(f'  {min_years}+ years: {count:3d} symbols ({pct:5.1f}%)')

    print()

    # Show new additions
    print('5 new additions:')
    new_additions = final_800[final_800['years'] < 10.0]
    for idx, row in new_additions.iterrows():
        print(f"  {row['symbol']:6s} {row['years']:5.1f}y ${row['avg_volume']/1e6:8.1f}M/day")

    print()
    print('[SUCCESS] Built final 800-stock universe!')
    print('Quality: 795 with 10+ years, 5 with 9.3 years')

    return 0


if __name__ == '__main__':
    sys.exit(main())
