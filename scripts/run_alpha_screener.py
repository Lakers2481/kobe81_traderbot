#!/usr/bin/env python3
"""
Alpha Screener CLI
==================

Run walk-forward alpha screening on the universe to discover profitable alphas.

Usage:
    # Screen all registered alphas
    python scripts/run_alpha_screener.py --universe data/universe/optionable_liquid_900.csv

    # Screen specific alphas
    python scripts/run_alpha_screener.py --alphas rsi2_oversold,momentum_breakout

    # Custom date range and splits
    python scripts/run_alpha_screener.py --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 63

    # Show detailed leaderboard
    python scripts/run_alpha_screener.py --top 20 --verbose

    # Save results to CSV
    python scripts/run_alpha_screener.py --output reports/alpha_screener_results.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# from research.screener import (
#     AlphaScreener,
#     ScreenerConfig,
#     list_available_alphas,
# )
from research.alphas import ALPHA_REGISTRY
from data.universe.loader import load_universe


def main():
    parser = argparse.ArgumentParser(
        description="Run walk-forward alpha screening to discover profitable alphas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Screen all alphas on 900-stock universe
    python scripts/run_alpha_screener.py --universe data/universe/optionable_liquid_900.csv

    # Screen specific alphas with custom settings
    python scripts/run_alpha_screener.py --alphas rsi2_oversold,ibs_rsi_oversold --train-days 504 --test-days 126

    # Output detailed results
    python scripts/run_alpha_screener.py --top 25 --verbose --output reports/alpha_results.csv
"""
    )

    # Data settings
    parser.add_argument(
        '--universe',
        type=str,
        default='data/universe/optionable_liquid_900.csv',
        help='Path to universe CSV file (default: data/universe/optionable_liquid_900.csv)',
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2015-01-01',
        help='Start date for screening (default: 2015-01-01)',
    )
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date for screening (default: today)',
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='data/cache',
        help='Directory for cached price data (default: data/cache)',
    )

    # Screening settings
    parser.add_argument(
        '--alphas',
        type=str,
        default=None,
        help='Comma-separated list of alphas to screen (default: all registered alphas)',
    )
    parser.add_argument(
        '--train-days',
        type=int,
        default=252,
        help='Training window in trading days (default: 252 = 1 year)',
    )
    parser.add_argument(
        '--test-days',
        type=int,
        default=63,
        help='Test window in trading days (default: 63 = 1 quarter)',
    )
    parser.add_argument(
        '--min-symbols',
        type=int,
        default=10,
        help='Minimum symbols for cross-sectional analysis (default: 10)',
    )

    # Output settings
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top alphas to display (default: 10)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to CSV file',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output',
    )
    parser.add_argument(
        '--list-alphas',
        action='store_true',
        help='List all available alphas and exit',
    )

    args = parser.parse_args()

    print("ERROR: Alpha screening functionality is not fully implemented or available.")
    print("The 'AlphaScreener' class and related components are missing from research.screener.py.")
    print("Please check research/screener.py and research/alphas.py for available functionality.")
    sys.exit(1)


if __name__ == '__main__':
    main()
