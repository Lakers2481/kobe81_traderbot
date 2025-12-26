#!/usr/bin/env python3
"""
Position Correlation Analysis Script for Kobe Trading System

Analyzes correlations between portfolio positions:
- Calculates correlation matrix between positions
- Identifies highly correlated pairs (>0.7)
- Shows sector-level correlations
- Warns on concentration risk

Usage:
    python correlation.py --dotenv /path/to/.env
    python correlation.py --matrix
    python correlation.py --warnings
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.env_loader import load_env


# Sector mappings for common stocks
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology',
    'CRM': 'Technology', 'ORCL': 'Technology', 'ADBE': 'Technology', 'CSCO': 'Technology',
    'AVGO': 'Technology', 'TXN': 'Technology', 'QCOM': 'Technology', 'MU': 'Technology',
    'NOW': 'Technology', 'INTU': 'Technology', 'AMAT': 'Technology', 'LRCX': 'Technology',

    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare',
    'ABBV': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
    'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'MDT': 'Healthcare',
    'GILD': 'Healthcare', 'CVS': 'Healthcare', 'ISRG': 'Healthcare', 'VRTX': 'Healthcare',

    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
    'MS': 'Financials', 'C': 'Financials', 'BLK': 'Financials', 'SCHW': 'Financials',
    'AXP': 'Financials', 'USB': 'Financials', 'PNC': 'Financials', 'TFC': 'Financials',
    'BK': 'Financials', 'COF': 'Financials', 'CME': 'Financials', 'ICE': 'Financials',

    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'LOW': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
    'BKNG': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary', 'GM': 'Consumer Discretionary',

    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
    'MO': 'Consumer Staples', 'CL': 'Consumer Staples', 'MDLZ': 'Consumer Staples',
    'KHC': 'Consumer Staples', 'GIS': 'Consumer Staples', 'K': 'Consumer Staples',

    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
    'OXY': 'Energy', 'PXD': 'Energy', 'HAL': 'Energy', 'DVN': 'Energy',

    # Industrials
    'UNP': 'Industrials', 'CAT': 'Industrials', 'HON': 'Industrials', 'BA': 'Industrials',
    'DE': 'Industrials', 'RTX': 'Industrials', 'LMT': 'Industrials', 'GE': 'Industrials',
    'MMM': 'Industrials', 'UPS': 'Industrials', 'FDX': 'Industrials', 'EMR': 'Industrials',

    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities', 'XEL': 'Utilities',

    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
    'PSA': 'Real Estate', 'SPG': 'Real Estate', 'O': 'Real Estate', 'WELL': 'Real Estate',

    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'FCX': 'Materials',
    'NEM': 'Materials', 'ECL': 'Materials', 'NUE': 'Materials', 'DOW': 'Materials',

    # Communication Services
    'DIS': 'Communication Services', 'NFLX': 'Communication Services', 'CMCSA': 'Communication Services',
    'VZ': 'Communication Services', 'T': 'Communication Services', 'TMUS': 'Communication Services',
    'CHTR': 'Communication Services', 'ATVI': 'Communication Services',

    # ETFs
    'SPY': 'ETF-Broad', 'QQQ': 'ETF-Tech', 'IWM': 'ETF-SmallCap', 'DIA': 'ETF-Broad',
    'XLF': 'ETF-Financials', 'XLE': 'ETF-Energy', 'XLK': 'ETF-Tech', 'XLV': 'ETF-Healthcare',
    'XLY': 'ETF-ConsDisc', 'XLP': 'ETF-ConsStaples', 'XLI': 'ETF-Industrials',
    'XLU': 'ETF-Utilities', 'XLB': 'ETF-Materials', 'XLRE': 'ETF-RealEstate',
}


@dataclass
class CorrelationWarning:
    """Container for correlation warnings."""
    type: str
    severity: str  # 'low', 'medium', 'high'
    message: str
    symbols: List[str]
    value: float


def fetch_polygon_bars(symbol: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch daily bars from Polygon API."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        results = data.get('results', [])
        if not results:
            return pd.DataFrame()
        rows = []
        for r in results:
            rows.append({
                'timestamp': pd.to_datetime(r.get('t'), unit='ms'),
                'close': float(r.get('c', 0)),
            })
        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def get_alpaca_positions(api_key: str, api_secret: str, base_url: str) -> List[Dict]:
    """Fetch current positions from Alpaca."""
    url = f"{base_url}/v2/positions"
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret,
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        print(f"Error fetching Alpaca positions: {e}")
        return []


def load_positions(positions_file: Path, use_broker: bool = False) -> List[Dict]:
    """Load positions from file or broker."""
    # Try local state file first
    if positions_file.exists() and not use_broker:
        try:
            data = json.loads(positions_file.read_text())
            if isinstance(data, list):
                return data
            return []
        except Exception:
            pass

    # Try broker
    if use_broker:
        api_key = os.getenv('ALPACA_API_KEY_ID', '')
        api_secret = os.getenv('ALPACA_API_SECRET_KEY', '')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')

        if api_key and api_secret:
            return get_alpaca_positions(api_key, api_secret, base_url)

    return []


class CorrelationAnalyzer:
    """Portfolio correlation analysis engine."""

    HIGH_CORRELATION_THRESHOLD = 0.70
    SECTOR_CONCENTRATION_THRESHOLD = 0.40  # 40% of portfolio in one sector

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.positions: List[Dict] = []
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.returns_df: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None

    def load_positions(self, positions: List[Dict]) -> None:
        """Load position data."""
        self.positions = positions

    def get_symbols(self) -> List[str]:
        """Get list of position symbols."""
        return [p.get('symbol', '').upper() for p in self.positions if p.get('symbol')]

    def fetch_price_data(self, lookback_days: int = 120) -> bool:
        """Fetch historical price data for all positions."""
        symbols = self.get_symbols()
        if not symbols:
            print("No positions to analyze.")
            return False

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        print(f"Fetching price data for {len(symbols)} positions...")

        for symbol in symbols:
            df = fetch_polygon_bars(symbol, start_date, end_date, self.api_key)
            if not df.empty:
                self.price_data[symbol] = df
            else:
                print(f"  Warning: No data for {symbol}")

        if len(self.price_data) < 2:
            print("Need at least 2 positions with data for correlation analysis.")
            return False

        return True

    def calculate_returns(self) -> pd.DataFrame:
        """Calculate daily returns for all positions."""
        if not self.price_data:
            return pd.DataFrame()

        # Combine all price series
        combined = pd.DataFrame()
        for symbol, df in self.price_data.items():
            combined[symbol] = df['close']

        # Forward fill missing values, then calculate returns
        combined = combined.ffill().bfill()
        self.returns_df = combined.pct_change().dropna()

        return self.returns_df

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between positions."""
        if self.returns_df is None or self.returns_df.empty:
            self.calculate_returns()

        if self.returns_df is None or self.returns_df.empty:
            return pd.DataFrame()

        self.correlation_matrix = self.returns_df.corr()
        return self.correlation_matrix

    def get_high_correlation_pairs(self) -> List[Tuple[str, str, float]]:
        """Identify pairs with correlation above threshold."""
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()

        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return []

        pairs = []
        symbols = self.correlation_matrix.columns.tolist()

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = self.correlation_matrix.iloc[i, j]
                if abs(corr) >= self.HIGH_CORRELATION_THRESHOLD:
                    pairs.append((symbols[i], symbols[j], corr))

        # Sort by absolute correlation descending
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs

    def get_sector_allocations(self) -> Dict[str, Dict]:
        """Calculate sector allocations from positions."""
        sectors: Dict[str, Dict] = {}

        total_value = sum(float(p.get('market_value', 0)) for p in self.positions)

        for pos in self.positions:
            symbol = pos.get('symbol', '').upper()
            market_value = float(pos.get('market_value', 0))

            sector = SECTOR_MAP.get(symbol, 'Unknown')

            if sector not in sectors:
                sectors[sector] = {
                    'symbols': [],
                    'total_value': 0.0,
                    'pct_of_portfolio': 0.0
                }

            sectors[sector]['symbols'].append(symbol)
            sectors[sector]['total_value'] += market_value

        # Calculate percentages
        for sector in sectors:
            if total_value > 0:
                sectors[sector]['pct_of_portfolio'] = sectors[sector]['total_value'] / total_value

        return sectors

    def calculate_sector_correlations(self) -> pd.DataFrame:
        """Calculate average correlations between sectors."""
        if self.correlation_matrix is None:
            self.calculate_correlation_matrix()

        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return pd.DataFrame()

        # Map symbols to sectors
        symbol_to_sector = {}
        for symbol in self.correlation_matrix.columns:
            symbol_to_sector[symbol] = SECTOR_MAP.get(symbol, 'Unknown')

        # Get unique sectors
        sectors = list(set(symbol_to_sector.values()))
        sector_corr = pd.DataFrame(index=sectors, columns=sectors, dtype=float)

        for s1 in sectors:
            for s2 in sectors:
                # Get symbols in each sector
                syms1 = [s for s, sec in symbol_to_sector.items() if sec == s1]
                syms2 = [s for s, sec in symbol_to_sector.items() if sec == s2]

                if not syms1 or not syms2:
                    sector_corr.loc[s1, s2] = np.nan
                    continue

                # Average correlation between sector symbols
                corrs = []
                for sym1 in syms1:
                    for sym2 in syms2:
                        if sym1 != sym2 and sym1 in self.correlation_matrix.columns and sym2 in self.correlation_matrix.columns:
                            corrs.append(self.correlation_matrix.loc[sym1, sym2])

                if corrs:
                    sector_corr.loc[s1, s2] = np.mean(corrs)
                else:
                    sector_corr.loc[s1, s2] = 1.0 if s1 == s2 else np.nan

        return sector_corr

    def generate_warnings(self) -> List[CorrelationWarning]:
        """Generate all correlation and concentration warnings."""
        warnings = []

        # High correlation warnings
        high_corr_pairs = self.get_high_correlation_pairs()
        for sym1, sym2, corr in high_corr_pairs:
            severity = 'high' if corr > 0.85 else 'medium' if corr > 0.75 else 'low'
            warnings.append(CorrelationWarning(
                type='high_correlation',
                severity=severity,
                message=f"{sym1} and {sym2} have {corr:.1%} correlation - consider reducing overlap",
                symbols=[sym1, sym2],
                value=corr
            ))

        # Sector concentration warnings
        sectors = self.get_sector_allocations()
        for sector, data in sectors.items():
            pct = data['pct_of_portfolio']
            if pct >= self.SECTOR_CONCENTRATION_THRESHOLD:
                severity = 'high' if pct > 0.50 else 'medium'
                warnings.append(CorrelationWarning(
                    type='sector_concentration',
                    severity=severity,
                    message=f"{sector} sector represents {pct:.1%} of portfolio ({len(data['symbols'])} positions)",
                    symbols=data['symbols'],
                    value=pct
                ))

        # Few unique sectors warning
        unique_sectors = len([s for s in sectors if s != 'Unknown'])
        if unique_sectors < 3 and len(self.positions) >= 5:
            warnings.append(CorrelationWarning(
                type='low_diversification',
                severity='medium',
                message=f"Portfolio concentrated in only {unique_sectors} sector(s)",
                symbols=self.get_symbols(),
                value=float(unique_sectors)
            ))

        # High average portfolio correlation
        if self.correlation_matrix is not None and not self.correlation_matrix.empty:
            # Get off-diagonal average
            mask = np.ones(self.correlation_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_corr = self.correlation_matrix.values[mask].mean()

            if avg_corr > 0.5:
                severity = 'high' if avg_corr > 0.65 else 'medium'
                warnings.append(CorrelationWarning(
                    type='high_portfolio_correlation',
                    severity=severity,
                    message=f"Average portfolio correlation is {avg_corr:.1%} - positions move together",
                    symbols=self.get_symbols(),
                    value=avg_corr
                ))

        return warnings


def print_correlation_matrix(matrix: pd.DataFrame) -> None:
    """Print formatted correlation matrix."""
    if matrix.empty:
        print("No correlation matrix available.")
        return

    print("\n" + "=" * 80)
    print("               POSITION CORRELATION MATRIX")
    print("=" * 80)

    # Truncate to max 10 symbols for readability
    if len(matrix.columns) > 10:
        print(f"(Showing top 10 of {len(matrix.columns)} positions)")
        matrix = matrix.iloc[:10, :10]

    # Print header
    header = "          |" + "|".join(f"{s:>7s}" for s in matrix.columns)
    print(header)
    print("-" * len(header))

    # Print rows
    for idx, row in matrix.iterrows():
        row_str = f"{str(idx):>10s}|"
        for col in matrix.columns:
            val = row[col]
            if pd.isna(val):
                row_str += "    N/A|"
            elif idx == col:
                row_str += "   1.00|"
            else:
                # Color coding via symbols
                if val > 0.7:
                    marker = "!!"
                elif val > 0.5:
                    marker = "+ "
                elif val < -0.3:
                    marker = "- "
                else:
                    marker = "  "
                row_str += f"{marker}{val:5.2f}|"
        print(row_str)

    print("-" * len(header))
    print("\nLegend: !! = High correlation (>0.7), + = Moderate positive (>0.5), - = Negative (<-0.3)")
    print()


def print_high_correlation_pairs(pairs: List[Tuple[str, str, float]]) -> None:
    """Print highly correlated pairs."""
    print("\n" + "=" * 60)
    print("         HIGHLY CORRELATED PAIRS (>70%)")
    print("=" * 60)

    if not pairs:
        print("\n  No highly correlated pairs found. Portfolio is well-diversified!")
        print()
        return

    print(f"\n{'Symbol 1':>10s} | {'Symbol 2':>10s} | {'Correlation':>12s} | Risk Level")
    print("-" * 60)

    for sym1, sym2, corr in pairs:
        if corr > 0.85:
            risk = "HIGH"
        elif corr > 0.75:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        print(f"{sym1:>10s} | {sym2:>10s} | {corr:>11.1%} | {risk}")

    print("-" * 60)
    print(f"\nTotal high-correlation pairs: {len(pairs)}")
    print()


def print_sector_analysis(sectors: Dict[str, Dict]) -> None:
    """Print sector allocation analysis."""
    print("\n" + "=" * 70)
    print("                    SECTOR ALLOCATION")
    print("=" * 70)

    # Sort by allocation descending
    sorted_sectors = sorted(sectors.items(), key=lambda x: x[1]['pct_of_portfolio'], reverse=True)

    print(f"\n{'Sector':>25s} | {'% of Portfolio':>14s} | {'# Positions':>11s} | {'Symbols'}")
    print("-" * 70)

    for sector, data in sorted_sectors:
        pct = data['pct_of_portfolio']
        count = len(data['symbols'])
        symbols = ', '.join(data['symbols'][:5])
        if len(data['symbols']) > 5:
            symbols += f" (+{len(data['symbols']) - 5} more)"

        # Warning indicator
        warning = " !!" if pct > 0.40 else ""
        print(f"{sector:>25s} | {pct:>13.1%}{warning} | {count:>11d} | {symbols}")

    print("-" * 70)
    print("\n!! = Concentration warning (>40% in single sector)")
    print()


def print_warnings(warnings: List[CorrelationWarning]) -> None:
    """Print correlation warnings."""
    print("\n" + "=" * 70)
    print("                   RISK WARNINGS")
    print("=" * 70)

    if not warnings:
        print("\n  No correlation or concentration warnings.")
        print("  Portfolio appears well-diversified!")
        print()
        return

    # Group by severity
    high = [w for w in warnings if w.severity == 'high']
    medium = [w for w in warnings if w.severity == 'medium']
    low = [w for w in warnings if w.severity == 'low']

    if high:
        print("\n[HIGH SEVERITY]")
        print("-" * 50)
        for w in high:
            print(f"  * {w.message}")

    if medium:
        print("\n[MEDIUM SEVERITY]")
        print("-" * 50)
        for w in medium:
            print(f"  * {w.message}")

    if low:
        print("\n[LOW SEVERITY]")
        print("-" * 50)
        for w in low:
            print(f"  * {w.message}")

    print("\n" + "-" * 70)
    print(f"Total Warnings: {len(warnings)} ({len(high)} high, {len(medium)} medium, {len(low)} low)")
    print()


def print_summary(analyzer: CorrelationAnalyzer) -> None:
    """Print summary report."""
    print("\n" + "=" * 70)
    print("           PORTFOLIO CORRELATION SUMMARY")
    print("=" * 70)

    positions = len(analyzer.positions)
    symbols = analyzer.get_symbols()

    print(f"\n  Positions Analyzed: {positions}")
    print(f"  Unique Symbols: {len(set(symbols))}")

    if analyzer.correlation_matrix is not None and not analyzer.correlation_matrix.empty:
        # Average correlation (excluding diagonal)
        mask = np.ones(analyzer.correlation_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = analyzer.correlation_matrix.values[mask].mean()
        max_corr = analyzer.correlation_matrix.values[mask].max()
        min_corr = analyzer.correlation_matrix.values[mask].min()

        print(f"\n  Correlation Statistics:")
        print(f"    Average: {avg_corr:.2%}")
        print(f"    Maximum: {max_corr:.2%}")
        print(f"    Minimum: {min_corr:.2%}")

    high_corr_pairs = analyzer.get_high_correlation_pairs()
    print(f"\n  High Correlation Pairs (>70%): {len(high_corr_pairs)}")

    sectors = analyzer.get_sector_allocations()
    print(f"  Unique Sectors: {len([s for s in sectors if s != 'Unknown'])}")

    concentrated = [s for s, d in sectors.items() if d['pct_of_portfolio'] > 0.40]
    if concentrated:
        print(f"  Concentrated Sectors (>40%): {', '.join(concentrated)}")

    warnings = analyzer.generate_warnings()
    print(f"\n  Risk Warnings: {len(warnings)}")

    print()


def main():
    ap = argparse.ArgumentParser(description='Position Correlation Analysis for Kobe Trading System')
    ap.add_argument('--dotenv', type=str, default='C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env',
                    help='Path to .env file')
    ap.add_argument('--positions', type=str, default='state/positions.json',
                    help='Path to positions JSON file')
    ap.add_argument('--broker', action='store_true',
                    help='Fetch positions from broker instead of file')
    ap.add_argument('--matrix', action='store_true',
                    help='Show full correlation matrix')
    ap.add_argument('--warnings', action='store_true',
                    help='Show only warnings')
    ap.add_argument('--json', action='store_true',
                    help='Output as JSON')
    ap.add_argument('--lookback', type=int, default=120,
                    help='Days of history for correlation (default: 120)')
    args = ap.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    api_key = os.getenv('POLYGON_API_KEY', '')
    if not api_key:
        print("Error: POLYGON_API_KEY not found in environment")
        sys.exit(1)

    # Load positions
    positions_file = Path(args.positions)
    positions = load_positions(positions_file, use_broker=args.broker)

    if not positions:
        print("\n" + "=" * 50)
        print("  No positions found.")
        print("=" * 50)
        print("\n  Checked locations:")
        print(f"    - File: {positions_file}")
        if args.broker:
            print("    - Alpaca broker API")
        print("\n  Portfolio is empty or positions could not be loaded.")
        print("  Run reconcile_alpaca.py first to save positions.")
        print()
        sys.exit(0)

    # Initialize analyzer
    analyzer = CorrelationAnalyzer(api_key)
    analyzer.load_positions(positions)

    # Fetch data
    if not analyzer.fetch_price_data(lookback_days=args.lookback):
        print("Error: Could not fetch price data")
        sys.exit(1)

    # Calculate correlations
    analyzer.calculate_correlation_matrix()

    # Output based on flags
    if args.json:
        output = {
            'positions': len(positions),
            'symbols': analyzer.get_symbols(),
            'correlation_matrix': analyzer.correlation_matrix.to_dict() if analyzer.correlation_matrix is not None else {},
            'high_correlation_pairs': [
                {'symbol1': s1, 'symbol2': s2, 'correlation': c}
                for s1, s2, c in analyzer.get_high_correlation_pairs()
            ],
            'sector_allocations': analyzer.get_sector_allocations(),
            'warnings': [
                {'type': w.type, 'severity': w.severity, 'message': w.message, 'value': w.value}
                for w in analyzer.generate_warnings()
            ],
            'timestamp': datetime.now().isoformat()
        }
        print(json.dumps(output, indent=2, default=str))
    elif args.warnings:
        warnings = analyzer.generate_warnings()
        print_warnings(warnings)
    elif args.matrix:
        if analyzer.correlation_matrix is not None:
            print_correlation_matrix(analyzer.correlation_matrix)
        print_high_correlation_pairs(analyzer.get_high_correlation_pairs())
    else:
        # Full report
        print_summary(analyzer)
        print_correlation_matrix(analyzer.correlation_matrix)
        print_high_correlation_pairs(analyzer.get_high_correlation_pairs())
        print_sector_analysis(analyzer.get_sector_allocations())
        print_warnings(analyzer.generate_warnings())


if __name__ == '__main__':
    main()
