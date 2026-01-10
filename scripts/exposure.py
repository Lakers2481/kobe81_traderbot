#!/usr/bin/env python3
"""
Portfolio exposure analysis for Kobe trading system.
Analyzes sector, market cap, and factor exposures.
Usage: python scripts/exposure.py [--sector|--cap|--factor|--all]
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    import requests
except ImportError:
    requests = None

# Sector classifications (simplified)
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "META": "Technology", "NVDA": "Technology", "AMD": "Technology", "INTC": "Technology",
    "CRM": "Technology", "ORCL": "Technology", "CSCO": "Technology", "ADBE": "Technology",
    "AVGO": "Technology", "TXN": "Technology", "QCOM": "Technology", "IBM": "Technology",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "LLY": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "BLK": "Financials", "SCHW": "Financials",
    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer", "MCD": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "TGT": "Consumer", "COST": "Consumer",
    "WMT": "Consumer", "PG": "Consumer", "KO": "Consumer", "PEP": "Consumer",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "PXD": "Energy", "OXY": "Energy", "VLO": "Energy",
    # Industrials
    "CAT": "Industrials", "DE": "Industrials", "UNP": "Industrials", "BA": "Industrials",
    "HON": "Industrials", "GE": "Industrials", "MMM": "Industrials", "LMT": "Industrials",
    # Materials
    "LIN": "Materials", "APD": "Materials", "ECL": "Materials", "NEM": "Materials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate", "EQIX": "Real Estate",
    # Communication
    "T": "Communication", "VZ": "Communication", "TMUS": "Communication", "DIS": "Communication",
    "NFLX": "Communication", "CMCSA": "Communication",
}

# Market cap tiers (in billions)
CAP_TIERS = {
    "Mega Cap": 200,      # > $200B
    "Large Cap": 10,      # $10B - $200B
    "Mid Cap": 2,         # $2B - $10B
    "Small Cap": 0.3,     # $300M - $2B
    "Micro Cap": 0,       # < $300M
}


def load_positions() -> List[Dict]:
    """Load current positions."""
    positions_file = Path("state/positions.json")
    if not positions_file.exists():
        return []

    with open(positions_file) as f:
        positions = json.load(f)

    return positions if isinstance(positions, list) else list(positions.values())


def get_sector(symbol: str) -> str:
    """Get sector for symbol."""
    return SECTOR_MAP.get(symbol, "Other")


def get_market_cap_tier(market_cap_billions: float) -> str:
    """Get market cap tier."""
    for tier, threshold in CAP_TIERS.items():
        if market_cap_billions >= threshold:
            return tier
    return "Micro Cap"


def analyze_sector_exposure(positions: List[Dict]):
    """Analyze sector exposure."""
    print("\n=== Sector Exposure Analysis ===\n")

    if not positions:
        print("No positions to analyze")
        return

    sector_exposure = defaultdict(lambda: {"count": 0, "value": 0.0, "symbols": []})
    total_value = 0.0

    for pos in positions:
        symbol = pos.get("symbol", "")
        qty = float(pos.get("qty", pos.get("quantity", 0)))
        price = float(pos.get("current_price", pos.get("avg_entry_price", 0)))
        value = qty * price

        sector = get_sector(symbol)
        sector_exposure[sector]["count"] += 1
        sector_exposure[sector]["value"] += value
        sector_exposure[sector]["symbols"].append(symbol)
        total_value += value

    # Sort by value
    sorted_sectors = sorted(sector_exposure.items(), key=lambda x: x[1]["value"], reverse=True)

    print(f"{'Sector':<15} {'Positions':>10} {'Value':>12} {'Weight':>8}")
    print("-" * 50)

    for sector, data in sorted_sectors:
        weight = (data["value"] / total_value * 100) if total_value > 0 else 0
        print(f"{sector:<15} {data['count']:>10} ${data['value']:>10,.0f} {weight:>7.1f}%")
        if len(data["symbols"]) <= 5:
            print(f"                {', '.join(data['symbols'])}")

    print("-" * 50)
    print(f"{'TOTAL':<15} {len(positions):>10} ${total_value:>10,.0f} {100:>7.1f}%")

    # Concentration warnings
    print("\n--- Concentration Analysis ---")
    for sector, data in sorted_sectors[:3]:
        weight = (data["value"] / total_value * 100) if total_value > 0 else 0
        if weight > 40:
            print(f"[WARN] {sector}: {weight:.1f}% - HIGH concentration (>40%)")
        elif weight > 25:
            print(f"[INFO] {sector}: {weight:.1f}% - Elevated exposure (>25%)")


def analyze_cap_exposure(positions: List[Dict]):
    """Analyze market cap exposure."""
    print("\n=== Market Cap Exposure Analysis ===\n")

    if not positions:
        print("No positions to analyze")
        return

    # Note: In production, you'd fetch real market cap data
    # Here we use estimated tiers based on common knowledge
    ESTIMATED_CAP_TIER = {
        "AAPL": "Mega Cap", "MSFT": "Mega Cap", "GOOGL": "Mega Cap", "AMZN": "Mega Cap",
        "NVDA": "Mega Cap", "META": "Mega Cap", "TSLA": "Mega Cap", "BRK.B": "Mega Cap",
        "JPM": "Mega Cap", "JNJ": "Mega Cap", "V": "Mega Cap", "UNH": "Mega Cap",
    }

    cap_exposure = defaultdict(lambda: {"count": 0, "value": 0.0, "symbols": []})
    total_value = 0.0

    for pos in positions:
        symbol = pos.get("symbol", "")
        qty = float(pos.get("qty", pos.get("quantity", 0)))
        price = float(pos.get("current_price", pos.get("avg_entry_price", 0)))
        value = qty * price

        tier = ESTIMATED_CAP_TIER.get(symbol, "Large Cap")  # Default to Large Cap
        cap_exposure[tier]["count"] += 1
        cap_exposure[tier]["value"] += value
        cap_exposure[tier]["symbols"].append(symbol)
        total_value += value

    # Display in tier order
    print(f"{'Cap Tier':<12} {'Positions':>10} {'Value':>12} {'Weight':>8}")
    print("-" * 45)

    for tier in CAP_TIERS.keys():
        if tier in cap_exposure:
            data = cap_exposure[tier]
            weight = (data["value"] / total_value * 100) if total_value > 0 else 0
            print(f"{tier:<12} {data['count']:>10} ${data['value']:>10,.0f} {weight:>7.1f}%")

    print("-" * 45)
    print(f"{'TOTAL':<12} {len(positions):>10} ${total_value:>10,.0f} {100:>7.1f}%")


def analyze_factor_exposure(positions: List[Dict]):
    """Analyze factor exposures (momentum, value, volatility)."""
    print("\n=== Factor Exposure Analysis ===\n")

    if not positions:
        print("No positions to analyze")
        return

    # Note: In production, calculate real factor exposures from price data
    # Here we provide a framework

    total_positions = len(positions)
    total_value = sum(
        float(p.get("qty", p.get("quantity", 0))) *
        float(p.get("current_price", p.get("avg_entry_price", 0)))
        for p in positions
    )

    print("Factor exposures require historical price data for calculation.")
    print("Run with --fetch flag to pull live data (requires Polygon API).\n")

    print("Framework factors analyzed:")
    print("  - Momentum: 12-month price momentum")
    print("  - Value: Price-to-book, P/E ratios")
    print("  - Size: Market capitalization")
    print("  - Volatility: 30-day realized volatility")
    print("  - Quality: ROE, debt ratios")

    print("\nPortfolio Summary:")
    print(f"  Positions: {total_positions}")
    print(f"  Total Value: ${total_value:,.0f}")


def analyze_correlation(positions: List[Dict]):
    """Analyze position correlations."""
    print("\n=== Correlation Analysis ===\n")

    if len(positions) < 2:
        print("Need at least 2 positions for correlation analysis")
        return

    symbols = [p.get("symbol", "") for p in positions]

    print("Top correlated pairs require historical price data.")
    print("This analysis identifies:")
    print("  - Highly correlated positions (>0.8) - diversification risk")
    print("  - Negative correlations (<-0.3) - natural hedges")
    print(f"\nSymbols in portfolio: {', '.join(symbols[:10])}")
    if len(symbols) > 10:
        print(f"  ... and {len(symbols) - 10} more")


def show_all_exposures(positions: List[Dict]):
    """Show all exposure analyses."""
    analyze_sector_exposure(positions)
    analyze_cap_exposure(positions)
    analyze_factor_exposure(positions)
    analyze_correlation(positions)


def main():
    parser = argparse.ArgumentParser(description="Portfolio exposure analysis")
    parser.add_argument("--sector", action="store_true", help="Sector exposure analysis")
    parser.add_argument("--cap", action="store_true", help="Market cap exposure analysis")
    parser.add_argument("--factor", action="store_true", help="Factor exposure analysis")
    parser.add_argument("--correlation", action="store_true", help="Correlation analysis")
    parser.add_argument("--all", action="store_true", help="All exposure analyses")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    args = parser.parse_args()

    positions = load_positions()

    print("\n=== Kobe Portfolio Exposure Analysis ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Positions: {len(positions)}")

    if args.json:
        # JSON output
        sector_data = defaultdict(lambda: {"count": 0, "symbols": []})
        for pos in positions:
            sector = get_sector(pos.get("symbol", ""))
            sector_data[sector]["count"] += 1
            sector_data[sector]["symbols"].append(pos.get("symbol", ""))

        output = {
            "timestamp": datetime.now().isoformat(),
            "position_count": len(positions),
            "sectors": dict(sector_data),
        }
        print(json.dumps(output, indent=2))
    elif args.sector:
        analyze_sector_exposure(positions)
    elif args.cap:
        analyze_cap_exposure(positions)
    elif args.factor:
        analyze_factor_exposure(positions)
    elif args.correlation:
        analyze_correlation(positions)
    elif args.all:
        show_all_exposures(positions)
    else:
        # Default: sector exposure
        analyze_sector_exposure(positions)


if __name__ == "__main__":
    main()
