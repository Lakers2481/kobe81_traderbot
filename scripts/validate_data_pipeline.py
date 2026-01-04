#!/usr/bin/env python3
"""
Data Pipeline Validation Script
Validates all 7 layers of data quality for paper trading readiness
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import requests
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / '.env')

from core.structured_log import jlog
from data.providers.polygon_eod import fetch_daily_bars_polygon, PolygonConfig


class DataQualityValidator:
    """Comprehensive data quality validator for trading pipeline"""

    def __init__(self, universe_file: str, cache_dir: str):
        self.universe_file = Path(universe_file)
        self.cache_dir = Path(cache_dir)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "health_score": 0.0,
            "checks_run": 0,
            "passed": 0,
            "warnings": 0,
            "critical": 0,
            "layers": {}
        }

    def validate_all_layers(self) -> Dict:
        """Run all 7 validation layers"""
        print("=" * 80)
        print("DATA QUALITY VALIDATION - 7 LAYERS")
        print("=" * 80)

        # Layer 1: Source Validation
        self._validate_layer1_sources()

        # Layer 2: Schema Validation
        self._validate_layer2_schema()

        # Layer 3: Range Validation
        self._validate_layer3_ranges()

        # Layer 4: Consistency Validation
        self._validate_layer4_consistency()

        # Layer 5: Cross-Source Validation
        self._validate_layer5_cross_source()

        # Layer 6: Temporal Validation
        self._validate_layer6_temporal()

        # Layer 7: Statistical Validation
        self._validate_layer7_statistical()

        # Calculate final health score
        self._calculate_health_score()

        return self.results

    def _validate_layer1_sources(self):
        """Layer 1: Source Validation - API connections and authentication"""
        print("\n[LAYER 1] Source Validation")
        print("-" * 80)

        layer_results = {"name": "Source Validation", "checks": []}

        # Check Polygon API key
        polygon_key = os.getenv('POLYGON_API_KEY', '')
        check = {
            "name": "Polygon API Key",
            "status": "PASS" if polygon_key else "CRITICAL",
            "message": "Valid" if polygon_key else "Missing POLYGON_API_KEY",
            "action": None if polygon_key else "HALT - Add API key to .env"
        }
        layer_results["checks"].append(check)
        self._update_counts(check["status"])
        print(f"  {'PASS' if polygon_key else 'CRITICAL'}: Polygon API Key - {check['message']}")

        # Check Alpaca API keys
        alpaca_key_id = os.getenv('ALPACA_API_KEY_ID', '')
        alpaca_secret = os.getenv('ALPACA_API_SECRET_KEY', '')
        check = {
            "name": "Alpaca API Keys",
            "status": "PASS" if (alpaca_key_id and alpaca_secret) else "CRITICAL",
            "message": "Valid" if (alpaca_key_id and alpaca_secret) else "Missing Alpaca credentials",
            "action": None if (alpaca_key_id and alpaca_secret) else "HALT - Add Alpaca keys to .env"
        }
        layer_results["checks"].append(check)
        self._update_counts(check["status"])
        print(f"  {'PASS' if (alpaca_key_id and alpaca_secret) else 'CRITICAL'}: Alpaca API Keys - {check['message']}")

        # Test Polygon connection with SPY
        if polygon_key:
            try:
                PolygonConfig(api_key=polygon_key)
                test_url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/2024-01-01/2024-01-05?apiKey={polygon_key}"
                response = requests.get(test_url, timeout=10)

                check = {
                    "name": "Polygon API Response",
                    "status": "PASS" if response.status_code == 200 else "CRITICAL",
                    "message": f"HTTP {response.status_code}",
                    "action": None if response.status_code == 200 else "HALT - API not responding"
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'PASS' if response.status_code == 200 else 'CRITICAL'}: Polygon API Response - {check['message']}")

            except Exception as e:
                check = {
                    "name": "Polygon API Response",
                    "status": "CRITICAL",
                    "message": f"Connection failed: {str(e)}",
                    "action": "HALT - Cannot reach Polygon API"
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  CRITICAL: Polygon API Response - {check['message']}")

        self.results["layers"]["layer1"] = layer_results

    def _validate_layer2_schema(self):
        """Layer 2: Schema Validation - Required fields and data types"""
        print("\n[LAYER 2] Schema Validation")
        print("-" * 80)

        layer_results = {"name": "Schema Validation", "checks": []}

        # Check universe file exists and format
        check = {
            "name": "Universe File Exists",
            "status": "PASS" if self.universe_file.exists() else "CRITICAL",
            "message": str(self.universe_file) if self.universe_file.exists() else "Not found",
            "action": None if self.universe_file.exists() else "HALT - Universe file missing"
        }
        layer_results["checks"].append(check)
        self._update_counts(check["status"])
        print(f"  {'PASS' if self.universe_file.exists() else 'CRITICAL'}: Universe File - {check['message']}")

        if self.universe_file.exists():
            # Load and validate universe format
            try:
                universe_df = pd.read_csv(self.universe_file)

                # Check for 'symbol' column
                has_symbol_col = 'symbol' in universe_df.columns
                check = {
                    "name": "Universe File Schema",
                    "status": "PASS" if has_symbol_col else "CRITICAL",
                    "message": f"Columns: {list(universe_df.columns)}" if has_symbol_col else "Missing 'symbol' column",
                    "action": None if has_symbol_col else "HALT - Invalid universe format"
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'PASS' if has_symbol_col else 'CRITICAL'}: Universe Schema - {check['message']}")

                # Count symbols
                symbol_count = len(universe_df)
                check = {
                    "name": "Universe Symbol Count",
                    "status": "PASS",
                    "message": f"{symbol_count} symbols loaded",
                    "action": None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  PASS: Symbol Count - {symbol_count} symbols")

            except Exception as e:
                check = {
                    "name": "Universe File Schema",
                    "status": "CRITICAL",
                    "message": f"Load failed: {str(e)}",
                    "action": "HALT - Cannot parse universe file"
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  CRITICAL: Universe Schema - {check['message']}")

        # Check cache directory structure
        cache_exists = self.cache_dir.exists()
        check = {
            "name": "Cache Directory",
            "status": "PASS" if cache_exists else "WARN",
            "message": str(self.cache_dir) if cache_exists else "Not found - will be created",
            "action": None
        }
        layer_results["checks"].append(check)
        self._update_counts(check["status"])
        print(f"  {'PASS' if cache_exists else 'WARN'}: Cache Directory - {check['message']}")

        # Check sample cache file schema (SPY)
        if cache_exists:
            spy_cache = list(self.cache_dir.glob("SPY_*.csv"))
            if spy_cache:
                try:
                    sample_df = pd.read_csv(spy_cache[0])
                    required_cols = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
                    has_required = required_cols.issubset(set(sample_df.columns))

                    check = {
                        "name": "Cache File Schema",
                        "status": "PASS" if has_required else "CRITICAL",
                        "message": f"Columns: {list(sample_df.columns)}" if has_required else f"Missing: {required_cols - set(sample_df.columns)}",
                        "action": None if has_required else "HALT - Invalid cache format"
                    }
                    layer_results["checks"].append(check)
                    self._update_counts(check["status"])
                    print(f"  {'PASS' if has_required else 'CRITICAL'}: Cache Schema - {check['message']}")

                except Exception as e:
                    check = {
                        "name": "Cache File Schema",
                        "status": "WARN",
                        "message": f"Cannot read sample: {str(e)}",
                        "action": None
                    }
                    layer_results["checks"].append(check)
                    self._update_counts(check["status"])
                    print(f"  WARN: Cache Schema - {check['message']}")

        self.results["layers"]["layer2"] = layer_results

    def _validate_layer3_ranges(self):
        """Layer 3: Range Validation - Price/volume bounds"""
        print("\n[LAYER 3] Range Validation")
        print("-" * 80)

        layer_results = {"name": "Range Validation", "checks": []}

        # Sample validation on SPY cache
        spy_cache = list(self.cache_dir.glob("SPY_*.csv"))
        if spy_cache:
            try:
                df = pd.read_csv(spy_cache[0])

                # Check for negative prices
                negative_prices = (df[['open', 'high', 'low', 'close']] < 0).any().any()
                check = {
                    "name": "Negative Prices",
                    "status": "CRITICAL" if negative_prices else "PASS",
                    "message": "Found negative prices!" if negative_prices else "All prices positive",
                    "action": "HALT - Data corruption" if negative_prices else None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'CRITICAL' if negative_prices else 'PASS'}: Negative Prices - {check['message']}")

                # Check for zero/null prices
                zero_prices = (df[['open', 'high', 'low', 'close']] == 0).any().any()
                null_prices = df[['open', 'high', 'low', 'close']].isnull().any().any()
                check = {
                    "name": "Zero/Null Prices",
                    "status": "CRITICAL" if (zero_prices or null_prices) else "PASS",
                    "message": "Found zero/null prices!" if (zero_prices or null_prices) else "All prices valid",
                    "action": "HALT - Missing data" if (zero_prices or null_prices) else None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'CRITICAL' if (zero_prices or null_prices) else 'PASS'}: Zero/Null Prices - {check['message']}")

                # Check for negative volume
                negative_vol = (df['volume'] < 0).any()
                check = {
                    "name": "Negative Volume",
                    "status": "CRITICAL" if negative_vol else "PASS",
                    "message": "Found negative volume!" if negative_vol else "All volumes non-negative",
                    "action": "HALT - Data corruption" if negative_vol else None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'CRITICAL' if negative_vol else 'PASS'}: Negative Volume - {check['message']}")

                # Price reasonableness (SPY should be $300-$600 range as of 2024)
                min_price = df[['open', 'high', 'low', 'close']].min().min()
                max_price = df[['open', 'high', 'low', 'close']].max().max()
                reasonable = (100 < min_price < 700) and (100 < max_price < 700)
                check = {
                    "name": "SPY Price Range",
                    "status": "PASS" if reasonable else "WARN",
                    "message": f"${min_price:.2f} - ${max_price:.2f}",
                    "action": None if reasonable else "Verify - Prices outside expected range"
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'PASS' if reasonable else 'WARN'}: SPY Price Range - {check['message']}")

            except Exception as e:
                check = {
                    "name": "Range Validation",
                    "status": "WARN",
                    "message": f"Cannot validate: {str(e)}",
                    "action": None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  WARN: Range Validation - {check['message']}")

        self.results["layers"]["layer3"] = layer_results

    def _validate_layer4_consistency(self):
        """Layer 4: Consistency Validation - OHLC relationships"""
        print("\n[LAYER 4] Consistency Validation")
        print("-" * 80)

        layer_results = {"name": "Consistency Validation", "checks": []}

        # Sample validation on SPY cache
        spy_cache = list(self.cache_dir.glob("SPY_*.csv"))
        if spy_cache:
            try:
                df = pd.read_csv(spy_cache[0])

                # OHLC consistency: High >= Open, Close, Low; Low <= Open, Close, High
                violations_h = ((df['high'] < df['open']) |
                               (df['high'] < df['close']) |
                               (df['high'] < df['low'])).sum()

                violations_l = ((df['low'] > df['open']) |
                               (df['low'] > df['close']) |
                               (df['low'] > df['high'])).sum()

                total_violations = violations_h + violations_l

                check = {
                    "name": "OHLC Consistency",
                    "status": "CRITICAL" if total_violations > 0 else "PASS",
                    "message": f"{total_violations} violations found" if total_violations > 0 else "All OHLC relationships valid",
                    "action": "HALT - Data corruption" if total_violations > 0 else None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'CRITICAL' if total_violations > 0 else 'PASS'}: OHLC Consistency - {check['message']}")

            except Exception as e:
                check = {
                    "name": "OHLC Consistency",
                    "status": "WARN",
                    "message": f"Cannot validate: {str(e)}",
                    "action": None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  WARN: OHLC Consistency - {check['message']}")

        self.results["layers"]["layer4"] = layer_results

    def _validate_layer5_cross_source(self):
        """Layer 5: Cross-Source Validation - Compare providers if available"""
        print("\n[LAYER 5] Cross-Source Validation")
        print("-" * 80)

        layer_results = {"name": "Cross-Source Validation", "checks": []}

        # This would compare Polygon vs Alpaca, but requires both sources
        # For now, just verify cache vs universe consistency

        if self.universe_file.exists() and self.cache_dir.exists():
            try:
                universe_df = pd.read_csv(self.universe_file)
                cached_symbols = set()

                for cache_file in self.cache_dir.glob("*_*.csv"):
                    symbol = cache_file.stem.split('_')[0]
                    cached_symbols.add(symbol)

                universe_symbols = set(universe_df['symbol'].tolist())
                coverage = len(cached_symbols & universe_symbols) / len(universe_symbols) * 100 if universe_symbols else 0

                check = {
                    "name": "Cache Coverage",
                    "status": "PASS" if coverage > 50 else "WARN",
                    "message": f"{coverage:.1f}% of universe cached ({len(cached_symbols)} / {len(universe_symbols)} symbols)",
                    "action": "Run prefetch_polygon_universe.py" if coverage < 50 else None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'PASS' if coverage > 50 else 'WARN'}: Cache Coverage - {check['message']}")

            except Exception as e:
                check = {
                    "name": "Cache Coverage",
                    "status": "WARN",
                    "message": f"Cannot validate: {str(e)}",
                    "action": None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  WARN: Cache Coverage - {check['message']}")

        self.results["layers"]["layer5"] = layer_results

    def _validate_layer6_temporal(self):
        """Layer 6: Temporal Validation - Freshness and gaps"""
        print("\n[LAYER 6] Temporal Validation")
        print("-" * 80)

        layer_results = {"name": "Temporal Validation", "checks": []}

        # Check cache freshness
        spy_cache = list(self.cache_dir.glob("SPY_*.csv"))
        if spy_cache:
            try:
                df = pd.read_csv(spy_cache[0], parse_dates=['timestamp'])

                # Most recent data point
                latest_date = df['timestamp'].max()
                days_old = (datetime.now() - latest_date).days

                check = {
                    "name": "Data Freshness",
                    "status": "PASS" if days_old <= 5 else "WARN",
                    "message": f"Latest: {latest_date.strftime('%Y-%m-%d')} ({days_old} days old)",
                    "action": "Update cache - data is stale" if days_old > 5 else None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'PASS' if days_old <= 5 else 'WARN'}: Data Freshness - {check['message']}")

                # Check for gaps (trading days should be ~252/year)
                df_sorted = df.sort_values('timestamp')
                time_diffs = df_sorted['timestamp'].diff()
                large_gaps = (time_diffs > pd.Timedelta(days=7)).sum()  # Gaps > 1 week

                check = {
                    "name": "Time Series Gaps",
                    "status": "WARN" if large_gaps > 5 else "PASS",
                    "message": f"{large_gaps} gaps > 7 days found",
                    "action": "Verify - May include holidays/market closures" if large_gaps > 5 else None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'WARN' if large_gaps > 5 else 'PASS'}: Time Series Gaps - {check['message']}")

            except Exception as e:
                check = {
                    "name": "Temporal Validation",
                    "status": "WARN",
                    "message": f"Cannot validate: {str(e)}",
                    "action": None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  WARN: Temporal Validation - {check['message']}")

        self.results["layers"]["layer6"] = layer_results

    def _validate_layer7_statistical(self):
        """Layer 7: Statistical Validation - Outliers and anomalies"""
        print("\n[LAYER 7] Statistical Validation")
        print("-" * 80)

        layer_results = {"name": "Statistical Validation", "checks": []}

        # Sample validation on SPY cache
        spy_cache = list(self.cache_dir.glob("SPY_*.csv"))
        if spy_cache:
            try:
                df = pd.read_csv(spy_cache[0])

                # Calculate daily returns
                df['return'] = df['close'].pct_change()

                # Flag extreme moves (>5% in one day for SPY is unusual)
                extreme_moves = (abs(df['return']) > 0.05).sum()

                check = {
                    "name": "Extreme Price Moves",
                    "status": "PASS" if extreme_moves < 10 else "WARN",
                    "message": f"{extreme_moves} days with >5% move",
                    "action": "Verify - May include splits/dividends" if extreme_moves > 10 else None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'PASS' if extreme_moves < 10 else 'WARN'}: Extreme Moves - {check['message']}")

                # Volume consistency (check for zero volume days)
                zero_volume_days = (df['volume'] == 0).sum()

                check = {
                    "name": "Zero Volume Days",
                    "status": "WARN" if zero_volume_days > 0 else "PASS",
                    "message": f"{zero_volume_days} days with zero volume",
                    "action": "Verify - SPY should always have volume" if zero_volume_days > 0 else None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  {'WARN' if zero_volume_days > 0 else 'PASS'}: Zero Volume - {check['message']}")

            except Exception as e:
                check = {
                    "name": "Statistical Validation",
                    "status": "WARN",
                    "message": f"Cannot validate: {str(e)}",
                    "action": None
                }
                layer_results["checks"].append(check)
                self._update_counts(check["status"])
                print(f"  WARN: Statistical Validation - {check['message']}")

        self.results["layers"]["layer7"] = layer_results

    def _validate_vix_source(self):
        """Special check for VIX data availability - CRITICAL for position sizing"""
        print("\n[SPECIAL] VIX Data Validation")
        print("-" * 80)

        vix_symbols = ['VIX', 'VIXY', 'VXX']  # Try multiple VIX proxies
        vix_available = False

        for vix_symbol in vix_symbols:
            vix_cache = list(self.cache_dir.glob(f"{vix_symbol}_*.csv"))
            if vix_cache:
                try:
                    df = pd.read_csv(vix_cache[0])

                    # Check for zero/null VIX values (CRITICAL BUG!)
                    zero_vix = (df['close'] == 0).sum()
                    null_vix = df['close'].isnull().sum()

                    if zero_vix == 0 and null_vix == 0:
                        vix_available = True
                        print(f"  PASS: {vix_symbol} data available and valid")

                        # Check VIX range (should be 8-80 typically)
                        vix_min = df['close'].min()
                        vix_max = df['close'].max()
                        if 5 < vix_min < 100 and 5 < vix_max < 100:
                            print(f"  PASS: {vix_symbol} range: {vix_min:.2f} - {vix_max:.2f}")
                        else:
                            print(f"  WARN: {vix_symbol} range unusual: {vix_min:.2f} - {vix_max:.2f}")
                        break
                    else:
                        print(f"  CRITICAL: {vix_symbol} has {zero_vix} zero values, {null_vix} null values")

                except Exception as e:
                    print(f"  WARN: Cannot validate {vix_symbol}: {str(e)}")

        if not vix_available:
            print("  CRITICAL: No valid VIX data found! Position sizing will fail!")
            print("  ACTION: HALT - Fetch VIX data before trading")

    def _update_counts(self, status: str):
        """Update validation counters"""
        self.results["checks_run"] += 1
        if status == "PASS":
            self.results["passed"] += 1
        elif status == "WARN":
            self.results["warnings"] += 1
        elif status == "CRITICAL":
            self.results["critical"] += 1

    def _calculate_health_score(self):
        """Calculate overall health score"""
        total = self.results["checks_run"]
        if total == 0:
            self.results["health_score"] = 0.0
            return

        # Weight: PASS = 1.0, WARN = 0.5, CRITICAL = 0.0
        score = (self.results["passed"] * 1.0 + self.results["warnings"] * 0.5) / total * 100
        self.results["health_score"] = round(score, 1)

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 80)
        print("DATA QUALITY REPORT")
        print("=" * 80)
        print(f"Health Score: {self.results['health_score']}%")
        print(f"Checks Run: {self.results['checks_run']}")
        print(f"Passed: {self.results['passed']} | Warnings: {self.results['warnings']} | Critical: {self.results['critical']}")

        if self.results["critical"] > 0:
            print("\nCRITICAL ISSUES (MUST FIX BEFORE TRADING):")
            for layer_name, layer_data in self.results["layers"].items():
                for check in layer_data["checks"]:
                    if check["status"] == "CRITICAL":
                        print(f"  - [{layer_data['name']}] {check['name']}: {check['message']}")
                        if check["action"]:
                            print(f"    ACTION: {check['action']}")

        if self.results["warnings"] > 0:
            print("\nWARNINGS (RECOMMENDED TO FIX):")
            for layer_name, layer_data in self.results["layers"].items():
                for check in layer_data["checks"]:
                    if check["status"] == "WARN":
                        print(f"  - [{layer_data['name']}] {check['name']}: {check['message']}")
                        if check["action"]:
                            print(f"    ACTION: {check['action']}")

        print("\nVERIFIED HEALTHY:")
        healthy_count = 0
        for layer_name, layer_data in self.results["layers"].items():
            layer_pass = sum(1 for c in layer_data["checks"] if c["status"] == "PASS")
            if layer_pass > 0:
                print(f"  - {layer_data['name']}: {layer_pass} checks passed")
                healthy_count += layer_pass

        print("\nRECOMMENDATIONS:")
        if self.results["critical"] > 0:
            print("  1. FIX CRITICAL ISSUES - Trading cannot proceed safely")
        if self.results["warnings"] > 0:
            print("  2. Address warnings to improve data quality")
        if self.results["health_score"] < 70:
            print("  3. Health score < 70% - Run prefetch and verify data sources")
        else:
            print("  1. Data pipeline appears ready for paper trading")
            print("  2. Monitor for freshness - update cache daily")
            print("  3. Verify VIX data availability before position sizing")

        print("=" * 80)

    def save_report(self, output_file: str):
        """Save validation report to JSON"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nReport saved to: {output_file}")


def main():
    """Main validation routine"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate data pipeline for paper trading")
    parser.add_argument("--universe", default="data/universe/optionable_liquid_900.csv",
                       help="Universe file path")
    parser.add_argument("--cache", default="data/cache",
                       help="Cache directory path")
    parser.add_argument("--output", default="reports/data_pipeline_audit.json",
                       help="Output report file")

    args = parser.parse_args()

    # Create validator
    validator = DataQualityValidator(
        universe_file=args.universe,
        cache_dir=args.cache
    )

    # Run all validations
    results = validator.validate_all_layers()

    # Special VIX check
    validator._validate_vix_source()

    # Print summary
    validator.print_summary()

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validator.save_report(str(output_path))

    # Exit code based on health
    if results["critical"] > 0:
        print("\nEXIT CODE 2: Critical issues found - trading HALTED")
        sys.exit(2)
    elif results["warnings"] > 0:
        print("\nEXIT CODE 1: Warnings found - proceed with caution")
        sys.exit(1)
    else:
        print("\nEXIT CODE 0: All checks passed - ready for paper trading")
        sys.exit(0)


if __name__ == "__main__":
    main()
