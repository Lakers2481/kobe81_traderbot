#!/usr/bin/env python3
"""
Environment variable management for Kobe trading system.
Usage: python scripts/env.py [--check|--list|--validate|--export FILE]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Required environment variables for trading
REQUIRED_VARS = [
    "POLYGON_API_KEY",
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "ALPACA_BASE_URL",
]

# Optional but recommended
OPTIONAL_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "LOG_LEVEL",
]


def load_dotenv(dotenv_path: Path) -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    if dotenv_path.exists():
        with open(dotenv_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars


def check_env_vars(dotenv_path: Path = None) -> Tuple[List[str], List[str], List[str]]:
    """Check which environment variables are set, missing, or optional."""
    if dotenv_path:
        env_vars = load_dotenv(dotenv_path)
        # Merge with actual environment (env file takes precedence for display)
        for key in REQUIRED_VARS + OPTIONAL_VARS:
            if key not in env_vars and key in os.environ:
                env_vars[key] = os.environ[key]
    else:
        env_vars = dict(os.environ)

    present = []
    missing = []
    optional_missing = []

    for var in REQUIRED_VARS:
        if var in env_vars and env_vars[var]:
            present.append(var)
        else:
            missing.append(var)

    for var in OPTIONAL_VARS:
        if var not in env_vars or not env_vars[var]:
            optional_missing.append(var)

    return present, missing, optional_missing


def mask_value(value: str) -> str:
    """Mask sensitive value for display."""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def list_env_vars(dotenv_path: Path = None, show_values: bool = False):
    """List all environment variables."""
    if dotenv_path:
        env_vars = load_dotenv(dotenv_path)
    else:
        env_vars = {}

    print("\n=== Required Environment Variables ===\n")
    for var in REQUIRED_VARS:
        value = env_vars.get(var) or os.environ.get(var)
        if value:
            display_value = value if show_values else mask_value(value)
            print(f"  {var}: {display_value}")
        else:
            print(f"  {var}: [NOT SET]")

    print("\n=== Optional Environment Variables ===\n")
    for var in OPTIONAL_VARS:
        value = env_vars.get(var) or os.environ.get(var)
        if value:
            display_value = value if show_values else mask_value(value)
            print(f"  {var}: {display_value}")
        else:
            print(f"  {var}: [NOT SET]")


def validate_env_vars(dotenv_path: Path = None) -> bool:
    """Validate environment variables for trading readiness."""
    present, missing, optional_missing = check_env_vars(dotenv_path)

    print("\n=== Environment Validation ===\n")

    if present:
        print("Present (required):")
        for var in present:
            print(f"  [OK] {var}")

    if missing:
        print("\nMissing (required):")
        for var in missing:
            print(f"  [ERROR] {var}")

    if optional_missing:
        print("\nMissing (optional):")
        for var in optional_missing:
            print(f"  [WARN] {var}")

    # Validate specific values
    alpaca_url = os.environ.get("ALPACA_BASE_URL", "")
    if alpaca_url:
        if "paper" in alpaca_url.lower():
            print("\n[INFO] Alpaca mode: PAPER TRADING")
        elif "api.alpaca.markets" in alpaca_url:
            print("\n[WARN] Alpaca mode: LIVE TRADING (REAL MONEY)")

    is_valid = len(missing) == 0
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    return is_valid


def export_template(output_path: Path):
    """Export .env template file."""
    template = '''# Kobe Trading System Environment Variables
# Copy this file to .env and fill in your values

# === Required ===

# Polygon.io API key for market data
POLYGON_API_KEY=your_polygon_api_key_here

# Alpaca API credentials for trading
ALPACA_API_KEY_ID=your_alpaca_key_id_here
ALPACA_API_SECRET_KEY=your_alpaca_secret_key_here

# Alpaca base URL
# Paper trading: https://paper-api.alpaca.markets
# Live trading: https://api.alpaca.markets
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# === Optional ===

# Telegram notifications (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
'''
    with open(output_path, "w") as f:
        f.write(template)
    print(f"Template exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Environment variable management")
    parser.add_argument("--check", action="store_true", help="Quick check of required vars")
    parser.add_argument("--list", action="store_true", help="List all env vars")
    parser.add_argument("--show-values", action="store_true", help="Show actual values (not masked)")
    parser.add_argument("--validate", action="store_true", help="Full validation")
    parser.add_argument("--export", type=str, metavar="FILE", help="Export .env template")
    parser.add_argument("--dotenv", type=str, help="Path to .env file")

    args = parser.parse_args()
    dotenv_path = Path(args.dotenv) if args.dotenv else None

    # Load .env if specified
    if dotenv_path and dotenv_path.exists():
        env_vars = load_dotenv(dotenv_path)
        for key, value in env_vars.items():
            os.environ[key] = value

    if args.export:
        export_template(Path(args.export))
    elif args.list:
        list_env_vars(dotenv_path, args.show_values)
    elif args.validate:
        is_valid = validate_env_vars(dotenv_path)
        sys.exit(0 if is_valid else 1)
    elif args.check:
        present, missing, _ = check_env_vars(dotenv_path)
        print(f"Required vars: {len(present)}/{len(REQUIRED_VARS)} present")
        if missing:
            print(f"Missing: {', '.join(missing)}")
            sys.exit(1)
        print("All required environment variables are set.")
    else:
        # Default: validate
        is_valid = validate_env_vars(dotenv_path)
        sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
