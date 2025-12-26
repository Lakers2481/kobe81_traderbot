#!/usr/bin/env python3
"""
API key validation and rotation for Kobe trading system.
Usage: python scripts/secrets.py [--validate|--test|--rotate SERVICE]
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None


def load_dotenv(dotenv_path: Path):
    """Load environment variables from .env file."""
    if dotenv_path.exists():
        with open(dotenv_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")


def validate_polygon_key() -> dict:
    """Validate Polygon API key."""
    api_key = os.environ.get("POLYGON_API_KEY", "")
    result = {"service": "Polygon.io", "key_set": bool(api_key), "valid": False, "message": ""}

    if not api_key:
        result["message"] = "API key not set"
        return result

    if not requests:
        result["message"] = "requests library not installed"
        return result

    try:
        # Test with a simple API call
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={api_key}"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            result["valid"] = True
            result["message"] = "API key is valid"
        elif resp.status_code == 401:
            result["message"] = "Invalid API key (401 Unauthorized)"
        elif resp.status_code == 403:
            result["message"] = "API key forbidden (403) - check subscription tier"
        else:
            result["message"] = f"Unexpected response: {resp.status_code}"
    except Exception as e:
        result["message"] = f"Connection error: {e}"

    return result


def validate_alpaca_key() -> dict:
    """Validate Alpaca API credentials."""
    key_id = os.environ.get("ALPACA_API_KEY_ID", "")
    secret_key = os.environ.get("ALPACA_API_SECRET_KEY", "")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    result = {
        "service": "Alpaca",
        "key_set": bool(key_id and secret_key),
        "valid": False,
        "message": "",
        "mode": "paper" if "paper" in base_url.lower() else "live",
    }

    if not key_id or not secret_key:
        result["message"] = "API credentials not set"
        return result

    if not requests:
        result["message"] = "requests library not installed"
        return result

    try:
        # Test with account endpoint
        url = f"{base_url}/v2/account"
        headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        }
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            account = resp.json()
            result["valid"] = True
            result["message"] = f"Valid - Account: {account.get('account_number', 'N/A')}"
            result["buying_power"] = account.get("buying_power", "N/A")
            result["equity"] = account.get("equity", "N/A")
        elif resp.status_code == 401:
            result["message"] = "Invalid API credentials (401 Unauthorized)"
        elif resp.status_code == 403:
            result["message"] = "API access forbidden (403)"
        else:
            result["message"] = f"Unexpected response: {resp.status_code}"
    except Exception as e:
        result["message"] = f"Connection error: {e}"

    return result


def validate_telegram_key() -> dict:
    """Validate Telegram bot token."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    result = {
        "service": "Telegram",
        "key_set": bool(bot_token),
        "valid": False,
        "message": "",
    }

    if not bot_token:
        result["message"] = "Bot token not set (optional)"
        return result

    if not requests:
        result["message"] = "requests library not installed"
        return result

    try:
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            if data.get("ok"):
                result["valid"] = True
                bot_name = data.get("result", {}).get("username", "N/A")
                result["message"] = f"Valid - Bot: @{bot_name}"
                result["chat_id_set"] = bool(chat_id)
        elif resp.status_code == 401:
            result["message"] = "Invalid bot token"
        else:
            result["message"] = f"Unexpected response: {resp.status_code}"
    except Exception as e:
        result["message"] = f"Connection error: {e}"

    return result


def validate_all_secrets():
    """Validate all API keys and secrets."""
    print("\n=== Kobe API Key Validation ===\n")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    results = []

    # Polygon
    print("Checking Polygon.io...")
    polygon_result = validate_polygon_key()
    results.append(polygon_result)
    status = "[OK]" if polygon_result["valid"] else "[FAIL]"
    print(f"  {status} {polygon_result['message']}")

    # Alpaca
    print("\nChecking Alpaca...")
    alpaca_result = validate_alpaca_key()
    results.append(alpaca_result)
    status = "[OK]" if alpaca_result["valid"] else "[FAIL]"
    print(f"  {status} {alpaca_result['message']}")
    if alpaca_result["valid"]:
        print(f"      Mode: {alpaca_result['mode'].upper()}")
        print(f"      Equity: ${alpaca_result.get('equity', 'N/A')}")

    # Telegram (optional)
    print("\nChecking Telegram (optional)...")
    telegram_result = validate_telegram_key()
    results.append(telegram_result)
    if telegram_result["key_set"]:
        status = "[OK]" if telegram_result["valid"] else "[FAIL]"
        print(f"  {status} {telegram_result['message']}")
    else:
        print(f"  [SKIP] Not configured")

    # Summary
    required_valid = polygon_result["valid"] and alpaca_result["valid"]
    print("\n" + "=" * 40)
    print(f"Overall: {'READY FOR TRADING' if required_valid else 'NOT READY'}")

    return required_valid


def test_data_access():
    """Test actual data access with validated keys."""
    print("\n=== Testing Data Access ===\n")

    polygon_key = os.environ.get("POLYGON_API_KEY", "")
    if polygon_key and requests:
        try:
            # Test fetching AAPL data
            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={polygon_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    bar = data["results"][0]
                    print(f"Polygon data test:")
                    print(f"  Symbol: AAPL")
                    print(f"  Close: ${bar.get('c', 'N/A')}")
                    print(f"  Volume: {bar.get('v', 'N/A'):,}")
                    print(f"  [OK] Data fetch successful")
                else:
                    print(f"  [WARN] No data returned")
        except Exception as e:
            print(f"  [FAIL] Error: {e}")

    alpaca_key = os.environ.get("ALPACA_API_KEY_ID", "")
    alpaca_secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
    alpaca_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if alpaca_key and alpaca_secret and requests:
        try:
            url = f"{alpaca_url}/v2/positions"
            headers = {
                "APCA-API-KEY-ID": alpaca_key,
                "APCA-API-SECRET-KEY": alpaca_secret,
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                positions = resp.json()
                print(f"\nAlpaca positions test:")
                print(f"  Open positions: {len(positions)}")
                print(f"  [OK] Broker connection successful")
        except Exception as e:
            print(f"\n  [FAIL] Broker error: {e}")


def main():
    parser = argparse.ArgumentParser(description="API key validation and management")
    parser.add_argument("--validate", action="store_true", help="Validate all API keys")
    parser.add_argument("--test", action="store_true", help="Test data access with keys")
    parser.add_argument("--rotate", type=str, metavar="SERVICE", help="Instructions to rotate key for service")
    parser.add_argument("--dotenv", type=str, help="Path to .env file")

    args = parser.parse_args()

    # Load .env if specified
    if args.dotenv:
        load_dotenv(Path(args.dotenv))

    if args.rotate:
        service = args.rotate.lower()
        print(f"\n=== Key Rotation Instructions: {service.upper()} ===\n")
        if service == "polygon":
            print("1. Log in to https://polygon.io/dashboard")
            print("2. Go to API Keys section")
            print("3. Generate new key or regenerate existing")
            print("4. Update POLYGON_API_KEY in your .env file")
            print("5. Run: python scripts/secrets.py --validate")
        elif service == "alpaca":
            print("1. Log in to https://app.alpaca.markets (or paper-)")
            print("2. Go to API Keys section")
            print("3. Generate new key pair")
            print("4. Update ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY in .env")
            print("5. Run: python scripts/secrets.py --validate")
        elif service == "telegram":
            print("1. Message @BotFather on Telegram")
            print("2. Use /newbot or /token to get new token")
            print("3. Update TELEGRAM_BOT_TOKEN in .env")
            print("4. Run: python scripts/secrets.py --validate")
        else:
            print(f"Unknown service: {service}")
            print("Valid services: polygon, alpaca, telegram")
    elif args.test:
        validate_all_secrets()
        test_data_access()
    elif args.validate:
        is_valid = validate_all_secrets()
        sys.exit(0 if is_valid else 1)
    else:
        # Default: validate
        is_valid = validate_all_secrets()
        sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
