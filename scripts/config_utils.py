#!/usr/bin/env python3
"""
Configuration Management for Kobe Trading System

Shows current config with signature, validates against schema, and compares with defaults.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env
from core.config_pin import sha256_file


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_DOTENV = "C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env"
CONFIG_FILE = ROOT / "config" / "settings.json"

# Default configuration schema/values for comparison
DEFAULT_CONFIG = {
    "polygon": {
        "base_url": "https://api.polygon.io",
        "adjusted": True,
        "sort": "asc",
        "limit": 50000
    },
    "backtest": {
        "initial_cash": 100000,
        "slippage_bps": 5
    },
    "risk": {
        "max_notional_per_order": 75.0,
        "max_daily_notional": 1000.0,
        "min_price": 3.0,
        "max_price": 1000.0,
        "allow_shorts": False
    },
    "execution": {
        "order_type": "IOC_LIMIT",
        "limit_offset_pct": 0.1,
        "timeout_seconds": 10
    }
}

# Schema for validation
CONFIG_SCHEMA = {
    "polygon": {
        "type": "object",
        "required": ["base_url"],
        "properties": {
            "base_url": {"type": "string"},
            "adjusted": {"type": "boolean"},
            "sort": {"type": "string", "enum": ["asc", "desc"]},
            "limit": {"type": "integer", "min": 1, "max": 100000}
        }
    },
    "backtest": {
        "type": "object",
        "properties": {
            "initial_cash": {"type": "number", "min": 0},
            "slippage_bps": {"type": "number", "min": 0, "max": 100}
        }
    },
    "risk": {
        "type": "object",
        "properties": {
            "max_notional_per_order": {"type": "number", "min": 0},
            "max_daily_notional": {"type": "number", "min": 0},
            "min_price": {"type": "number", "min": 0},
            "max_price": {"type": "number", "min": 0},
            "allow_shorts": {"type": "boolean"}
        }
    },
    "execution": {
        "type": "object",
        "properties": {
            "order_type": {"type": "string"},
            "limit_offset_pct": {"type": "number"},
            "timeout_seconds": {"type": "integer", "min": 1}
        }
    }
}


# -----------------------------------------------------------------------------
# Config Loading
# -----------------------------------------------------------------------------
def load_config(config_path: Path = CONFIG_FILE) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load configuration file and return (config, error)."""
    if not config_path.exists():
        return None, f"Config file not found: {config_path}"

    try:
        content = config_path.read_text(encoding="utf-8")
        # Handle BOM if present
        if content.startswith('\ufeff'):
            content = content[1:]
        config = json.loads(content)
        return config, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    except Exception as e:
        return None, f"Error reading config: {e}"


def get_config_signature(config_path: Path = CONFIG_FILE) -> Optional[str]:
    """Get SHA-256 signature of config file."""
    if not config_path.exists():
        return None
    try:
        return sha256_file(config_path)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Config Display
# -----------------------------------------------------------------------------
def show_config(config_path: Path = CONFIG_FILE, show_env: bool = True) -> Dict[str, Any]:
    """Show current configuration with signature."""
    config, error = load_config(config_path)
    signature = get_config_signature(config_path)

    result = {
        "action": "show_config",
        "config_file": str(config_path),
        "exists": config_path.exists(),
        "signature": signature,
        "last_modified": None,
        "config": None,
        "error": error
    }

    if config_path.exists():
        stat = config_path.stat()
        result["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        result["size_bytes"] = stat.st_size

    if config:
        result["config"] = config
        result["status"] = "PASS"
    else:
        result["status"] = "FAIL"

    # Show relevant environment variables
    if show_env:
        env_vars = {}
        relevant_keys = [
            "ALPACA_BASE_URL", "ALPACA_API_KEY_ID",
            "POLYGON_API_KEY", "KOBE_ENV", "KOBE_LOG_LEVEL"
        ]
        for key in relevant_keys:
            val = os.getenv(key)
            if val:
                # Mask sensitive values
                if "SECRET" in key or "KEY" in key.upper():
                    env_vars[key] = val[:4] + "****" + val[-4:] if len(val) > 8 else "****"
                else:
                    env_vars[key] = val
        result["environment"] = env_vars

    return result


# -----------------------------------------------------------------------------
# Config Validation
# -----------------------------------------------------------------------------
def validate_type(value: Any, expected_type: str) -> bool:
    """Validate value against expected type."""
    type_map = {
        "string": str,
        "boolean": bool,
        "integer": int,
        "number": (int, float),
        "object": dict,
        "array": list
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return True
    return isinstance(value, expected)


def validate_config(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Validate configuration against schema."""
    config, error = load_config(config_path)

    if error:
        return {
            "action": "validate_config",
            "status": "FAIL",
            "error": error,
            "issues": []
        }

    issues = []
    warnings = []

    # Validate each section
    for section_name, section_schema in CONFIG_SCHEMA.items():
        section_config = config.get(section_name)

        # Check if section exists (warning if optional)
        if section_config is None:
            warnings.append(f"Section '{section_name}' not present in config")
            continue

        # Check section type
        if not isinstance(section_config, dict):
            issues.append(f"Section '{section_name}' must be an object")
            continue

        # Check required fields
        required = section_schema.get("required", [])
        for field in required:
            if field not in section_config:
                issues.append(f"Missing required field: {section_name}.{field}")

        # Validate properties
        properties = section_schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name not in section_config:
                continue

            value = section_config[prop_name]
            prop_type = prop_schema.get("type")

            # Type check
            if prop_type and not validate_type(value, prop_type):
                issues.append(
                    f"Type mismatch: {section_name}.{prop_name} "
                    f"expected {prop_type}, got {type(value).__name__}"
                )
                continue

            # Range checks for numbers
            if prop_type in ("number", "integer"):
                min_val = prop_schema.get("min")
                max_val = prop_schema.get("max")
                if min_val is not None and value < min_val:
                    issues.append(f"{section_name}.{prop_name} below minimum ({min_val})")
                if max_val is not None and value > max_val:
                    issues.append(f"{section_name}.{prop_name} above maximum ({max_val})")

            # Enum check
            if "enum" in prop_schema and value not in prop_schema["enum"]:
                issues.append(
                    f"Invalid value for {section_name}.{prop_name}: "
                    f"'{value}' not in {prop_schema['enum']}"
                )

    status = "FAIL" if issues else ("WARN" if warnings else "PASS")

    return {
        "action": "validate_config",
        "status": status,
        "config_file": str(config_path),
        "signature": get_config_signature(config_path),
        "issues": issues,
        "warnings": warnings,
        "sections_validated": list(CONFIG_SCHEMA.keys())
    }


# -----------------------------------------------------------------------------
# Config Comparison
# -----------------------------------------------------------------------------
def compare_with_defaults(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Compare current config with default values."""
    config, error = load_config(config_path)

    if error:
        return {
            "action": "compare_config",
            "status": "FAIL",
            "error": error,
            "differences": []
        }

    differences = []
    additions = []
    missing = []

    def compare_nested(current: Dict, default: Dict, path: str = ""):
        """Recursively compare nested dictionaries."""
        # Find differences and additions
        for key, curr_val in current.items():
            full_path = f"{path}.{key}" if path else key

            if key not in default:
                additions.append({
                    "path": full_path,
                    "value": curr_val,
                    "type": "addition"
                })
            elif isinstance(curr_val, dict) and isinstance(default.get(key), dict):
                compare_nested(curr_val, default[key], full_path)
            elif curr_val != default.get(key):
                differences.append({
                    "path": full_path,
                    "current": curr_val,
                    "default": default.get(key),
                    "type": "modified"
                })

        # Find missing keys
        for key, def_val in default.items():
            full_path = f"{path}.{key}" if path else key
            if key not in current:
                missing.append({
                    "path": full_path,
                    "default_value": def_val,
                    "type": "missing"
                })

    compare_nested(config, DEFAULT_CONFIG)

    has_issues = bool(differences or additions or missing)

    return {
        "action": "compare_config",
        "status": "WARN" if has_issues else "PASS",
        "config_file": str(config_path),
        "signature": get_config_signature(config_path),
        "summary": {
            "modified_values": len(differences),
            "additions": len(additions),
            "missing_from_current": len(missing)
        },
        "differences": differences,
        "additions": additions,
        "missing": missing
    }


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def print_result(result: Dict[str, Any], verbose: bool = False) -> int:
    """Print result and return exit code."""
    action = result.get("action", "unknown")
    status = result.get("status", "UNKNOWN")

    print("=" * 70)
    print(f"KOBE CONFIG MANAGEMENT - {action.upper()}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 70)

    if status == "FAIL":
        status_str = "[FAIL]"
    elif status == "WARN":
        status_str = "[WARN]"
    else:
        status_str = "[PASS]"

    print(f"\nStatus: {status_str}")

    # Config file info
    print(f"\nConfig File: {result.get('config_file', 'N/A')}")
    if result.get("exists", True):
        print(f"Signature (SHA-256): {result.get('signature', 'N/A')}")
        if result.get("last_modified"):
            print(f"Last Modified: {result.get('last_modified')}")
        if result.get("size_bytes"):
            print(f"Size: {result.get('size_bytes')} bytes")
    else:
        print("  (file does not exist)")

    # Error
    if result.get("error"):
        print(f"\nError: {result.get('error')}")

    # Action-specific output
    if action == "show_config" and result.get("config"):
        print("\n" + "-" * 50)
        print("Configuration Contents:")
        print("-" * 50)
        print(json.dumps(result["config"], indent=2))

        if result.get("environment"):
            print("\n" + "-" * 50)
            print("Environment Variables:")
            print("-" * 50)
            for key, val in result["environment"].items():
                print(f"  {key}: {val}")

    elif action == "validate_config":
        print("\n" + "-" * 50)
        print("Validation Results:")
        print("-" * 50)

        issues = result.get("issues", [])
        warnings = result.get("warnings", [])

        if issues:
            print("\nIssues (must fix):")
            for issue in issues:
                print(f"  - {issue}")

        if warnings:
            print("\nWarnings (review):")
            for warning in warnings:
                print(f"  - {warning}")

        if not issues and not warnings:
            print("\n  All validations passed!")

        print(f"\nSections validated: {', '.join(result.get('sections_validated', []))}")

    elif action == "compare_config":
        print("\n" + "-" * 50)
        print("Comparison with Defaults:")
        print("-" * 50)

        summary = result.get("summary", {})
        print(f"\n  Modified values: {summary.get('modified_values', 0)}")
        print(f"  Additions:       {summary.get('additions', 0)}")
        print(f"  Missing:         {summary.get('missing_from_current', 0)}")

        differences = result.get("differences", [])
        if differences:
            print("\nModified values:")
            for diff in differences:
                print(f"  {diff['path']}:")
                print(f"    current: {diff['current']}")
                print(f"    default: {diff['default']}")

        additions = result.get("additions", [])
        if additions:
            print("\nAdditions (not in defaults):")
            for add in additions:
                print(f"  {add['path']}: {add['value']}")

        missing = result.get("missing", [])
        if missing:
            print("\nMissing (in defaults but not current):")
            for miss in missing:
                print(f"  {miss['path']}: {miss['default_value']}")

    print("\n" + "=" * 70)

    if status == "FAIL":
        return 2
    elif status == "WARN":
        return 1
    return 0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Kobe Configuration Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python config.py --show              # Show current configuration
  python config.py --validate          # Validate against schema
  python config.py --diff              # Compare with defaults
  python config.py --show --validate   # Show and validate
        """
    )
    parser.add_argument("--dotenv", type=str, default=DEFAULT_DOTENV,
                        help="Path to .env file")
    parser.add_argument("--config", type=str, default=str(CONFIG_FILE),
                        help="Path to config file")
    parser.add_argument("--show", action="store_true",
                        help="Show current configuration with signature")
    parser.add_argument("--validate", action="store_true",
                        help="Validate configuration against schema")
    parser.add_argument("--diff", action="store_true",
                        help="Compare with default configuration")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    # Load environment
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        loaded = load_env(dotenv)
        if args.verbose:
            print(f"Loaded {len(loaded)} env vars from {dotenv}")

    config_path = Path(args.config)

    # Default to --show if no flags
    if not args.show and not args.validate and not args.diff:
        args.show = True

    try:
        exit_code = 0

        if args.show:
            result = show_config(config_path)
            code = print_result(result, verbose=args.verbose)
            exit_code = max(exit_code, code)

        if args.validate:
            result = validate_config(config_path)
            code = print_result(result, verbose=args.verbose)
            exit_code = max(exit_code, code)

        if args.diff:
            result = compare_with_defaults(config_path)
            code = print_result(result, verbose=args.verbose)
            exit_code = max(exit_code, code)

        sys.exit(exit_code)
    except Exception as e:
        print(f"[ERROR] Config operation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
