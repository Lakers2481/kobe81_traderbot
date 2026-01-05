"""
Typed Settings Schema (Pydantic)
================================

Provides typed, validated configuration for Kobe trading system.
Fail-closed in live mode, warn-only in paper mode.

FIX (2026-01-05): Created for config validation and type safety.

Usage:
    from config.settings_schema import load_validated_settings, get_validated_setting

    # Load and validate all settings
    settings = load_validated_settings()

    # Access typed settings
    mode = settings.system.mode
    max_notional = settings.modes.real.max_notional_per_order

    # Validate in context
    validate_settings_for_mode("live")  # Raises if invalid for live
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import yaml

logger = logging.getLogger(__name__)

# Try to import pydantic v2 first, then v1
try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseModel, Field, ValidationError, validator
        PYDANTIC_V2 = False
    except ImportError:
        PYDANTIC_V2 = None
        logger.warning(
            "pydantic not installed. Schema validation will be skipped. "
            "Install with: pip install pydantic"
        )


# ============================================================================
# Schema Definitions
# ============================================================================

if PYDANTIC_V2 is not None:

    class SystemConfig(BaseModel):
        """System-level configuration."""
        name: str = Field(default="Kobe", description="System name")
        version: str = Field(default="2.0.0", description="System version")
        mode: Literal["paper", "live", "micro"] = Field(
            default="paper",
            description="Trading mode"
        )
        timezone: str = Field(default="America/New_York", description="System timezone")

    class ModeConfig(BaseModel):
        """Configuration for a specific trading mode."""
        max_notional_per_order: float = Field(
            ge=0,
            description="Maximum notional value per order"
        )
        max_daily_notional: float = Field(
            ge=0,
            description="Maximum daily notional exposure"
        )
        max_positions: int = Field(
            ge=1, le=100,
            description="Maximum concurrent positions"
        )
        risk_per_trade_pct: float = Field(
            ge=0, le=0.10,
            description="Risk per trade as percentage (0.02 = 2%)"
        )
        max_notional_pct: Optional[float] = Field(
            default=None, ge=0, le=1.0,
            description="Max notional as percentage of account"
        )
        max_daily_exposure_pct: Optional[float] = Field(
            default=None, ge=0, le=1.0,
            description="Max daily exposure as percentage"
        )
        description: Optional[str] = None

    class ModesConfig(BaseModel):
        """All trading mode configurations."""
        micro: ModeConfig = Field(
            default_factory=lambda: ModeConfig(
                max_notional_per_order=1000,
                max_daily_notional=3000,
                max_positions=3,
                risk_per_trade_pct=0.005,
            )
        )
        paper: ModeConfig = Field(
            default_factory=lambda: ModeConfig(
                max_notional_per_order=15000,
                max_daily_notional=45000,
                max_positions=5,
                risk_per_trade_pct=0.01,
            )
        )
        real: ModeConfig = Field(
            default_factory=lambda: ModeConfig(
                max_notional_per_order=11000,
                max_daily_notional=22000,
                max_positions=2,
                risk_per_trade_pct=0.02,
                max_notional_pct=0.10,
                max_daily_exposure_pct=0.20,
            )
        )

    class DataConfig(BaseModel):
        """Data provider configuration."""
        provider: Literal["polygon", "stooq", "yfinance"] = "polygon"
        cache_dir: str = "data/cache"
        universe_file: str = "data/universe/optionable_liquid_900.csv"

    class BacktestConfig(BaseModel):
        """Backtest configuration."""
        initial_capital: float = Field(default=100000, ge=1000)
        slippage_pct: float = Field(default=0.001, ge=0, le=0.10)

    class RiskConfig(BaseModel):
        """Risk management configuration."""
        max_position_size: float = Field(default=0.10, ge=0, le=1.0)
        max_daily_loss: float = Field(default=1000, ge=0)
        max_order_value: float = Field(default=75, ge=0)
        max_open_positions: int = Field(default=10, ge=1, le=100)
        macro_blackout_enabled: bool = True
        max_concurrent_trades: int = Field(default=2, ge=1)

    class ExecutionConfig(BaseModel):
        """Execution configuration."""
        broker: Literal["alpaca", "ibkr"] = "alpaca"
        order_type: str = "ioc_limit"
        limit_buffer: float = Field(default=0.001, ge=0, le=0.10)

    class VixConfig(BaseModel):
        """VIX-based trading pause configuration."""
        pause_enabled: bool = True
        pause_threshold: float = Field(default=30.0, ge=0)
        elevated_threshold: float = Field(default=25.0, ge=0)
        extreme_threshold: float = Field(default=40.0, ge=0)
        fallback_vix: float = Field(default=20.0, ge=0)

    class Settings(BaseModel):
        """Complete settings schema."""
        system: SystemConfig = Field(default_factory=SystemConfig)
        trading_mode: Literal["micro", "paper", "real"] = "paper"
        modes: ModesConfig = Field(default_factory=ModesConfig)
        data: DataConfig = Field(default_factory=DataConfig)
        backtest: BacktestConfig = Field(default_factory=BacktestConfig)
        risk: RiskConfig = Field(default_factory=RiskConfig)
        execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
        vix: VixConfig = Field(default_factory=VixConfig)

        class Config:
            extra = "allow"  # Allow extra fields not in schema

else:
    # Fallback: No pydantic - use dicts
    class Settings:
        """Fallback settings container when pydantic is not available."""
        def __init__(self, data: Dict[str, Any]):
            self._data = data
            # Create nested attribute access
            self.system = type("SystemConfig", (), data.get("system", {}))()
            self.modes = type("ModesConfig", (), data.get("modes", {}))()
            self.data = type("DataConfig", (), data.get("data", {}))()
            self.risk = type("RiskConfig", (), data.get("risk", {}))()

        def model_dump(self) -> Dict[str, Any]:
            return self._data


# ============================================================================
# Validation Functions
# ============================================================================

def _load_yaml_config() -> Dict[str, Any]:
    """Load raw YAML configuration."""
    config_path = os.getenv("KOBE_CONFIG_PATH")
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).parent / "base.yaml"

    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_validated_settings() -> Settings:
    """
    Load and validate settings from base.yaml.

    Returns:
        Validated Settings object

    Raises:
        ValidationError: If settings are invalid (in strict mode)
    """
    raw = _load_yaml_config()

    if PYDANTIC_V2 is not None:
        try:
            return Settings(**raw)
        except ValidationError as e:
            logger.error(f"Settings validation failed: {e}")
            raise
    else:
        # Fallback without pydantic
        return Settings(raw)


def validate_settings_for_mode(mode: str) -> tuple[bool, list[str]]:
    """
    Validate settings for a specific trading mode.

    Args:
        mode: Trading mode ("live", "paper", "micro")

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    try:
        settings = load_validated_settings()

        # Check mode consistency
        if hasattr(settings, 'system') and hasattr(settings.system, 'mode'):
            if settings.system.mode != mode:
                errors.append(
                    f"Settings mode ({settings.system.mode}) does not match requested mode ({mode})"
                )

        # Live mode specific checks
        if mode == "live":
            # Require certain settings to be present
            if hasattr(settings, 'modes') and hasattr(settings.modes, 'real'):
                real_mode = settings.modes.real
                if hasattr(real_mode, 'max_notional_pct'):
                    if real_mode.max_notional_pct is None or real_mode.max_notional_pct > 0.20:
                        errors.append(
                            f"max_notional_pct must be <= 0.20 for live mode (got {real_mode.max_notional_pct})"
                        )
                if hasattr(real_mode, 'max_positions'):
                    if real_mode.max_positions > 5:
                        errors.append(
                            f"max_positions must be <= 5 for live mode (got {real_mode.max_positions})"
                        )

            # Check environment variables
            if not os.getenv("ALPACA_API_KEY_ID"):
                errors.append("ALPACA_API_KEY_ID not set")
            if not os.getenv("ALPACA_API_SECRET_KEY"):
                errors.append("ALPACA_API_SECRET_KEY not set")

    except ValidationError as e:
        errors.extend([str(err) for err in e.errors()])
    except Exception as e:
        errors.append(f"Unexpected error: {e}")

    return len(errors) == 0, errors


def get_validated_mode_config(mode: str) -> Dict[str, Any]:
    """
    Get validated configuration for a specific mode.

    Args:
        mode: Trading mode

    Returns:
        Configuration dict for the mode
    """
    settings = load_validated_settings()

    mode_map = {
        "micro": settings.modes.micro if hasattr(settings.modes, 'micro') else None,
        "paper": settings.modes.paper if hasattr(settings.modes, 'paper') else None,
        "live": settings.modes.real if hasattr(settings.modes, 'real') else None,
        "real": settings.modes.real if hasattr(settings.modes, 'real') else None,
    }

    mode_config = mode_map.get(mode)
    if mode_config is None:
        return {}

    if PYDANTIC_V2 is not None and hasattr(mode_config, 'model_dump'):
        return mode_config.model_dump()
    elif hasattr(mode_config, '__dict__'):
        return {k: v for k, v in mode_config.__dict__.items() if not k.startswith('_')}
    return {}


# ============================================================================
# Strict Validation (for live mode)
# ============================================================================

class SettingsValidationError(Exception):
    """Raised when settings fail validation in strict mode."""
    pass


def require_valid_settings(mode: str) -> Settings:
    """
    Require valid settings or raise an error.

    Use this in live mode to fail-closed on invalid config.

    Args:
        mode: Trading mode

    Returns:
        Validated Settings

    Raises:
        SettingsValidationError: If settings are invalid
    """
    is_valid, errors = validate_settings_for_mode(mode)

    if not is_valid:
        error_msg = f"Settings validation failed for {mode} mode:\n" + "\n".join(f"  - {e}" for e in errors)
        if mode == "live":
            raise SettingsValidationError(error_msg)
        else:
            logger.warning(error_msg)

    return load_validated_settings()
