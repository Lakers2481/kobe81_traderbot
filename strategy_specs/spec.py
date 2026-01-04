"""
Strategy Spec - Declarative Strategy Definition
================================================

StrategySpec is a JSON-serializable definition of a trading strategy.
It decouples strategy logic from implementation code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IndicatorDef:
    """Definition of a technical indicator."""
    name: str                          # e.g., "rsi_2"
    indicator: str                     # e.g., "RSI"
    params: Dict[str, Any]             # e.g., {"period": 2}
    source: str = "close"              # price column to use


@dataclass
class ConditionDef:
    """Definition of an entry/exit condition."""
    left: str                          # e.g., "ibs"
    operator: str                      # e.g., "<"
    right: str                         # e.g., "0.08" or "sma_200"
    combine: str = "and"               # "and" or "or" with next condition


@dataclass
class EntryRules:
    """Entry rules for the strategy."""
    direction: str = "long"            # "long" or "short"
    conditions: List[ConditionDef] = field(default_factory=list)
    indicators: List[IndicatorDef] = field(default_factory=list)
    min_confidence: float = 0.5        # Minimum confidence score


@dataclass
class ExitRules:
    """Exit rules for the strategy."""
    stop_type: str = "atr"             # "atr", "fixed", "trailing"
    stop_multiplier: float = 2.0       # ATR multiplier or fixed %
    target_type: str = "atr"           # "atr", "fixed", "none"
    target_multiplier: float = 3.0     # ATR multiplier or fixed %
    time_stop_bars: int = 7            # Max bars to hold
    trailing_atr_mult: float = 0.0     # Trailing stop ATR mult (0 = disabled)


@dataclass
class SizingRules:
    """Position sizing rules."""
    method: str = "risk_pct"           # "risk_pct", "fixed", "kelly"
    risk_per_trade_pct: float = 2.0    # Risk per trade %
    max_position_pct: float = 10.0     # Max position size %
    max_daily_risk_pct: float = 6.0    # Max daily risk %


@dataclass
class RiskLimits:
    """Risk limits for the strategy."""
    max_drawdown_pct: float = 25.0     # Max drawdown before halt
    max_trades_per_day: int = 3        # Max trades per day
    max_positions: int = 5             # Max concurrent positions
    max_sector_exposure_pct: float = 30.0  # Max in single sector


@dataclass
class DataRequirements:
    """Data requirements for the strategy."""
    min_history_days: int = 200        # Minimum bars needed
    required_columns: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    universe: str = "optionable_liquid_900"  # Universe file


@dataclass
class StrategySpec:
    """
    Complete strategy specification.

    This is the DSL for defining trading strategies.
    It can be serialized to JSON and loaded back.
    """

    # Identification
    id: str
    name: str
    version: str
    family: str                        # Strategy family (e.g., "mean_reversion")
    description: str = ""
    author: str = ""

    # Rules
    entry: EntryRules = field(default_factory=EntryRules)
    exit: ExitRules = field(default_factory=ExitRules)
    sizing: SizingRules = field(default_factory=SizingRules)
    risk: RiskLimits = field(default_factory=RiskLimits)
    data: DataRequirements = field(default_factory=DataRequirements)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "draft"              # draft, testing, validated, production, archived
    parameters_count: int = 0          # For multiple testing gate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategySpec":
        """Create from dictionary."""
        # Convert nested dataclasses
        if "entry" in data and isinstance(data["entry"], dict):
            entry_data = data["entry"]
            if "conditions" in entry_data:
                entry_data["conditions"] = [
                    ConditionDef(**c) if isinstance(c, dict) else c
                    for c in entry_data["conditions"]
                ]
            if "indicators" in entry_data:
                entry_data["indicators"] = [
                    IndicatorDef(**i) if isinstance(i, dict) else i
                    for i in entry_data["indicators"]
                ]
            data["entry"] = EntryRules(**entry_data)

        if "exit" in data and isinstance(data["exit"], dict):
            data["exit"] = ExitRules(**data["exit"])

        if "sizing" in data and isinstance(data["sizing"], dict):
            data["sizing"] = SizingRules(**data["sizing"])

        if "risk" in data and isinstance(data["risk"], dict):
            data["risk"] = RiskLimits(**data["risk"])

        if "data" in data and isinstance(data["data"], dict):
            data["data"] = DataRequirements(**data["data"])

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "StrategySpec":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def count_parameters(self) -> int:
        """Count tunable parameters in the spec."""
        count = 0

        # Entry indicators
        for ind in self.entry.indicators:
            count += len(ind.params)

        # Entry conditions
        count += len(self.entry.conditions)

        # Exit params
        count += 4  # stop_mult, target_mult, time_stop, trailing

        # Sizing params
        count += 3  # risk_pct, max_pos, max_daily

        self.parameters_count = count
        return count


def load_spec(filepath: str) -> StrategySpec:
    """
    Load a StrategySpec from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        StrategySpec instance
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return StrategySpec.from_dict(data)


def save_spec(spec: StrategySpec, filepath: str) -> None:
    """
    Save a StrategySpec to JSON file.

    Args:
        spec: StrategySpec to save
        filepath: Output path
    """
    spec.updated_at = datetime.now().isoformat()
    with open(filepath, "w") as f:
        f.write(spec.to_json())


def validate_spec(spec: StrategySpec) -> List[str]:
    """
    Validate a StrategySpec.

    Args:
        spec: StrategySpec to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Required fields
    if not spec.id:
        errors.append("Missing id")
    if not spec.name:
        errors.append("Missing name")
    if not spec.version:
        errors.append("Missing version")
    if not spec.family:
        errors.append("Missing family")

    # Entry validation
    if not spec.entry.conditions:
        errors.append("No entry conditions defined")

    # Exit validation
    if spec.exit.stop_multiplier <= 0:
        errors.append("Stop multiplier must be positive")

    # Sizing validation
    if spec.sizing.risk_per_trade_pct <= 0 or spec.sizing.risk_per_trade_pct > 10:
        errors.append("Risk per trade must be 0-10%")

    # Risk validation
    if spec.risk.max_drawdown_pct <= 0 or spec.risk.max_drawdown_pct > 50:
        errors.append("Max drawdown must be 0-50%")

    return errors


# =============================================================================
# Example Spec Builders
# =============================================================================

def create_ibs_rsi_spec() -> StrategySpec:
    """Create the IBS+RSI mean reversion spec."""
    return StrategySpec(
        id="ibs_rsi_v2.6",
        name="IBS+RSI Mean Reversion",
        version="2.6",
        family="mean_reversion",
        description="Buy when IBS<0.08 and RSI(2)<5, above SMA(200)",
        author="Kobe Trading System",
        entry=EntryRules(
            direction="long",
            conditions=[
                ConditionDef(left="ibs", operator="<", right="0.08"),
                ConditionDef(left="rsi_2", operator="<", right="5"),
                ConditionDef(left="close", operator=">", right="sma_200"),
            ],
            indicators=[
                IndicatorDef(name="ibs", indicator="IBS", params={}),
                IndicatorDef(name="rsi_2", indicator="RSI", params={"period": 2}),
                IndicatorDef(name="sma_200", indicator="SMA", params={"period": 200}),
            ],
            min_confidence=0.6,
        ),
        exit=ExitRules(
            stop_type="atr",
            stop_multiplier=2.0,
            target_type="none",
            target_multiplier=0.0,
            time_stop_bars=7,
        ),
        sizing=SizingRules(
            method="risk_pct",
            risk_per_trade_pct=2.0,
            max_position_pct=10.0,
        ),
        status="production",
    )


def create_turtle_soup_spec() -> StrategySpec:
    """Create the Turtle Soup spec."""
    return StrategySpec(
        id="turtle_soup_v2.5",
        name="Turtle Soup ICT",
        version="2.5",
        family="ict_pattern",
        description="ICT Turtle Soup - liquidity sweep at 20-day low",
        author="Kobe Trading System",
        entry=EntryRules(
            direction="long",
            conditions=[
                ConditionDef(left="low", operator="<", right="low_20"),
                ConditionDef(left="close", operator=">", right="low_20"),
                ConditionDef(left="sweep_atr", operator=">=", right="0.3"),
            ],
            indicators=[
                IndicatorDef(name="low_20", indicator="LOWEST", params={"period": 20, "source": "low"}),
                IndicatorDef(name="atr_14", indicator="ATR", params={"period": 14}),
            ],
            min_confidence=0.6,
        ),
        exit=ExitRules(
            stop_type="atr",
            stop_multiplier=2.0,
            target_type="atr",
            target_multiplier=3.0,
            time_stop_bars=10,
        ),
        sizing=SizingRules(
            method="risk_pct",
            risk_per_trade_pct=2.0,
            max_position_pct=10.0,
        ),
        status="production",
    )
