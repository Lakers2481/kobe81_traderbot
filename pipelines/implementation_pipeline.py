"""
Implementation Pipeline - Generate strategy code from specs.

This pipeline takes validated StrategySpecs and generates
Python code that can be backtested and deployed.

Schedule: On-demand (triggered by spec validation)

Author: Kobe Trading System
Version: 1.0.0
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pipelines.base import Pipeline


class ImplementationPipeline(Pipeline):
    """Pipeline for generating strategy implementations."""

    @property
    def name(self) -> str:
        return "implementation"

    def execute(self) -> bool:
        """
        Execute code generation.

        Returns:
            True if implementations generated successfully
        """
        self.logger.info("Generating strategy implementations...")

        # Load validated specs
        specs = self._load_validated_specs()
        if not specs:
            self.logger.info("No validated specs to implement")
            self.set_metric("implementations_generated", 0)
            return True

        implementations = 0
        for spec in specs:
            if self._generate_implementation(spec):
                implementations += 1

        self.set_metric("specs_processed", len(specs))
        self.set_metric("implementations_generated", implementations)

        self.logger.info(f"Generated {implementations} implementations")
        return True

    def _load_validated_specs(self) -> List[Dict]:
        """Load specs that are ready for implementation."""
        specs_file = self.state_dir / "specs" / "specs.jsonl"
        if not specs_file.exists():
            return []

        specs = []
        with open(specs_file) as f:
            for line in f:
                spec = json.loads(line)
                if spec.get("status") in ["validated", "approved"]:
                    if spec.get("confidence", 0) >= 0.6:
                        specs.append(spec)

        return specs[:3]  # Process 3 at a time

    def _generate_implementation(self, spec: Dict) -> bool:
        """Generate Python code for a strategy spec."""
        try:
            spec_id = spec.get("spec_id", "unknown")
            name = spec.get("name", "UnnamedStrategy")

            # Sanitize name for Python class
            class_name = "".join(word.title() for word in name.split())
            class_name = "".join(c for c in class_name if c.isalnum())

            # Generate code
            code = self._generate_strategy_code(spec, class_name)

            # Save to strategies directory
            strategies_dir = self.project_root / "strategies" / "generated"
            strategies_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{spec_id.replace('-', '_')}.py"
            filepath = strategies_dir / filename
            filepath.write_text(code)

            self.add_artifact(str(filepath))
            self.logger.info(f"Generated: {filepath}")
            return True

        except Exception as e:
            self.add_warning(f"Failed to generate implementation: {e}")
            return False

    def _generate_strategy_code(self, spec: Dict, class_name: str) -> str:
        """Generate Python strategy code from spec."""
        entry_conditions = spec.get("entry_conditions", [])
        exit_conditions = spec.get("exit_conditions", {})
        filters = spec.get("filters", [])
        position_sizing = spec.get("position_sizing", {})

        code = f'''"""
Auto-generated strategy: {spec.get("name", "Unknown")}

Spec ID: {spec.get("spec_id", "unknown")}
Version: {spec.get("version", "1.0.0")}
Generated: {datetime.utcnow().isoformat()}

Description:
{spec.get("description", "No description")}

WARNING: This is auto-generated code. Review before use.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class {class_name}Params:
    """Parameters for {class_name}."""
    # Position sizing
    risk_pct: float = {position_sizing.get("risk_pct", 0.02)}
    max_pct: float = {position_sizing.get("max_pct", 0.10)}

    # Exit parameters
    stop_mult: float = {exit_conditions.get("stop_loss", {}).get("multiplier", 2.0)}
    target_mult: float = {exit_conditions.get("take_profit", {}).get("multiplier", 3.0)}
    max_bars: int = {exit_conditions.get("time_stop", {}).get("max_bars", 7)}


class {class_name}:
    """
    {spec.get("name", "Generated Strategy")}

    Entry Conditions:
    {self._format_conditions(entry_conditions)}

    Exit Conditions:
    - Stop Loss: {exit_conditions.get("stop_loss", {})}
    - Take Profit: {exit_conditions.get("take_profit", {})}
    - Time Stop: {exit_conditions.get("time_stop", {})}

    Filters:
    {self._format_conditions(filters)}
    """

    def __init__(self, params: Optional[{class_name}Params] = None):
        self.params = params or {class_name}Params()
        self._name = "{spec.get("name", "Unknown")}"

    @property
    def name(self) -> str:
        return self._name

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from OHLCV data.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            DataFrame with signal columns added
        """
        df = df.copy()

        # Add required indicators
        df = self._add_indicators(df)

        # Apply filters
        df = self._apply_filters(df)

        # Generate entry signals
        df = self._generate_entry_signals(df)

        # Calculate exits
        df = self._calculate_exits(df)

        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        # ATR for stops
        df["tr"] = pd.concat([
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        ], axis=1).max(axis=1)
        df["atr"] = df["tr"].rolling(14).mean()

        # SMA for trend filter
        df["sma_200"] = df["close"].rolling(200).mean()

        return df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-trade filters."""
        df["filter_pass"] = True

        # Trend filter
        if "sma_200" in df.columns:
            df["filter_pass"] = df["filter_pass"] & (df["close"] > df["sma_200"])

        return df

    def _generate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate entry signals based on conditions."""
        df["signal"] = 0

        # Entry logic (customize based on spec)
        # This is a template - actual logic depends on entry_conditions

        return df

    def _calculate_exits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate stop loss and take profit levels."""
        df["stop_loss"] = df["close"] - (df["atr"] * self.params.stop_mult)
        df["take_profit"] = df["close"] + (df["atr"] * self.params.target_mult)

        return df

    def scan_signals_over_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scan for signals over historical data (for backtesting).

        Uses shift(1) to prevent lookahead bias.
        """
        df = self.generate_signals(df)

        # Shift signals to prevent lookahead
        df["signal_safe"] = df["signal"].shift(1)

        return df


# Usage example
if __name__ == "__main__":
    # Load sample data
    # df = pd.read_csv("sample_data.csv")
    # strategy = {class_name}()
    # signals = strategy.generate_signals(df)
    pass
'''
        return code

    def _format_conditions(self, conditions: List[Dict]) -> str:
        """Format conditions list as string."""
        if not conditions:
            return "    - None"
        return "\n".join(f"    - {c}" for c in conditions)
