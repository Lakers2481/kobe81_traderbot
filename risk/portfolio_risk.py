"""
Portfolio-level risk gate for KOBE81.

Enforces position limits, sector concentration, and correlation constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import csv

import numpy as np
import pandas as pd


class PortfolioRiskStatus(Enum):
    """Result status for portfolio risk checks."""
    APPROVED = auto()
    REJECTED = auto()
    WARNING = auto()


@dataclass
class PortfolioRiskLimits:
    """Configuration for portfolio risk limits."""
    max_gross_exposure_pct: float = 100.0  # % of NAV
    max_single_name_pct: float = 10.0      # Max single position as % of NAV
    max_sector_pct: float = 30.0           # Max sector exposure as % of NAV
    max_correlated_basket_pct: float = 40.0  # Max correlated positions as % of NAV
    max_simultaneous_positions: int = 20   # Max number of open positions
    correlation_threshold: float = 0.70    # Threshold for "correlated"
    correlation_window_days: int = 20      # Rolling window for correlation


@dataclass
class PortfolioRiskCheck:
    """Result of a portfolio risk check."""
    status: PortfolioRiskStatus
    approved: bool
    symbol: str
    side: str
    requested_qty: int
    requested_notional: float
    rejection_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.name,
            "approved": self.approved,
            "symbol": self.symbol,
            "side": self.side,
            "requested_qty": self.requested_qty,
            "requested_notional": self.requested_notional,
            "rejection_reasons": self.rejection_reasons,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


@dataclass
class PortfolioPosition:
    """Represents an open position in the portfolio."""
    symbol: str
    qty: int
    avg_price: float
    current_price: float
    side: str  # "long" or "short"
    sector: str = "Unknown"

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.current_price

    @property
    def pnl(self) -> float:
        if self.side == "long":
            return self.qty * (self.current_price - self.avg_price)
        else:
            return self.qty * (self.avg_price - self.current_price)


@dataclass
class PortfolioState:
    """Current state of the portfolio."""
    nav: float  # Net Asset Value (total account value)
    cash: float
    positions: List[PortfolioPosition] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    @property
    def long_exposure(self) -> float:
        return sum(p.notional for p in self.positions if p.side == "long")

    @property
    def short_exposure(self) -> float:
        return sum(p.notional for p in self.positions if p.side == "short")

    @property
    def gross_exposure(self) -> float:
        return self.long_exposure + self.short_exposure

    @property
    def net_exposure(self) -> float:
        return self.long_exposure - self.short_exposure

    @property
    def position_count(self) -> int:
        return len(self.positions)


class SectorMapper:
    """Maps symbols to sectors."""

    def __init__(self, sector_map_path: Optional[Path] = None):
        self._sector_map: Dict[str, str] = {}
        if sector_map_path and sector_map_path.exists():
            self._load_sector_map(sector_map_path)

    def _load_sector_map(self, path: Path) -> None:
        """Load sector mappings from CSV."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    symbol = row.get("symbol", "").strip().upper()
                    sector = row.get("sector", "Unknown").strip()
                    if symbol:
                        self._sector_map[symbol] = sector
        except Exception:
            pass

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol."""
        return self._sector_map.get(symbol.upper(), "Unknown")

    def add_mapping(self, symbol: str, sector: str) -> None:
        """Add or update a sector mapping."""
        self._sector_map[symbol.upper()] = sector


class CorrelationAnalyzer:
    """Analyzes correlations between positions."""

    def __init__(self, window_days: int = 20, threshold: float = 0.70):
        self.window_days = window_days
        self.threshold = threshold
        self._price_cache: Dict[str, pd.Series] = {}

    def set_price_history(self, symbol: str, prices: pd.Series) -> None:
        """Cache price history for a symbol."""
        self._price_cache[symbol.upper()] = prices

    def compute_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Compute correlation between two symbols."""
        s1 = symbol1.upper()
        s2 = symbol2.upper()

        if s1 not in self._price_cache or s2 not in self._price_cache:
            return None

        p1 = self._price_cache[s1].tail(self.window_days)
        p2 = self._price_cache[s2].tail(self.window_days)

        if len(p1) < 5 or len(p2) < 5:
            return None

        # Align series
        combined = pd.DataFrame({"p1": p1, "p2": p2}).dropna()
        if len(combined) < 5:
            return None

        # Compute returns correlation
        r1 = combined["p1"].pct_change().dropna()
        r2 = combined["p2"].pct_change().dropna()

        if len(r1) < 3:
            return None

        return float(r1.corr(r2))

    def identify_correlation_clusters(
        self,
        symbols: List[str],
    ) -> List[List[str]]:
        """
        Identify clusters of highly correlated symbols.

        Returns list of clusters, where each cluster is a list of symbols
        that are all correlated above the threshold.
        """
        if len(symbols) < 2:
            return []

        # Build correlation matrix
        n = len(symbols)
        corr_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                corr = self.compute_correlation(symbols[i], symbols[j])
                if corr is not None:
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        # Simple clustering: group symbols that are all mutually correlated
        clusters: List[Set[str]] = []
        assigned = set()

        for i in range(n):
            if symbols[i] in assigned:
                continue

            cluster = {symbols[i]}
            for j in range(i + 1, n):
                if symbols[j] in assigned:
                    continue
                if corr_matrix[i, j] >= self.threshold:
                    # Check if correlated with all current cluster members
                    all_correlated = True
                    for k, s in enumerate(symbols):
                        if s in cluster and s != symbols[i]:
                            if corr_matrix[j, k] < self.threshold:
                                all_correlated = False
                                break
                    if all_correlated:
                        cluster.add(symbols[j])

            if len(cluster) > 1:
                clusters.append(cluster)
                assigned.update(cluster)

        return [list(c) for c in clusters]


class PortfolioRiskGate:
    """
    Portfolio-level risk gate.

    Enforces position limits, sector concentration, and correlation constraints
    before allowing new trades.
    """

    def __init__(
        self,
        limits: Optional[PortfolioRiskLimits] = None,
        sector_map_path: Optional[Path] = None,
        enabled: bool = True,
    ):
        self.limits = limits or PortfolioRiskLimits()
        self.enabled = enabled
        self.sector_mapper = SectorMapper(sector_map_path)
        self.correlation_analyzer = CorrelationAnalyzer(
            window_days=self.limits.correlation_window_days,
            threshold=self.limits.correlation_threshold,
        )

    def check(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        price: float,
        qty: int,
        state: PortfolioState,
    ) -> PortfolioRiskCheck:
        """
        Check if a proposed trade passes portfolio risk limits.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            price: Proposed price
            qty: Proposed quantity
            state: Current portfolio state

        Returns:
            PortfolioRiskCheck with approval status and details
        """
        if not self.enabled:
            return PortfolioRiskCheck(
                status=PortfolioRiskStatus.APPROVED,
                approved=True,
                symbol=symbol,
                side=side,
                requested_qty=qty,
                requested_notional=price * qty,
                metrics={"gate_enabled": False},
            )

        rejection_reasons = []
        warnings = []
        metrics = {}

        proposed_notional = price * qty
        nav = state.nav

        if nav <= 0:
            return PortfolioRiskCheck(
                status=PortfolioRiskStatus.REJECTED,
                approved=False,
                symbol=symbol,
                side=side,
                requested_qty=qty,
                requested_notional=proposed_notional,
                rejection_reasons=["NAV is zero or negative"],
            )

        # 1. Check gross exposure limit
        new_gross = state.gross_exposure + proposed_notional
        gross_pct = (new_gross / nav) * 100
        metrics["gross_exposure_pct"] = gross_pct
        metrics["max_gross_exposure_pct"] = self.limits.max_gross_exposure_pct

        if gross_pct > self.limits.max_gross_exposure_pct:
            rejection_reasons.append(
                f"Gross exposure {gross_pct:.1f}% exceeds limit {self.limits.max_gross_exposure_pct}%"
            )

        # 2. Check single-name concentration
        # Find existing position in this symbol
        existing_notional = 0.0
        for pos in state.positions:
            if pos.symbol.upper() == symbol.upper():
                existing_notional = pos.notional
                break

        new_single_notional = existing_notional + proposed_notional
        single_pct = (new_single_notional / nav) * 100
        metrics["single_name_pct"] = single_pct
        metrics["max_single_name_pct"] = self.limits.max_single_name_pct

        if single_pct > self.limits.max_single_name_pct:
            rejection_reasons.append(
                f"Single-name exposure {single_pct:.1f}% exceeds limit {self.limits.max_single_name_pct}%"
            )

        # 3. Check sector concentration
        sector = self.sector_mapper.get_sector(symbol)
        sector_exposure = self._compute_sector_exposure(state, sector)
        new_sector_exposure = sector_exposure + proposed_notional
        sector_pct = (new_sector_exposure / nav) * 100
        metrics["sector"] = sector
        metrics["sector_exposure_pct"] = sector_pct
        metrics["max_sector_pct"] = self.limits.max_sector_pct

        if sector_pct > self.limits.max_sector_pct:
            rejection_reasons.append(
                f"Sector '{sector}' exposure {sector_pct:.1f}% exceeds limit {self.limits.max_sector_pct}%"
            )

        # 4. Check position count
        # Only count as new position if not already holding
        is_new_position = not any(
            p.symbol.upper() == symbol.upper() for p in state.positions
        )
        new_position_count = state.position_count + (1 if is_new_position else 0)
        metrics["position_count"] = new_position_count
        metrics["max_positions"] = self.limits.max_simultaneous_positions

        if new_position_count > self.limits.max_simultaneous_positions:
            rejection_reasons.append(
                f"Position count {new_position_count} exceeds limit {self.limits.max_simultaneous_positions}"
            )

        # 5. Check correlation clusters
        current_symbols = [p.symbol.upper() for p in state.positions]
        all_symbols = current_symbols + [symbol.upper()]
        clusters = self.correlation_analyzer.identify_correlation_clusters(all_symbols)

        for cluster in clusters:
            if symbol.upper() in cluster:
                cluster_exposure = sum(
                    p.notional for p in state.positions
                    if p.symbol.upper() in cluster
                ) + proposed_notional
                cluster_pct = (cluster_exposure / nav) * 100
                metrics["correlated_cluster"] = cluster
                metrics["correlated_basket_pct"] = cluster_pct

                if cluster_pct > self.limits.max_correlated_basket_pct:
                    rejection_reasons.append(
                        f"Correlated basket {cluster} exposure {cluster_pct:.1f}% "
                        f"exceeds limit {self.limits.max_correlated_basket_pct}%"
                    )
                elif cluster_pct > self.limits.max_correlated_basket_pct * 0.8:
                    warnings.append(
                        f"Correlated basket approaching limit ({cluster_pct:.1f}%)"
                    )
                break

        # 6. Add warnings for approaching limits
        if gross_pct > self.limits.max_gross_exposure_pct * 0.9:
            warnings.append(f"Gross exposure approaching limit ({gross_pct:.1f}%)")

        if single_pct > self.limits.max_single_name_pct * 0.8:
            warnings.append(f"Single-name concentration high ({single_pct:.1f}%)")

        # Determine final status
        if rejection_reasons:
            status = PortfolioRiskStatus.REJECTED
            approved = False
        elif warnings:
            status = PortfolioRiskStatus.WARNING
            approved = True
        else:
            status = PortfolioRiskStatus.APPROVED
            approved = True

        return PortfolioRiskCheck(
            status=status,
            approved=approved,
            symbol=symbol,
            side=side,
            requested_qty=qty,
            requested_notional=proposed_notional,
            rejection_reasons=rejection_reasons,
            warnings=warnings,
            metrics=metrics,
        )

    def _compute_sector_exposure(self, state: PortfolioState, sector: str) -> float:
        """Compute total exposure to a sector."""
        total = 0.0
        for pos in state.positions:
            pos_sector = self.sector_mapper.get_sector(pos.symbol)
            if pos_sector == sector:
                total += pos.notional
        return total

    def compute_gross_exposure(self, state: PortfolioState) -> float:
        """Compute current gross exposure."""
        return state.gross_exposure

    def compute_sector_exposures(self, state: PortfolioState) -> Dict[str, float]:
        """Compute exposure by sector."""
        exposures: Dict[str, float] = {}
        for pos in state.positions:
            sector = self.sector_mapper.get_sector(pos.symbol)
            exposures[sector] = exposures.get(sector, 0.0) + pos.notional
        return exposures

    def set_price_history(self, symbol: str, prices: pd.Series) -> None:
        """Set price history for correlation analysis."""
        self.correlation_analyzer.set_price_history(symbol, prices)

    def identify_correlation_clusters(
        self,
        state: PortfolioState,
    ) -> List[List[str]]:
        """Identify correlation clusters in current portfolio."""
        symbols = [p.symbol.upper() for p in state.positions]
        return self.correlation_analyzer.identify_correlation_clusters(symbols)

    @classmethod
    def from_config(
        cls,
        config: Optional[Dict[str, Any]] = None,
        sector_map_path: Optional[Path] = None,
    ) -> PortfolioRiskGate:
        """Create gate from configuration dictionary."""
        if config is None:
            try:
                from config.settings_loader import load_settings
                settings = load_settings()
                config = settings.get("portfolio_risk", {})
            except Exception:
                config = {}

        enabled = config.get("enabled", True)
        limits_cfg = config.get("limits", {})

        limits = PortfolioRiskLimits(
            max_gross_exposure_pct=limits_cfg.get("max_gross_exposure_pct", 100.0),
            max_single_name_pct=limits_cfg.get("max_single_name_pct", 10.0),
            max_sector_pct=limits_cfg.get("max_sector_pct", 30.0),
            max_correlated_basket_pct=limits_cfg.get("max_correlated_basket_pct", 40.0),
            max_simultaneous_positions=limits_cfg.get("max_simultaneous_positions", 20),
            correlation_threshold=limits_cfg.get("correlation_threshold", 0.70),
            correlation_window_days=limits_cfg.get("correlation_window_days", 20),
        )

        if sector_map_path is None:
            sector_map_path = Path("data/sector_map.csv")

        return cls(
            limits=limits,
            sector_map_path=sector_map_path,
            enabled=enabled,
        )
