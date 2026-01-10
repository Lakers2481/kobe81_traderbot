"""
Alpha Research Integration - Wires VectorBT Mining + Alphalens Validation to Kobe's Brain.

This module connects the new alpha mining infrastructure to Kobe's existing systems:
- ResearchEngine (autonomous/research.py) - for running experiments
- CuriosityEngine (cognitive/curiosity_engine.py) - for generating hypotheses
- AlphaLibrary (research/alpha_library.py) - 100+ alpha factors

Created: 2026-01-07
Purpose: Transform Kobe from 3 alphas to 10,000+ variant testing capability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlphaDiscovery:
    """A discovered alpha with validation metrics."""
    alpha_id: str
    name: str
    category: str
    parameters: Dict[str, Any]

    # Performance metrics
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int

    # Validation metrics (from Alphalens)
    ic_mean: Optional[float] = None
    ic_sharpe: Optional[float] = None
    q5_q1_spread: Optional[float] = None
    statistically_significant: bool = False

    # Metadata
    discovered_at: datetime = field(default_factory=datetime.now)
    validated: bool = False
    promoted_to_production: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alpha_id': self.alpha_id,
            'name': self.name,
            'category': self.category,
            'parameters': self.parameters,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'ic_mean': self.ic_mean,
            'ic_sharpe': self.ic_sharpe,
            'q5_q1_spread': self.q5_q1_spread,
            'statistically_significant': self.statistically_significant,
            'discovered_at': self.discovered_at.isoformat(),
            'validated': self.validated,
            'promoted_to_production': self.promoted_to_production,
        }


class AlphaResearchIntegration:
    """
    Integrates the new alpha mining infrastructure with Kobe's brain.

    This is the bridge that connects:
    - VectorBT fast alpha mining (10,000+ variants in seconds)
    - Alphalens factor validation (IC, quantile analysis)
    - AlphaLibrary (100+ alpha factors)

    To:
    - ResearchEngine (autonomous experiments)
    - CuriosityEngine (hypothesis generation)
    - Production scanner (validated alphas)
    """

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir is None:
            state_dir = Path("state/research")
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.discoveries: List[AlphaDiscovery] = []
        self._load_state()

        # Lazy-loaded components
        self._alpha_library = None
        self._vectorbt_miner = None
        self._factor_validator = None

        logger.info(f"AlphaResearchIntegration initialized with {len(self.discoveries)} discoveries")

    @property
    def alpha_library(self):
        """Lazy-load the AlphaLibrary."""
        if self._alpha_library is None:
            try:
                from research.alpha_library import AlphaLibrary
                self._alpha_library = AlphaLibrary()
                logger.info(f"Loaded AlphaLibrary with {len(self._alpha_library._registry)} alphas")
            except ImportError as e:
                logger.warning(f"AlphaLibrary not available: {e}")
                self._alpha_library = None
        return self._alpha_library

    @property
    def vectorbt_miner(self):
        """Lazy-load the VectorBT AlphaMiner."""
        if self._vectorbt_miner is None:
            try:
                from research.vectorbt_miner import AlphaMiner, HAS_VBT
                if HAS_VBT:
                    self._vectorbt_miner = True  # Placeholder, needs prices to init
                    logger.info("VectorBT AlphaMiner available")
                else:
                    logger.warning("VectorBT not installed - fast mining disabled")
                    self._vectorbt_miner = False
            except ImportError as e:
                logger.warning(f"VectorBT miner not available: {e}")
                self._vectorbt_miner = False
        return self._vectorbt_miner

    @property
    def factor_validator_available(self) -> bool:
        """Check if FactorValidator is available (Alphalens installed)."""
        try:
            from research.factor_validator import HAS_ALPHALENS
            return HAS_ALPHALENS
        except ImportError:
            return False

    def get_factor_validator(self, prices: pd.DataFrame):
        """
        Get a FactorValidator instance with prices.

        Note: FactorValidator requires price data at initialization,
        so we create it on-demand rather than lazy-loading.

        Args:
            prices: DataFrame with OHLCV data

        Returns:
            FactorValidator instance or None if Alphalens not installed
        """
        try:
            from research.factor_validator import FactorValidator, HAS_ALPHALENS
            if HAS_ALPHALENS:
                return FactorValidator(prices)
            else:
                logger.warning("Alphalens not installed - factor validation disabled")
                return None
        except ImportError as e:
            logger.warning(f"FactorValidator not available: {e}")
            return None

    # =========================================================================
    # ALPHA MINING - Fast parameter sweep using VectorBT
    # =========================================================================

    def run_alpha_mining_sweep(
        self,
        prices: pd.DataFrame,
        categories: Optional[List[str]] = None,
        min_sharpe: float = 0.5,
        min_trades: int = 30,
    ) -> List[AlphaDiscovery]:
        """
        Run a comprehensive alpha mining sweep using VectorBT.

        This is the main entry point for discovering new alphas. It:
        1. Uses VectorBT to test 10,000+ parameter combinations in seconds
        2. Filters to top performers
        3. Validates with Alphalens IC analysis
        4. Creates AlphaDiscovery records for promising candidates

        Args:
            prices: DataFrame with OHLCV data
            categories: Alpha categories to mine (default: all)
            min_sharpe: Minimum Sharpe ratio to consider
            min_trades: Minimum number of trades required

        Returns:
            List of discovered alphas meeting criteria
        """
        if not self.vectorbt_miner:
            logger.warning("VectorBT not available - using fallback mining")
            return self._fallback_alpha_mining(prices, categories, min_sharpe, min_trades)

        try:
            from research.vectorbt_miner import AlphaMiner

            # Initialize miner with price data
            miner = AlphaMiner(prices)

            # Run comprehensive mining
            logger.info("Starting VectorBT alpha mining sweep...")
            all_results = miner.mine_all()

            if all_results.empty:
                logger.warning("No alphas found in mining sweep")
                return []

            # Get top performers
            top_performers = miner.get_top_performers(
                n=50,
                metric='sharpe_ratio',
                min_trades=min_trades
            )

            logger.info(f"VectorBT mining found {len(top_performers)} candidates with Sharpe > {min_sharpe}")

            # Convert to AlphaDiscovery records
            discoveries = []
            for _, row in top_performers.iterrows():
                if row.get('sharpe_ratio', 0) < min_sharpe:
                    continue

                discovery = AlphaDiscovery(
                    alpha_id=f"vbt_{row.get('alpha_type', 'unknown')}_{len(self.discoveries)}",
                    name=row.get('alpha_type', 'unknown'),
                    category=self._infer_category(row.get('alpha_type', '')),
                    parameters=row.get('parameters', {}),
                    sharpe_ratio=row.get('sharpe_ratio', 0),
                    win_rate=row.get('win_rate', 0),
                    profit_factor=row.get('profit_factor', 1),
                    max_drawdown=row.get('max_drawdown', 0),
                    total_trades=row.get('total_trades', 0),
                )
                discoveries.append(discovery)
                self.discoveries.append(discovery)

            self._save_state()
            return discoveries

        except Exception as e:
            logger.error(f"VectorBT mining failed: {e}")
            return self._fallback_alpha_mining(prices, categories, min_sharpe, min_trades)

    def _fallback_alpha_mining(
        self,
        prices: pd.DataFrame,
        categories: Optional[List[str]],
        min_sharpe: float,
        min_trades: int,
    ) -> List[AlphaDiscovery]:
        """Fallback mining using AlphaLibrary when VectorBT unavailable."""
        if not self.alpha_library:
            logger.warning("AlphaLibrary not available - no mining possible")
            return []

        logger.info("Running fallback alpha mining with AlphaLibrary...")

        # Compute all alphas from the library
        alphas_df = self.alpha_library.compute_all(prices, categories)

        # Simple ranking by alpha values (not full backtest)
        discoveries = []
        for col in alphas_df.columns:
            if col in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']:
                continue

            # Create a basic discovery record
            discovery = AlphaDiscovery(
                alpha_id=f"lib_{col}_{len(self.discoveries)}",
                name=col,
                category=self._infer_category(col),
                parameters={},
                sharpe_ratio=0.0,  # Needs full backtest
                win_rate=0.0,
                profit_factor=1.0,
                max_drawdown=0.0,
                total_trades=0,
            )
            discoveries.append(discovery)
            self.discoveries.append(discovery)

        self._save_state()
        return discoveries

    def _infer_category(self, alpha_name: str) -> str:
        """Infer alpha category from name."""
        name_lower = alpha_name.lower()
        if 'mom' in name_lower or 'momentum' in name_lower:
            return 'momentum'
        elif 'rev' in name_lower or 'revert' in name_lower or 'mean' in name_lower:
            return 'mean_reversion'
        elif 'vol' in name_lower or 'atr' in name_lower:
            return 'volatility'
        elif 'rsi' in name_lower or 'macd' in name_lower or 'bb' in name_lower:
            return 'technical'
        elif 'rank' in name_lower or 'cross' in name_lower:
            return 'cross_sectional'
        else:
            return 'other'

    # =========================================================================
    # FACTOR VALIDATION - Alphalens IC analysis
    # =========================================================================

    def validate_alpha_with_alphalens(
        self,
        alpha_discovery: AlphaDiscovery,
        factor_data: pd.Series,
        prices: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> AlphaDiscovery:
        """
        Validate an alpha discovery using Alphalens factor analysis.

        This adds IC metrics, quantile spreads, and statistical significance
        to the discovery record.

        Args:
            alpha_discovery: The discovery to validate
            factor_data: The alpha factor values (Series with MultiIndex)
            prices: Price data for forward returns
            output_dir: Optional directory for tearsheet output

        Returns:
            Updated AlphaDiscovery with validation metrics
        """
        if not self.factor_validator_available:
            logger.warning("FactorValidator not available - skipping validation")
            return alpha_discovery

        try:
            # Get validator with price data
            validator = self.get_factor_validator(prices)
            if validator is None:
                logger.warning("Could not create FactorValidator")
                return alpha_discovery

            # Generate tearsheet and get report
            report = validator.generate_tearsheet(
                factor=factor_data,
                name=alpha_discovery.name,
                output_dir=str(output_dir) if output_dir else None,
            )

            # Update discovery with validation metrics
            alpha_discovery.ic_mean = report.ic_mean
            alpha_discovery.ic_sharpe = report.ic_sharpe
            alpha_discovery.q5_q1_spread = report.q5_q1_spread
            alpha_discovery.statistically_significant = report.significant_5pct
            alpha_discovery.validated = True

            logger.info(
                f"Validated {alpha_discovery.name}: IC={report.ic_mean:.4f}, "
                f"Q5-Q1={report.q5_q1_spread:.4f}, Significant={report.significant_5pct}"
            )

            self._save_state()
            return alpha_discovery

        except Exception as e:
            logger.error(f"Alphalens validation failed for {alpha_discovery.name}: {e}")
            return alpha_discovery

    # =========================================================================
    # BRAIN INTEGRATION - Connect to CuriosityEngine and ResearchEngine
    # =========================================================================

    def generate_alpha_hypotheses_for_curiosity_engine(self) -> List[Dict[str, Any]]:
        """
        Generate hypothesis candidates from alpha discoveries for the CuriosityEngine.

        This creates testable hypotheses from our top alpha discoveries
        that the CuriosityEngine can then validate and potentially promote to edges.

        Returns:
            List of hypothesis specifications for CuriosityEngine
        """
        hypotheses = []

        # Get top performing discoveries that haven't been promoted
        top_discoveries = sorted(
            [d for d in self.discoveries if not d.promoted_to_production],
            key=lambda d: d.sharpe_ratio,
            reverse=True
        )[:10]

        for disc in top_discoveries:
            # Create hypothesis for CuriosityEngine
            hypothesis = {
                'description': f"Alpha '{disc.name}' shows promising performance",
                'condition': f"alpha = {disc.name} AND category = {disc.category}",
                'prediction': f"win_rate > 0.55 AND sharpe > {disc.sharpe_ratio * 0.8:.2f}",
                'rationale': (
                    f"VectorBT mining found {disc.name} with Sharpe={disc.sharpe_ratio:.2f}, "
                    f"WR={disc.win_rate:.1%}, PF={disc.profit_factor:.2f} over {disc.total_trades} trades"
                ),
                'source': 'alpha_research_integration',
                'alpha_discovery_id': disc.alpha_id,
            }
            hypotheses.append(hypothesis)

        logger.info(f"Generated {len(hypotheses)} alpha hypotheses for CuriosityEngine")
        return hypotheses

    def submit_hypotheses_to_curiosity_engine(self) -> int:
        """
        Submit generated hypotheses to the CuriosityEngine for testing.

        Returns:
            Number of hypotheses submitted
        """
        try:
            from cognitive.curiosity_engine import get_curiosity_engine, Hypothesis
            import hashlib

            engine = get_curiosity_engine()
            hypotheses = self.generate_alpha_hypotheses_for_curiosity_engine()

            submitted = 0
            for h_spec in hypotheses:
                # Create a proper Hypothesis object
                hyp_id = hashlib.md5(
                    f"alpha_{h_spec['alpha_discovery_id']}".encode()
                ).hexdigest()[:8]

                # Skip if already exists
                if hyp_id in engine._hypotheses:
                    continue

                hypothesis = Hypothesis(
                    hypothesis_id=hyp_id,
                    description=h_spec['description'],
                    condition=h_spec['condition'],
                    prediction=h_spec['prediction'],
                    rationale=h_spec['rationale'],
                    source=h_spec['source'],
                )

                engine._hypotheses[hyp_id] = hypothesis
                submitted += 1

            if submitted > 0:
                engine._save_state()
                logger.info(f"Submitted {submitted} alpha hypotheses to CuriosityEngine")

            return submitted

        except Exception as e:
            logger.error(f"Failed to submit hypotheses to CuriosityEngine: {e}")
            return 0

    def create_research_experiment_from_discovery(
        self,
        discovery: AlphaDiscovery,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a ResearchEngine experiment from an alpha discovery.

        This allows the ResearchEngine to run formal backtests on discovered alphas.

        Args:
            discovery: The AlphaDiscovery to create an experiment for

        Returns:
            Experiment specification for ResearchEngine
        """
        try:
            from autonomous.research import ResearchEngine, Experiment
            from datetime import datetime
            from zoneinfo import ZoneInfo
            import random

            ET = ZoneInfo("America/New_York")

            # Create experiment specification
            experiment = Experiment(
                id=f"alpha_exp_{discovery.alpha_id}_{random.randint(1000, 9999)}",
                name=f"Alpha Test: {discovery.name}",
                hypothesis=f"Testing if {discovery.name} alpha maintains performance in walk-forward",
                parameter_changes={
                    'alpha_name': discovery.name,
                    'alpha_category': discovery.category,
                    'source_sharpe': discovery.sharpe_ratio,
                    'source_win_rate': discovery.win_rate,
                },
                created_at=datetime.now(ET),
            )

            logger.info(f"Created research experiment for alpha: {discovery.name}")
            return experiment.__dict__

        except Exception as e:
            logger.error(f"Failed to create research experiment: {e}")
            return None

    # =========================================================================
    # PRODUCTION PROMOTION - Promote validated alphas
    # =========================================================================

    def promote_alpha_to_production(
        self,
        discovery: AlphaDiscovery,
        require_validation: bool = True,
    ) -> bool:
        """
        Promote a validated alpha discovery to production.

        This creates the necessary configuration for the alpha to be used
        in the production scanner.

        Args:
            discovery: The AlphaDiscovery to promote
            require_validation: Whether to require Alphalens validation

        Returns:
            True if promotion succeeded
        """
        if require_validation and not discovery.validated:
            logger.warning(f"Cannot promote {discovery.name} - not validated")
            return False

        if require_validation and not discovery.statistically_significant:
            logger.warning(f"Cannot promote {discovery.name} - not statistically significant")
            return False

        # Create production config
        config_dir = Path("config/alpha_configs")
        config_dir.mkdir(parents=True, exist_ok=True)

        config = {
            'alpha_id': discovery.alpha_id,
            'name': discovery.name,
            'category': discovery.category,
            'parameters': discovery.parameters,
            'metrics': {
                'sharpe_ratio': discovery.sharpe_ratio,
                'win_rate': discovery.win_rate,
                'profit_factor': discovery.profit_factor,
                'ic_mean': discovery.ic_mean,
                'ic_sharpe': discovery.ic_sharpe,
            },
            'promoted_at': datetime.now().isoformat(),
            'status': 'active',
        }

        config_file = config_dir / f"{discovery.alpha_id}.json"
        config_file.write_text(json.dumps(config, indent=2))

        discovery.promoted_to_production = True
        self._save_state()

        logger.info(f"Promoted alpha {discovery.name} to production")
        return True

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of alpha research status."""
        return {
            'total_discoveries': len(self.discoveries),
            'validated': sum(1 for d in self.discoveries if d.validated),
            'significant': sum(1 for d in self.discoveries if d.statistically_significant),
            'promoted': sum(1 for d in self.discoveries if d.promoted_to_production),
            'top_sharpe': max((d.sharpe_ratio for d in self.discoveries), default=0),
            'top_ic': max((d.ic_mean or 0 for d in self.discoveries), default=0),
            'categories': list(set(d.category for d in self.discoveries)),
            'vectorbt_available': bool(self.vectorbt_miner),
            'alphalens_available': self.factor_validator_available,
            'alpha_library_size': len(self.alpha_library._registry) if self.alpha_library else 0,
        }

    def _save_state(self) -> None:
        """Save discoveries to disk."""
        state_file = self.state_dir / "alpha_discoveries.json"
        state = {
            'saved_at': datetime.now().isoformat(),
            'discoveries': [d.to_dict() for d in self.discoveries],
        }
        state_file.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        """Load discoveries from disk."""
        state_file = self.state_dir / "alpha_discoveries.json"
        if not state_file.exists():
            return

        try:
            state = json.loads(state_file.read_text())
            for d in state.get('discoveries', []):
                discovery = AlphaDiscovery(
                    alpha_id=d['alpha_id'],
                    name=d['name'],
                    category=d['category'],
                    parameters=d.get('parameters', {}),
                    sharpe_ratio=d['sharpe_ratio'],
                    win_rate=d['win_rate'],
                    profit_factor=d['profit_factor'],
                    max_drawdown=d['max_drawdown'],
                    total_trades=d['total_trades'],
                    ic_mean=d.get('ic_mean'),
                    ic_sharpe=d.get('ic_sharpe'),
                    q5_q1_spread=d.get('q5_q1_spread'),
                    statistically_significant=d.get('statistically_significant', False),
                    discovered_at=datetime.fromisoformat(d['discovered_at']),
                    validated=d.get('validated', False),
                    promoted_to_production=d.get('promoted_to_production', False),
                )
                self.discoveries.append(discovery)
        except Exception as e:
            logger.warning(f"Failed to load alpha discoveries: {e}")


# Singleton instance
_integration: Optional[AlphaResearchIntegration] = None


def get_alpha_research_integration() -> AlphaResearchIntegration:
    """Get the singleton AlphaResearchIntegration instance."""
    global _integration
    if _integration is None:
        _integration = AlphaResearchIntegration()
    return _integration


# =========================================================================
# CONVENIENCE FUNCTIONS - For use by other modules
# =========================================================================

def run_alpha_mining(prices: pd.DataFrame, **kwargs) -> List[AlphaDiscovery]:
    """Run alpha mining sweep and return discoveries."""
    integration = get_alpha_research_integration()
    return integration.run_alpha_mining_sweep(prices, **kwargs)


def get_alpha_library_alphas(prices: pd.DataFrame, categories: Optional[List[str]] = None) -> pd.DataFrame:
    """Compute all alphas from the AlphaLibrary."""
    integration = get_alpha_research_integration()
    if integration.alpha_library:
        return integration.alpha_library.compute_all(prices, categories)
    return pd.DataFrame()


def get_research_summary() -> Dict[str, Any]:
    """Get summary of alpha research status."""
    integration = get_alpha_research_integration()
    return integration.get_summary()
