"""
Alpha-Seeding Generative Agent - Reflexivity-Based Trading

IMPORTANT ETHICAL AND LEGAL NOTICE:
===================================
This module is for RESEARCH AND SIMULATION PURPOSES ONLY.

The concept of "seeding" market narratives could constitute market manipulation
if actually implemented. This code is designed to:
1. Study reflexivity in financial markets (Soros theory)
2. Simulate how information spreads through markets
3. Research narrative-driven price movements
4. NEVER actually publish content to influence real markets

Any production use would require extensive legal review and likely violates
SEC regulations on market manipulation. Use responsibly for research only.

Author: Kobe Trading System
Date: 2026-01-07
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import hashlib

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state" / "alpha_seeder"


class NarrativePhase(Enum):
    """Phases of a narrative's lifecycle."""
    DISCOVERED = "discovered"      # Pattern identified internally
    VALIDATED = "validated"        # Statistical validation complete
    CONTENT_READY = "content_ready"  # Research content generated
    SIMULATED = "simulated"        # Spread simulation run
    TRACKING = "tracking"          # Monitoring real-world adoption
    EXPIRED = "expired"            # Narrative has played out


@dataclass
class TradingPattern:
    """A discovered trading pattern."""
    pattern_id: str
    name: str
    description: str
    hypothesis: str
    symbols_affected: List[str]
    direction: str  # "LONG" or "SHORT"
    win_rate: float
    sample_size: int
    profit_factor: float
    p_value: float
    discovered_at: str
    validated: bool = False
    confidence: float = 0.0


@dataclass
class GeneratedContent:
    """Generated research content for a pattern."""
    content_id: str
    pattern_id: str
    title: str
    summary: str
    key_findings: List[str]
    supporting_data: Dict[str, Any]
    charts_needed: List[str]
    target_audience: str  # "retail", "institutional", "quant"
    generated_at: str
    word_count: int = 0


@dataclass
class NarrativeSimulation:
    """Simulation of how a narrative might spread."""
    simulation_id: str
    pattern_id: str
    initial_reach: int  # Estimated initial readers
    virality_factor: float  # 0-1 likelihood of sharing
    time_to_peak_days: int  # Days to maximum spread
    price_impact_estimate: float  # Expected price move %
    optimal_entry_day: int  # Best day to enter after publication
    optimal_exit_day: int  # Best day to exit
    simulated_at: str


@dataclass
class AlphaSeed:
    """Complete alpha seed with all components."""
    seed_id: str
    pattern: TradingPattern
    content: Optional[GeneratedContent]
    simulation: Optional[NarrativeSimulation]
    phase: NarrativePhase
    created_at: str
    updated_at: str


class PatternDiscovery:
    """
    Discovers novel trading patterns from data.

    Integrates with CuriosityEngine and experiment_analyzer to find
    patterns that are:
    1. Statistically significant
    2. Not widely known (no recent public discussion)
    3. Actionable (clear entry/exit criteria)
    """

    def __init__(self):
        self.discovered_patterns: Dict[str, TradingPattern] = {}
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load previously discovered patterns."""
        patterns_file = STATE_DIR / "patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    for p_data in data.get("patterns", {}).values():
                        self.discovered_patterns[p_data["pattern_id"]] = TradingPattern(**p_data)
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")

    def _save_patterns(self) -> None:
        """Save discovered patterns."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        patterns_file = STATE_DIR / "patterns.json"
        data = {
            "patterns": {k: v.__dict__ for k, v in self.discovered_patterns.items()},
            "updated_at": datetime.now().isoformat(),
        }
        with open(patterns_file, 'w') as f:
            json.dump(data, f, indent=2)

    def discover_from_experiments(self) -> List[TradingPattern]:
        """
        Discover patterns from experiment results.

        Integrates with research/experiment_analyzer.py.
        """
        try:
            from research.experiment_analyzer import ExperimentAnalyzer
            analyzer = ExperimentAnalyzer()
            results = analyzer.run_analysis()

            patterns = []
            for result in results:
                if result.is_significant and result.improvement_pct > 10:
                    pattern = TradingPattern(
                        pattern_id=f"EXP_{hashlib.md5(result.parameter_name.encode()).hexdigest()[:8]}",
                        name=f"{result.parameter_name} Effect",
                        description=f"Changing {result.parameter_name} from {result.control_value} to {result.experiment_value}",
                        hypothesis=f"Higher {result.parameter_name} leads to better performance",
                        symbols_affected=["BROAD_MARKET"],
                        direction="LONG",
                        win_rate=result.experiment_win_rate,
                        sample_size=result.experiment_trades,
                        profit_factor=result.experiment_avg_pnl / abs(result.control_avg_pnl) if result.control_avg_pnl != 0 else 1.5,
                        p_value=result.p_value,
                        discovered_at=datetime.now().isoformat(),
                        validated=True,
                        confidence=0.8,
                    )
                    patterns.append(pattern)
                    self.discovered_patterns[pattern.pattern_id] = pattern

            self._save_patterns()
            return patterns

        except ImportError:
            logger.warning("ExperimentAnalyzer not available")
            return []

    def discover_from_curiosity(self) -> List[TradingPattern]:
        """
        Discover patterns from CuriosityEngine hypotheses.

        Integrates with cognitive/curiosity_engine.py.
        """
        try:
            from cognitive.curiosity_engine import get_curiosity_engine
            engine = get_curiosity_engine()
            hypotheses = engine.get_validated_hypotheses()

            patterns = []
            for hyp in hypotheses:
                if hyp.get("validation_result", {}).get("significant", False):
                    pattern = TradingPattern(
                        pattern_id=f"CUR_{hyp.get('hypothesis_id', '')[:8]}",
                        name=hyp.get("name", "Unknown Pattern"),
                        description=hyp.get("description", ""),
                        hypothesis=hyp.get("hypothesis", ""),
                        symbols_affected=hyp.get("symbols", []),
                        direction=hyp.get("direction", "LONG"),
                        win_rate=hyp.get("win_rate", 0.5),
                        sample_size=hyp.get("sample_size", 0),
                        profit_factor=hyp.get("profit_factor", 1.0),
                        p_value=hyp.get("p_value", 1.0),
                        discovered_at=datetime.now().isoformat(),
                        validated=True,
                        confidence=hyp.get("confidence", 0.5),
                    )
                    patterns.append(pattern)
                    self.discovered_patterns[pattern.pattern_id] = pattern

            self._save_patterns()
            return patterns

        except ImportError:
            logger.warning("CuriosityEngine not available")
            return []

    def create_manual_pattern(
        self,
        name: str,
        description: str,
        hypothesis: str,
        symbols: List[str],
        direction: str,
        win_rate: float,
        sample_size: int,
        profit_factor: float,
        p_value: float,
    ) -> TradingPattern:
        """Create a pattern manually from research."""
        pattern = TradingPattern(
            pattern_id=f"MAN_{hashlib.md5(name.encode()).hexdigest()[:8]}",
            name=name,
            description=description,
            hypothesis=hypothesis,
            symbols_affected=symbols,
            direction=direction,
            win_rate=win_rate,
            sample_size=sample_size,
            profit_factor=profit_factor,
            p_value=p_value,
            discovered_at=datetime.now().isoformat(),
            validated=True,
            confidence=0.7,
        )

        self.discovered_patterns[pattern.pattern_id] = pattern
        self._save_patterns()
        return pattern


class ContentGenerator:
    """
    Generates research content for discovered patterns.

    SIMULATION ONLY: This generates content for research purposes.
    Content should NEVER be published to manipulate markets.
    """

    def generate_research_content(
        self,
        pattern: TradingPattern,
        style: str = "analytical"
    ) -> GeneratedContent:
        """
        Generate research content for a pattern.

        Args:
            pattern: The trading pattern to write about
            style: "analytical", "narrative", or "brief"

        Returns:
            GeneratedContent with article structure
        """
        # Generate title
        title = self._generate_title(pattern)

        # Generate summary
        summary = self._generate_summary(pattern)

        # Generate key findings
        key_findings = self._generate_findings(pattern)

        # Identify supporting data needed
        supporting_data = {
            "win_rate": pattern.win_rate,
            "sample_size": pattern.sample_size,
            "profit_factor": pattern.profit_factor,
            "p_value": pattern.p_value,
            "symbols": pattern.symbols_affected,
        }

        # Identify charts needed
        charts_needed = [
            f"equity_curve_{pattern.pattern_id}",
            f"win_rate_by_year_{pattern.pattern_id}",
            f"sample_distribution_{pattern.pattern_id}",
        ]

        content = GeneratedContent(
            content_id=f"CONT_{pattern.pattern_id}_{datetime.now().strftime('%Y%m%d')}",
            pattern_id=pattern.pattern_id,
            title=title,
            summary=summary,
            key_findings=key_findings,
            supporting_data=supporting_data,
            charts_needed=charts_needed,
            target_audience="quant" if pattern.p_value < 0.01 else "retail",
            generated_at=datetime.now().isoformat(),
            word_count=len(summary.split()) + sum(len(f.split()) for f in key_findings),
        )

        return content

    def _generate_title(self, pattern: TradingPattern) -> str:
        """Generate article title."""
        if pattern.win_rate > 0.7:
            return f"High-Probability Pattern: {pattern.name} ({pattern.win_rate:.0%} Win Rate)"
        elif pattern.profit_factor > 2:
            return f"Alpha Discovery: {pattern.name} (Profit Factor {pattern.profit_factor:.1f})"
        else:
            return f"Market Pattern Analysis: {pattern.name}"

    def _generate_summary(self, pattern: TradingPattern) -> str:
        """Generate article summary."""
        symbols_str = ", ".join(pattern.symbols_affected[:5])
        if len(pattern.symbols_affected) > 5:
            symbols_str += f" and {len(pattern.symbols_affected) - 5} more"

        return (
            f"Our quantitative research has identified a statistically significant "
            f"trading pattern with compelling performance characteristics. The {pattern.name} "
            f"pattern, tested on {pattern.sample_size} historical instances, demonstrates "
            f"a {pattern.win_rate:.1%} win rate with a profit factor of {pattern.profit_factor:.2f}. "
            f"This edge is particularly relevant for {symbols_str}. "
            f"The hypothesis underlying this pattern is: {pattern.hypothesis}"
        )

    def _generate_findings(self, pattern: TradingPattern) -> List[str]:
        """Generate key findings."""
        findings = []

        findings.append(
            f"Win rate of {pattern.win_rate:.1%} across {pattern.sample_size} trades, "
            f"statistically significant at p={pattern.p_value:.4f}"
        )

        findings.append(
            f"Profit factor of {pattern.profit_factor:.2f} indicates favorable "
            f"risk-reward characteristics"
        )

        if pattern.direction == "LONG":
            findings.append(
                "Pattern indicates bullish setups with defined entry and exit criteria"
            )
        else:
            findings.append(
                "Pattern indicates bearish setups suitable for short positions or hedging"
            )

        if len(pattern.symbols_affected) > 10:
            findings.append(
                f"Broad applicability across {len(pattern.symbols_affected)} symbols "
                f"suggests systematic rather than idiosyncratic edge"
            )

        return findings


class NarrativeSimulator:
    """
    Simulates how a narrative might spread through markets.

    SIMULATION ONLY: This models information diffusion without
    any actual content publication.
    """

    def simulate_spread(
        self,
        content: GeneratedContent,
        publication_channel: str = "blog"
    ) -> NarrativeSimulation:
        """
        Simulate how content might spread through markets.

        Args:
            content: The generated content
            publication_channel: "blog", "twitter", "substack", "academic"

        Returns:
            NarrativeSimulation with spread predictions
        """
        # Base parameters by channel
        channel_params = {
            "blog": {"reach": 500, "virality": 0.05, "time_to_peak": 14},
            "twitter": {"reach": 5000, "virality": 0.15, "time_to_peak": 3},
            "substack": {"reach": 2000, "virality": 0.08, "time_to_peak": 7},
            "academic": {"reach": 200, "virality": 0.02, "time_to_peak": 60},
        }

        params = channel_params.get(publication_channel, channel_params["blog"])

        # Adjust for content quality
        if content.word_count > 2000:
            params["virality"] *= 1.2  # More thorough = more shareable
        if "High-Probability" in content.title:
            params["virality"] *= 1.5  # Attention-grabbing

        # Calculate price impact estimate
        # Based on market microstructure research on information diffusion
        price_impact = min(0.05, params["virality"] * 0.1)  # Max 5% impact

        # Optimal timing
        optimal_entry = params["time_to_peak"] // 4  # Enter early
        optimal_exit = params["time_to_peak"] + params["time_to_peak"] // 2  # Exit after peak

        simulation = NarrativeSimulation(
            simulation_id=f"SIM_{content.content_id}",
            pattern_id=content.pattern_id,
            initial_reach=params["reach"],
            virality_factor=params["virality"],
            time_to_peak_days=params["time_to_peak"],
            price_impact_estimate=price_impact,
            optimal_entry_day=optimal_entry,
            optimal_exit_day=optimal_exit,
            simulated_at=datetime.now().isoformat(),
        )

        return simulation


class AlphaSeeder:
    """
    Main Alpha-Seeding Agent.

    RESEARCH AND SIMULATION ONLY.

    This class orchestrates:
    1. Pattern discovery
    2. Content generation
    3. Spread simulation
    4. Trading signal timing

    It does NOT actually publish content or manipulate markets.
    """

    def __init__(self):
        self.discovery = PatternDiscovery()
        self.content_gen = ContentGenerator()
        self.simulator = NarrativeSimulator()
        self.seeds: Dict[str, AlphaSeed] = {}
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_seeds()

    def _load_seeds(self) -> None:
        """Load existing seeds."""
        seeds_file = STATE_DIR / "seeds.json"
        if seeds_file.exists():
            try:
                with open(seeds_file, 'r') as f:
                    data = json.load(f)
                    for seed_data in data.get("seeds", {}).values():
                        # Reconstruct seed (simplified)
                        pattern = TradingPattern(**seed_data["pattern"])
                        seed = AlphaSeed(
                            seed_id=seed_data["seed_id"],
                            pattern=pattern,
                            content=GeneratedContent(**seed_data["content"]) if seed_data.get("content") else None,
                            simulation=NarrativeSimulation(**seed_data["simulation"]) if seed_data.get("simulation") else None,
                            phase=NarrativePhase(seed_data["phase"]),
                            created_at=seed_data["created_at"],
                            updated_at=seed_data["updated_at"],
                        )
                        self.seeds[seed.seed_id] = seed
            except Exception as e:
                logger.warning(f"Failed to load seeds: {e}")

    def _save_seeds(self) -> None:
        """Save seeds."""
        seeds_file = STATE_DIR / "seeds.json"
        data = {
            "seeds": {
                k: {
                    "seed_id": v.seed_id,
                    "pattern": v.pattern.__dict__,
                    "content": v.content.__dict__ if v.content else None,
                    "simulation": v.simulation.__dict__ if v.simulation else None,
                    "phase": v.phase.value,
                    "created_at": v.created_at,
                    "updated_at": v.updated_at,
                }
                for k, v in self.seeds.items()
            },
            "updated_at": datetime.now().isoformat(),
        }
        with open(seeds_file, 'w') as f:
            json.dump(data, f, indent=2)

    def discover_new_patterns(self) -> List[TradingPattern]:
        """Run pattern discovery from all sources."""
        patterns = []

        # From experiments
        exp_patterns = self.discovery.discover_from_experiments()
        patterns.extend(exp_patterns)

        # From curiosity engine
        cur_patterns = self.discovery.discover_from_curiosity()
        patterns.extend(cur_patterns)

        logger.info(f"Discovered {len(patterns)} new patterns")
        return patterns

    def create_seed(self, pattern: TradingPattern) -> AlphaSeed:
        """
        Create a complete alpha seed from a pattern.

        This generates content and runs simulation but does NOT
        publish anything.
        """
        # Generate content
        content = self.content_gen.generate_research_content(pattern)

        # Run simulation
        simulation = self.simulator.simulate_spread(content)

        # Create seed
        seed = AlphaSeed(
            seed_id=f"SEED_{pattern.pattern_id}",
            pattern=pattern,
            content=content,
            simulation=simulation,
            phase=NarrativePhase.SIMULATED,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        self.seeds[seed.seed_id] = seed
        self._save_seeds()

        return seed

    def get_trading_signal(self, seed_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trading signal timing from a seed's simulation.

        This provides timing guidance based on the simulated
        narrative spread, WITHOUT actually publishing anything.
        """
        seed = self.seeds.get(seed_id)
        if not seed or not seed.simulation:
            return None

        sim = seed.simulation

        # Calculate hypothetical timing
        # (In reality, this would track actual narrative spread)
        signal = {
            "seed_id": seed_id,
            "pattern_name": seed.pattern.name,
            "symbols": seed.pattern.symbols_affected,
            "direction": seed.pattern.direction,
            "optimal_entry_window": {
                "start_day": sim.optimal_entry_day,
                "end_day": sim.optimal_entry_day + 3,
            },
            "optimal_exit_window": {
                "start_day": sim.optimal_exit_day - 2,
                "end_day": sim.optimal_exit_day + 2,
            },
            "expected_price_impact": sim.price_impact_estimate,
            "confidence": seed.pattern.confidence * 0.8,  # Discount for simulation
            "warning": "SIMULATION ONLY - Do not use for actual trading without proper validation",
        }

        return signal

    def get_status(self) -> Dict[str, Any]:
        """Get seeder status."""
        return {
            "discovered_patterns": len(self.discovery.discovered_patterns),
            "active_seeds": len(self.seeds),
            "seeds_by_phase": {
                phase.value: len([s for s in self.seeds.values() if s.phase == phase])
                for phase in NarrativePhase
            },
        }


# Singleton instance
_seeder: Optional[AlphaSeeder] = None


def get_alpha_seeder() -> AlphaSeeder:
    """Get or create singleton seeder."""
    global _seeder
    if _seeder is None:
        _seeder = AlphaSeeder()
    return _seeder


# Example usage - RESEARCH ONLY
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("ALPHA SEEDER - RESEARCH AND SIMULATION MODE")
    print("WARNING: For research purposes only. Do not use to manipulate markets.")
    print("=" * 70)

    seeder = get_alpha_seeder()

    # Create a sample pattern manually
    pattern = seeder.discovery.create_manual_pattern(
        name="Tech R&D Treasury Effect",
        description="Tech stocks with high R&D spending outperform when 10-year Treasury yields drop",
        hypothesis="Lower discount rates benefit growth companies with high R&D investment",
        symbols=["NVDA", "AMD", "GOOGL", "META", "MSFT"],
        direction="LONG",
        win_rate=0.68,
        sample_size=47,
        profit_factor=1.85,
        p_value=0.023,
    )

    print(f"\nPattern created: {pattern.name}")
    print(f"  Win Rate: {pattern.win_rate:.1%}")
    print(f"  Sample Size: {pattern.sample_size}")
    print(f"  P-Value: {pattern.p_value}")

    # Create seed
    seed = seeder.create_seed(pattern)

    print(f"\nSeed created: {seed.seed_id}")
    print(f"  Content Title: {seed.content.title}")
    print(f"  Simulated Reach: {seed.simulation.initial_reach}")
    print(f"  Time to Peak: {seed.simulation.time_to_peak_days} days")
    print(f"  Expected Impact: {seed.simulation.price_impact_estimate:.1%}")

    # Get trading signal
    signal = seeder.get_trading_signal(seed.seed_id)
    if signal:
        print(f"\nSimulated Trading Signal:")
        print(f"  Entry Window: Days {signal['optimal_entry_window']['start_day']}-{signal['optimal_entry_window']['end_day']}")
        print(f"  Exit Window: Days {signal['optimal_exit_window']['start_day']}-{signal['optimal_exit_window']['end_day']}")
        print(f"  Expected Impact: {signal['expected_price_impact']:.1%}")
        print(f"\n  WARNING: {signal['warning']}")

    print(f"\nStatus: {seeder.get_status()}")
