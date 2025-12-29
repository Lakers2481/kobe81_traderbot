"""
LLM-based pattern narrator using Claude Sonnet 4.

Generates human-readable explanations for discovered patterns.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Try to import Anthropic client
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class PatternNarrative:
    """Human-readable explanation of a trading pattern."""
    pattern_id: str
    title: str
    executive_summary: str
    market_conditions: str
    entry_setup: str
    edge_explanation: str
    risk_factors: List[str]
    position_sizing: str
    historical_evidence: str
    confidence_score: float
    generated_at: datetime = field(default_factory=datetime.utcnow)
    llm_model: str = "claude-sonnet-4-20250514"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pattern_id': self.pattern_id,
            'title': self.title,
            'executive_summary': self.executive_summary,
            'market_conditions': self.market_conditions,
            'entry_setup': self.entry_setup,
            'edge_explanation': self.edge_explanation,
            'risk_factors': self.risk_factors,
            'position_sizing': self.position_sizing,
            'historical_evidence': self.historical_evidence,
            'confidence_score': self.confidence_score,
            'generated_at': self.generated_at.isoformat(),
            'llm_model': self.llm_model,
        }


@dataclass
class PatternPlaybook:
    """Collection of validated pattern narratives."""
    playbook_id: str
    generated_at: datetime
    patterns: List[PatternNarrative]
    market_regime_summary: Dict[str, List[str]] = field(default_factory=dict)
    total_patterns: int = 0
    avg_win_rate: float = 0.0
    methodology_notes: str = ""

    def __post_init__(self):
        self.total_patterns = len(self.patterns)
        if self.patterns:
            self.avg_win_rate = sum(p.confidence_score for p in self.patterns) / len(self.patterns)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'playbook_id': self.playbook_id,
            'generated_at': self.generated_at.isoformat(),
            'patterns': [p.to_dict() for p in self.patterns],
            'market_regime_summary': self.market_regime_summary,
            'total_patterns': self.total_patterns,
            'avg_win_rate': self.avg_win_rate,
            'methodology_notes': self.methodology_notes,
        }


class PatternNarrator:
    """
    Generates LLM explanations for discovered patterns using Claude.

    Extends pattern data with human-readable narratives that explain
    WHY patterns work, not just THAT they work.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2000,
        fallback_enabled: bool = True,
    ):
        """
        Initialize narrator.

        Args:
            model: Claude model to use
            max_tokens: Maximum tokens in response
            fallback_enabled: Use deterministic fallback if API unavailable
        """
        self.model = model
        self.max_tokens = max_tokens
        self.fallback_enabled = fallback_enabled
        self._client = None

    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None and ANTHROPIC_AVAILABLE:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def narrate_pattern(
        self,
        cluster: Any,  # PatternCluster
        sample_trades: Optional[List[Dict]] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> PatternNarrative:
        """
        Generate narrative for a single pattern cluster.

        Args:
            cluster: PatternCluster object
            sample_trades: Sample trades from the cluster
            market_context: Optional market context data

        Returns:
            PatternNarrative with LLM-generated explanation
        """
        # Try LLM first
        client = self._get_client()
        if client:
            try:
                narrative = self._generate_llm_narrative(cluster, sample_trades, market_context)
                if narrative:
                    return narrative
            except Exception as e:
                logger.warning(f"LLM narrative failed: {e}")

        # Fallback to deterministic
        if self.fallback_enabled:
            return self._deterministic_narrative(cluster, sample_trades)

        raise RuntimeError("Unable to generate narrative")

    def _generate_llm_narrative(
        self,
        cluster: Any,
        sample_trades: Optional[List[Dict]],
        market_context: Optional[Dict[str, Any]],
    ) -> Optional[PatternNarrative]:
        """Generate narrative using Claude LLM."""
        client = self._get_client()
        if not client:
            return None

        # Build prompt
        prompt = self._build_prompt(cluster, sample_trades, market_context)

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            return self._parse_response(content, cluster)
        except Exception as e:
            logger.warning(f"Claude API error: {e}")
            return None

    def _build_prompt(
        self,
        cluster: Any,
        sample_trades: Optional[List[Dict]],
        market_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build prompt for Claude."""
        # Get top features
        top_features = sorted(
            cluster.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        features_str = "\n".join([f"- {k}: {v:.3f}" for k, v in top_features])

        # Format sample trades
        trades_str = "None available"
        if sample_trades:
            trades_str = "\n".join([
                f"- {t.get('symbol', 'N/A')}: PnL={t.get('pnl', 0):.2f}"
                for t in sample_trades[:3]
            ])

        prompt = f"""You are a senior quantitative trading analyst explaining a discovered pattern.

PATTERN STATISTICS:
- Cluster ID: {cluster.cluster_id}
- Sample Size: {cluster.n_samples} trades
- Win Rate: {cluster.win_rate:.1%}
- Profit Factor: {cluster.profit_factor:.2f}
- Avg Return: {cluster.avg_return:.2f}%

DEFINING FEATURES (most important):
{features_str}

SAMPLE TRADES:
{trades_str}

REGIME DISTRIBUTION:
{cluster.regime_distribution}

Generate a comprehensive but concise analysis:
1. TITLE: A memorable 3-5 word name for this pattern
2. EXECUTIVE SUMMARY: 2-3 sentences describing the pattern
3. MARKET CONDITIONS: When does this pattern occur?
4. ENTRY SETUP: What specific conditions trigger entry?
5. EDGE EXPLANATION: WHY does this pattern work?
6. RISK FACTORS: What can invalidate this pattern?
7. POSITION SIZING: How much to allocate?
8. CONFIDENCE SCORE: 0-100 based on statistical robustness

Format as JSON with keys: title, executive_summary, market_conditions, entry_setup, edge_explanation, risk_factors (list), position_sizing, confidence_score (int)."""

        return prompt

    def _parse_response(self, content: str, cluster: Any) -> PatternNarrative:
        """Parse LLM response into PatternNarrative."""
        import json

        try:
            # Try to extract JSON from response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                return PatternNarrative(
                    pattern_id=cluster.cluster_id,
                    title=data.get('title', f'Pattern {cluster.cluster_id}'),
                    executive_summary=data.get('executive_summary', ''),
                    market_conditions=data.get('market_conditions', ''),
                    entry_setup=data.get('entry_setup', ''),
                    edge_explanation=data.get('edge_explanation', ''),
                    risk_factors=data.get('risk_factors', []),
                    position_sizing=data.get('position_sizing', '2% risk per trade'),
                    historical_evidence=f"{cluster.n_samples} trades, {cluster.win_rate:.1%} WR, {cluster.profit_factor:.2f} PF",
                    confidence_score=float(data.get('confidence_score', cluster.confidence * 100)),
                    llm_model=self.model,
                )
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        # Fallback
        return self._deterministic_narrative(cluster, None)

    def _deterministic_narrative(
        self,
        cluster: Any,
        sample_trades: Optional[List[Dict]],
    ) -> PatternNarrative:
        """Generate deterministic narrative without LLM."""
        # Get top features for title
        top_features = sorted(
            cluster.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:2]
        feature_names = [f[0].replace('_', ' ').title() for f in top_features]
        title = f"{feature_names[0]} Pattern" if feature_names else f"Pattern {cluster.cluster_id}"

        return PatternNarrative(
            pattern_id=cluster.cluster_id,
            title=title,
            executive_summary=f"A pattern with {cluster.win_rate:.1%} win rate based on {cluster.n_samples} historical trades.",
            market_conditions=f"Regime distribution: {cluster.regime_distribution}",
            entry_setup=f"Key features: {', '.join([f[0] for f in top_features])}",
            edge_explanation="Pattern identified through ML clustering of similar trade setups.",
            risk_factors=["Market regime change", "Low sample size", "Feature drift"],
            position_sizing="2% risk per trade, sized by ATR",
            historical_evidence=f"{cluster.n_samples} trades, {cluster.win_rate:.1%} WR, {cluster.profit_factor:.2f} PF",
            confidence_score=cluster.confidence * 100,
            llm_model="deterministic_fallback",
        )

    def build_playbook(
        self,
        clusters: List[Any],
        regime_data: Optional[Dict[str, Any]] = None,
    ) -> PatternPlaybook:
        """
        Build complete playbook from all validated patterns.

        Args:
            clusters: List of PatternCluster objects
            regime_data: Optional market regime context

        Returns:
            PatternPlaybook with narratives for all patterns
        """
        narratives = []
        for cluster in clusters:
            try:
                narrative = self.narrate_pattern(
                    cluster,
                    sample_trades=cluster.sample_trades,
                    market_context=regime_data,
                )
                narratives.append(narrative)
            except Exception as e:
                logger.warning(f"Failed to narrate cluster {cluster.cluster_id}: {e}")

        # Organize by regime
        regime_summary = {}
        for n in narratives:
            # Extract regimes from narrative if available
            for regime in ['BULL', 'BEAR', 'NEUTRAL']:
                if regime.lower() in n.market_conditions.lower():
                    if regime not in regime_summary:
                        regime_summary[regime] = []
                    regime_summary[regime].append(n.pattern_id)

        playbook_id = f"playbook_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        return PatternPlaybook(
            playbook_id=playbook_id,
            generated_at=datetime.utcnow(),
            patterns=narratives,
            market_regime_summary=regime_summary,
            methodology_notes="Patterns discovered via ML clustering, validated with backtest data.",
        )
