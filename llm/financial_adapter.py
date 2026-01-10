"""
Financial LLM Adapter - Domain-Specific Enhancement
====================================================

Adapts LLM outputs for the financial domain without full fine-tuning.

This module provides a FinGPT-style approach to improving LLM outputs
for trading analysis through:

1. Financial system prompts (domain knowledge injection)
2. Few-shot examples of high-quality financial reasoning
3. Output validation against market data
4. Feedback loop from trade outcomes (implicit RLHF)

This is more practical than full RLHF while still achieving significant
improvements in financial reasoning quality.

Usage:
    from llm.financial_adapter import FinancialLLMAdapter, get_financial_adapter

    adapter = get_financial_adapter()
    response = adapter.chat([
        LLMMessage(role="user", content="Should I buy AAPL given it's down 5% today?")
    ])

    # Record outcomes for learning
    adapter.record_outcome(query="...", response="...", outcome={"pnl": 150, "win": True})
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Record of a trade outcome for feedback learning."""
    query: str
    response: str
    outcome: Dict[str, Any]  # {pnl, r_multiple, win, etc.}
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FewShotExample:
    """A high-quality example for few-shot prompting."""
    input_query: str
    good_response: str
    bad_response: str
    category: str = "general"  # general, technical, fundamental, risk


class FinancialLLMAdapter:
    """
    Adapts LLM outputs for the financial domain.

    This is a lightweight alternative to full fine-tuning that still
    significantly improves financial reasoning through:

    1. Expert system prompts that inject domain knowledge
    2. Few-shot examples demonstrating ideal reasoning patterns
    3. Output validation to catch hallucinations
    4. Feedback collection for continuous improvement

    The adapter wraps any base LLM provider and enhances its outputs
    specifically for trading analysis tasks.
    """

    # Comprehensive financial system prompt
    FINANCIAL_SYSTEM_PROMPT = """You are a senior quantitative trading analyst with deep expertise in:

**Technical Analysis:**
- Price action patterns (support/resistance, breakouts, consolidations)
- Momentum indicators (RSI, MACD, Stochastic, ADX)
- Volatility metrics (ATR, Bollinger Bands, VIX, historical volatility)
- Volume analysis (OBV, VWAP, volume profiles, accumulation/distribution)

**Market Microstructure:**
- Order flow dynamics and liquidity
- Bid-ask spreads and market depth
- Dark pool activity and institutional positioning
- Market making and price discovery

**Risk Management:**
- Position sizing (Kelly criterion, fixed fractional, volatility-based)
- Drawdown limits and portfolio VaR
- Correlation analysis and diversification
- Stop-loss strategies (fixed, trailing, volatility-adjusted)

**Behavioral Finance:**
- Sentiment indicators (put/call ratio, VIX term structure, fear/greed)
- Institutional vs retail positioning
- News and social media sentiment analysis
- Market cycles and regime detection

**Statistical Methods:**
- Mean reversion and momentum strategies
- Factor models and alpha decomposition
- Statistical significance testing
- Walk-forward validation and overfitting detection

**Your Analysis Standards:**
1. ALWAYS cite specific data points (prices, percentages, dates, sample sizes)
2. ALWAYS quantify risk/reward with explicit entry, stop, and target prices
3. ALWAYS consider the current market regime (bull/bear/neutral/volatile)
4. ALWAYS provide confidence levels backed by evidence
5. NEVER recommend a trade without clear risk parameters
6. NEVER present opinions as facts - distinguish between data and interpretation

**Risk-First Mentality:**
- Capital preservation > profit maximization
- If uncertain, recommend STAND_DOWN
- Size positions based on risk, not conviction
- Consider worst-case scenarios explicitly"""

    # Few-shot examples for high-quality reasoning
    FEW_SHOT_EXAMPLES = [
        FewShotExample(
            input_query="Should I buy AAPL? It's down 5% today.",
            good_response="""**AAPL Analysis - Down 5% Day**

**Current State (as of market close):**
- Price: $175.50 (down 5.2% from $185.15 open)
- RSI(2): 8.3 (extreme oversold - below 10)
- IBS: 0.12 (closing near day's low)
- Volume: 2.3x average daily volume (145M vs 63M ADV)

**Historical Pattern Analysis:**
- 5%+ single-day drops in AAPL: 23 occurrences over past 2 years
- Next-day bounce rate: 74% (17/23 positive next-day returns)
- Average next-day return when bounced: +2.1%
- Median recovery time to prior high: 8 trading days

**Risk Assessment:**
- Nearest support: $172.00 (prior consolidation zone, June 2024)
- If support breaks: Next support $165 (200-day SMA)
- ATR(14): $3.80, suggesting normal daily range
- Current drop = 1.4 ATR (significant but not extreme)

**Trade Parameters (if entering):**
- Entry: $175.50 (current price)
- Stop: $171.80 (1% below support at $172)
- Target: $179.20 (+2.1% based on historical average bounce)
- R:R Ratio: 1.8:1 ($3.70 profit potential vs $3.70 risk)
- Position size: 1% of portfolio (reduced due to elevated volatility)

**Recommendation: CONDITIONAL BUY**
- Confidence: 65%
- Conditions:
  1. Wait for first 30 minutes to settle (avoid opening volatility)
  2. Verify no negative news catalyst (earnings miss? guidance cut?)
  3. Check VIX level - if >30, reduce size further
  4. Ensure no earnings in next 5 days

**What Could Go Wrong:**
1. This is part of a broader market selloff (check SPY)
2. News-driven decline with more news to come
3. Technical damage suggests longer recovery
4. Support at $172 fails, triggering larger decline""",
            bad_response="Yes, buy AAPL. It's oversold and will bounce. These big down days are always buying opportunities.",
            category="technical",
        ),
        FewShotExample(
            input_query="The HMM regime detector says BULL but my technical indicators say SELL. What should I do?",
            good_response="""**Signal Conflict Resolution Analysis**

**Understanding the Contradiction:**
- HMM Regime: BULL (likely based on 10-20 day momentum, volatility, and breadth)
- Technical Signals: SELL (likely RSI overbought, MACD divergence, or price rejection)

**Key Insight:** These signals operate on different timeframes:
- HMM Regime: Medium-term market state (days to weeks)
- Technical Indicators: Short-term entry timing (hours to days)

**This is NOT necessarily a contradiction.** You can be:
- In a BULL regime (medium-term uptrend)
- But at a short-term overbought point (need pullback)

**Resolution Framework:**

1. **If HMM = BULL and Technicals = SELL:**
   - Do NOT initiate new LONG positions at current levels
   - Wait for pullback to support / RSI normalization
   - The BULL regime suggests buying the dip, not chasing

2. **For existing positions:**
   - Don't panic sell in a BULL regime due to short-term overbought
   - Consider partial profit-taking if significantly overbought
   - Tighten stops but don't exit completely

3. **Historical Resolution:**
   - Check which signal source has been more accurate historically
   - In similar past conflicts, what was the outcome?

**Recommendation: STAND_DOWN on new longs**
- Wait for RSI to normalize (drop below 70)
- Wait for price to test support
- The regime gives permission to buy dips; technicals say "not yet"

**Action Items:**
1. Set alert for RSI < 70 or price at key support
2. If in position, trail stop to lock in gains
3. Re-evaluate in 2-3 days when signals align""",
            bad_response="Just follow the HMM - it uses machine learning so it's more accurate than simple indicators.",
            category="conflict_resolution",
        ),
    ]

    def __init__(
        self,
        base_provider=None,
        feedback_store_path: Optional[Path] = None,
        use_few_shot: bool = True,
        validate_outputs: bool = True,
    ):
        """
        Initialize Financial LLM Adapter.

        Args:
            base_provider: Base LLM provider to wrap (lazy-loaded if None)
            feedback_store_path: Path to store outcome feedback
            use_few_shot: Whether to include few-shot examples
            validate_outputs: Whether to validate numerical claims
        """
        self._provider = base_provider
        self._feedback_store_path = feedback_store_path or Path("state/llm_feedback.jsonl")
        self._use_few_shot = use_few_shot
        self._validate_outputs = validate_outputs
        self._feedback_store: List[TradeOutcome] = []

        # Load existing feedback
        self._load_feedback()

        logger.info("FinancialLLMAdapter initialized")

    @property
    def provider(self):
        """Lazy-load base provider."""
        if self._provider is None:
            from llm.router import get_provider
            self._provider = get_provider(task_type="reasoning")
        return self._provider

    def chat(
        self,
        messages: List,
        temperature: float = 0.7,
        **kwargs,
    ):
        """
        Enhanced chat with financial domain adaptation.

        Args:
            messages: List of LLMMessage objects
            temperature: Sampling temperature
            **kwargs: Additional provider arguments

        Returns:
            LLMResponse with enhanced financial reasoning
        """
        from llm.provider_base import LLMMessage

        # Build adapted messages
        adapted_messages = []

        # 1. Add financial system prompt
        adapted_messages.append(
            LLMMessage(role="system", content=self.FINANCIAL_SYSTEM_PROMPT)
        )

        # 2. Add few-shot examples if enabled and relevant
        if self._use_few_shot and self._is_trading_query(messages):
            few_shot_messages = self._get_relevant_few_shot(messages)
            adapted_messages.extend(few_shot_messages)

        # 3. Add user messages
        adapted_messages.extend(messages)

        # 4. Get response from base provider
        response = self.provider.chat(adapted_messages, temperature=temperature, **kwargs)

        # 5. Validate if enabled
        if self._validate_outputs:
            response = self._validate_and_correct(response)

        return response

    def _is_trading_query(self, messages: List) -> bool:
        """Check if the query is trading-related."""
        trading_keywords = [
            'buy', 'sell', 'trade', 'stock', 'position', 'signal',
            'bullish', 'bearish', 'long', 'short', 'entry', 'exit',
            'stop', 'target', 'rsi', 'macd', 'regime', 'vix'
        ]

        # Check the last user message
        for msg in reversed(messages):
            if hasattr(msg, 'role') and msg.role == "user":
                content = getattr(msg, 'content', '').lower()
                if any(kw in content for kw in trading_keywords):
                    return True
                break

        return False

    def _get_relevant_few_shot(self, messages: List) -> List:
        """Get relevant few-shot examples for the query."""
        from llm.provider_base import LLMMessage

        few_shot_messages = []

        # Get query content
        query_content = ""
        for msg in reversed(messages):
            if hasattr(msg, 'role') and msg.role == "user":
                query_content = getattr(msg, 'content', '').lower()
                break

        # Select relevant examples (up to 2)
        selected = []

        # Check for conflict resolution query
        if any(word in query_content for word in ['conflict', 'contradiction', 'disagree', 'hmm', 'regime']):
            conflict_examples = [e for e in self.FEW_SHOT_EXAMPLES if e.category == "conflict_resolution"]
            selected.extend(conflict_examples[:1])

        # Check for technical analysis query
        if any(word in query_content for word in ['buy', 'sell', 'down', 'up', 'price', 'rsi']):
            technical_examples = [e for e in self.FEW_SHOT_EXAMPLES if e.category == "technical"]
            selected.extend(technical_examples[:1])

        # Convert to messages
        for example in selected[:2]:  # Max 2 examples
            few_shot_messages.append(
                LLMMessage(role="user", content=f"Example query: {example.input_query}")
            )
            few_shot_messages.append(
                LLMMessage(role="assistant", content=example.good_response)
            )

        return few_shot_messages

    def _validate_and_correct(self, response) -> Any:
        """Validate numerical claims and correct if needed."""
        try:
            from cognitive.llm_validator import LLMValidator

            validator = LLMValidator()
            issues = validator.validate_response(response.content)

            # Log issues
            for issue in issues:
                logger.debug(f"Validation issue: {issue}")

            # For critical issues, we could regenerate
            # For now, just log them
            critical_issues = [i for i in issues if getattr(i, 'severity', '') == 'CRITICAL']
            if critical_issues:
                logger.warning(f"Response has {len(critical_issues)} critical validation issues")

        except ImportError:
            pass  # Validator not available

        return response

    def record_outcome(
        self,
        query: str,
        response: str,
        outcome: Dict[str, Any],
    ) -> None:
        """
        Record a trade outcome for implicit feedback learning.

        This enables a simple form of RLHF where we track which
        response patterns led to profitable outcomes.

        Args:
            query: The original query
            response: The LLM's response
            outcome: Trade outcome {pnl, r_multiple, win, etc.}
        """
        record = TradeOutcome(
            query=query,
            response=response,
            outcome=outcome,
        )

        self._feedback_store.append(record)
        self._save_feedback(record)

        # Periodically analyze patterns
        if len(self._feedback_store) >= 100 and len(self._feedback_store) % 50 == 0:
            self._analyze_feedback_patterns()

        logger.debug(f"Recorded trade outcome: win={outcome.get('win', 'N/A')}")

    def _load_feedback(self) -> None:
        """Load existing feedback from disk."""
        if self._feedback_store_path.exists():
            try:
                with open(self._feedback_store_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self._feedback_store.append(TradeOutcome(**data))
                logger.info(f"Loaded {len(self._feedback_store)} feedback records")
            except Exception as e:
                logger.warning(f"Failed to load feedback: {e}")

    def _save_feedback(self, record: TradeOutcome) -> None:
        """Save a feedback record to disk."""
        try:
            self._feedback_store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._feedback_store_path, 'a') as f:
                f.write(json.dumps({
                    'query': record.query,
                    'response': record.response,
                    'outcome': record.outcome,
                    'timestamp': record.timestamp,
                }) + '\n')
        except Exception as e:
            logger.warning(f"Failed to save feedback: {e}")

    def _analyze_feedback_patterns(self) -> None:
        """Analyze feedback to identify successful response patterns."""
        if not self._feedback_store:
            return

        # Separate wins and losses
        wins = [r for r in self._feedback_store if r.outcome.get('win', False)]
        losses = [r for r in self._feedback_store if not r.outcome.get('win', True)]

        if len(wins) < 10 or len(losses) < 10:
            return

        # Calculate win rate
        win_rate = len(wins) / len(self._feedback_store)
        logger.info(f"Feedback analysis: {len(wins)} wins, {len(losses)} losses ({win_rate:.0%} WR)")

        # Look for patterns in winning responses
        # (This is a simplified analysis - could be made more sophisticated)
        winning_patterns = self._extract_patterns(wins)
        losing_patterns = self._extract_patterns(losses)

        # Log findings
        if winning_patterns:
            logger.info(f"Common patterns in winning responses: {winning_patterns[:5]}")
        if losing_patterns:
            logger.info(f"Common patterns in losing responses: {losing_patterns[:5]}")

    def _extract_patterns(self, records: List[TradeOutcome]) -> List[str]:
        """Extract common patterns from response records."""
        from collections import Counter

        # Simple keyword counting
        keywords = []
        for record in records:
            response = record.response.lower()

            # Check for key phrases
            if 'r:r ratio' in response or 'risk/reward' in response:
                keywords.append('has_risk_reward')
            if 'stop' in response and '$' in response:
                keywords.append('has_explicit_stop')
            if 'confidence' in response:
                keywords.append('has_confidence')
            if 'stand_down' in response or 'stand down' in response:
                keywords.append('recommended_stand_down')
            if any(word in response for word in ['caution', 'careful', 'risk']):
                keywords.append('mentions_caution')

        counter = Counter(keywords)
        return [pattern for pattern, count in counter.most_common(10)]

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about recorded feedback."""
        if not self._feedback_store:
            return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}

        wins = sum(1 for r in self._feedback_store if r.outcome.get('win', False))
        losses = len(self._feedback_store) - wins

        return {
            'total': len(self._feedback_store),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(self._feedback_store) if self._feedback_store else 0.0,
            'avg_pnl': sum(r.outcome.get('pnl', 0) for r in self._feedback_store) / len(self._feedback_store),
        }


# Singleton instance
_financial_adapter_instance: Optional[FinancialLLMAdapter] = None


def get_financial_adapter(
    use_few_shot: bool = True,
    validate_outputs: bool = True,
) -> FinancialLLMAdapter:
    """
    Get the singleton Financial LLM Adapter instance.

    Args:
        use_few_shot: Whether to include few-shot examples
        validate_outputs: Whether to validate numerical claims

    Returns:
        FinancialLLMAdapter instance
    """
    global _financial_adapter_instance
    if _financial_adapter_instance is None:
        _financial_adapter_instance = FinancialLLMAdapter(
            use_few_shot=use_few_shot,
            validate_outputs=validate_outputs,
        )
    return _financial_adapter_instance


def financial_chat(
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Convenience function for financial-enhanced chat.

    Args:
        query: User's trading question
        context: Optional market context

    Returns:
        Enhanced response string
    """
    from llm.provider_base import LLMMessage

    adapter = get_financial_adapter()

    messages = []
    if context:
        context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
        messages.append(LLMMessage(role="user", content=f"Context:\n{context_str}"))

    messages.append(LLMMessage(role="user", content=query))

    response = adapter.chat(messages)
    return response.content
