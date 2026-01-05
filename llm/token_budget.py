"""
Token Budget Management
=======================

Tracks and enforces daily token/cost limits for LLM API calls.
Prevents runaway API costs with hard limits and alerts.

State persists across sessions via JSON file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Pricing (per 1K tokens)
# =============================================================================

LLM_PRICING: Dict[str, Dict[str, float]] = {
    # Claude 3 Haiku (cheapest, fastest)
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": {"input": 0.00025, "output": 0.00125},
    # Claude 3 Sonnet
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    # Claude Sonnet 4 (latest)
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    # Claude 3 Opus (most expensive)
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    # GPT-4 Turbo
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    # GPT-4o
    "gpt-4o": {"input": 0.005, "output": 0.015},
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Local models (free)
    "llama3.2": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "codellama": {"input": 0.0, "output": 0.0},
    "deepseek-r1": {"input": 0.0, "output": 0.0},
    "qwen": {"input": 0.0, "output": 0.0},
}

# Default pricing for unknown models
DEFAULT_PRICING = {"input": 0.003, "output": 0.015}


def calculate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Calculate USD cost for an LLM API call.

    Args:
        model: Model name (e.g., "claude-sonnet-4-20250514")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD (0.0 for local models)
    """
    # Check for local models (free)
    model_lower = model.lower()
    if any(local in model_lower for local in ["llama", "mistral", "codellama", "deepseek", "qwen", "phi"]):
        return 0.0

    pricing = LLM_PRICING.get(model, DEFAULT_PRICING)
    cost = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])
    return round(cost, 6)


@dataclass
class TokenBudget:
    """
    Track and enforce daily token budget for LLM API calls.

    Prevents runaway API costs by enforcing both:
    - Daily token limit (default: 100K tokens)
    - Daily USD limit (default: $50)

    State persists across sessions via JSON file.
    """

    daily_limit: int = 100_000
    max_daily_usd: float = 50.0
    alert_threshold_usd: float = 25.0
    state_file: str = "state/llm_token_usage.json"

    # Runtime state (loaded from file)
    used_today: int = 0
    cost_usd_today: float = 0.0
    input_tokens_today: int = 0
    output_tokens_today: int = 0
    calls_today: int = 0
    last_reset: str = ""
    _alert_sent: bool = False

    def __post_init__(self):
        self._load_state()
        self._check_reset()

    def _load_state(self) -> None:
        """Load state from file if exists."""
        path = Path(self.state_file)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self.used_today = data.get('used_today', 0)
                self.last_reset = data.get('last_reset', '')
                self.cost_usd_today = data.get('cost_usd_today', 0.0)
                self.input_tokens_today = data.get('input_tokens_today', 0)
                self.output_tokens_today = data.get('output_tokens_today', 0)
                self.calls_today = data.get('calls_today', 0)
            except Exception as e:
                logger.warning(f"Failed to load token budget state: {e}")

    def _save_state(self) -> None:
        """Save state to file."""
        path = Path(self.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, 'w') as f:
                json.dump({
                    'used_today': self.used_today,
                    'last_reset': self.last_reset,
                    'daily_limit': self.daily_limit,
                    'cost_usd_today': self.cost_usd_today,
                    'input_tokens_today': self.input_tokens_today,
                    'output_tokens_today': self.output_tokens_today,
                    'calls_today': self.calls_today,
                    'max_daily_usd': self.max_daily_usd,
                    'alert_threshold_usd': self.alert_threshold_usd,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save token budget state: {e}")

    def _check_reset(self) -> None:
        """Reset counter if new day."""
        today = date.today().isoformat()
        if self.last_reset != today:
            self.used_today = 0
            self.cost_usd_today = 0.0
            self.input_tokens_today = 0
            self.output_tokens_today = 0
            self.calls_today = 0
            self._alert_sent = False
            self.last_reset = today
            self._save_state()
            logger.info(f"Token budget reset for new day: {today}")

    def can_use(self, estimated_tokens: int = 2000) -> bool:
        """
        Check if we can use estimated tokens without exceeding budget.

        Args:
            estimated_tokens: Expected token usage

        Returns:
            True if within budget, False otherwise
        """
        self._check_reset()

        # Check token limit
        if self.used_today + estimated_tokens > self.daily_limit:
            logger.warning(
                f"Token limit would be exceeded: {self.used_today} + {estimated_tokens} > {self.daily_limit}"
            )
            return False

        # Check USD limit
        if self.cost_usd_today >= self.max_daily_usd:
            logger.warning(
                f"Daily USD limit reached: ${self.cost_usd_today:.2f} >= ${self.max_daily_usd:.2f}"
            )
            return False

        return True

    def record_usage(
        self,
        tokens: int,
        model: str = "claude-sonnet-4-20250514",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """
        Record token usage and persist.

        Args:
            tokens: Total tokens (for backward compatibility)
            model: Model name for pricing lookup
            input_tokens: Input tokens (for accurate cost)
            output_tokens: Output tokens (for accurate cost)
        """
        self._check_reset()
        self.used_today += tokens
        self.calls_today += 1

        # Calculate cost
        if input_tokens > 0 or output_tokens > 0:
            self.input_tokens_today += input_tokens
            self.output_tokens_today += output_tokens
            cost = calculate_cost_usd(model, input_tokens, output_tokens)
        else:
            # Estimate based on total tokens (assume 30% input, 70% output)
            est_input = int(tokens * 0.3)
            est_output = tokens - est_input
            cost = calculate_cost_usd(model, est_input, est_output)

        self.cost_usd_today += cost

        # Check for alert threshold
        if not self._alert_sent and self.cost_usd_today >= self.alert_threshold_usd:
            logger.warning(
                f"LLM cost alert: ${self.cost_usd_today:.2f} >= ${self.alert_threshold_usd:.2f} threshold"
            )
            self._alert_sent = True

        self._save_state()

    def get_remaining_budget(self) -> Dict[str, float]:
        """Get remaining budget."""
        self._check_reset()
        return {
            "tokens_remaining": self.daily_limit - self.used_today,
            "usd_remaining": self.max_daily_usd - self.cost_usd_today,
            "tokens_used": self.used_today,
            "usd_used": self.cost_usd_today,
            "calls_today": self.calls_today,
        }

    def get_status(self) -> Dict[str, any]:
        """Get full budget status."""
        self._check_reset()
        return {
            "date": self.last_reset,
            "tokens_used": self.used_today,
            "tokens_limit": self.daily_limit,
            "tokens_pct": round(100 * self.used_today / self.daily_limit, 1),
            "cost_usd": round(self.cost_usd_today, 4),
            "cost_limit_usd": self.max_daily_usd,
            "cost_pct": round(100 * self.cost_usd_today / self.max_daily_usd, 1),
            "calls_today": self.calls_today,
            "input_tokens": self.input_tokens_today,
            "output_tokens": self.output_tokens_today,
            "can_use": self.can_use(2000),
        }

    def get_remaining_percent(self) -> float:
        """
        Get percentage of budget remaining.

        FIX (2026-01-05): Added for preflight checks.

        Returns:
            Percentage remaining (0-100), based on the more limiting factor
            (either tokens or USD).
        """
        self._check_reset()

        tokens_remaining_pct = 100 * (self.daily_limit - self.used_today) / self.daily_limit
        usd_remaining_pct = 100 * (self.max_daily_usd - self.cost_usd_today) / self.max_daily_usd

        # Return the more limiting factor
        return min(tokens_remaining_pct, usd_remaining_pct)


# Global instance (lazy initialization)
_global_budget: TokenBudget | None = None


def get_token_budget() -> TokenBudget:
    """Get or create global token budget instance."""
    global _global_budget
    if _global_budget is None:
        _global_budget = TokenBudget()
    return _global_budget
