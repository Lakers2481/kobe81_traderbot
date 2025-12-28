"""
YAML settings loader for Kobe81 Trading Bot.
Provides config-gated access to base.yaml settings.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_settings_cache: Optional[Dict[str, Any]] = None


def get_config_path() -> Path:
    """Return path to base.yaml config file."""
    # Check environment variable first, then default to project config
    env_path = os.getenv("KOBE_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    # Default: relative to this module
    return Path(__file__).parent / "base.yaml"


def load_settings(force_reload: bool = False) -> Dict[str, Any]:
    """Load and cache settings from base.yaml."""
    global _settings_cache
    if _settings_cache is not None and not force_reload:
        return _settings_cache

    config_path = get_config_path()
    if not config_path.exists():
        _settings_cache = {}
        return _settings_cache

    with open(config_path, "r", encoding="utf-8") as f:
        _settings_cache = yaml.safe_load(f) or {}
    return _settings_cache


def get_setting(path: str, default: Any = None) -> Any:
    """
    Get a nested setting by dot-notation path.
    Example: get_setting("execution.clamp.enabled", False)
    """
    settings = load_settings()
    keys = path.split(".")
    value = settings
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# Specific config accessors for clarity

def is_clamp_enabled() -> bool:
    """Check if LULD/volatility clamp is enabled."""
    return bool(get_setting("execution.clamp.enabled", False))


def get_clamp_max_pct() -> float:
    """Max percentage from quote for clamp."""
    return float(get_setting("execution.clamp.max_pct_from_quote", 0.02))


def get_clamp_use_atr() -> bool:
    """Whether to use ATR-based clamp instead of fixed percentage."""
    return bool(get_setting("execution.clamp.use_atr", False))


def get_clamp_atr_multiple() -> float:
    """ATR multiple for ATR-based clamp."""
    return float(get_setting("execution.clamp.atr_multiple", 2.0))


def is_rate_limiter_enabled() -> bool:
    """Check if order rate limiter is enabled."""
    return bool(get_setting("execution.rate_limiter.enabled", False))


def get_rate_limiter_config() -> Dict[str, Any]:
    """Get full rate limiter configuration."""
    return {
        "enabled": is_rate_limiter_enabled(),
        "max_orders_per_minute": int(get_setting("execution.rate_limiter.max_orders_per_minute", 120)),
        "retry_on_429": bool(get_setting("execution.rate_limiter.retry_on_429", True)),
        "max_retries": int(get_setting("execution.rate_limiter.max_retries", 3)),
        "base_delay_ms": int(get_setting("execution.rate_limiter.base_delay_ms", 500)),
    }


def is_earnings_filter_enabled() -> bool:
    """Check if earnings proximity filter is enabled."""
    return bool(get_setting("filters.earnings.enabled", False))


def get_earnings_filter_config() -> Dict[str, Any]:
    """Get full earnings filter configuration."""
    return {
        "enabled": is_earnings_filter_enabled(),
        "days_before": int(get_setting("filters.earnings.days_before", 2)),
        "days_after": int(get_setting("filters.earnings.days_after", 1)),
        "cache_file": str(get_setting("filters.earnings.cache_file", "state/earnings_cache.json")),
    }


def is_metrics_endpoint_enabled() -> bool:
    """Check if /metrics endpoint is enabled."""
    return bool(get_setting("health.metrics.enabled", True))


def get_metrics_config() -> Dict[str, Any]:
    """Get full metrics endpoint configuration."""
    return {
        "enabled": is_metrics_endpoint_enabled(),
        "include_performance": bool(get_setting("health.metrics.include_performance", True)),
        "include_requests": bool(get_setting("health.metrics.include_requests", True)),
    }


def get_commission_config() -> Dict[str, Any]:
    """Get backtest commission configuration."""
    return {
        "enabled": bool(get_setting("backtest.commissions.enabled", False)),
        "per_share": float(get_setting("backtest.commissions.per_share", 0.0)),
        "min_per_order": float(get_setting("backtest.commissions.min_per_order", 0.0)),
        "bps": float(get_setting("backtest.commissions.bps", 0.0)),
        "sec_fee_per_dollar": float(get_setting("backtest.commissions.sec_fee_per_dollar", 0.0000278)),
        "taf_fee_per_share": float(get_setting("backtest.commissions.taf_fee_per_share", 0.000166)),
    }


def get_regime_filter_config() -> Dict[str, Any]:
    """Get regime filter configuration."""
    return {
        "enabled": bool(get_setting("regime_filter.enabled", False)),
        "trend_fast": int(get_setting("regime_filter.trend.fast", 20)),
        "trend_slow": int(get_setting("regime_filter.trend.slow", 200)),
        "require_above_slow": bool(get_setting("regime_filter.trend.require_above_slow", True)),
        "vol_window": int(get_setting("regime_filter.vol.window", 20)),
        "max_ann_vol": get_setting("regime_filter.vol.max_ann_vol", None),
    }


def is_regime_filter_enabled() -> bool:
    """Check if regime filter is enabled."""
    return bool(get_setting("regime_filter.enabled", False))


def get_selection_config() -> Dict[str, Any]:
    """Get signal selection configuration."""
    return {
        "enabled": bool(get_setting("selection.enabled", False)),
        "top_n": int(get_setting("selection.top_n", 10)),
        "score_weights": {
            "rsi2": float(get_setting("selection.score_weights.rsi2", 0.6)),
            "ibs": float(get_setting("selection.score_weights.ibs", 0.4)),
        },
        "include_and_guard": bool(get_setting("selection.include_and_guard", True)),
        "min_price": float(get_setting("selection.min_price", 5.0)),
    }


def is_selection_enabled() -> bool:
    """Check if signal selection is enabled."""
    return bool(get_setting("selection.enabled", False))


def get_sizing_config() -> Dict[str, Any]:
    """Get volatility-targeted sizing configuration."""
    return {
        "enabled": bool(get_setting("sizing.enabled", False)),
        "risk_per_trade_pct": float(get_setting("sizing.risk_per_trade_pct", 0.005)),
        "atr_multiple_for_stop": float(get_setting("sizing.atr_multiple_for_stop", 2.0)),
    }


def is_sizing_enabled() -> bool:
    """Check if volatility sizing is enabled."""
    return bool(get_setting("sizing.enabled", False))


# ============================================================================
# Cognitive Architecture Settings
# ============================================================================

def is_cognitive_enabled() -> bool:
    """Check if cognitive layer is enabled."""
    return bool(get_setting("cognitive.enabled", True))


def get_cognitive_brain_config() -> Dict[str, Any]:
    """Get CognitiveBrain configuration."""
    return {
        "min_confidence_to_act": float(get_setting("cognitive.brain.min_confidence_to_act", 0.5)),
        "max_processing_time_ms": int(get_setting("cognitive.brain.max_processing_time_ms", 5000)),
    }


def get_cognitive_governor_config() -> Dict[str, Any]:
    """Get MetacognitiveGovernor configuration."""
    return {
        "fast_confidence_threshold": float(get_setting("cognitive.governor.fast_confidence_threshold", 0.75)),
        "slow_confidence_threshold": float(get_setting("cognitive.governor.slow_confidence_threshold", 0.50)),
        "stand_down_threshold": float(get_setting("cognitive.governor.stand_down_threshold", 0.30)),
        "default_fast_budget_ms": int(get_setting("cognitive.governor.default_fast_budget_ms", 100)),
        "default_slow_budget_ms": int(get_setting("cognitive.governor.default_slow_budget_ms", 5000)),
        "high_stakes_position_pct": float(get_setting("cognitive.governor.high_stakes_position_pct", 0.03)),
    }


def get_cognitive_reflection_config() -> Dict[str, Any]:
    """Get ReflectionEngine configuration."""
    return {
        "max_reflections": int(get_setting("cognitive.reflection.max_reflections", 100)),
        "lookback_hours": int(get_setting("cognitive.reflection.lookback_hours", 24)),
    }


def get_cognitive_self_model_config() -> Dict[str, Any]:
    """Get SelfModel configuration."""
    return {
        "state_dir": str(get_setting("cognitive.self_model.state_dir", "state/cognitive")),
        "auto_persist": bool(get_setting("cognitive.self_model.auto_persist", True)),
        "poor_capability_threshold": float(get_setting("cognitive.self_model.poor_capability_threshold", 0.35)),
        "excellent_capability_threshold": float(get_setting("cognitive.self_model.excellent_capability_threshold", 0.65)),
        "min_trades_for_rating": int(get_setting("cognitive.self_model.min_trades_for_rating", 10)),
        "max_calibration_error": float(get_setting("cognitive.self_model.max_calibration_error", 0.25)),
    }


def get_cognitive_knowledge_boundary_config() -> Dict[str, Any]:
    """Get KnowledgeBoundary configuration."""
    return {
        "uncertainty_threshold": float(get_setting("cognitive.knowledge_boundary.uncertainty_threshold", 0.3)),
        "stand_down_threshold": float(get_setting("cognitive.knowledge_boundary.stand_down_threshold", 0.6)),
        "min_samples_for_confidence": int(get_setting("cognitive.knowledge_boundary.min_samples_for_confidence", 5)),
    }


def get_cognitive_episodic_memory_config() -> Dict[str, Any]:
    """Get EpisodicMemory configuration."""
    return {
        "storage_dir": str(get_setting("cognitive.episodic_memory.storage_dir", "state/cognitive/episodes")),
        "max_active_episodes": int(get_setting("cognitive.episodic_memory.max_active_episodes", 100)),
        "max_completed_episodes": int(get_setting("cognitive.episodic_memory.max_completed_episodes", 1000)),
        "auto_persist": bool(get_setting("cognitive.episodic_memory.auto_persist", True)),
    }


def get_cognitive_semantic_memory_config() -> Dict[str, Any]:
    """Get SemanticMemory configuration."""
    return {
        "state_dir": str(get_setting("cognitive.semantic_memory.state_dir", "state/cognitive")),
        "min_rule_confidence": float(get_setting("cognitive.semantic_memory.min_rule_confidence", 0.4)),
        "max_rules": int(get_setting("cognitive.semantic_memory.max_rules", 500)),
    }


def get_cognitive_curiosity_config() -> Dict[str, Any]:
    """Get CuriosityEngine configuration."""
    return {
        "max_hypotheses": int(get_setting("cognitive.curiosity.max_hypotheses", 50)),
        "validation_threshold": float(get_setting("cognitive.curiosity.validation_threshold", 0.7)),
    }


def get_market_mood_config() -> Dict[str, Any]:
    """Get MarketMoodAnalyzer configuration."""
    return {
        "vix_weight": float(get_setting("cognitive.market_mood.vix_weight", 0.6)),
        "sentiment_weight": float(get_setting("cognitive.market_mood.sentiment_weight", 0.4)),
        "extreme_threshold": float(get_setting("cognitive.market_mood.extreme_threshold", 0.7)),
        "vix_fear_level": float(get_setting("cognitive.market_mood.vix_fear_level", 25.0)),
        "vix_greed_level": float(get_setting("cognitive.market_mood.vix_greed_level", 15.0)),
        "vix_extreme_fear_level": float(get_setting("cognitive.market_mood.vix_extreme_fear_level", 35.0)),
        "vix_extreme_greed_level": float(get_setting("cognitive.market_mood.vix_extreme_greed_level", 12.0)),
    }
