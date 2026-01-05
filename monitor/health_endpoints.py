from __future__ import annotations

from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import threading
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings_loader import get_metrics_config

logger = logging.getLogger(__name__)


# ============================================================================
# FIX (2026-01-05): Token Bucket Rate Limiter for /metrics endpoints
# Prevents flood attacks that could impact process performance
# ============================================================================

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for protecting endpoints.

    Uses sliding window to track requests per time period.
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self._lock = threading.Lock()

    def is_allowed(self) -> bool:
        """
        Check if a request is allowed.

        Returns:
            True if allowed, False if rate limited
        """
        with self._lock:
            now = time.time()

            # Remove old requests outside window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()

            # Check if under limit
            if len(self.requests) >= self.max_requests:
                return False

            # Record this request
            self.requests.append(now)
            return True

    def get_retry_after(self) -> int:
        """Get seconds until a request would be allowed."""
        with self._lock:
            if not self.requests:
                return 0

            oldest = self.requests[0]
            wait_time = self.window_seconds - (time.time() - oldest)
            return max(1, int(wait_time))


# Global rate limiter for metrics endpoints
# 60 requests per minute should be sufficient for monitoring
_metrics_rate_limiter: Optional[TokenBucketRateLimiter] = None
_rate_limiter_lock = threading.Lock()
_metrics_throttled_count = 0  # Track throttled requests


def get_metrics_rate_limiter() -> TokenBucketRateLimiter:
    """Get or create the metrics rate limiter."""
    global _metrics_rate_limiter
    if _metrics_rate_limiter is None:
        with _rate_limiter_lock:
            if _metrics_rate_limiter is None:
                _metrics_rate_limiter = TokenBucketRateLimiter(
                    max_requests=60,
                    window_seconds=60,
                )
    return _metrics_rate_limiter


def is_rate_limiting_enabled() -> bool:
    """Check if rate limiting should be enabled (live mode only)."""
    try:
        from config.settings_loader import get_setting
        return get_setting("system.mode", "paper") == "live"
    except ImportError:
        import os
        return os.getenv("KOBE_MODE", "paper").lower() == "live"


def increment_throttle_counter() -> None:
    """Increment the throttled requests counter."""
    global _metrics_throttled_count
    _metrics_throttled_count += 1
    # Also update Prometheus counter if available
    try:
        from trade_logging.prometheus_metrics import METRICS_THROTTLED
        METRICS_THROTTLED.inc()
    except ImportError:
        pass
    except Exception:
        pass  # Ignore errors in metrics


# Global metrics storage (thread-safe via GIL for simple operations)
_metrics: Dict[str, Any] = {
    "requests": {
        "total": 0,
        "orders_submitted": 0,
        "orders_rejected": 0,
        "orders_filled": 0,
        # IOC-specific counters for fill visibility
        "ioc_submitted": 0,
        "ioc_filled": 0,
        "ioc_cancelled": 0,
        "liquidity_blocked": 0,
    },
    "performance": {
        "win_rate": None,
        "profit_factor": None,
        "sharpe": None,
        "last_updated": None,
    },
    "system": {
        "start_time": time.time(),
        "uptime_seconds": 0,
    },
    "timestamps": {
        "last_submit_ts": None,
        "last_fill_ts": None,
        "last_trade_event_ts": None,
    },
    # Phase 3: AI Reliability Metrics (Calibration, Conformal, LLM)
    "calibration": {
        "brier_score": None,
        "expected_calibration_error": None,
        "max_calibration_error": None,
        "n_samples": 0,
        "last_updated": None,
    },
    "conformal": {
        "coverage_rate": None,
        "target_coverage": None,
        "avg_interval_width": None,
        "n_predictions": 0,
        "last_updated": None,
    },
    "llm": {
        "tokens_used_today": 0,
        "token_budget_remaining": None,
        "token_usage_pct": None,
        "llm_calls_saved": 0,
        "cache_hit_rate": None,
        "selective_mode_enabled": False,
        # Phase 4: USD Cost Tracking (Codex #2)
        "cost_usd_today": 0.0,
        "cost_budget_usd": None,
        "cost_remaining_usd": None,
        "cost_usage_pct": None,
        "input_tokens_today": 0,
        "output_tokens_today": 0,
        "calls_today": 0,
        "avg_latency_ms": None,
        "model": None,
        "last_updated": None,
    },
    "uncertainty": {
        "avg_uncertainty_score": None,
        "high_uncertainty_trades_blocked": 0,
        "uncertainty_threshold": 0.7,
    },
    # Phase 4: Execution Bandit Metrics
    "execution_bandit": {
        "enabled": False,
        "algorithm": None,
        "strategies": [],
        "total_selections": 0,
        "strategy_stats": {},  # {strategy: {selections, avg_reward, ucb_score}}
        "cumulative_regret": 0.0,
        "last_updated": None,
    },
    # Gemini: Strategy Foundry Metrics
    "strategy_foundry": {
        "enabled": False,
        "population_size": 0,
        "generations_run": 0,
        "best_fitness": None,
        "strategies_discovered": 0,
        "last_evolution": None,
    },
    # TCA (Transaction Cost Analysis) Metrics
    "tca": {
        "total_trades_7d": 0,
        "avg_slippage_bps": 0.0,
        "avg_spread_capture_bps": 0.0,
        "total_cost_usd_7d": 0.0,
        "last_updated": None,
    },
}


def update_request_counter(counter_name: str, increment: int = 1) -> None:
    """Increment a request counter."""
    if counter_name in _metrics["requests"]:
        _metrics["requests"][counter_name] += increment
    _metrics["requests"]["total"] += increment


def update_trade_event(kind: str) -> None:
    """
    Update metrics for a trade event.

    Args:
        kind: One of "ioc_submitted", "ioc_filled", "ioc_cancelled", "liquidity_blocked"
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _metrics["timestamps"]["last_trade_event_ts"] = now

    if kind in _metrics["requests"]:
        _metrics["requests"][kind] += 1
        _metrics["requests"]["total"] += 1

    # Update specific timestamps
    if kind == "ioc_submitted":
        _metrics["timestamps"]["last_submit_ts"] = now
        _metrics["requests"]["orders_submitted"] += 1
    elif kind == "ioc_filled":
        _metrics["timestamps"]["last_fill_ts"] = now
        _metrics["requests"]["orders_filled"] += 1
    elif kind == "ioc_cancelled":
        _metrics["requests"]["orders_rejected"] += 1


def update_performance_metrics(
    win_rate: Optional[float] = None,
    profit_factor: Optional[float] = None,
    sharpe: Optional[float] = None,
) -> None:
    """Update performance metrics from last backtest/run."""
    if win_rate is not None:
        _metrics["performance"]["win_rate"] = win_rate
    if profit_factor is not None:
        _metrics["performance"]["profit_factor"] = profit_factor
    if sharpe is not None:
        _metrics["performance"]["sharpe"] = sharpe
    _metrics["performance"]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_performance_from_summary(summary_path: str | Path) -> None:
    """Load performance metrics from a summary.json file."""
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        update_performance_metrics(
            win_rate=summary.get("win_rate"),
            profit_factor=summary.get("profit_factor"),
            sharpe=summary.get("sharpe"),
        )
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        pass


# =============================================================================
# Phase 3: AI Reliability Metrics Update Functions
# =============================================================================

def update_calibration_metrics(
    brier_score: Optional[float] = None,
    ece: Optional[float] = None,
    mce: Optional[float] = None,
    n_samples: Optional[int] = None,
) -> None:
    """
    Update calibration metrics from probability calibration analysis.

    Args:
        brier_score: Mean squared error of probabilities (lower is better)
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        n_samples: Number of samples used for calibration
    """
    if brier_score is not None:
        _metrics["calibration"]["brier_score"] = brier_score
    if ece is not None:
        _metrics["calibration"]["expected_calibration_error"] = ece
    if mce is not None:
        _metrics["calibration"]["max_calibration_error"] = mce
    if n_samples is not None:
        _metrics["calibration"]["n_samples"] = n_samples
    _metrics["calibration"]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def update_conformal_metrics(
    coverage_rate: Optional[float] = None,
    target_coverage: Optional[float] = None,
    avg_interval_width: Optional[float] = None,
    n_predictions: Optional[int] = None,
) -> None:
    """
    Update conformal prediction metrics.

    Args:
        coverage_rate: Actual coverage rate (% of actuals within prediction interval)
        target_coverage: Target coverage (e.g., 0.90 for 90%)
        avg_interval_width: Average prediction interval width
        n_predictions: Number of predictions made
    """
    if coverage_rate is not None:
        _metrics["conformal"]["coverage_rate"] = coverage_rate
    if target_coverage is not None:
        _metrics["conformal"]["target_coverage"] = target_coverage
    if avg_interval_width is not None:
        _metrics["conformal"]["avg_interval_width"] = avg_interval_width
    if n_predictions is not None:
        _metrics["conformal"]["n_predictions"] = n_predictions
    _metrics["conformal"]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def update_llm_metrics(
    tokens_used: Optional[int] = None,
    budget_remaining: Optional[int] = None,
    usage_pct: Optional[float] = None,
    calls_saved: Optional[int] = None,
    cache_hit_rate: Optional[float] = None,
    selective_mode: Optional[bool] = None,
    # Phase 4: USD cost tracking fields
    cost_usd_today: Optional[float] = None,
    cost_budget_usd: Optional[float] = None,
    cost_remaining_usd: Optional[float] = None,
    cost_usage_pct: Optional[float] = None,
    input_tokens_today: Optional[int] = None,
    output_tokens_today: Optional[int] = None,
    calls_today: Optional[int] = None,
    avg_latency_ms: Optional[float] = None,
    model: Optional[str] = None,
) -> None:
    """
    Update LLM usage metrics including USD cost tracking.

    Args:
        tokens_used: Tokens used today
        budget_remaining: Remaining token budget
        usage_pct: Token budget usage percentage
        calls_saved: Number of LLM calls saved by selective mode
        cache_hit_rate: Cache hit rate percentage
        selective_mode: Whether selective mode is enabled
        cost_usd_today: USD spent today
        cost_budget_usd: Daily USD budget limit
        cost_remaining_usd: Remaining USD budget
        cost_usage_pct: USD budget usage percentage
        input_tokens_today: Input tokens used today
        output_tokens_today: Output tokens used today
        calls_today: Number of LLM calls today
        avg_latency_ms: Average response latency in ms
        model: Current model being used
    """
    if tokens_used is not None:
        _metrics["llm"]["tokens_used_today"] = tokens_used
    if budget_remaining is not None:
        _metrics["llm"]["token_budget_remaining"] = budget_remaining
    if usage_pct is not None:
        _metrics["llm"]["token_usage_pct"] = usage_pct
    if calls_saved is not None:
        _metrics["llm"]["llm_calls_saved"] = calls_saved
    if cache_hit_rate is not None:
        _metrics["llm"]["cache_hit_rate"] = cache_hit_rate
    if selective_mode is not None:
        _metrics["llm"]["selective_mode_enabled"] = selective_mode
    # Phase 4: USD cost tracking
    if cost_usd_today is not None:
        _metrics["llm"]["cost_usd_today"] = cost_usd_today
    if cost_budget_usd is not None:
        _metrics["llm"]["cost_budget_usd"] = cost_budget_usd
    if cost_remaining_usd is not None:
        _metrics["llm"]["cost_remaining_usd"] = cost_remaining_usd
    if cost_usage_pct is not None:
        _metrics["llm"]["cost_usage_pct"] = cost_usage_pct
    if input_tokens_today is not None:
        _metrics["llm"]["input_tokens_today"] = input_tokens_today
    if output_tokens_today is not None:
        _metrics["llm"]["output_tokens_today"] = output_tokens_today
    if calls_today is not None:
        _metrics["llm"]["calls_today"] = calls_today
    if avg_latency_ms is not None:
        _metrics["llm"]["avg_latency_ms"] = avg_latency_ms
    if model is not None:
        _metrics["llm"]["model"] = model
    _metrics["llm"]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def update_uncertainty_metrics(
    avg_uncertainty: Optional[float] = None,
    blocked_count: Optional[int] = None,
    threshold: Optional[float] = None,
) -> None:
    """
    Update uncertainty-based filtering metrics.

    Args:
        avg_uncertainty: Average uncertainty score across predictions
        blocked_count: Number of trades blocked due to high uncertainty
        threshold: Uncertainty threshold for blocking trades
    """
    if avg_uncertainty is not None:
        _metrics["uncertainty"]["avg_uncertainty_score"] = avg_uncertainty
    if blocked_count is not None:
        _metrics["uncertainty"]["high_uncertainty_trades_blocked"] = blocked_count
    if threshold is not None:
        _metrics["uncertainty"]["uncertainty_threshold"] = threshold


def increment_uncertainty_blocked() -> None:
    """Increment the high uncertainty trades blocked counter."""
    _metrics["uncertainty"]["high_uncertainty_trades_blocked"] += 1


# =============================================================================
# Phase 4: Execution Bandit Metrics Update Functions
# =============================================================================

def update_execution_bandit_metrics(
    enabled: Optional[bool] = None,
    algorithm: Optional[str] = None,
    strategies: Optional[list] = None,
    total_selections: Optional[int] = None,
    strategy_stats: Optional[Dict[str, Any]] = None,
    cumulative_regret: Optional[float] = None,
) -> None:
    """
    Update execution bandit metrics from the ExecutionBandit.

    Args:
        enabled: Whether bandit is enabled
        algorithm: Current algorithm (thompson, ucb, epsilon_greedy)
        strategies: List of available strategies
        total_selections: Total number of strategy selections
        strategy_stats: Per-strategy stats {strategy: {selections, avg_reward}}
        cumulative_regret: Cumulative regret estimate
    """
    if enabled is not None:
        _metrics["execution_bandit"]["enabled"] = enabled
    if algorithm is not None:
        _metrics["execution_bandit"]["algorithm"] = algorithm
    if strategies is not None:
        _metrics["execution_bandit"]["strategies"] = strategies
    if total_selections is not None:
        _metrics["execution_bandit"]["total_selections"] = total_selections
    if strategy_stats is not None:
        _metrics["execution_bandit"]["strategy_stats"] = strategy_stats
    if cumulative_regret is not None:
        _metrics["execution_bandit"]["cumulative_regret"] = cumulative_regret
    _metrics["execution_bandit"]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sync_bandit_metrics_from_instance() -> None:
    """
    Sync execution bandit metrics from the global ExecutionBandit instance.
    Call this periodically or after trades.
    """
    try:
        from execution.execution_bandit import get_execution_bandit
        bandit = get_execution_bandit()
        stats = bandit.get_stats()
        update_execution_bandit_metrics(
            enabled=True,
            algorithm=bandit.algorithm,
            strategies=list(bandit.strategies),
            total_selections=stats.total_pulls,
            strategy_stats=stats.arm_stats,
            cumulative_regret=stats.regret_estimate,
        )
    except Exception:
        pass  # Bandit not available or not initialized


def sync_llm_metrics_from_analyzer() -> None:
    """
    Sync LLM usage metrics from the LLMTradeAnalyzer/TokenBudget.
    Call this periodically or after LLM calls.
    """
    try:
        from cognitive.llm_trade_analyzer import get_llm_cost_stats
        stats = get_llm_cost_stats()
        update_llm_metrics(
            tokens_used=stats.get("tokens_used_today"),
            budget_remaining=stats.get("token_budget_remaining"),
            usage_pct=stats.get("token_usage_pct"),
            calls_saved=stats.get("calls_saved"),
            cost_usd_today=stats.get("cost_usd_today"),
            cost_budget_usd=stats.get("cost_budget_usd"),
            cost_remaining_usd=stats.get("cost_remaining_usd"),
            cost_usage_pct=stats.get("cost_usage_pct"),
            input_tokens_today=stats.get("input_tokens_today"),
            output_tokens_today=stats.get("output_tokens_today"),
            calls_today=stats.get("calls_today"),
            model=stats.get("model"),
        )
    except ImportError:
        pass  # LLM analyzer not available
    except Exception:
        pass  # LLM analyzer not initialized


# =============================================================================
# Gemini: Strategy Foundry Metrics Update Functions
# =============================================================================

def update_strategy_foundry_metrics(
    enabled: Optional[bool] = None,
    population_size: Optional[int] = None,
    generations_run: Optional[int] = None,
    best_fitness: Optional[float] = None,
    strategies_discovered: Optional[int] = None,
) -> None:
    """
    Update Strategy Foundry (GP) metrics.

    Args:
        enabled: Whether foundry is running
        population_size: Current population size
        generations_run: Number of generations completed
        best_fitness: Best fitness score achieved
        strategies_discovered: Number of viable strategies found
    """
    if enabled is not None:
        _metrics["strategy_foundry"]["enabled"] = enabled
    if population_size is not None:
        _metrics["strategy_foundry"]["population_size"] = population_size
    if generations_run is not None:
        _metrics["strategy_foundry"]["generations_run"] = generations_run
    if best_fitness is not None:
        _metrics["strategy_foundry"]["best_fitness"] = best_fitness
    if strategies_discovered is not None:
        _metrics["strategy_foundry"]["strategies_discovered"] = strategies_discovered
    _metrics["strategy_foundry"]["last_evolution"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# =============================================================================
# TCA (Transaction Cost Analysis) Metrics Update Functions
# =============================================================================

def update_tca_metrics(
    total_trades: Optional[int] = None,
    avg_slippage_bps: Optional[float] = None,
    avg_spread_capture_bps: Optional[float] = None,
    total_cost_usd: Optional[float] = None,
) -> None:
    """
    Update TCA (Transaction Cost Analysis) metrics.

    Args:
        total_trades: Number of trades in lookback period
        avg_slippage_bps: Average slippage in basis points
        avg_spread_capture_bps: Average spread capture in basis points
        total_cost_usd: Total transaction costs in USD
    """
    if total_trades is not None:
        _metrics["tca"]["total_trades_7d"] = total_trades
    if avg_slippage_bps is not None:
        _metrics["tca"]["avg_slippage_bps"] = avg_slippage_bps
    if avg_spread_capture_bps is not None:
        _metrics["tca"]["avg_spread_capture_bps"] = avg_spread_capture_bps
    if total_cost_usd is not None:
        _metrics["tca"]["total_cost_usd_7d"] = total_cost_usd
    _metrics["tca"]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sync_tca_metrics(lookback_days: int = 7) -> None:
    """
    Sync TCA metrics from the global TransactionCostAnalyzer instance.
    Call this periodically or after trades.
    """
    try:
        from execution.tca.transaction_cost_analyzer import get_tca_analyzer
        tca = get_tca_analyzer()
        summary = tca.get_summary_tca_metrics(lookback_days=lookback_days)
        update_tca_metrics(
            total_trades=summary.get("total_trades", 0),
            avg_slippage_bps=summary.get("avg_slippage_bps", 0.0),
            avg_spread_capture_bps=summary.get("avg_spread_capture_bps", 0.0),
            total_cost_usd=summary.get("total_cost_usd", 0.0),
        )
    except Exception:
        pass  # TCA not available or not initialized


def get_metrics() -> Dict[str, Any]:
    """Get current metrics snapshot including AI reliability metrics."""
    cfg = get_metrics_config()

    metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": {
            "uptime_seconds": int(time.time() - _metrics["system"]["start_time"]),
        },
    }

    if cfg.get("include_requests", True):
        metrics["requests"] = _metrics["requests"].copy()

    if cfg.get("include_performance", True):
        metrics["performance"] = _metrics["performance"].copy()

    # Always include timestamps for trade events
    metrics["timestamps"] = _metrics["timestamps"].copy()

    # Phase 3: Include AI Reliability Metrics
    # Only include if they have been populated (not all None)
    if _metrics["calibration"]["brier_score"] is not None:
        metrics["calibration"] = _metrics["calibration"].copy()

    if _metrics["conformal"]["coverage_rate"] is not None:
        metrics["conformal"] = _metrics["conformal"].copy()

    # Always include LLM metrics if token tracking is active
    if _metrics["llm"]["tokens_used_today"] > 0 or _metrics["llm"]["selective_mode_enabled"]:
        metrics["llm"] = _metrics["llm"].copy()

    # Include uncertainty metrics if any trades were blocked
    if _metrics["uncertainty"]["avg_uncertainty_score"] is not None or \
       _metrics["uncertainty"]["high_uncertainty_trades_blocked"] > 0:
        metrics["uncertainty"] = _metrics["uncertainty"].copy()

    # Phase 4: Include execution bandit metrics if enabled
    if _metrics["execution_bandit"]["enabled"] or \
       _metrics["execution_bandit"]["total_selections"] > 0:
        metrics["execution_bandit"] = _metrics["execution_bandit"].copy()

    # Gemini: Include strategy foundry metrics if running
    if _metrics["strategy_foundry"]["enabled"] or \
       _metrics["strategy_foundry"]["generations_run"] > 0:
        metrics["strategy_foundry"] = _metrics["strategy_foundry"].copy()

    # TCA: Include transaction cost analysis metrics if any trades recorded
    if _metrics["tca"]["total_trades_7d"] > 0 or \
       _metrics["tca"]["last_updated"] is not None:
        metrics["tca"] = _metrics["tca"].copy()

    return metrics


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/readiness":
            self._json({"ready": True})
        elif self.path == "/liveness":
            self._json({"alive": True})
        elif self.path == "/health":
            self._json({
                "status": "healthy",
                "ready": True,
                "alive": True,
            })
        elif self.path == "/metrics":
            # FIX (2026-01-05): Rate limit /metrics in live mode
            if not self._check_rate_limit():
                return
            cfg = get_metrics_config()
            if not cfg.get("enabled", True):
                self.send_response(404)
                self.end_headers()
                return
            self._json(get_metrics())
        elif self.path == "/metrics/prometheus":
            # FIX (2026-01-05): Rate limit /metrics/prometheus in live mode
            if not self._check_rate_limit():
                return
            # Prometheus text format endpoint
            self._prometheus()
        else:
            self.send_response(404)
            self.end_headers()

    def _check_rate_limit(self) -> bool:
        """
        Check rate limit for metrics endpoints.

        FIX (2026-01-05): Added rate limiting to prevent flood attacks.

        Returns:
            True if request is allowed, False if rate limited (429 sent)
        """
        if not is_rate_limiting_enabled():
            return True  # Rate limiting disabled in paper mode

        rate_limiter = get_metrics_rate_limiter()
        if rate_limiter.is_allowed():
            return True

        # Rate limited - send 429
        increment_throttle_counter()
        retry_after = rate_limiter.get_retry_after()

        body = json.dumps({
            "error": "Too Many Requests",
            "message": "Rate limit exceeded for metrics endpoint",
            "retry_after": retry_after,
        }).encode("utf-8")

        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Retry-After", str(retry_after))
        self.end_headers()
        self.wfile.write(body)
        return False

    def _prometheus(self):
        """Serve Prometheus text format metrics."""
        try:
            from trade_logging.prometheus_metrics import get_metrics_text, get_content_type
            output = get_metrics_text()
            self.send_response(200)
            self.send_header("Content-Type", get_content_type())
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)
        except ImportError:
            # Fallback if prometheus_client not installed
            body = b"# prometheus_client not installed\n# Install with: pip install prometheus-client\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            body = f"# Error: {e}\n".encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        return  # quiet

    def _json(self, payload):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_health_server(port: int = 8000, bind_localhost_only: bool = None) -> HTTPServer:
    """
    Start HTTP health check server with endpoints:
    - /health - Overall health status
    - /readiness - Kubernetes readiness probe
    - /liveness - Kubernetes liveness probe
    - /metrics - Performance and request metrics (config-gated)

    SECURITY FIX (2026-01-04): In production mode, binds to 127.0.0.1 (localhost only).
    In paper mode, binds to 0.0.0.0 (all interfaces) for easier debugging.
    """
    # Determine binding address
    if bind_localhost_only is None:
        # Auto-detect based on mode
        try:
            from config.settings_loader import get_setting
            is_live = get_setting("system.mode", "paper") == "live"
        except ImportError:
            import os
            is_live = os.getenv("KOBE_MODE", "paper").lower() == "live"
        bind_localhost_only = is_live

    host = "127.0.0.1" if bind_localhost_only else "0.0.0.0"
    logger.info(f"Starting health server on {host}:{port} (localhost_only={bind_localhost_only})")
    server = HTTPServer((host, port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def reset_metrics() -> None:
    """Reset all metrics (for testing)."""
    global _metrics
    _metrics = {
        "requests": {
            "total": 0,
            "orders_submitted": 0,
            "orders_rejected": 0,
            "orders_filled": 0,
            "ioc_submitted": 0,
            "ioc_filled": 0,
            "ioc_cancelled": 0,
            "liquidity_blocked": 0,
        },
        "performance": {
            "win_rate": None,
            "profit_factor": None,
            "sharpe": None,
            "last_updated": None,
        },
        "system": {
            "start_time": time.time(),
            "uptime_seconds": 0,
        },
        "timestamps": {
            "last_submit_ts": None,
            "last_fill_ts": None,
            "last_trade_event_ts": None,
        },
        # Phase 3: AI Reliability Metrics
        "calibration": {
            "brier_score": None,
            "expected_calibration_error": None,
            "max_calibration_error": None,
            "n_samples": 0,
            "last_updated": None,
        },
        "conformal": {
            "coverage_rate": None,
            "target_coverage": None,
            "avg_interval_width": None,
            "n_predictions": 0,
            "last_updated": None,
        },
        "llm": {
            "tokens_used_today": 0,
            "token_budget_remaining": None,
            "token_usage_pct": None,
            "llm_calls_saved": 0,
            "cache_hit_rate": None,
            "selective_mode_enabled": False,
            # Phase 4: USD Cost Tracking
            "cost_usd_today": 0.0,
            "cost_budget_usd": None,
            "cost_remaining_usd": None,
            "cost_usage_pct": None,
            "input_tokens_today": 0,
            "output_tokens_today": 0,
            "calls_today": 0,
            "avg_latency_ms": None,
            "model": None,
            "last_updated": None,
        },
        "uncertainty": {
            "avg_uncertainty_score": None,
            "high_uncertainty_trades_blocked": 0,
            "uncertainty_threshold": 0.7,
        },
        # Phase 4: Execution Bandit Metrics
        "execution_bandit": {
            "enabled": False,
            "algorithm": None,
            "strategies": [],
            "total_selections": 0,
            "strategy_stats": {},
            "cumulative_regret": 0.0,
            "last_updated": None,
        },
        # Gemini: Strategy Foundry Metrics
        "strategy_foundry": {
            "enabled": False,
            "population_size": 0,
            "generations_run": 0,
            "best_fitness": None,
            "strategies_discovered": 0,
            "last_evolution": None,
        },
        # TCA (Transaction Cost Analysis) Metrics
        "tca": {
            "total_trades_7d": 0,
            "avg_slippage_bps": 0.0,
            "avg_spread_capture_bps": 0.0,
            "total_cost_usd_7d": 0.0,
            "last_updated": None,
        },
    }

