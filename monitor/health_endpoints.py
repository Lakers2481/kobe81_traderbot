from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings_loader import get_metrics_config


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


def get_metrics() -> Dict[str, Any]:
    """Get current metrics snapshot."""
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
            cfg = get_metrics_config()
            if not cfg.get("enabled", True):
                self.send_response(404)
                self.end_headers()
                return
            self._json(get_metrics())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return  # quiet

    def _json(self, payload):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_health_server(port: int = 8000) -> HTTPServer:
    """
    Start HTTP health check server with endpoints:
    - /health - Overall health status
    - /readiness - Kubernetes readiness probe
    - /liveness - Kubernetes liveness probe
    - /metrics - Performance and request metrics (config-gated)
    """
    server = HTTPServer(("0.0.0.0", port), _Handler)
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
    }

