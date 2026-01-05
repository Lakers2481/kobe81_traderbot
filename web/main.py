"""
Web Dashboard - FastAPI Application
=====================================

This module provides a FastAPI web application to serve as a monitoring and
introspection dashboard for the trading bot. It exposes various API endpoints
to query the real-time status, performance metrics, and cognitive state of the AI.

Features:
- **Bot Status:** General health and operational status.
- **Cognitive Status:** Detailed insights into the Cognitive Brain's activity.
- **Recent Reflections:** Access to the latest learning insights, including LLM critiques.
- **Transaction Cost Analysis (TCA):** Metrics on execution quality and slippage.
- **Self-Model Description:** The AI's self-awareness of its strengths and weaknesses.

This dashboard provides a "window into the AI's mind," making it easier to
monitor, debug, and understand the complex behavior of the trading robot.

FIX (2026-01-04): Docs UI (/docs, /redoc, /openapi.json) is now disabled in
live mode to prevent information disclosure. Only available in paper mode.

Usage:
    To run the FastAPI application:
    uvicorn web.main:app --reload --port 8000

    Then open your browser to http://localhost:8000/docs for the API documentation.
    (Note: /docs is only available in paper mode for security)
"""

import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

# Import necessary components from the cognitive architecture
from cognitive.signal_processor import get_signal_processor
from cognitive.reflection_engine import get_reflection_engine
from cognitive.self_model import get_self_model
from execution.tca.transaction_cost_analyzer import get_tca_analyzer

# Import for real status values (SECURITY FIX 2026-01-04: replace placeholders)
from core.kill_switch import is_kill_switch_active
from config.settings_loader import get_setting

# Import webhook router
try:
    from web.api.webhooks import router as webhooks_router
    WEBHOOKS_AVAILABLE = True
except ImportError:
    WEBHOOKS_AVAILABLE = False
    webhooks_router = None

logger = logging.getLogger(__name__)

# FIX (2026-01-04): Disable docs UI in live mode to prevent information disclosure
# This hides /docs, /redoc, and /openapi.json endpoints when trading with real money
_mode = get_setting("system.mode", "paper")
_is_live_mode = _mode == "live"

if _is_live_mode:
    logger.info("Live mode detected - disabling docs UI for security")

app = FastAPI(
    title="Kobe81 Traderbot Dashboard API",
    description="API for monitoring and introspecting the Kobe81 Traderbot's status and cognitive state.",
    version="1.0.0",
    # Disable docs in live mode to prevent information disclosure
    docs_url=None if _is_live_mode else "/docs",
    redoc_url=None if _is_live_mode else "/redoc",
    openapi_url=None if _is_live_mode else "/openapi.json",
)

# Mount webhook router if available
if WEBHOOKS_AVAILABLE and webhooks_router:
    app.include_router(webhooks_router)
    logger.info("Webhook endpoints mounted at /webhook/*")


# =============================================================================
# API Security - Protect sensitive endpoints in live mode
# FIX (2026-01-05): Added to prevent unauthorized access to trading picks
# =============================================================================

async def require_api_key_in_production(x_api_key: Optional[str] = Header(None)):
    """
    Require API key for data endpoints in production (live) mode.

    In paper mode, endpoints are open for development convenience.
    In live mode, requires X-API-Key header matching WEB_API_KEY env var.

    FIX (2026-01-05): Added to protect /status, /top3_picks, /trade_of_day
    """
    if _is_live_mode:
        expected_key = os.getenv("WEB_API_KEY")
        if not expected_key:
            logger.error("WEB_API_KEY not configured in live mode")
            raise HTTPException(
                status_code=500,
                detail="WEB_API_KEY not configured. Set this environment variable for live mode."
            )
        if not x_api_key or x_api_key != expected_key:
            logger.warning(f"Unauthorized API access attempt to protected endpoint")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing X-API-Key header"
            )


# Rate limiter for API endpoints (reuse existing pattern from webhooks)
from collections import deque
import time

class APIRateLimiter:
    """Simple token bucket rate limiter for API endpoints."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, deque] = {}  # IP -> timestamps

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request from client_ip is allowed."""
        now = time.time()

        if client_ip not in self._requests:
            self._requests[client_ip] = deque()

        # Remove old entries
        while self._requests[client_ip] and self._requests[client_ip][0] < now - self.window_seconds:
            self._requests[client_ip].popleft()

        if len(self._requests[client_ip]) >= self.max_requests:
            return False

        self._requests[client_ip].append(now)
        return True

    def get_retry_after(self) -> int:
        """Return seconds to wait before retry."""
        return self.window_seconds


_api_rate_limiter = APIRateLimiter(max_requests=100, window_seconds=60)


async def check_rate_limit(request: Request):
    """Rate limiting dependency for API endpoints."""
    if not _is_live_mode:
        return  # No rate limiting in paper mode

    # Get client IP (respecting X-Forwarded-For)
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
    if "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    if not _api_rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait before retrying.",
            headers={"Retry-After": str(_api_rate_limiter.get_retry_after())}
        )

@app.get("/", response_class=HTMLResponse, summary="Home Page")
async def read_root():
    """
    Returns a simple HTML page with links to the API documentation and key endpoints.
    """
    # Build docs links only for paper mode (hidden in live mode for security)
    docs_links = ""
    if not _is_live_mode:
        docs_links = """
                <li><a href="/docs">API Documentation (Swagger UI)</a></li>
                <li><a href="/redoc">Alternative API Documentation (ReDoc)</a></li>"""

    mode_banner = f"<p><strong>Mode:</strong> {'🔴 LIVE TRADING' if _is_live_mode else '📝 Paper Trading'}</p>"

    html_content = f"""
    <html>
        <head>
            <title>Kobe81 Traderbot Dashboard</title>
            <link rel="icon" href="https://raw.githubusercontent.com/mljar/mljar-supervised/master/docs/img/mljar_logo.png" />
        </head>
        <body>
            <h1>Kobe81 Traderbot Dashboard</h1>
            {mode_banner}
            <p>Welcome to the monitoring API for your intelligent trading robot.</p>
            <ul>{docs_links}
                <li><a href="/status">General Bot Status</a></li>
                <li><a href="/cognitive_status">Cognitive System Status</a></li>
                <li><a href="/recent_reflections">Recent Reflections (Learning Log)</a></li>
                <li><a href="/recent_tca">Recent Transaction Cost Analysis (TCA)</a></li>
                <li><a href="/self_model_description">AI's Self-Description</a></li>
                <li><a href="/morning_briefing">Morning Briefing (LLM Analysis)</a></li>
            </ul>
            <h3>LLM-Powered Endpoints</h3>
            <ul>
                <li><a href="/morning_briefing">Morning Briefing</a> - Claude's daily market analysis</li>
                <li><a href="/live_narrative/AAPL">Live Narrative (Example: AAPL)</a> - Real-time symbol analysis</li>
            </ul>
            <p>Use the API endpoints to explore available data.</p>
        </body>
    </html>
    """
    return html_content

@app.get(
    "/status",
    summary="Get overall bot status",
    dependencies=[Depends(require_api_key_in_production), Depends(check_rate_limit)]
)
async def get_bot_status() -> Dict[str, Any]:
    """
    Returns a high-level overview of the bot's operational status.
    """
    try:
        signal_processor = get_signal_processor()
        # This will be refined as more components are exposed
        # SECURITY FIX 2026-01-04: Use real status values, not placeholders
        mode = get_setting("system.mode", "paper")
        status = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "operational",
            "kill_switch_active": is_kill_switch_active(),
            "paper_mode": mode == "paper",
            "mode": mode,
            "active_cognitive_episodes": len(signal_processor._active_episodes),
            "log_level": logging.getLevelName(logger.getEffectiveLevel()),
        }
        return status
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/cognitive_status",
    summary="Get cognitive system status",
    dependencies=[Depends(require_api_key_in_production), Depends(check_rate_limit)]
)
async def get_cognitive_system_status() -> Dict[str, Any]:
    """
    Returns a detailed status of the CognitiveSignalProcessor and its underlying Brain.
    """
    try:
        processor = get_signal_processor()
        return processor.get_cognitive_status()
    except Exception as e:
        logger.error(f"Error getting cognitive status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recent_reflections", summary="Get recent learning reflections")
async def get_recent_reflections() -> List[Dict[str, Any]]:
    """
    Returns a list of the most recent learning reflections, including any LLM critiques.
    """
    try:
        reflection_engine = get_reflection_engine()
        # Returns a list of Reflection objects converted to dicts
        return [r.to_dict() for r in reflection_engine.get_recent_reflections(limit=10)]
    except Exception as e:
        logger.error(f"Error getting recent reflections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recent_tca", summary="Get recent Transaction Cost Analysis (TCA) metrics")
async def get_recent_tca() -> Dict[str, Any]:
    """
    Returns a summary of recent Transaction Cost Analysis (TCA) metrics.
    """
    try:
        tca_analyzer = get_tca_analyzer()
        return tca_analyzer.get_summary_tca_metrics(lookback_days=7)
    except Exception as e:
        logger.error(f"Error getting recent TCA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/self_model_description", summary="Get AI's self-description")
async def get_ai_self_description() -> Dict[str, str]:
    """
    Returns the AI's natural language self-description of its capabilities and limitations.
    """
    try:
        self_model = get_self_model()
        return {"self_description": self_model.get_self_description()}
    except Exception as e:
        logger.error(f"Error getting self-model description: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LLM-Powered Endpoints (Human-Like Reasoning from Claude)
# =============================================================================

@app.get(
    "/morning_briefing",
    summary="Get daily morning briefing with LLM analysis",
    dependencies=[Depends(require_api_key_in_production), Depends(check_rate_limit)]
)
async def get_morning_briefing() -> Dict[str, Any]:
    """
    Returns the daily insights report with Claude LLM-generated narratives.

    The morning briefing includes:
    - Market summary with regime assessment
    - Top-3 picks with human-like reasoning explanations
    - Trade of the Day deep analysis
    - Key findings and patterns discovered
    - Sentiment interpretation of recent news
    - Risk warnings and opportunities

    This endpoint reads from the daily_insights.json file generated by:
    `python scripts/scan.py --top3 --narrative`

    If the file doesn't exist, returns an error with instructions.
    """
    insights_path = Path("logs/daily_insights.json")

    if not insights_path.exists():
        return {
            "success": False,
            "error": "Daily insights not generated yet",
            "instructions": "Run: python scripts/scan.py --top3 --narrative",
            "timestamp": datetime.now().isoformat(),
        }

    try:
        with open(insights_path, 'r', encoding='utf-8') as f:
            insights = json.load(f)

        return {
            "success": True,
            "insights": insights,
            "generated_at": insights.get("timestamp"),
            "generation_method": insights.get("generation_method", "unknown"),
            "retrieved_at": datetime.now().isoformat(),
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing daily insights JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON in insights file: {e}")
    except Exception as e:
        logger.error(f"Error reading morning briefing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/live_narrative/{symbol}",
    summary="Get real-time LLM narrative for a symbol",
    dependencies=[Depends(require_api_key_in_production), Depends(check_rate_limit)]
)
async def get_live_narrative(symbol: str) -> Dict[str, Any]:
    """
    Generates a real-time LLM analysis for a specific symbol.

    This endpoint:
    1. Fetches recent news for the symbol
    2. Analyzes sentiment using VADER
    3. Generates a Claude LLM interpretation of the news impact

    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, TSLA)

    Returns:
        Dict containing:
        - symbol: The requested symbol
        - news_narrative: LLM-generated interpretation
        - sentiment: Aggregated sentiment scores
        - article_count: Number of articles analyzed
        - generation_method: "claude" or "deterministic"
    """
    symbol = symbol.upper().strip()

    if not symbol or len(symbol) > 10:
        raise HTTPException(status_code=400, detail="Invalid symbol format")

    try:
        from altdata.news_processor import get_news_processor

        news_processor = get_news_processor()
        interpretation_result = news_processor.get_narrative_interpretation(
            symbols=[symbol],
            lookback_minutes=120,  # Last 2 hours of news
        )

        return {
            "success": True,
            "symbol": symbol,
            "news_narrative": interpretation_result.get("interpretation", ""),
            "sentiment": interpretation_result.get("aggregated_sentiment", {}),
            "article_count": len(interpretation_result.get("articles", [])),
            "generation_method": interpretation_result.get("generation_method", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }

    except ImportError as e:
        logger.error(f"News processor not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="News processor module not available. Check dependencies."
        )
    except Exception as e:
        logger.error(f"Error generating live narrative for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/top3_picks",
    summary="Get current Top-3 picks with narratives",
    dependencies=[Depends(require_api_key_in_production), Depends(check_rate_limit)]
)
async def get_top3_picks() -> Dict[str, Any]:
    """
    Returns the current Top-3 trading picks with LLM-generated explanations.

    This is a subset of the morning briefing focused only on actionable trades.
    """
    insights_path = Path("logs/daily_insights.json")

    if not insights_path.exists():
        # Try to read the raw picks file
        picks_path = Path("logs/daily_picks.csv")
        if picks_path.exists():
            import pandas as pd
            try:
                picks_df = pd.read_csv(picks_path)
                return {
                    "success": True,
                    "picks": picks_df.to_dict(orient='records'),
                    "has_narratives": False,
                    "note": "Raw picks available. Run with --narrative for LLM analysis.",
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                logger.error(f"Error reading picks CSV: {e}")

        return {
            "success": False,
            "error": "No picks available",
            "instructions": "Run: python scripts/scan.py --top3 --narrative",
        }

    try:
        with open(insights_path, 'r', encoding='utf-8') as f:
            insights = json.load(f)

        top3 = insights.get("top3_narratives", [])

        return {
            "success": True,
            "picks": top3,
            "has_narratives": bool(top3),
            "count": len(top3),
            "generated_at": insights.get("timestamp"),
            "generation_method": insights.get("generation_method", "unknown"),
        }
    except Exception as e:
        logger.error(f"Error getting top 3 picks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/trade_of_day",
    summary="Get Trade of the Day with deep analysis",
    dependencies=[Depends(require_api_key_in_production), Depends(check_rate_limit)]
)
async def get_trade_of_day() -> Dict[str, Any]:
    """
    Returns the Trade of the Day with extended LLM-generated analysis.

    The TOTD is the highest-conviction trade from the Top-3 picks.
    """
    insights_path = Path("logs/daily_insights.json")
    totd_csv_path = Path("logs/trade_of_day.csv")

    # Try insights file first (has narratives)
    if insights_path.exists():
        try:
            with open(insights_path, 'r', encoding='utf-8') as f:
                insights = json.load(f)

            top3 = insights.get("top3_narratives", [])
            totd_analysis = insights.get("totd_deep_analysis", "")

            if top3:
                totd = top3[0]  # First pick is TOTD
                return {
                    "success": True,
                    "trade_of_day": totd,
                    "deep_analysis": totd_analysis,
                    "has_narrative": bool(totd_analysis),
                    "generated_at": insights.get("timestamp"),
                    "generation_method": insights.get("generation_method", "unknown"),
                }
        except Exception as e:
            logger.warning(f"Could not read insights file: {e}")

    # Fallback to CSV
    if totd_csv_path.exists():
        import pandas as pd
        try:
            totd_df = pd.read_csv(totd_csv_path)
            if not totd_df.empty:
                return {
                    "success": True,
                    "trade_of_day": totd_df.iloc[0].to_dict(),
                    "deep_analysis": "",
                    "has_narrative": False,
                    "note": "Run with --narrative for LLM analysis",
                }
        except Exception as e:
            logger.error(f"Error reading TOTD CSV: {e}")

    return {
        "success": False,
        "error": "No Trade of the Day available",
        "instructions": "Run: python scripts/scan.py --top3 --narrative",
    }


if __name__ == "__main__":
    # This block is for local testing and development outside of uvicorn command.
    # In production, you would run with 'uvicorn web.main:app --host 0.0.0.0 --port 8000'
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
