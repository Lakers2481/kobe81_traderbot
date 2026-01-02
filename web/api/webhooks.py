"""
Webhook Endpoints for External Signal Ingestion.

Provides REST API endpoints for receiving trading signals from:
- TradingView alerts
- Custom alerting systems
- Third-party signal providers

Security:
- HMAC signature validation for authenticated sources
- Rate limiting per source IP
- Request validation and sanitization
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Depends, Header
from pydantic import BaseModel, Field, validator

from web.api.signal_queue import (
    ExternalSignal,
    SignalSource,
    SignalStatus,
    get_signal_queue,
)
from core.kill_switch import is_kill_switch_active

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhooks"])


# =============================================================================
# Configuration
# =============================================================================

# HMAC secret for TradingView (set via environment variable)
TRADINGVIEW_WEBHOOK_SECRET = os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "")
CUSTOM_WEBHOOK_SECRET = os.getenv("CUSTOM_WEBHOOK_SECRET", "")

# Rate limiting
RATE_LIMIT_WINDOW_S = 60  # 1 minute window
RATE_LIMIT_MAX_REQUESTS = 30  # Max requests per window per IP

# Request tracking for rate limiting
_request_counts: Dict[str, list] = {}


# =============================================================================
# Request Models
# =============================================================================

class TradingViewAlert(BaseModel):
    """
    TradingView webhook alert payload.

    TradingView alerts can be configured with custom JSON payloads.
    This model supports the recommended format:

    {
        "symbol": "{{ticker}}",
        "side": "buy",
        "action": "open",
        "price": {{close}},
        "strategy": "MyStrategy",
        "message": "{{message}}"
    }
    """
    symbol: str = Field(..., description="Stock/crypto symbol")
    side: str = Field(..., description="buy, sell, long, short")
    action: str = Field(default="open", description="open, close, scale_in, scale_out")
    price: Optional[float] = Field(default=None, description="Alert trigger price")
    stop_loss: Optional[float] = Field(default=None, description="Stop loss price")
    take_profit: Optional[float] = Field(default=None, description="Take profit price")
    qty: Optional[int] = Field(default=None, description="Position size in shares")
    risk_pct: Optional[float] = Field(default=None, description="Risk percentage (0.01 = 1%)")
    strategy: Optional[str] = Field(default=None, description="Strategy name")
    timeframe: Optional[str] = Field(default=None, description="Timeframe (1h, 4h, 1d)")
    message: Optional[str] = Field(default=None, description="Alert message")
    confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Confidence 0-1")

    @validator("symbol")
    def validate_symbol(cls, v):
        """Validate and normalize symbol."""
        v = v.upper().strip()
        if not v or len(v) > 10:
            raise ValueError("Invalid symbol format")
        # Remove common suffixes from TradingView
        for suffix in [".US", ".NASDAQ", ".NYSE", ".ARCA"]:
            if v.endswith(suffix):
                v = v[:-len(suffix)]
        return v

    @validator("side")
    def validate_side(cls, v):
        """Validate side."""
        v = v.lower().strip()
        if v not in ("buy", "sell", "long", "short"):
            raise ValueError("Side must be buy, sell, long, or short")
        return v

    @validator("action")
    def validate_action(cls, v):
        """Validate action."""
        v = v.lower().strip()
        if v not in ("open", "close", "scale_in", "scale_out"):
            raise ValueError("Action must be open, close, scale_in, or scale_out")
        return v


class CustomSignalPayload(BaseModel):
    """Generic webhook payload for custom signal sources."""
    symbol: str
    side: str
    action: str = "open"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    qty: Optional[int] = None
    risk_pct: Optional[float] = None
    strategy: Optional[str] = None
    timeframe: Optional[str] = None
    confidence: Optional[float] = None
    idempotency_key: Optional[str] = None
    expires_in_seconds: Optional[int] = Field(default=300, le=3600)  # Max 1 hour
    source_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WebhookResponse(BaseModel):
    """Standard webhook response."""
    success: bool
    signal_id: Optional[str] = None
    status: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# =============================================================================
# Security Utilities
# =============================================================================

def validate_hmac_signature(
    payload: bytes,
    signature: str,
    secret: str,
    algorithm: str = "sha256",
) -> bool:
    """
    Validate HMAC signature.

    Args:
        payload: Raw request body
        signature: Signature from header
        secret: HMAC secret
        algorithm: Hash algorithm (sha256, sha1)

    Returns:
        True if signature is valid
    """
    if not secret:
        logger.warning("HMAC secret not configured, skipping validation")
        return True  # Allow if not configured (dev mode)

    if not signature:
        return False

    # Handle different signature formats
    if signature.startswith("sha256="):
        signature = signature[7:]
    elif signature.startswith("sha1="):
        signature = signature[5:]

    try:
        if algorithm == "sha256":
            expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        else:
            expected = hmac.new(secret.encode(), payload, hashlib.sha1).hexdigest()

        return hmac.compare_digest(signature.lower(), expected.lower())
    except Exception as e:
        logger.error(f"HMAC validation error: {e}")
        return False


def check_rate_limit(client_ip: str) -> bool:
    """
    Check if client is within rate limit.

    Returns:
        True if within limit, False if exceeded
    """
    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW_S

    # Clean old entries
    if client_ip in _request_counts:
        _request_counts[client_ip] = [
            t for t in _request_counts[client_ip] if t > window_start
        ]
    else:
        _request_counts[client_ip] = []

    # Check limit
    if len(_request_counts[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False

    # Record request
    _request_counts[client_ip].append(current_time)
    return True


async def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    # Check forwarded headers for proxied requests
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# =============================================================================
# Webhook Endpoints
# =============================================================================

@router.post("/tradingview", response_model=WebhookResponse)
async def tradingview_webhook(
    request: Request,
    alert: TradingViewAlert,
    x_tradingview_signature: Optional[str] = Header(None, alias="X-TradingView-Signature"),
):
    """
    Receive TradingView alert webhook.

    TradingView can send webhooks when alert conditions are triggered.
    Configure your TradingView alert to POST JSON to this endpoint.

    **Security:** If TRADINGVIEW_WEBHOOK_SECRET is set, the request
    must include a valid HMAC signature in X-TradingView-Signature header.

    **Example TradingView Alert Message:**
    ```json
    {
        "symbol": "{{ticker}}",
        "side": "buy",
        "action": "open",
        "price": {{close}},
        "strategy": "ICT_TurtleSoup"
    }
    ```
    """
    # Check kill switch
    if is_kill_switch_active():
        return WebhookResponse(
            success=False,
            status="rejected",
            message="Trading system kill switch is active",
        )

    # Rate limiting
    client_ip = await get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before sending more requests.",
        )

    # HMAC validation (if configured)
    if TRADINGVIEW_WEBHOOK_SECRET:
        body = await request.body()
        if not validate_hmac_signature(
            payload=body,
            signature=x_tradingview_signature or "",
            secret=TRADINGVIEW_WEBHOOK_SECRET,
        ):
            logger.warning(f"Invalid TradingView signature from {client_ip}")
            raise HTTPException(status_code=401, detail="Invalid signature")

    # Create external signal
    signal = ExternalSignal(
        symbol=alert.symbol,
        side=alert.side,
        action=alert.action,
        source=SignalSource.TRADINGVIEW,
        source_name="TradingView",
        entry_price=alert.price,
        stop_loss=alert.stop_loss,
        take_profit=alert.take_profit,
        qty=alert.qty,
        risk_pct=alert.risk_pct,
        strategy=alert.strategy,
        timeframe=alert.timeframe,
        confidence=alert.confidence,
        raw_payload=alert.dict(),
    )

    # Enqueue signal
    queue = get_signal_queue()
    success = queue.enqueue(signal)

    if success:
        logger.info(
            f"TradingView alert queued: {alert.symbol} {alert.side} {alert.action} "
            f"from {client_ip}"
        )
        return WebhookResponse(
            success=True,
            signal_id=signal.signal_id,
            status="queued",
            message=f"Signal {signal.signal_id} queued for processing",
        )
    else:
        return WebhookResponse(
            success=False,
            signal_id=signal.signal_id,
            status=signal.status.value,
            message=signal.result_notes or "Signal rejected",
        )


@router.post("/signal", response_model=WebhookResponse)
async def custom_signal_webhook(
    request: Request,
    payload: CustomSignalPayload,
    x_webhook_signature: Optional[str] = Header(None, alias="X-Webhook-Signature"),
):
    """
    Receive custom trading signal webhook.

    Generic endpoint for custom signal sources. Supports additional
    fields like idempotency_key and custom metadata.

    **Security:** If CUSTOM_WEBHOOK_SECRET is set, the request
    must include a valid HMAC signature in X-Webhook-Signature header.
    """
    # Check kill switch
    if is_kill_switch_active():
        return WebhookResponse(
            success=False,
            status="rejected",
            message="Trading system kill switch is active",
        )

    # Rate limiting
    client_ip = await get_client_ip(request)
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # HMAC validation (if configured)
    if CUSTOM_WEBHOOK_SECRET:
        body = await request.body()
        if not validate_hmac_signature(
            payload=body,
            signature=x_webhook_signature or "",
            secret=CUSTOM_WEBHOOK_SECRET,
        ):
            logger.warning(f"Invalid custom webhook signature from {client_ip}")
            raise HTTPException(status_code=401, detail="Invalid signature")

    # Create external signal
    from datetime import timedelta

    expires_at = None
    if payload.expires_in_seconds:
        expires_at = datetime.utcnow() + timedelta(seconds=payload.expires_in_seconds)

    signal = ExternalSignal(
        symbol=payload.symbol,
        side=payload.side,
        action=payload.action,
        source=SignalSource.CUSTOM_WEBHOOK,
        source_name=payload.source_name or "Custom",
        idempotency_key=payload.idempotency_key,
        expires_at=expires_at,
        entry_price=payload.entry_price,
        stop_loss=payload.stop_loss,
        take_profit=payload.take_profit,
        qty=payload.qty,
        risk_pct=payload.risk_pct,
        strategy=payload.strategy,
        timeframe=payload.timeframe,
        confidence=payload.confidence,
        raw_payload=payload.dict(),
    )

    # Enqueue signal
    queue = get_signal_queue()
    success = queue.enqueue(signal)

    if success:
        logger.info(
            f"Custom signal queued: {payload.symbol} {payload.side} {payload.action} "
            f"from {payload.source_name or 'Custom'} ({client_ip})"
        )
        return WebhookResponse(
            success=True,
            signal_id=signal.signal_id,
            status="queued",
            message=f"Signal {signal.signal_id} queued for processing",
        )
    else:
        return WebhookResponse(
            success=False,
            signal_id=signal.signal_id,
            status=signal.status.value,
            message=signal.result_notes or "Signal rejected",
        )


@router.get("/status")
async def webhook_status():
    """
    Get webhook system status.

    Returns queue statistics and configuration status.
    """
    queue = get_signal_queue()
    stats = queue.get_stats()

    return {
        "status": "operational",
        "kill_switch_active": is_kill_switch_active(),
        "queue_stats": stats,
        "tradingview_hmac_configured": bool(TRADINGVIEW_WEBHOOK_SECRET),
        "custom_hmac_configured": bool(CUSTOM_WEBHOOK_SECRET),
        "rate_limit": {
            "window_seconds": RATE_LIMIT_WINDOW_S,
            "max_requests": RATE_LIMIT_MAX_REQUESTS,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/queue")
async def get_queue_contents():
    """
    Get pending signals in queue (for monitoring).

    Returns list of pending signals waiting to be processed.
    """
    queue = get_signal_queue()
    pending = queue.get_pending()

    return {
        "pending_count": len(pending),
        "signals": [s.to_dict() for s in pending],
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/test", response_model=WebhookResponse)
async def test_webhook(request: Request):
    """
    Test endpoint for webhook connectivity.

    Use this to verify your webhook setup without creating real signals.
    Does not require authentication or affect the signal queue.
    """
    body = await request.body()
    client_ip = await get_client_ip(request)

    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = {"raw": body.decode("utf-8", errors="replace")}

    logger.info(f"Test webhook received from {client_ip}: {payload}")

    return WebhookResponse(
        success=True,
        status="received",
        message=f"Test webhook received from {client_ip}. Payload logged.",
    )
