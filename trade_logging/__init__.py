"""
Trade Logging Package
======================

Provides comprehensive trade audit and observability infrastructure:

1. DecisionCardLogger: JSON audit trail for all trade decisions
2. PrometheusMetrics: Real-time metrics for Grafana dashboards

Usage:
    from trade_logging import DecisionCardLogger, get_card_logger

    logger = get_card_logger()
    card = logger.create_card(
        symbol="AAPL",
        side="long",
        strategy="ibs_rsi",
        plan=TradePlan(...),
        signals=[...],
        risk_checks=[...],
    )
    logger.save_card(card)

    # After trade execution:
    logger.update_tca(card.card_id, tca_result)

    # After trade closes:
    logger.update_result(card.card_id, trade_result)

Prometheus Metrics:
    from trade_logging.prometheus_metrics import (
        inc_order, set_pnl, record_scan_completed
    )

    inc_order("AAPL", "long", "ibs_rsi", filled=True)
    set_pnl(daily=150.50, total=1250.00)
"""

from trade_logging.decision_card_logger import (
    DecisionCard,
    DecisionCardLogger,
    TradePlan,
    SignalDriver,
    ExecutionContext,
    RiskCheck,
    ModelInfo,
    TCAResult,
    TradeResult,
    get_card_logger,
)

from trade_logging.prometheus_metrics import (
    PROMETHEUS_AVAILABLE,
    get_metrics_text,
    get_content_type,
    inc_order,
    inc_signal,
    inc_decision,
    inc_error,
    inc_scan,
    set_pnl,
    set_performance,
    set_portfolio,
    set_data_freshness,
    record_scan_completed,
    record_trade_completed,
    record_signal_score,
    set_uptime,
    set_vix,
)

__all__ = [
    # Decision Card Logger
    'DecisionCard',
    'DecisionCardLogger',
    'TradePlan',
    'SignalDriver',
    'ExecutionContext',
    'RiskCheck',
    'ModelInfo',
    'TCAResult',
    'TradeResult',
    'get_card_logger',
    # Prometheus Metrics
    'PROMETHEUS_AVAILABLE',
    'get_metrics_text',
    'get_content_type',
    'inc_order',
    'inc_signal',
    'inc_decision',
    'inc_error',
    'inc_scan',
    'set_pnl',
    'set_performance',
    'set_portfolio',
    'set_data_freshness',
    'record_scan_completed',
    'record_trade_completed',
    'record_signal_score',
    'set_uptime',
    'set_vix',
]
