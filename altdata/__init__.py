"""Alternative data sources and utilities (sentiment, news, etc.)."""

# News and sentiment
from altdata.news_processor import NewsProcessor, get_news_processor

# Congressional trades (Quiver Quant API)
from altdata.congressional_trades import (
    CongressionalTradesClient,
    CongressionalTrade,
    CongressionalTradeSummary,
    get_congressional_client,
)

# Insider activity (SEC EDGAR Form 4)
from altdata.insider_activity import (
    InsiderActivityClient,
    InsiderTrade,
    InsiderActivitySummary,
    get_insider_client,
)

# Options flow and unusual activity
from altdata.options_flow import (
    OptionsFlowClient,
    UnusualOptionActivity,
    OptionsFlowSummary,
    get_options_flow_client,
)

__all__ = [
    # News
    'NewsProcessor',
    'get_news_processor',
    # Congressional
    'CongressionalTradesClient',
    'CongressionalTrade',
    'CongressionalTradeSummary',
    'get_congressional_client',
    # Insider
    'InsiderActivityClient',
    'InsiderTrade',
    'InsiderActivitySummary',
    'get_insider_client',
    # Options
    'OptionsFlowClient',
    'UnusualOptionActivity',
    'OptionsFlowSummary',
    'get_options_flow_client',
]

