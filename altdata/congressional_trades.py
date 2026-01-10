"""
Congressional Trades - Quiver Quant API Integration
====================================================

This module fetches congressional trading data from Quiver Quant API.
Congressional trades are useful for detecting unusual activity by
government officials who may have access to non-public information.

Features:
- Fetches recent congressional trades for specific symbols
- Aggregates net buying/selling activity
- Identifies notable officials trading a stock
- Provides buy/sell signal interpretation

Usage:
    from altdata.congressional_trades import CongressionalTradesClient

    client = CongressionalTradesClient()

    # Get recent trades for PLTR
    trades = client.fetch_recent_trades('PLTR', days_back=90)

    # Get summary of congressional activity
    summary = client.get_trade_summary('PLTR')
    print(f"Net activity: {summary['net_activity']}")

API Key:
    Requires QUIVER_API_KEY in .env file.
    Free tier: 100 requests/month
    Get key at: https://www.quiverquant.com/
"""

import logging
import os
import requests
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import time

logger = logging.getLogger(__name__)

# Quiver Quant API configuration
QUIVER_BASE_URL = "https://api.quiverquant.com/beta"
QUIVER_RATE_LIMIT = 10  # requests per minute (conservative for free tier)
_last_request_time: float = 0.0


@dataclass
class CongressionalTrade:
    """Represents a single congressional trade transaction."""
    transaction_id: str
    representative: str
    party: str  # 'R', 'D', or 'I'
    house: str  # 'House' or 'Senate'
    symbol: str
    transaction_type: str  # 'Purchase', 'Sale', 'Exchange'
    transaction_date: datetime
    disclosure_date: datetime
    amount_range: str  # e.g., "$1,001 - $15,000"
    amount_min: float = 0.0
    amount_max: float = 0.0
    asset_description: str = ""
    cap_gains_over_200: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "representative": self.representative,
            "party": self.party,
            "house": self.house,
            "symbol": self.symbol,
            "transaction_type": self.transaction_type,
            "transaction_date": self.transaction_date.isoformat(),
            "disclosure_date": self.disclosure_date.isoformat(),
            "amount_range": self.amount_range,
            "amount_min": self.amount_min,
            "amount_max": self.amount_max,
            "asset_description": self.asset_description,
            "cap_gains_over_200": self.cap_gains_over_200,
        }

    @property
    def is_buy(self) -> bool:
        return self.transaction_type.lower() in ('purchase', 'buy')

    @property
    def is_sell(self) -> bool:
        return self.transaction_type.lower() in ('sale', 'sell', 'sale (full)', 'sale (partial)')

    @property
    def estimated_value(self) -> float:
        """Return midpoint of amount range."""
        return (self.amount_min + self.amount_max) / 2


@dataclass
class CongressionalTradeSummary:
    """Aggregated summary of congressional trading activity for a symbol."""
    symbol: str
    total_trades: int
    buy_count: int
    sell_count: int
    net_activity: str  # 'NET_BUY', 'NET_SELL', 'NEUTRAL'
    total_buy_value: float
    total_sell_value: float
    unique_representatives: int
    notable_officials: List[str]
    party_breakdown: Dict[str, int]  # {'R': x, 'D': y, 'I': z}
    recent_trades: List[CongressionalTrade]
    data_source: str  # 'quiver' or 'simulated'
    as_of_date: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "total_trades": self.total_trades,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "net_activity": self.net_activity,
            "total_buy_value": self.total_buy_value,
            "total_sell_value": self.total_sell_value,
            "unique_representatives": self.unique_representatives,
            "notable_officials": self.notable_officials,
            "party_breakdown": self.party_breakdown,
            "recent_trades": [t.to_dict() for t in self.recent_trades],
            "data_source": self.data_source,
            "as_of_date": self.as_of_date.isoformat(),
        }


def _parse_amount_range(amount_str: str) -> tuple:
    """
    Parse Quiver amount range string to (min, max) tuple.

    Examples:
        "$1,001 - $15,000" -> (1001.0, 15000.0)
        "$50,001 - $100,000" -> (50001.0, 100000.0)
        "$1,000,001 - $5,000,000" -> (1000001.0, 5000000.0)
    """
    try:
        # Remove $ and commas, split by -
        cleaned = amount_str.replace('$', '').replace(',', '')
        parts = cleaned.split('-')
        if len(parts) == 2:
            return float(parts[0].strip()), float(parts[1].strip())
        else:
            # Single value (e.g., "$15,000+")
            val = float(parts[0].strip().replace('+', ''))
            return val, val * 2  # Estimate upper bound
    except (ValueError, AttributeError):
        return 0.0, 0.0


class CongressionalTradesClient:
    """
    Fetches congressional trading data from Quiver Quant API.

    Uses Quiver Quant API for real-time data with fallback to simulated
    data when API is unavailable or for testing purposes.
    """

    def __init__(self, use_real_api: bool = True):
        """
        Initialize the CongressionalTradesClient.

        Args:
            use_real_api: If True, try to use Quiver Quant API first.
                          If False, always use simulated data.
        """
        self._use_real_api = use_real_api
        self._api_key = os.getenv("QUIVER_API_KEY")
        self._api_available = bool(self._api_key)

        if self._api_available and self._use_real_api:
            logger.info("CongressionalTradesClient initialized with Quiver Quant API.")
        else:
            if not self._api_available:
                logger.warning("QUIVER_API_KEY not found. Using simulated congressional data.")
            else:
                logger.info("CongressionalTradesClient initialized with simulated data (API disabled).")

    def _rate_limit(self) -> None:
        """Enforce rate limiting for API requests."""
        global _last_request_time
        min_interval = 60.0 / QUIVER_RATE_LIMIT
        elapsed = time.time() - _last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_request_time = time.time()

    def _fetch_from_quiver(
        self,
        symbol: str,
        days_back: int = 90,
    ) -> List[CongressionalTrade]:
        """
        Fetch congressional trades from Quiver Quant API.

        Args:
            symbol: Stock symbol (e.g., 'PLTR')
            days_back: Number of days to look back

        Returns:
            List of CongressionalTrade objects

        Raises:
            requests.RequestException: If the API request fails
        """
        self._rate_limit()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

        # Quiver endpoint for congressional trading
        url = f"{QUIVER_BASE_URL}/historical/congresstrading/{symbol}"

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        trades = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        for item in data:
            try:
                # Parse transaction date
                trans_date_str = item.get('TransactionDate', '')
                disclosure_date_str = item.get('DisclosureDate', '')

                trans_date = datetime.fromisoformat(trans_date_str) if trans_date_str else datetime.now()
                disclosure_date = datetime.fromisoformat(disclosure_date_str) if disclosure_date_str else datetime.now()

                # Skip if older than cutoff
                if trans_date < cutoff_date:
                    continue

                # Parse amount range
                amount_str = item.get('Amount', '$0 - $0')
                amount_min, amount_max = _parse_amount_range(amount_str)

                trade = CongressionalTrade(
                    transaction_id=f"{item.get('Representative', 'UNK')}_{trans_date.strftime('%Y%m%d')}_{symbol}",
                    representative=item.get('Representative', 'Unknown'),
                    party=item.get('Party', 'U')[:1],  # First letter: R, D, I
                    house=item.get('House', 'House'),
                    symbol=symbol,
                    transaction_type=item.get('Transaction', 'Unknown'),
                    transaction_date=trans_date,
                    disclosure_date=disclosure_date,
                    amount_range=amount_str,
                    amount_min=amount_min,
                    amount_max=amount_max,
                    asset_description=item.get('Asset', ''),
                    cap_gains_over_200=item.get('CapGainsOver200', False),
                )
                trades.append(trade)

            except Exception as e:
                logger.warning(f"Failed to parse congressional trade: {e}")
                continue

        logger.info(f"Fetched {len(trades)} congressional trades for {symbol} from Quiver API")
        return trades

    def _get_simulated_trades(
        self,
        symbol: str,
        days_back: int = 90,
    ) -> List[CongressionalTrade]:
        """
        Generate simulated congressional trades for testing.

        Returns realistic-looking data based on common patterns.
        """
        now = datetime.now()
        trades = []

        # Simulated trade patterns for common symbols
        simulated_data = {
            'PLTR': [
                ('Rep. Nancy Pelosi', 'D', 'House', 'Purchase', '$50,001 - $100,000', 15),
                ('Rep. Dan Crenshaw', 'R', 'House', 'Purchase', '$15,001 - $50,000', 30),
                ('Sen. Tommy Tuberville', 'R', 'Senate', 'Purchase', '$100,001 - $250,000', 45),
            ],
            'NVDA': [
                ('Rep. Michael McCaul', 'R', 'House', 'Purchase', '$250,001 - $500,000', 10),
                ('Rep. Nancy Pelosi', 'D', 'House', 'Sale', '$500,001 - $1,000,000', 20),
                ('Sen. Sheldon Whitehouse', 'D', 'Senate', 'Purchase', '$15,001 - $50,000', 35),
            ],
            'AAPL': [
                ('Rep. Kevin Hern', 'R', 'House', 'Purchase', '$1,001 - $15,000', 5),
                ('Sen. Richard Burr', 'R', 'Senate', 'Sale', '$50,001 - $100,000', 25),
            ],
            'TSLA': [
                ('Rep. Ro Khanna', 'D', 'House', 'Purchase', '$15,001 - $50,000', 12),
                ('Rep. Josh Gottheimer', 'D', 'House', 'Sale', '$50,001 - $100,000', 40),
            ],
            'MSFT': [
                ('Sen. Mark Warner', 'D', 'Senate', 'Purchase', '$100,001 - $250,000', 8),
                ('Rep. French Hill', 'R', 'House', 'Purchase', '$15,001 - $50,000', 22),
            ],
        }

        # Get trades for this symbol, or generate random ones
        symbol_trades = simulated_data.get(symbol, [
            ('Rep. Generic Member', 'D', 'House', 'Purchase', '$1,001 - $15,000', 30),
        ])

        for rep, party, house, trans_type, amount, days_ago in symbol_trades:
            if days_ago <= days_back:
                amount_min, amount_max = _parse_amount_range(amount)
                trade = CongressionalTrade(
                    transaction_id=f"sim_{rep.replace(' ', '_')}_{symbol}_{days_ago}",
                    representative=rep,
                    party=party,
                    house=house,
                    symbol=symbol,
                    transaction_type=trans_type,
                    transaction_date=now - timedelta(days=days_ago),
                    disclosure_date=now - timedelta(days=max(0, days_ago - 45)),  # Disclosure delay
                    amount_range=amount,
                    amount_min=amount_min,
                    amount_max=amount_max,
                    asset_description=f"{symbol} Common Stock",
                    cap_gains_over_200=False,
                )
                trades.append(trade)

        logger.debug(f"Generated {len(trades)} simulated congressional trades for {symbol}")
        return trades

    def fetch_recent_trades(
        self,
        symbol: str,
        days_back: int = 90,
    ) -> List[CongressionalTrade]:
        """
        Fetch recent congressional trades for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'PLTR')
            days_back: Number of days to look back (default 90)

        Returns:
            List of CongressionalTrade objects, sorted by date (most recent first)
        """
        symbol = symbol.upper()

        # Try real API if available
        if self._api_available and self._use_real_api:
            try:
                trades = self._fetch_from_quiver(symbol, days_back)
                return sorted(trades, key=lambda t: t.transaction_date, reverse=True)
            except requests.RequestException as e:
                logger.warning(f"Quiver API request failed: {e}. Falling back to simulated data.")
            except Exception as e:
                logger.error(f"Unexpected error fetching congressional trades: {e}")

        # Fall back to simulated data
        trades = self._get_simulated_trades(symbol, days_back)
        return sorted(trades, key=lambda t: t.transaction_date, reverse=True)

    def get_trade_summary(
        self,
        symbol: str,
        days_back: int = 90,
    ) -> CongressionalTradeSummary:
        """
        Get aggregated summary of congressional trading activity.

        Args:
            symbol: Stock symbol (e.g., 'PLTR')
            days_back: Number of days to look back

        Returns:
            CongressionalTradeSummary with aggregated metrics
        """
        symbol = symbol.upper()
        trades = self.fetch_recent_trades(symbol, days_back)

        buy_count = 0
        sell_count = 0
        total_buy_value = 0.0
        total_sell_value = 0.0
        representatives = set()
        party_breakdown = {'R': 0, 'D': 0, 'I': 0}

        for trade in trades:
            representatives.add(trade.representative)

            if trade.party in party_breakdown:
                party_breakdown[trade.party] += 1

            if trade.is_buy:
                buy_count += 1
                total_buy_value += trade.estimated_value
            elif trade.is_sell:
                sell_count += 1
                total_sell_value += trade.estimated_value

        # Determine net activity
        if total_buy_value > total_sell_value * 1.5:
            net_activity = 'NET_BUY'
        elif total_sell_value > total_buy_value * 1.5:
            net_activity = 'NET_SELL'
        else:
            net_activity = 'NEUTRAL'

        # Identify notable officials (those with large trades)
        notable = []
        for trade in trades:
            if trade.amount_max >= 100000:
                notable.append(trade.representative)
        notable = list(set(notable))[:5]  # Top 5 unique

        data_source = 'quiver' if (self._api_available and self._use_real_api) else 'simulated'

        return CongressionalTradeSummary(
            symbol=symbol,
            total_trades=len(trades),
            buy_count=buy_count,
            sell_count=sell_count,
            net_activity=net_activity,
            total_buy_value=total_buy_value,
            total_sell_value=total_sell_value,
            unique_representatives=len(representatives),
            notable_officials=notable,
            party_breakdown=party_breakdown,
            recent_trades=trades[:10],  # Include top 10 most recent
            data_source=data_source,
        )

    def introspect(self) -> str:
        """Generates an introspection report for the CongressionalTradesClient."""
        api_status = "connected" if (self._api_available and self._use_real_api) else "simulated"
        return (
            "--- Congressional Trades Client Introspection ---\n"
            f"Data source: Quiver Quant API ({api_status})\n"
            "My role is to track congressional trading activity for symbols.\n"
            "I provide insights into government official buying/selling patterns.\n"
            "I support symbol-specific trade summaries with party breakdown."
        )


# Singleton instance
_congressional_client_instance: Optional[CongressionalTradesClient] = None


def get_congressional_client() -> CongressionalTradesClient:
    """Factory function to get the singleton instance of CongressionalTradesClient."""
    global _congressional_client_instance
    if _congressional_client_instance is None:
        _congressional_client_instance = CongressionalTradesClient()
    return _congressional_client_instance
