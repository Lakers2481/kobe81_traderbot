"""
Options Chain Fetcher for Live Trading.

Fetches real-time options chains from data providers:
- Polygon.io (primary)
- Alpaca (fallback)

Provides:
- Full options chain for a symbol
- Filtered chains by expiration, strike, delta
- Greeks from market data
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Single option contract data."""
    # Identification
    symbol: str               # Underlying symbol
    contract_symbol: str      # Full OCC symbol (e.g., "AAPL230120C00150000")
    option_type: OptionType
    expiration: date
    strike: float

    # Pricing
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    mid: Optional[float] = None
    volume: int = 0
    open_interest: int = 0

    # Greeks (may be None if not available)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None

    # Underlying info
    underlying_price: Optional[float] = None

    # Metadata
    timestamp: Optional[datetime] = None
    source: str = "polygon"

    def __post_init__(self):
        """Calculate mid price if not set."""
        if self.mid is None and self.bid is not None and self.ask is not None:
            self.mid = (self.bid + self.ask) / 2

    @property
    def days_to_expiration(self) -> int:
        """Days until expiration."""
        return (self.expiration - date.today()).days

    @property
    def is_itm(self) -> bool:
        """Check if option is in-the-money."""
        if self.underlying_price is None:
            return False
        if self.option_type == OptionType.CALL:
            return self.underlying_price > self.strike
        else:
            return self.underlying_price < self.strike

    @property
    def moneyness(self) -> Optional[float]:
        """Calculate moneyness (S/K for calls, K/S for puts)."""
        if self.underlying_price is None or self.strike == 0:
            return None
        if self.option_type == OptionType.CALL:
            return self.underlying_price / self.strike
        else:
            return self.strike / self.underlying_price

    @property
    def spread_pct(self) -> Optional[float]:
        """Bid-ask spread as percentage of mid."""
        if self.bid is None or self.ask is None or self.mid is None or self.mid == 0:
            return None
        return ((self.ask - self.bid) / self.mid) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "contract_symbol": self.contract_symbol,
            "option_type": self.option_type.value,
            "expiration": self.expiration.isoformat(),
            "strike": self.strike,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "mid": self.mid,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "implied_volatility": self.implied_volatility,
            "underlying_price": self.underlying_price,
            "days_to_expiration": self.days_to_expiration,
            "is_itm": self.is_itm,
        }


@dataclass
class OptionsChain:
    """Complete options chain for a symbol."""
    symbol: str
    underlying_price: float
    timestamp: datetime
    calls: List[OptionContract] = field(default_factory=list)
    puts: List[OptionContract] = field(default_factory=list)
    expirations: List[date] = field(default_factory=list)
    source: str = "polygon"

    @property
    def all_contracts(self) -> List[OptionContract]:
        """Get all contracts (calls and puts)."""
        return self.calls + self.puts

    def get_expiration(self, target_dte: int) -> Optional[date]:
        """
        Find expiration closest to target DTE.

        Args:
            target_dte: Target days to expiration

        Returns:
            Closest expiration date or None
        """
        if not self.expirations:
            return None

        today = date.today()
        closest = None
        min_diff = float("inf")

        for exp in self.expirations:
            dte = (exp - today).days
            diff = abs(dte - target_dte)
            if diff < min_diff:
                min_diff = diff
                closest = exp

        return closest

    def get_contracts_by_expiration(
        self,
        expiration: date,
        option_type: Optional[OptionType] = None,
    ) -> List[OptionContract]:
        """Get contracts for a specific expiration."""
        contracts = self.all_contracts if option_type is None else (
            self.calls if option_type == OptionType.CALL else self.puts
        )
        return [c for c in contracts if c.expiration == expiration]

    def get_atm_strike(self, expiration: date) -> Optional[float]:
        """Get at-the-money strike for an expiration."""
        contracts = self.get_contracts_by_expiration(expiration)
        if not contracts:
            return None

        strikes = sorted(set(c.strike for c in contracts))
        # Find strike closest to underlying price
        closest = min(strikes, key=lambda s: abs(s - self.underlying_price))
        return closest

    def get_contract_by_delta(
        self,
        expiration: date,
        option_type: OptionType,
        target_delta: float,
    ) -> Optional[OptionContract]:
        """
        Find contract closest to target delta.

        Args:
            expiration: Target expiration
            option_type: Call or put
            target_delta: Target delta (e.g., 0.30 for 30-delta)

        Returns:
            Contract closest to target delta
        """
        contracts = self.get_contracts_by_expiration(expiration, option_type)
        contracts_with_delta = [c for c in contracts if c.delta is not None]

        if not contracts_with_delta:
            return None

        # For puts, delta is negative, so we compare absolute values
        if option_type == OptionType.PUT:
            target_delta = -abs(target_delta)

        closest = min(
            contracts_with_delta,
            key=lambda c: abs((c.delta or 0) - target_delta),
        )
        return closest

    def filter_by_liquidity(
        self,
        min_volume: int = 10,
        min_open_interest: int = 100,
        max_spread_pct: float = 20.0,
    ) -> "OptionsChain":
        """
        Filter chain to liquid contracts only.

        Returns:
            New OptionsChain with only liquid contracts
        """
        def is_liquid(c: OptionContract) -> bool:
            if c.volume < min_volume:
                return False
            if c.open_interest < min_open_interest:
                return False
            if c.spread_pct is not None and c.spread_pct > max_spread_pct:
                return False
            return True

        return OptionsChain(
            symbol=self.symbol,
            underlying_price=self.underlying_price,
            timestamp=self.timestamp,
            calls=[c for c in self.calls if is_liquid(c)],
            puts=[c for c in self.puts if is_liquid(c)],
            expirations=self.expirations,
            source=self.source,
        )


class ChainFetcher:
    """
    Fetches options chains from data providers.

    Primary: Polygon.io
    Fallback: Alpaca (if available)
    """

    def __init__(
        self,
        polygon_api_key: Optional[str] = None,
        alpaca_api_key: Optional[str] = None,
        alpaca_secret: Optional[str] = None,
    ):
        """
        Initialize chain fetcher.

        Args:
            polygon_api_key: Polygon.io API key (or from POLYGON_API_KEY env)
            alpaca_api_key: Alpaca API key (or from ALPACA_API_KEY_ID env)
            alpaca_secret: Alpaca secret (or from ALPACA_API_SECRET_KEY env)
        """
        self.polygon_api_key = polygon_api_key or os.getenv("POLYGON_API_KEY", "")
        self.alpaca_api_key = alpaca_api_key or os.getenv("ALPACA_API_KEY_ID", "")
        self.alpaca_secret = alpaca_secret or os.getenv("ALPACA_API_SECRET_KEY", "")

        self._polygon_base = "https://api.polygon.io"
        self._alpaca_base = "https://data.alpaca.markets"

    def fetch_chain(
        self,
        symbol: str,
        expiration_range_days: int = 60,
        min_dte: int = 1,
        strike_range_pct: float = 0.20,  # +/- 20% from current price
        include_greeks: bool = True,
    ) -> Optional[OptionsChain]:
        """
        Fetch options chain for a symbol.

        Args:
            symbol: Underlying symbol
            expiration_range_days: Max days to expiration to include
            min_dte: Minimum days to expiration
            strike_range_pct: Strike range as percentage of underlying price
            include_greeks: Whether to fetch Greeks

        Returns:
            OptionsChain or None if fetch fails
        """
        # Try Polygon first
        if self.polygon_api_key:
            chain = self._fetch_polygon_chain(
                symbol,
                expiration_range_days,
                min_dte,
                strike_range_pct,
                include_greeks,
            )
            if chain:
                return chain

        # Fallback to Alpaca
        if self.alpaca_api_key:
            chain = self._fetch_alpaca_chain(
                symbol,
                expiration_range_days,
                min_dte,
            )
            if chain:
                return chain

        logger.error(f"Failed to fetch options chain for {symbol}")
        return None

    def _fetch_polygon_chain(
        self,
        symbol: str,
        expiration_range_days: int,
        min_dte: int,
        strike_range_pct: float,
        include_greeks: bool,
    ) -> Optional[OptionsChain]:
        """Fetch chain from Polygon.io."""
        try:
            # First get underlying price
            underlying_price = self._get_polygon_price(symbol)
            if underlying_price is None:
                return None

            # Calculate date range
            today = date.today()
            exp_from = today + timedelta(days=min_dte)
            exp_to = today + timedelta(days=expiration_range_days)

            # Calculate strike range
            min_strike = underlying_price * (1 - strike_range_pct)
            max_strike = underlying_price * (1 + strike_range_pct)

            # Fetch options contracts
            url = f"{self._polygon_base}/v3/reference/options/contracts"
            params = {
                "underlying_ticker": symbol.upper(),
                "expiration_date.gte": exp_from.isoformat(),
                "expiration_date.lte": exp_to.isoformat(),
                "strike_price.gte": min_strike,
                "strike_price.lte": max_strike,
                "limit": 1000,
                "apiKey": self.polygon_api_key,
            }

            response = requests.get(url, params=params, timeout=30)
            if response.status_code != 200:
                logger.error(f"Polygon options API error: {response.status_code}")
                return None

            data = response.json()
            results = data.get("results", [])

            if not results:
                logger.warning(f"No options contracts found for {symbol}")
                return None

            # Parse contracts
            calls = []
            puts = []
            expirations = set()

            for contract in results:
                option_type = OptionType.CALL if contract.get("contract_type") == "call" else OptionType.PUT
                exp_date = datetime.strptime(contract.get("expiration_date"), "%Y-%m-%d").date()
                expirations.add(exp_date)

                opt = OptionContract(
                    symbol=symbol.upper(),
                    contract_symbol=contract.get("ticker", ""),
                    option_type=option_type,
                    expiration=exp_date,
                    strike=contract.get("strike_price", 0),
                    underlying_price=underlying_price,
                    source="polygon",
                )

                # Fetch quote if available
                self._fetch_polygon_option_quote(opt)

                if option_type == OptionType.CALL:
                    calls.append(opt)
                else:
                    puts.append(opt)

            # Fetch Greeks if requested (snapshot endpoint)
            if include_greeks:
                self._enrich_with_greeks(symbol, calls + puts)

            return OptionsChain(
                symbol=symbol.upper(),
                underlying_price=underlying_price,
                timestamp=datetime.utcnow(),
                calls=sorted(calls, key=lambda c: (c.expiration, c.strike)),
                puts=sorted(puts, key=lambda c: (c.expiration, c.strike)),
                expirations=sorted(expirations),
                source="polygon",
            )

        except Exception as e:
            logger.error(f"Polygon chain fetch error: {e}")
            return None

    def _get_polygon_price(self, symbol: str) -> Optional[float]:
        """Get current underlying price from Polygon."""
        try:
            url = f"{self._polygon_base}/v2/aggs/ticker/{symbol.upper()}/prev"
            params = {"apiKey": self.polygon_api_key}
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results:
                    return results[0].get("c")  # Close price
        except Exception as e:
            logger.debug(f"Polygon price fetch error: {e}")

        return None

    def _fetch_polygon_option_quote(self, contract: OptionContract) -> None:
        """Fetch quote for a single option contract."""
        try:
            url = f"{self._polygon_base}/v3/quotes/{contract.contract_symbol}"
            params = {"apiKey": self.polygon_api_key, "limit": 1}
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results:
                    quote = results[0]
                    contract.bid = quote.get("bid_price")
                    contract.ask = quote.get("ask_price")
                    contract.timestamp = datetime.utcnow()
        except Exception:
            pass

    def _enrich_with_greeks(self, symbol: str, contracts: List[OptionContract]) -> None:
        """Fetch Greeks via Polygon snapshot endpoint."""
        try:
            url = f"{self._polygon_base}/v3/snapshot/options/{symbol.upper()}"
            params = {"apiKey": self.polygon_api_key, "limit": 1000}
            response = requests.get(url, params=params, timeout=30)

            if response.status_code != 200:
                return

            data = response.json()
            results = data.get("results", [])

            # Build lookup by contract symbol
            snapshot_map = {}
            for r in results:
                details = r.get("details", {})
                ticker = details.get("ticker")
                if ticker:
                    snapshot_map[ticker] = r

            # Enrich contracts
            for contract in contracts:
                snapshot = snapshot_map.get(contract.contract_symbol)
                if snapshot:
                    greeks = snapshot.get("greeks", {})
                    contract.delta = greeks.get("delta")
                    contract.gamma = greeks.get("gamma")
                    contract.theta = greeks.get("theta")
                    contract.vega = greeks.get("vega")
                    contract.implied_volatility = snapshot.get("implied_volatility")

                    # Update pricing from snapshot
                    quote = snapshot.get("last_quote", {})
                    if quote:
                        contract.bid = quote.get("bid")
                        contract.ask = quote.get("ask")
                        contract.mid = quote.get("midpoint")

                    contract.volume = snapshot.get("day", {}).get("volume", 0)
                    contract.open_interest = snapshot.get("open_interest", 0)

        except Exception as e:
            logger.debug(f"Greeks enrichment error: {e}")

    def _fetch_alpaca_chain(
        self,
        symbol: str,
        expiration_range_days: int,
        min_dte: int,
    ) -> Optional[OptionsChain]:
        """Fetch chain from Alpaca (simplified, as backup)."""
        # Note: Alpaca options API may require options trading approval
        logger.debug("Alpaca options chain fetch not implemented")
        return None

    def get_available_expirations(
        self,
        symbol: str,
        max_days: int = 90,
    ) -> List[date]:
        """Get available expiration dates for a symbol."""
        chain = self.fetch_chain(
            symbol,
            expiration_range_days=max_days,
            strike_range_pct=0.01,  # Minimal strike range to reduce data
            include_greeks=False,
        )
        return chain.expirations if chain else []


# Default fetcher instance
def get_chain_fetcher() -> ChainFetcher:
    """Get default chain fetcher instance."""
    return ChainFetcher()
