"""
News Processor - Real-Time News and Sentiment Analysis
======================================================

This module is responsible for fetching real-time financial news and performing
sentiment analysis on the headlines and articles. The resulting sentiment scores
can then be integrated into the market context for the Cognitive Brain, allowing
the AI to factor in qualitative news information into its trading decisions.

Features:
- Fetches news from Alpaca News API (real-time, production-grade).
- Falls back to simulated data when API unavailable.
- Performs sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).
- Provides aggregated sentiment scores for specific symbols or the overall market.

Usage:
    from altdata.news_processor import NewsProcessor

    processor = NewsProcessor()

    # Get news for a specific symbol
    aapl_news = processor.fetch_news(symbols=['AAPL'])
    aapl_sentiment = processor.get_aggregated_sentiment(symbols=['AAPL'])

    # Get overall market sentiment
    market_sentiment = processor.get_aggregated_sentiment()

    # This sentiment can then be added to the CognitiveBrain's context.
"""

import logging
import os
import requests
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import time

# VADER is already in requirements.txt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Alpaca News API configuration
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
ALPACA_NEWS_RATE_LIMIT = 200  # requests per minute
_last_request_time: float = 0.0


@dataclass
class NewsArticle:
    """Represents a single news article."""
    id: str
    headline: str
    summary: Optional[str] = None
    url: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    symbols: List[str] = field(default_factory=list)
    author: Optional[str] = None
    source: Optional[str] = None
    sentiment_score: Dict[str, float] = field(default_factory=dict) # VADER scores

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "url": self.url,
            "created_at": self.created_at.isoformat(),
            "symbols": self.symbols,
            "author": self.author,
            "source": self.source,
            "sentiment_score": self.sentiment_score,
        }


class NewsProcessor:
    """
    Fetches financial news and performs sentiment analysis.

    Uses Alpaca News API for real-time news with fallback to simulated data
    when API is unavailable or for testing purposes.
    """
    def __init__(self, use_real_api: bool = True):
        """
        Initialize the NewsProcessor.

        Args:
            use_real_api: If True, try to use Alpaca News API first.
                          If False, always use simulated data.
        """
        self._sentiment_analyzer = SentimentIntensityAnalyzer()
        self._use_real_api = use_real_api

        # Alpaca API credentials from environment
        self._api_key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
        self._api_secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

        # Check if API is available
        self._api_available = bool(self._api_key and self._api_secret)
        if self._api_available and self._use_real_api:
            logger.info("NewsProcessor initialized with Alpaca News API.")
        else:
            if not self._api_available:
                logger.warning("Alpaca API keys not found. NewsProcessor will use simulated data.")
            else:
                logger.info("NewsProcessor initialized with simulated data (API disabled).")

    def _get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Calculates VADER sentiment scores for a given text.
        Returns a dictionary with 'neg', 'neu', 'pos', 'compound' scores.
        """
        return self._sentiment_analyzer.polarity_scores(text)

    def _rate_limit(self) -> None:
        """Enforce rate limiting for API requests."""
        global _last_request_time
        min_interval = 60.0 / ALPACA_NEWS_RATE_LIMIT  # seconds between requests
        elapsed = time.time() - _last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_request_time = time.time()

    def _fetch_from_alpaca(
        self,
        symbols: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 20,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[NewsArticle]:
        """
        Fetch news from Alpaca News API.

        Raises:
            requests.RequestException: If the API request fails.
        """
        self._rate_limit()

        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
        }

        params: Dict[str, Any] = {"limit": min(limit, 50)}  # Alpaca max is 50

        if symbols:
            params["symbols"] = ",".join(symbols)

        if start_date:
            params["start"] = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        if end_date:
            params["end"] = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        response = requests.get(ALPACA_NEWS_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        articles = []

        for item in data.get("news", []):
            # Parse created_at timestamp
            created_str = item.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                # Convert to naive datetime for consistency
                created_at = created_at.replace(tzinfo=None)
            except (ValueError, AttributeError):
                created_at = datetime.now()

            article = NewsArticle(
                id=item.get("id", ""),
                headline=item.get("headline", ""),
                summary=item.get("summary", ""),
                url=item.get("url", ""),
                created_at=created_at,
                symbols=item.get("symbols", []),
                author=item.get("author", ""),
                source=item.get("source", ""),
            )

            # Apply sentiment analysis
            full_text = f"{article.headline}. {article.summary or ''}"
            article.sentiment_score = self._get_sentiment_scores(full_text)
            articles.append(article)

        logger.info(f"Fetched {len(articles)} articles from Alpaca News API")
        return articles

    def _get_simulated_news(
        self,
        symbols: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 20,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[NewsArticle]:
        """Get simulated news data for testing or when API is unavailable."""
        simulated_news_data = [
            NewsArticle(
                id="sim_1", headline="AAPL beats earnings expectations",
                summary="Apple reported strong Q3 results, exceeding analyst forecasts.",
                symbols=['AAPL'], source="Simulated",
                created_at=datetime.now() - timedelta(minutes=10)
            ),
            NewsArticle(
                id="sim_2", headline="Market uncertainty rises due to inflation fears",
                summary="Investors are concerned about persistent inflation pressures.",
                symbols=['SPY'], source="Simulated",
                created_at=datetime.now() - timedelta(minutes=30)
            ),
            NewsArticle(
                id="sim_3", headline="TSLA production ramp-up positive for Q4 outlook",
                summary="Tesla's new factory showing strong output, boosting delivery estimates.",
                symbols=['TSLA'], source="Simulated",
                created_at=datetime.now() - timedelta(hours=1)
            ),
            NewsArticle(
                id="sim_4", headline="GOOG faces antitrust scrutiny in Europe",
                summary="Google's advertising practices are under investigation.",
                symbols=['GOOG'], source="Simulated",
                created_at=datetime.now() - timedelta(hours=2)
            ),
            NewsArticle(
                id="sim_5", headline="New tech breakthrough for NVDA chips",
                summary="Nvidia announced a revolutionary new chip architecture.",
                symbols=['NVDA'], source="Simulated",
                created_at=datetime.now() - timedelta(hours=3)
            ),
            NewsArticle(
                id="sim_6", headline="Unexpected dip in consumer spending data",
                summary="Retail sales figures came in weaker than anticipated.",
                symbols=['SPY', 'XLY'], source="Simulated",
                created_at=datetime.now() - timedelta(hours=4)
            ),
        ]

        filtered_news = []
        for article in simulated_news_data:
            # Filter by symbols
            if symbols and not any(s in article.symbols for s in symbols):
                continue
            # Filter by query in headline/summary
            if query and query.lower() not in article.headline.lower() and \
               (article.summary and query.lower() not in article.summary.lower()):
                continue
            # Filter by date range
            if start_date and article.created_at < start_date:
                continue
            if end_date and article.created_at > end_date:
                continue
            filtered_news.append(article)

        # Apply sentiment analysis to the fetched articles
        for article in filtered_news:
            full_text = f"{article.headline}. {article.summary or ''}"
            article.sentiment_score = self._get_sentiment_scores(full_text)

        return filtered_news[:limit]

    def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 20,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[NewsArticle]:
        """
        Fetches news articles from Alpaca News API with fallback to simulated data.

        Args:
            symbols: List of stock symbols to filter by (e.g., ['AAPL', 'MSFT']).
            query: Text query to search for (not supported by Alpaca, used for simulated only).
            limit: Maximum number of articles to return.
            start_date: Start of date range to filter articles.
            end_date: End of date range to filter articles.

        Returns:
            List of NewsArticle objects with sentiment scores.
        """
        # Try real API if available and enabled
        if self._api_available and self._use_real_api:
            try:
                return self._fetch_from_alpaca(symbols, query, limit, start_date, end_date)
            except requests.RequestException as e:
                logger.warning(f"Alpaca News API request failed: {e}. Falling back to simulated data.")
            except Exception as e:
                logger.error(f"Unexpected error fetching news: {e}. Falling back to simulated data.")

        # Fall back to simulated data
        logger.debug(f"Using simulated news for symbols={symbols}, query='{query}'")
        return self._get_simulated_news(symbols, query, limit, start_date, end_date)

    def get_aggregated_sentiment(
        self,
        symbols: Optional[List[str]] = None,
        lookback_minutes: int = 60,
    ) -> Dict[str, float]:
        """
        Fetches recent news for given symbols and returns an aggregated sentiment score.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(minutes=lookback_minutes)
        
        news_articles = self.fetch_news(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            limit=50 # Fetch more to aggregate
        )
        
        if not news_articles:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        compound_scores = [a.sentiment_score.get('compound', 0.0) for a in news_articles]
        pos_scores = [a.sentiment_score.get('pos', 0.0) for a in news_articles]
        neg_scores = [a.sentiment_score.get('neg', 0.0) for a in news_articles]
        neu_scores = [a.sentiment_score.get('neu', 0.0) for a in news_articles]

        return {
            'compound': sum(compound_scores) / len(compound_scores),
            'positive': sum(pos_scores) / len(pos_scores),
            'negative': sum(neg_scores) / len(neg_scores),
            'neutral': sum(neu_scores) / len(neu_scores),
        }

    def get_narrative_interpretation(
        self,
        symbols: Optional[List[str]] = None,
        lookback_minutes: int = 60,
    ) -> Dict[str, Any]:
        """
        Get LLM narrative interpretation of recent news using Claude's human-like reasoning.

        This method combines:
        1. Recent news articles for the specified symbols
        2. VADER sentiment scores as quantitative baseline
        3. Claude LLM analysis for qualitative interpretation

        Args:
            symbols: List of symbols to focus on (optional)
            lookback_minutes: How far back to look for news

        Returns:
            Dict containing:
            - interpretation: Human-readable narrative (from Claude or deterministic)
            - articles: List of raw article dicts
            - aggregated_sentiment: VADER aggregated scores
            - generation_method: "claude" or "deterministic"
        """
        # Fetch recent news articles
        end_date = datetime.now()
        start_date = end_date - timedelta(minutes=lookback_minutes)

        articles = self.fetch_news(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            limit=20,
        )

        # Get aggregated VADER sentiment
        aggregated = self.get_aggregated_sentiment(symbols=symbols, lookback_minutes=lookback_minutes)

        if not articles:
            return {
                'interpretation': 'No recent news articles available for analysis.',
                'articles': [],
                'aggregated_sentiment': aggregated,
                'generation_method': 'none',
            }

        # Try to use LLM analyzer for narrative interpretation
        interpretation = ""
        generation_method = "deterministic"

        try:
            from cognitive.llm_trade_analyzer import get_trade_analyzer
            analyzer = get_trade_analyzer()

            # Convert articles to dicts for the analyzer
            article_dicts = [a.to_dict() for a in articles]

            interpretation = analyzer.interpret_sentiment(
                articles=article_dicts,
                aggregated_sentiment=aggregated,
                symbols=symbols,
            )

            if analyzer.api_available:
                generation_method = "claude"

        except ImportError:
            logger.debug("LLM trade analyzer not available, using deterministic interpretation")
            interpretation = self._deterministic_interpretation(articles, aggregated)
        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e}, using deterministic fallback")
            interpretation = self._deterministic_interpretation(articles, aggregated)

        return {
            'interpretation': interpretation,
            'articles': [a.to_dict() for a in articles],
            'aggregated_sentiment': aggregated,
            'generation_method': generation_method,
        }

    def _deterministic_interpretation(
        self,
        articles: List[NewsArticle],
        aggregated: Dict[str, float],
    ) -> str:
        """
        Generate a deterministic narrative interpretation when LLM is unavailable.

        Uses template-based generation based on sentiment scores and article content.
        """
        compound = aggregated.get('compound', 0.0)
        article_count = len(articles)

        # Determine overall tone
        if compound > 0.3:
            tone = "predominantly positive"
            impact = "supportive of bullish positioning"
        elif compound > 0.1:
            tone = "mildly positive"
            impact = "slightly supportive of risk-on trades"
        elif compound < -0.3:
            tone = "predominantly negative"
            impact = "suggesting caution and defensive positioning"
        elif compound < -0.1:
            tone = "mildly negative"
            impact = "warranting some caution on new longs"
        else:
            tone = "neutral to mixed"
            impact = "unlikely to significantly drive market direction"

        # Extract key headlines
        key_headlines = []
        for article in articles[:3]:
            headline = article.headline[:60]
            sent = article.sentiment_score.get('compound', 0)
            direction = "+" if sent > 0.1 else "-" if sent < -0.1 else "~"
            key_headlines.append(f"[{direction}] {headline}")

        # Build interpretation
        interpretation_parts = [
            f"News flow ({article_count} articles) is {tone} with aggregate sentiment of {compound:.2f}.",
            f"This reading is {impact}.",
        ]

        if key_headlines:
            interpretation_parts.append("Key headlines:")
            interpretation_parts.extend([f"  {h}" for h in key_headlines])

        # Add symbol-specific notes if articles mention specific tickers
        mentioned_symbols = set()
        for article in articles:
            mentioned_symbols.update(article.symbols)

        if mentioned_symbols:
            symbols_str = ", ".join(list(mentioned_symbols)[:5])
            interpretation_parts.append(f"Symbols in focus: {symbols_str}")

        return "\n".join(interpretation_parts)

    def introspect(self) -> str:
        """Generates an introspection report for the NewsProcessor."""
        api_status = "connected" if (self._api_available and self._use_real_api) else "simulated"
        return (
            "--- News Processor Introspection ---\n"
            f"Data source: Alpaca News API ({api_status})\n"
            "My role is to provide real-time qualitative market insights.\n"
            "I fetch news and analyze its sentiment using VADER to inform the CognitiveBrain.\n"
            "I support symbol-specific and market-wide sentiment aggregation."
        )


# Singleton instance for the NewsProcessor
_news_processor_instance: Optional[NewsProcessor] = None

def get_news_processor() -> NewsProcessor:
    """Factory function to get the singleton instance of the NewsProcessor."""
    global _news_processor_instance
    if _news_processor_instance is None:
        _news_processor_instance = NewsProcessor()
    return _news_processor_instance
