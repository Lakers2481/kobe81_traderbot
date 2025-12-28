"""
News Processor - Real-Time News and Sentiment Analysis
======================================================

This module is responsible for fetching real-time financial news and performing
sentiment analysis on the headlines and articles. The resulting sentiment scores
can then be integrated into the market context for the Cognitive Brain, allowing
the AI to factor in qualitative news information into its trading decisions.

Features:
- Fetches news from a configured API (e.g., Alpaca News API).
- Performs sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).
- Provides aggregated sentiment scores for specific symbols or the overall market.

Usage:
    from altdata.news_processor import NewsProcessor

    processor = NewsProcessor()

    # Get news for a specific symbol
    aapl_news = processor.fetch_news(symbol='AAPL')
    aapl_sentiment = processor.analyze_sentiment(aapl_news)

    # Get overall market sentiment
    market_news = processor.fetch_news(query='S&P 500')
    market_sentiment = processor.get_overall_sentiment(market_news)

    # This sentiment can then be added to the CognitiveBrain's context.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# VADER is already in requirements.txt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Alpaca API (assuming this is used for brokerage and can also fetch news)
# from alpaca.data.requests import NewsRequest
# from alpaca.data.client import NewsClient

logger = logging.getLogger(__name__)


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
    """
    def __init__(self):
        self._sentiment_analyzer = SentimentIntensityAnalyzer()
        # Initialize Alpaca News Client if API key is available
        # self._alpaca_news_client = NewsClient(api_key=os.getenv("APCA_API_KEY_ID"), secret_key=os.getenv("APCA_API_SECRET_KEY"))
        logger.info("NewsProcessor initialized. Sentiment analyzer ready.")

    def _get_sentiment_scores(self, text: str) -> Dict[str, float]:
        """
        Calculates VADER sentiment scores for a given text.
        Returns a dictionary with 'neg', 'neu', 'pos', 'compound' scores.
        """
        return self._sentiment_analyzer.polarity_scores(text)

    def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 20,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[NewsArticle]:
        """
        Fetches news articles. For now, this is a simulated fetch.
        In a real implementation, this would call an external API.
        """
        logger.info(f"Simulating news fetch for symbols={symbols}, query='{query}'")
        
        # --- SIMULATED NEWS DATA ---
        simulated_news_data = [
            NewsArticle(
                id="1", headline="AAPL beats earnings expectations",
                summary="Apple reported strong Q3 results, exceeding analyst forecasts.",
                symbols=['AAPL'], source="Financial Times",
                created_at=datetime.now() - timedelta(minutes=10)
            ),
            NewsArticle(
                id="2", headline="Market uncertainty rises due to inflation fears",
                summary="Investors are concerned about persistent inflation pressures.",
                symbols=['SPY'], source="Reuters",
                created_at=datetime.now() - timedelta(minutes=30)
            ),
            NewsArticle(
                id="3", headline="TSLA production ramp-up positive for Q4 outlook",
                summary="Tesla's new factory showing strong output, boosting delivery estimates.",
                symbols=['TSLA'], source="Bloomberg",
                created_at=datetime.now() - timedelta(hours=1)
            ),
            NewsArticle(
                id="4", headline="GOOG faces antitrust scrutiny in Europe",
                summary="Google's advertising practices are under investigation.",
                symbols=['GOOG'], source="Wall Street Journal",
                created_at=datetime.now() - timedelta(hours=2)
            ),
            NewsArticle(
                id="5", headline="New tech breakthrough for NVDA chips",
                summary="Nvidia announced a revolutionary new chip architecture.",
                symbols=['NVDA'], source="TechCrunch",
                created_at=datetime.now() - timedelta(hours=3)
            ),
            NewsArticle(
                id="6", headline="Unexpected dip in consumer spending data",
                summary="Retail sales figures came in weaker than anticipated.",
                symbols=['SPY', 'XLY'], source="MarketWatch",
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
        
    def introspect(self) -> str:
        """Generates an introspection report for the NewsProcessor."""
        return (
            "--- News Processor Introspection ---\n"
            "My role is to provide real-time qualitative market insights.\n"
            "I fetch news and analyze its sentiment to inform the CognitiveBrain."
        )


# Singleton instance for the NewsProcessor
_news_processor_instance: Optional[NewsProcessor] = None

def get_news_processor() -> NewsProcessor:
    """Factory function to get the singleton instance of the NewsProcessor."""
    global _news_processor_instance
    if _news_processor_instance is None:
        _news_processor_instance = NewsProcessor()
    return _news_processor_instance
