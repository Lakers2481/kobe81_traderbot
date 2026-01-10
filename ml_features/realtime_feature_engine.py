"""
Real-Time Feature Engine - Panopticon Data Matrix

Aggregates features from multiple alternative data sources for
real-time signal enhancement and intraday trigger decisions.

Data Sources:
- GitHub Activity (developer sentiment)
- Political Sentiment (regulatory/legislative)
- Social Media (market mood)
- Options Flow (unusual activity)
- Insider Activity (SEC filings)
- News Sentiment (real-time headlines)

Author: Kobe Trading System
Date: 2026-01-07
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CACHE_FILE = ROOT / "state" / "realtime_features.json"


@dataclass
class FeatureSnapshot:
    """Real-time feature snapshot for a symbol."""
    symbol: str
    timestamp: str

    # Technical features (from EOD)
    price: float = 0.0
    atr: float = 0.0
    rsi: float = 50.0
    volume_ratio: float = 1.0

    # GitHub features
    github_activity_score: float = 0.0
    github_sentiment: float = 0.0
    github_trend: str = "stable"

    # Political features
    political_direction: str = "NEUTRAL"
    political_strength: float = 0.0
    regulatory_risk: float = 0.0

    # Social features
    social_sentiment: float = 0.0
    social_volume: float = 0.0
    social_trend: str = "stable"

    # Options features
    put_call_ratio: float = 1.0
    unusual_volume: bool = False
    smart_money_flow: str = "neutral"

    # Insider features
    insider_buying: bool = False
    insider_selling: bool = False
    insider_net_shares: int = 0

    # News features
    news_sentiment: float = 0.0
    news_volume_24h: int = 0
    breaking_news: bool = False

    # Aggregate scores
    alt_data_score: float = 0.0  # Combined alt data signal
    confidence_boost: float = 0.0  # Suggested confidence adjustment

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TriggerRecommendation:
    """Intraday trigger recommendation."""
    symbol: str
    action: str  # "EXECUTE", "WAIT", "SKIP"
    confidence: float
    reasoning: List[str]
    features: Dict[str, Any]


class RealtimeFeatureEngine:
    """
    Aggregates real-time features from multiple alternative data sources.

    Provides:
    1. Feature snapshots for any symbol
    2. Intraday trigger recommendations
    3. Confidence adjustments based on alt data alignment
    """

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self._cache: Dict[str, FeatureSnapshot] = {}
        self._feature_providers: Dict[str, Callable] = {}
        self._register_providers()
        self._load_cache()

    def _register_providers(self) -> None:
        """Register feature provider functions."""
        # These will call actual provider modules when available
        self._feature_providers = {
            "github": self._get_github_features,
            "political": self._get_political_features,
            "social": self._get_social_features,
            "options": self._get_options_features,
            "insider": self._get_insider_features,
            "news": self._get_news_features,
        }

    def _load_cache(self) -> None:
        """Load cached features."""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    for symbol, feat_data in data.get("features", {}).items():
                        self._cache[symbol] = FeatureSnapshot(**feat_data)
            except Exception as e:
                logger.warning(f"Failed to load feature cache: {e}")

    def _save_cache(self) -> None:
        """Save feature cache."""
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "features": {k: v.to_dict() for k, v in self._cache.items()},
            "updated_at": datetime.now().isoformat(),
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_github_features(self, symbol: str) -> Dict[str, Any]:
        """Get GitHub activity features."""
        try:
            from altdata.github_activity import get_github_monitor
            monitor = get_github_monitor()
            activity = monitor.get_company_activity(symbol)
            if activity:
                return {
                    "github_activity_score": activity.get("activity_score", 0),
                    "github_sentiment": activity.get("avg_sentiment", 0),
                    "github_trend": activity.get("dominant_trend", "stable"),
                }
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"GitHub feature error for {symbol}: {e}")
        return {}

    def _get_political_features(self, symbol: str) -> Dict[str, Any]:
        """Get political sentiment features."""
        try:
            from altdata.political_sentiment import get_political_analyzer
            analyzer = get_political_analyzer()
            signal = analyzer.get_signal_for_symbol(symbol)
            if signal:
                return {
                    "political_direction": signal.direction,
                    "political_strength": signal.strength,
                    "regulatory_risk": 0.5 if signal.direction == "BEARISH" else 0.1,
                }
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Political feature error for {symbol}: {e}")
        return {}

    def _get_social_features(self, symbol: str) -> Dict[str, Any]:
        """Get social media sentiment features."""
        try:
            from altdata.sentiment import get_sentiment_analyzer
            analyzer = get_sentiment_analyzer()
            sentiment = analyzer.get_symbol_sentiment(symbol)
            if sentiment:
                return {
                    "social_sentiment": sentiment.get("score", 0),
                    "social_volume": sentiment.get("volume", 0),
                    "social_trend": sentiment.get("trend", "stable"),
                }
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Social feature error for {symbol}: {e}")
        return {}

    def _get_options_features(self, symbol: str) -> Dict[str, Any]:
        """Get options flow features."""
        try:
            from altdata.options_flow import get_options_analyzer
            analyzer = get_options_analyzer()
            flow = analyzer.get_symbol_flow(symbol)
            if flow:
                return {
                    "put_call_ratio": flow.get("put_call_ratio", 1.0),
                    "unusual_volume": flow.get("unusual", False),
                    "smart_money_flow": flow.get("smart_money", "neutral"),
                }
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Options feature error for {symbol}: {e}")
        return {}

    def _get_insider_features(self, symbol: str) -> Dict[str, Any]:
        """Get insider activity features."""
        try:
            from altdata.insider_activity import get_insider_tracker
            tracker = get_insider_tracker()
            activity = tracker.get_recent_activity(symbol, days=30)
            if activity:
                return {
                    "insider_buying": activity.get("net_buying", False),
                    "insider_selling": activity.get("net_selling", False),
                    "insider_net_shares": activity.get("net_shares", 0),
                }
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Insider feature error for {symbol}: {e}")
        return {}

    def _get_news_features(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment features."""
        try:
            from altdata.news_processor import get_news_processor
            processor = get_news_processor()
            news = processor.get_symbol_news(symbol, hours=24)
            if news:
                return {
                    "news_sentiment": news.get("avg_sentiment", 0),
                    "news_volume_24h": news.get("count", 0),
                    "breaking_news": news.get("breaking", False),
                }
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"News feature error for {symbol}: {e}")
        return {}

    def get_feature_snapshot(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> FeatureSnapshot:
        """
        Get real-time feature snapshot for a symbol.

        Args:
            symbol: Trading symbol
            force_refresh: Force refresh even if cached

        Returns:
            FeatureSnapshot with all available features
        """
        # Check cache (5 minute TTL)
        if not force_refresh and symbol in self._cache:
            cached = self._cache[symbol]
            cached_time = datetime.fromisoformat(cached.timestamp)
            if datetime.now() - cached_time < timedelta(minutes=5):
                return cached

        # Create new snapshot
        snapshot = FeatureSnapshot(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
        )

        # Fetch features from all providers in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(provider, symbol): name
                for name, provider in self._feature_providers.items()
            }

            for future in as_completed(futures):
                provider_name = futures[future]
                try:
                    features = future.result(timeout=10)
                    # Update snapshot with returned features
                    for key, value in features.items():
                        if hasattr(snapshot, key):
                            setattr(snapshot, key, value)
                except Exception as e:
                    logger.warning(f"Provider {provider_name} failed for {symbol}: {e}")

        # Calculate aggregate scores
        snapshot.alt_data_score = self._calculate_alt_data_score(snapshot)
        snapshot.confidence_boost = self._calculate_confidence_boost(snapshot)

        # Cache and save
        self._cache[symbol] = snapshot
        self._save_cache()

        return snapshot

    def _calculate_alt_data_score(self, snapshot: FeatureSnapshot) -> float:
        """
        Calculate aggregate alternative data score.

        Returns score from -1 (bearish) to +1 (bullish).
        """
        score = 0.0
        weights = {
            "github": 0.15,
            "political": 0.20,
            "social": 0.15,
            "options": 0.20,
            "insider": 0.15,
            "news": 0.15,
        }

        # GitHub score
        if snapshot.github_activity_score > 60:
            score += weights["github"] * (snapshot.github_sentiment + 0.5)
        elif snapshot.github_activity_score < 40:
            score -= weights["github"] * 0.3

        # Political score
        if snapshot.political_direction == "BULLISH":
            score += weights["political"] * snapshot.political_strength
        elif snapshot.political_direction == "BEARISH":
            score -= weights["political"] * snapshot.political_strength

        # Social score
        score += weights["social"] * snapshot.social_sentiment

        # Options score
        if snapshot.unusual_volume:
            if snapshot.smart_money_flow == "bullish":
                score += weights["options"] * 0.5
            elif snapshot.smart_money_flow == "bearish":
                score -= weights["options"] * 0.5

        # Insider score
        if snapshot.insider_buying:
            score += weights["insider"] * 0.5
        elif snapshot.insider_selling:
            score -= weights["insider"] * 0.3

        # News score
        score += weights["news"] * snapshot.news_sentiment

        return max(-1.0, min(1.0, score))

    def _calculate_confidence_boost(self, snapshot: FeatureSnapshot) -> float:
        """
        Calculate confidence adjustment based on alt data alignment.

        Returns adjustment from -0.15 to +0.10.
        """
        score = snapshot.alt_data_score

        if score > 0.5:
            return 0.10  # Strong bullish alignment
        elif score > 0.2:
            return 0.05  # Moderate bullish
        elif score < -0.5:
            return -0.15  # Strong bearish alignment
        elif score < -0.2:
            return -0.08  # Moderate bearish
        return 0.0  # Neutral

    def get_trigger_recommendation(
        self,
        symbol: str,
        base_signal: Dict[str, Any]
    ) -> TriggerRecommendation:
        """
        Get intraday trigger recommendation based on alt data.

        Args:
            symbol: Trading symbol
            base_signal: Base trading signal from EOD scan

        Returns:
            TriggerRecommendation with action and reasoning
        """
        snapshot = self.get_feature_snapshot(symbol)
        signal_direction = base_signal.get("side", "LONG")

        reasoning = []
        alignment_score = 0.0

        # Check GitHub alignment
        if signal_direction == "LONG":
            if snapshot.github_sentiment > 0.2 and snapshot.github_trend == "increasing":
                alignment_score += 0.15
                reasoning.append("GitHub: Positive sentiment with increasing activity")
            elif snapshot.github_sentiment < -0.2:
                alignment_score -= 0.1
                reasoning.append("GitHub: Negative developer sentiment")

        # Check political alignment
        if signal_direction == "LONG" and snapshot.political_direction == "BEARISH":
            alignment_score -= 0.2
            reasoning.append(f"Political: Bearish signal (strength {snapshot.political_strength:.1%})")
        elif signal_direction == "LONG" and snapshot.political_direction == "BULLISH":
            alignment_score += 0.15
            reasoning.append("Political: Supportive regulatory environment")

        # Check options flow
        if snapshot.unusual_volume:
            if snapshot.smart_money_flow == signal_direction.lower():
                alignment_score += 0.2
                reasoning.append(f"Options: Smart money aligned ({snapshot.smart_money_flow})")
            elif snapshot.smart_money_flow not in ["neutral", signal_direction.lower()]:
                alignment_score -= 0.25
                reasoning.append(f"Options: Smart money contrarian ({snapshot.smart_money_flow})")

        # Check insider activity
        if signal_direction == "LONG" and snapshot.insider_buying:
            alignment_score += 0.15
            reasoning.append("Insider: Recent insider buying detected")
        elif signal_direction == "LONG" and snapshot.insider_selling:
            alignment_score -= 0.1
            reasoning.append("Insider: Recent insider selling")

        # Check news
        if snapshot.breaking_news:
            reasoning.append("News: Breaking news detected - review manually")

        # Determine action
        base_confidence = base_signal.get("confidence", 0.6)
        adjusted_confidence = base_confidence + snapshot.confidence_boost + (alignment_score * 0.1)

        if alignment_score >= 0.3:
            action = "EXECUTE"
            reasoning.append(f"Alt data strongly aligned ({alignment_score:.2f})")
        elif alignment_score <= -0.3:
            action = "SKIP"
            reasoning.append(f"Alt data strongly contra ({alignment_score:.2f})")
        elif adjusted_confidence >= 0.65:
            action = "EXECUTE"
            reasoning.append(f"Adjusted confidence sufficient ({adjusted_confidence:.1%})")
        else:
            action = "WAIT"
            reasoning.append("Mixed signals - wait for better entry")

        return TriggerRecommendation(
            symbol=symbol,
            action=action,
            confidence=max(0, min(1, adjusted_confidence)),
            reasoning=reasoning,
            features=snapshot.to_dict(),
        )

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "cached_symbols": len(self._cache),
            "providers_registered": list(self._feature_providers.keys()),
            "last_update": max(
                (s.timestamp for s in self._cache.values()),
                default=None
            ),
        }


# Singleton instance
_engine: Optional[RealtimeFeatureEngine] = None


def get_realtime_engine() -> RealtimeFeatureEngine:
    """Get or create singleton engine."""
    global _engine
    if _engine is None:
        _engine = RealtimeFeatureEngine()
    return _engine


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = get_realtime_engine()

    print("Getting feature snapshot for NVDA...")
    snapshot = engine.get_feature_snapshot("NVDA")

    print(f"\nFeature Snapshot for {snapshot.symbol}:")
    print(f"  Alt Data Score: {snapshot.alt_data_score:.2f}")
    print(f"  Confidence Boost: {snapshot.confidence_boost:+.1%}")
    print(f"  GitHub Sentiment: {snapshot.github_sentiment:.2f}")
    print(f"  Political Direction: {snapshot.political_direction}")
    print(f"  Social Sentiment: {snapshot.social_sentiment:.2f}")

    # Test trigger recommendation
    print("\nTesting trigger recommendation...")
    base_signal = {
        "symbol": "NVDA",
        "side": "LONG",
        "confidence": 0.65,
        "score": 75,
    }

    recommendation = engine.get_trigger_recommendation("NVDA", base_signal)
    print(f"\nRecommendation: {recommendation.action}")
    print(f"Confidence: {recommendation.confidence:.1%}")
    print("Reasoning:")
    for reason in recommendation.reasoning:
        print(f"  - {reason}")
