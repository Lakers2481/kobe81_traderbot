from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Literal

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

NEWS_URL = "https://api.polygon.io/v2/reference/news"
CACHE_DIR = Path("data/sentiment")

# Sentiment model selection
# Set via environment variable: SENTIMENT_MODEL=fingpt or SENTIMENT_MODEL=vader
# Set via config: sentiment_provider in config/base.yaml
# Default: "vader" (stable, proven) - upgrade to "fingpt" after A/B validation
SENTIMENT_MODEL: Literal["vader", "fingpt", "ab_test"] = os.getenv("SENTIMENT_MODEL", "vader").lower()


@dataclass
class NewsItem:
    symbol: str
    published_utc: pd.Timestamp
    title: str
    description: str
    url: str


def fetch_polygon_news(symbol: str, start: str, end: str, api_key: str, limit: int = 50, timeout: int = 10) -> List[NewsItem]:
    params = {
        'ticker': symbol.upper(),
        'published_utc.gte': start,
        'published_utc.lte': end,
        'order': 'desc',
        'limit': limit,
        'apiKey': api_key,
    }
    items: List[NewsItem] = []
    try:
        r = requests.get(NEWS_URL, params=params, timeout=timeout)
        if r.status_code != 200:
            return items
        data = r.json()
        for res in data.get('results', []) or []:
            ts = pd.to_datetime(res.get('published_utc'), errors='coerce')
            items.append(NewsItem(
                symbol=symbol.upper(),
                published_utc=ts,
                title=str(res.get('title') or ''),
                description=str(res.get('description') or ''),
                url=str(res.get('article_url') or ''),
            ))
    except Exception:
        return items
    return items


def _cache_path_for(date_str: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"sentiment_{date_str}.csv"


def compute_daily_sentiment(items: List[NewsItem]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=['date','symbol','sent_mean','sent_count'])
    sid = SentimentIntensityAnalyzer()
    rows: List[Dict[str, Any]] = []
    for it in items:
        text = f"{it.title}. {it.description}".strip()
        if not text:
            score = 0.0
        else:
            s = sid.polarity_scores(text)
            score = float(s.get('compound', 0.0))
        rows.append({
            'date': it.published_utc.date() if pd.notna(it.published_utc) else None,
            'symbol': it.symbol,
            'score': score,
        })
    df = pd.DataFrame(rows)
    df = df.dropna(subset=['date'])
    agg = df.groupby(['date','symbol'])['score'].agg(sent_mean='mean', sent_count='count').reset_index()
    return agg


def write_daily_cache(df: pd.DataFrame, date_str: str) -> Path:
    p = _cache_path_for(date_str)
    df.to_csv(p, index=False)
    return p


def load_daily_cache(date_str: str) -> pd.DataFrame:
    p = _cache_path_for(date_str)
    if not p.exists():
        return pd.DataFrame(columns=['date','symbol','sent_mean','sent_count'])
    try:
        return pd.read_csv(p, parse_dates=['date'])
    except Exception:
        return pd.DataFrame(columns=['date','symbol','sent_mean','sent_count'])


def normalize_sentiment_to_conf(sent_mean: float) -> float:
    """Map compound score [-1,1] to confidence [0,1]."""
    return max(0.0, min((float(sent_mean) + 1.0) / 2.0, 1.0))


# =============================================================================
# FinGPT Integration (Fix #2 - 2026-01-08)
# =============================================================================

def analyze_sentiment(
    text: str,
    model: Literal["vader", "fingpt", "ab_test"] = None
) -> float:
    """
    Analyze sentiment using selected model.

    Args:
        text: Text to analyze
        model: Model to use ("vader", "fingpt", "ab_test")
               If None, uses SENTIMENT_MODEL environment variable

    Returns:
        Compound sentiment score [-1, 1]
    """
    # Determine which model to use
    selected_model = model if model else SENTIMENT_MODEL

    if selected_model == "fingpt":
        return _analyze_sentiment_fingpt(text)
    elif selected_model == "ab_test":
        return _analyze_sentiment_ab_test(text)
    else:  # vader (default)
        return _analyze_sentiment_vader(text)


def _analyze_sentiment_vader(text: str) -> float:
    """Analyze sentiment using VADER (baseline)."""
    if not text or len(text.strip()) < 10:
        return 0.0

    try:
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(text)
        return float(scores.get('compound', 0.0))
    except Exception as e:
        logger.warning(f"VADER analysis failed: {e}")
        return 0.0


def _analyze_sentiment_fingpt(text: str) -> float:
    """Analyze sentiment using FinGPT with fallback to VADER."""
    if not text or len(text.strip()) < 10:
        return 0.0

    try:
        # Import here to avoid loading model if not needed
        from altdata.sentiment_fingpt import analyze_sentiment_fingpt

        return analyze_sentiment_fingpt(text)

    except Exception as e:
        logger.warning(f"FinGPT analysis failed, falling back to VADER: {e}")
        return _analyze_sentiment_vader(text)


def _analyze_sentiment_ab_test(text: str) -> float:
    """
    Run both VADER and FinGPT, log comparison, return FinGPT result.

    This mode is for A/B testing - collects statistical data to validate
    whether FinGPT outperforms VADER on real trading data.
    """
    if not text or len(text.strip()) < 10:
        return 0.0

    # Run both models
    vader_score = _analyze_sentiment_vader(text)
    fingpt_score = _analyze_sentiment_fingpt(text)

    # Log comparison for statistical analysis
    _log_ab_comparison(text, vader_score, fingpt_score)

    # Return FinGPT score (testing candidate)
    return fingpt_score


def _log_ab_comparison(text: str, vader_score: float, fingpt_score: float) -> None:
    """Log A/B comparison for statistical analysis."""
    try:
        import json
        from datetime import datetime

        ab_log_file = Path("state/ab_tests/sentiment_vader_vs_fingpt.jsonl")
        ab_log_file.parent.mkdir(parents=True, exist_ok=True)

        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'text_preview': text[:100],  # First 100 chars for debugging
            'text_length': len(text),
            'vader_score': vader_score,
            'fingpt_score': fingpt_score,
            'difference': abs(vader_score - fingpt_score),
            'agreement': abs(vader_score - fingpt_score) < 0.2,  # Scores within 0.2 = agreement
        }

        with open(ab_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    except Exception as e:
        logger.warning(f"Failed to log A/B comparison: {e}")


def get_sentiment_model_info() -> Dict[str, Any]:
    """Get current sentiment model configuration."""
    info = {
        'active_model': SENTIMENT_MODEL,
        'available_models': ['vader', 'fingpt', 'ab_test'],
        'default_model': 'vader',
        'fingpt_available': False,
    }

    # Check if FinGPT is available
    try:
        from altdata.sentiment_fingpt import FinGPTSentimentAnalyzer
        _ = FinGPTSentimentAnalyzer.get_instance()
        info['fingpt_available'] = True
    except Exception:
        info['fingpt_available'] = False

    return info

