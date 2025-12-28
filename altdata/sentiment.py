from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


NEWS_URL = "https://api.polygon.io/v2/reference/news"
CACHE_DIR = Path("data/sentiment")


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
    # Map compound [-1,1] to [0,1]
    return max(0.0, min((float(sent_mean) + 1.0) / 2.0, 1.0))

