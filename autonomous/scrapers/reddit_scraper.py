"""
Reddit Strategy Scraper.

Fetches trading strategy discussions from Reddit using the JSON API.
NO authentication required for reading public posts.
Rate limit: 60 requests/minute.

All ideas found are converted to ExternalIdea objects for testing
with REAL backtest data - NO synthetic data ever.
"""

import time
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json

from core.structured_log import jlog


@dataclass
class RedditPost:
    """A Reddit post containing trading strategy discussion."""
    post_id: str
    title: str
    subreddit: str
    author: str
    content: str  # selftext or description
    url: str
    score: int
    num_comments: int
    created_utc: float
    top_comments: List[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


class RedditScraper:
    """
    Scrapes Reddit for trading strategy discussions.

    Uses Reddit's public JSON API (append .json to any URL).
    Rate limited to respect Reddit's guidelines.
    """

    # Subreddits to scrape - QUANT FOCUSED
    SUBREDDITS = [
        "algotrading",           # Primary algo trading
        "quant",                 # Quant finance
        "quantfinance",          # Quant discussions
        "RealDayTrading",        # Serious day trading
        "FuturesTrading",        # Futures strategies
        "Daytrading",            # Day trading strategies
    ]

    # Keywords to filter posts - QUANT LEVEL ONLY
    STRATEGY_KEYWORDS = [
        # ICT / Smart Money
        "ICT", "inner circle trader", "smart money",
        "order block", "fair value gap", "FVG",
        "liquidity sweep", "market structure",
        "breaker block", "mitigation block",

        # Quant Terms
        "backtest", "win rate", "profit factor",
        "sharpe ratio", "drawdown", "expectancy",
        "statistical edge", "alpha", "beta",

        # Strategy Types
        "mean reversion", "momentum", "trend following",
        "pairs trading", "statistical arbitrage",
        "breakout strategy", "range trading",

        # Technical
        "entry criteria", "exit rules", "stop loss",
        "take profit", "risk reward", "position sizing",
    ]

    # Minimum score - HIGHER for quality
    MIN_SCORE = 50

    # Rate limiting
    REQUEST_INTERVAL = 2.0  # seconds between requests

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("state/autonomous/scrapers/reddit")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "KobeTraderBot-Research/1.0 (Educational Research)"
        })

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get_cache_path(self, subreddit: str) -> Path:
        """Get cache file path for a subreddit."""
        return self.cache_dir / f"sub_{subreddit}.json"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 12) -> bool:
        """Check if cache is still valid."""
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < timedelta(hours=max_age_hours)

    def _has_strategy_keywords(self, text: str) -> bool:
        """Check if text contains strategy-related keywords."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.STRATEGY_KEYWORDS)

    def fetch_subreddit(self, subreddit: str, limit: int = 25) -> List[RedditPost]:
        """
        Fetch top posts from a subreddit.

        Args:
            subreddit: Name of subreddit (without r/)
            limit: Maximum posts to fetch

        Returns:
            List of RedditPost objects
        """
        cache_path = self._get_cache_path(subreddit)

        # Check cache first
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                jlog("reddit_cache_hit", level="DEBUG", subreddit=subreddit)
                return [RedditPost(**p) for p in cached["posts"][:limit]]
            except Exception as e:
                jlog("reddit_cache_error", level="WARNING", error=str(e))

        # Fetch from API
        self._rate_limit()

        try:
            # Fetch hot posts
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"
            params = {"limit": min(limit * 2, 100)}  # Fetch extra to filter

            resp = self._session.get(url, params=params, timeout=30)

            if resp.status_code == 429:
                jlog("reddit_rate_limited", level="WARNING")
                return []

            resp.raise_for_status()
            data = resp.json()

            posts = []
            for child in data.get("data", {}).get("children", []):
                post_data = child.get("data", {})

                # Skip low-quality posts
                if post_data.get("score", 0) < self.MIN_SCORE:
                    continue

                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")

                # Filter for strategy-related content
                if not self._has_strategy_keywords(f"{title} {selftext}"):
                    continue

                # Fetch top comments
                top_comments = self._fetch_top_comments(
                    subreddit, post_data.get("id", "")
                )

                post = RedditPost(
                    post_id=f"reddit:{subreddit}:{post_data.get('id', '')}",
                    title=title,
                    subreddit=subreddit,
                    author=post_data.get("author", "[deleted]"),
                    content=selftext[:5000] if selftext else "",
                    url=f"https://reddit.com{post_data.get('permalink', '')}",
                    score=post_data.get("score", 0),
                    num_comments=post_data.get("num_comments", 0),
                    created_utc=post_data.get("created_utc", 0),
                    top_comments=top_comments[:3]  # Top 3 comments
                )
                posts.append(post)

                if len(posts) >= limit:
                    break

            # Cache results
            with open(cache_path, "w") as f:
                json.dump({
                    "subreddit": subreddit,
                    "fetched_at": datetime.now().isoformat(),
                    "posts": [asdict(p) for p in posts]
                }, f, indent=2)

            jlog("reddit_fetch_complete", level="INFO",
                 subreddit=subreddit, posts_found=len(posts))

            return posts

        except requests.RequestException as e:
            jlog("reddit_fetch_error", level="ERROR",
                 subreddit=subreddit, error=str(e))
            return []

    def _fetch_top_comments(self, subreddit: str, post_id: str, limit: int = 3) -> List[str]:
        """Fetch top comments for a post."""
        if not post_id:
            return []

        self._rate_limit()

        try:
            url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
            params = {"limit": limit, "sort": "top"}

            resp = self._session.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                return []

            data = resp.json()

            comments = []
            if len(data) > 1:
                for child in data[1].get("data", {}).get("children", [])[:limit]:
                    body = child.get("data", {}).get("body", "")
                    if body and len(body) > 50:  # Skip short comments
                        comments.append(body[:1000])  # Truncate long comments

            return comments

        except Exception as e:
            jlog("reddit_comments_error", level="DEBUG",
                 post_id=post_id, error=str(e))
            return []

    def scrape_all(self, posts_per_subreddit: int = 10) -> List[RedditPost]:
        """
        Scrape all configured subreddits.

        Args:
            posts_per_subreddit: Max posts to fetch per subreddit

        Returns:
            List of unique RedditPost objects
        """
        all_posts = {}

        for subreddit in self.SUBREDDITS:
            posts = self.fetch_subreddit(subreddit, posts_per_subreddit)
            for post in posts:
                if post.post_id not in all_posts:
                    all_posts[post.post_id] = post

        jlog("reddit_scrape_complete", level="INFO",
             total_posts=len(all_posts),
             subreddits_scraped=len(self.SUBREDDITS))

        return list(all_posts.values())


def scrape_reddit_ideas() -> List[Dict[str, Any]]:
    """
    Handler function for autonomous brain task.

    Returns:
        List of strategy ideas extracted from Reddit posts
    """
    scraper = RedditScraper()
    posts = scraper.scrape_all(posts_per_subreddit=5)

    ideas = []
    for post in posts:
        # Combine content with top comments
        full_content = post.content
        if post.top_comments:
            full_content += "\n\n--- TOP COMMENTS ---\n" + "\n---\n".join(post.top_comments)

        ideas.append({
            "source_type": "reddit",
            "source_id": post.post_id,
            "source_url": post.url,
            "title": post.title,
            "description": f"r/{post.subreddit} | Score: {post.score} | Comments: {post.num_comments}",
            "content": full_content,
            "metadata": {
                "subreddit": post.subreddit,
                "score": post.score,
                "num_comments": post.num_comments,
                "author": post.author
            },
            "fetched_at": datetime.now().isoformat()
        })

    return ideas
