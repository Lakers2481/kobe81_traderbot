#!/usr/bin/env python3
"""
KOBE INTELLIGENT KNOWLEDGE SCRAPER
===================================
This scraper UNDERSTANDS Kobe's goals and ONLY looks for relevant content.

KOBE'S IDENTITY:
- Swing trading robot (holds 2-10 days)
- Uses IBS+RSI mean reversion and ICT Turtle Soup strategies
- Python-based with AI/ML components (LSTM, HMM, RL)
- Goal: Be the greatest trading robot ever created

WHAT WE'RE LOOKING FOR:
1. SWING TRADING strategies (NOT day trading, NOT 0DTE, NOT scalping)
2. MEAN REVERSION strategies (IBS, RSI, oversold/overbought)
3. ICT STRATEGIES (Smart Money, Order Blocks, FVG, Turtle Soup, liquidity)
4. RISK MANAGEMENT (position sizing, drawdown, stop losses, Kelly criterion)
5. AI/ML FOR TRADING (LSTM, transformers, RL, regime detection, ensemble)
6. PYTHON CODE (faster, cleaner, better architecture, best practices)
7. LLM/AI AGENTS (cognitive systems, self-improvement, autonomous agents)

WHAT WE EXCLUDE:
- 0DTE options, day trading, scalping, high-frequency
- Crypto gambling, meme stocks, WSB YOLO
- Forex, futures (we trade US equities)
- Generic beginner content
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo
import urllib.request
import urllib.parse
import re

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class KobeKnowledgeScraper:
    """
    Intelligent scraper that UNDERSTANDS Kobe's goals.
    Only collects knowledge that makes Kobe smarter, safer, more accurate.
    """

    # Kobe's identity - the scraper knows who it's serving
    KOBE_IDENTITY = {
        "name": "Kobe",
        "type": "swing_trading_robot",
        "holding_period": "2-10 days",
        "strategies": ["IBS+RSI mean reversion", "ICT Turtle Soup"],
        "tech_stack": ["Python", "LSTM", "HMM", "XGBoost", "RL"],
        "goal": "Be the greatest trading robot ever created",
    }

    # Keywords that MUST be present (at least one)
    INCLUDE_KEYWORDS = [
        # Swing Trading
        "swing trading", "swing trade", "multi-day", "position trading",
        "hold overnight", "2-5 days", "weekly options",

        # Mean Reversion (Kobe's core strategy)
        "mean reversion", "oversold", "overbought", "rsi strategy",
        "ibs indicator", "internal bar strength", "reversal",
        "bounce", "pullback", "dip buying",

        # ICT Strategies (Kobe's other strategy)
        "ict", "smart money", "order block", "fair value gap", "fvg",
        "turtle soup", "liquidity sweep", "liquidity grab",
        "institutional trading", "market structure", "break of structure",
        "change of character", "optimal trade entry", "ote",

        # Risk Management (safety)
        "risk management", "position sizing", "stop loss", "drawdown",
        "kelly criterion", "risk per trade", "max drawdown",
        "portfolio risk", "var", "value at risk", "sharpe ratio",
        "risk-adjusted", "capital preservation",

        # AI/ML for Trading (making Kobe smarter)
        "lstm trading", "machine learning trading", "ai trading",
        "reinforcement learning trading", "regime detection",
        "market regime", "ensemble model", "neural network trading",
        "transformer trading", "deep learning finance",
        "feature engineering trading", "alpha generation",

        # Python Trading Code (better code)
        "python trading", "backtesting python", "vectorbt",
        "pandas finance", "numpy trading", "zipline", "backtrader",
        "trading algorithm", "quantitative python",

        # LLM/AI Agents (cognitive improvements)
        "llm agent", "ai agent", "autonomous agent", "cognitive",
        "self-improving", "meta-learning", "reflection",
        "chain of thought", "reasoning", "planning agent",
    ]

    # Keywords that EXCLUDE content (if present, skip it)
    EXCLUDE_KEYWORDS = [
        # Day Trading (NOT Kobe)
        "0dte", "0 dte", "zero dte", "day trade", "daytrading",
        "scalping", "scalp", "intraday only", "close same day",

        # High Frequency (NOT Kobe)
        "high frequency", "hft", "microsecond", "nanosecond",
        "co-location", "market making",

        # Crypto/Forex (NOT Kobe's market)
        "crypto", "bitcoin", "ethereum", "forex", "fx trading",
        "currency pair", "futures trading",

        # Gambling/YOLO (NOT Kobe's style)
        "yolo", "moon", "diamond hands", "to the moon",
        "meme stock", "wallstreetbets style", "gambling",

        # Beginner/Generic (waste of time)
        "what is a stock", "how to start trading", "beginner guide",
        "first trade ever", "paper trading only",
    ]

    SOURCES = [
        "reddit_swing",      # Swing trading subreddits
        "reddit_quant",      # Quantitative trading
        "github_swing",      # Swing trading code
        "github_ml",         # ML for trading
        "github_risk",       # Risk management
        "arxiv_quant",       # Academic papers
        "stackoverflow",     # Python trading code
        "hackernews",        # Tech/AI news
    ]

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir is None:
            state_dir = Path("state/autonomous/scrapers")
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.current_source_idx = 0
        self._load_state()

    def _load_state(self):
        """Load scraper state."""
        state_file = self.state_dir / "scraper_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self.current_source_idx = data.get("current_source_idx", 0)
            except Exception:
                pass

    def save_state(self):
        """Save scraper state."""
        state_file = self.state_dir / "scraper_state.json"
        data = {
            "current_source_idx": self.current_source_idx,
            "last_updated": datetime.now(ET).isoformat(),
            "total_sources": len(self.SOURCES),
            "kobe_identity": self.KOBE_IDENTITY,
        }
        state_file.write_text(json.dumps(data, indent=2))

    def get_next_source(self) -> str:
        """Get next source in rotation."""
        # Bounds check in case SOURCES list changed
        if self.current_source_idx >= len(self.SOURCES):
            self.current_source_idx = 0
        source = self.SOURCES[self.current_source_idx]
        self.current_source_idx = (self.current_source_idx + 1) % len(self.SOURCES)
        self.save_state()
        return source

    def _fetch_url(self, url: str, timeout: int = 10) -> Optional[str]:
        """Fetch URL content safely."""
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode('utf-8', errors='ignore')
        except Exception as e:
            logger.debug(f"Fetch failed for {url}: {e}")
            return None

    def _is_relevant(self, text: str) -> bool:
        """Check if content is relevant to Kobe's goals."""
        text_lower = text.lower()

        # Check for EXCLUDE keywords first
        for keyword in self.EXCLUDE_KEYWORDS:
            if keyword in text_lower:
                return False

        # Check for at least one INCLUDE keyword
        for keyword in self.INCLUDE_KEYWORDS:
            if keyword in text_lower:
                return True

        return False

    def _get_relevance_score(self, text: str) -> float:
        """Score how relevant content is to Kobe (0-1)."""
        text_lower = text.lower()
        score = 0.0
        matches = 0

        for keyword in self.INCLUDE_KEYWORDS:
            if keyword in text_lower:
                matches += 1

        # Normalize: more matches = higher score
        if matches > 0:
            score = min(1.0, matches / 5.0)  # 5+ matches = max score

        return score

    def _categorize(self, text: str) -> str:
        """Categorize content by what it improves."""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["swing", "multi-day", "position"]):
            return "swing_strategy"
        elif any(kw in text_lower for kw in ["mean reversion", "rsi", "oversold", "ibs"]):
            return "mean_reversion"
        elif any(kw in text_lower for kw in ["ict", "smart money", "order block", "turtle soup", "liquidity"]):
            return "ict_strategy"
        elif any(kw in text_lower for kw in ["risk", "position size", "stop loss", "drawdown", "kelly"]):
            return "risk_management"
        elif any(kw in text_lower for kw in ["lstm", "machine learning", "ai trading", "regime", "neural"]):
            return "ai_ml_trading"
        elif any(kw in text_lower for kw in ["python", "backtest", "vectorbt", "pandas"]):
            return "python_code"
        elif any(kw in text_lower for kw in ["llm", "agent", "cognitive", "self-improv"]):
            return "ai_agents"
        else:
            return "general"

    # =========================================================================
    # REDDIT - SWING TRADING FOCUSED
    # =========================================================================
    def scrape_reddit_swing(self) -> List[Dict]:
        """Scrape Reddit for SWING TRADING content only."""
        # These subreddits are more relevant to swing trading
        subreddits = ["swingtrading", "stockmarket", "algotrading"]
        ideas = []

        for sub in subreddits:
            try:
                # Search for swing trading content
                url = f"https://www.reddit.com/r/{sub}/search.json?q=swing+trading+strategy&restrict_sr=1&limit=10&sort=relevance"
                content = self._fetch_url(url)
                if content:
                    data = json.loads(content)
                    posts = data.get("data", {}).get("children", [])
                    for post in posts:
                        p = post.get("data", {})
                        title = p.get("title", "")

                        # Only include if relevant
                        if self._is_relevant(title):
                            ideas.append({
                                "source": "reddit",
                                "subreddit": sub,
                                "title": title,
                                "score": p.get("score", 0),
                                "url": f"https://reddit.com{p.get('permalink', '')}",
                                "category": self._categorize(title),
                                "relevance": self._get_relevance_score(title),
                                "created": datetime.now(ET).isoformat(),
                            })
            except Exception as e:
                logger.debug(f"Reddit {sub} failed: {e}")
                continue

        # Save discoveries
        self._save_discoveries("reddit_swing", ideas)
        return ideas

    def scrape_reddit_quant(self) -> List[Dict]:
        """Scrape Reddit for quantitative/ML trading content."""
        ideas = []

        # Search r/algotrading for ML/AI content
        searches = [
            ("algotrading", "machine learning"),
            ("algotrading", "mean reversion"),
            ("quant", "risk management"),
            ("quant", "regime detection"),
        ]

        for sub, query in searches:
            try:
                url = f"https://www.reddit.com/r/{sub}/search.json?q={urllib.parse.quote(query)}&restrict_sr=1&limit=5&sort=top&t=month"
                content = self._fetch_url(url)
                if content:
                    data = json.loads(content)
                    posts = data.get("data", {}).get("children", [])
                    for post in posts:
                        p = post.get("data", {})
                        title = p.get("title", "")

                        if self._is_relevant(title):
                            ideas.append({
                                "source": "reddit",
                                "subreddit": sub,
                                "title": title,
                                "score": p.get("score", 0),
                                "url": f"https://reddit.com{p.get('permalink', '')}",
                                "category": self._categorize(title),
                                "relevance": self._get_relevance_score(title),
                                "query": query,
                                "created": datetime.now(ET).isoformat(),
                            })
            except Exception as e:
                logger.debug(f"Reddit quant failed: {e}")

        self._save_discoveries("reddit_quant", ideas)
        return ideas

    # =========================================================================
    # GITHUB - FOCUSED SEARCHES
    # =========================================================================
    def scrape_github_swing(self) -> List[Dict]:
        """Scrape GitHub for swing trading code."""
        ideas = []
        queries = [
            "swing trading python",
            "mean reversion strategy python",
            "RSI trading strategy",
            "oversold bounce strategy",
        ]

        query = random.choice(queries)
        try:
            url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&per_page=10"
            content = self._fetch_url(url)
            if content:
                data = json.loads(content)
                repos = data.get("items", [])
                for repo in repos[:5]:
                    desc = repo.get("description", "") or ""
                    name = repo.get("full_name", "")
                    combined = f"{name} {desc}"

                    if self._is_relevant(combined):
                        ideas.append({
                            "source": "github",
                            "name": name,
                            "description": desc,
                            "stars": repo.get("stargazers_count", 0),
                            "url": repo.get("html_url", ""),
                            "language": repo.get("language", ""),
                            "category": self._categorize(combined),
                            "relevance": self._get_relevance_score(combined),
                            "query": query,
                            "created": datetime.now(ET).isoformat(),
                        })
        except Exception as e:
            logger.debug(f"GitHub swing failed: {e}")

        self._save_discoveries("github_swing", ideas)
        return ideas

    def scrape_github_ml(self) -> List[Dict]:
        """Scrape GitHub for ML trading code."""
        ideas = []
        queries = [
            "LSTM stock prediction",
            "reinforcement learning trading",
            "market regime detection python",
            "trading neural network",
            "ensemble trading model",
        ]

        query = random.choice(queries)
        try:
            url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&per_page=10"
            content = self._fetch_url(url)
            if content:
                data = json.loads(content)
                repos = data.get("items", [])
                for repo in repos[:5]:
                    desc = repo.get("description", "") or ""
                    name = repo.get("full_name", "")

                    ideas.append({
                        "source": "github",
                        "name": name,
                        "description": desc,
                        "stars": repo.get("stargazers_count", 0),
                        "url": repo.get("html_url", ""),
                        "language": repo.get("language", ""),
                        "category": "ai_ml_trading",
                        "relevance": self._get_relevance_score(f"{name} {desc}"),
                        "query": query,
                        "created": datetime.now(ET).isoformat(),
                    })
        except Exception as e:
            logger.debug(f"GitHub ML failed: {e}")

        self._save_discoveries("github_ml", ideas)
        return ideas

    def scrape_github_risk(self) -> List[Dict]:
        """Scrape GitHub for risk management code."""
        ideas = []
        queries = [
            "position sizing python",
            "risk management trading",
            "Kelly criterion trading",
            "portfolio risk python",
            "drawdown analysis python",
        ]

        query = random.choice(queries)
        try:
            url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&per_page=10"
            content = self._fetch_url(url)
            if content:
                data = json.loads(content)
                repos = data.get("items", [])
                for repo in repos[:5]:
                    desc = repo.get("description", "") or ""
                    name = repo.get("full_name", "")

                    ideas.append({
                        "source": "github",
                        "name": name,
                        "description": desc,
                        "stars": repo.get("stargazers_count", 0),
                        "url": repo.get("html_url", ""),
                        "language": repo.get("language", ""),
                        "category": "risk_management",
                        "relevance": self._get_relevance_score(f"{name} {desc}"),
                        "query": query,
                        "created": datetime.now(ET).isoformat(),
                    })
        except Exception as e:
            logger.debug(f"GitHub risk failed: {e}")

        self._save_discoveries("github_risk", ideas)
        return ideas

    # =========================================================================
    # ARXIV - ACADEMIC PAPERS
    # =========================================================================
    def scrape_arxiv_quant(self) -> List[Dict]:
        """Scrape arXiv for quantitative finance papers."""
        ideas = []

        try:
            # Search for relevant papers
            queries = [
                "mean reversion trading",
                "regime switching market",
                "LSTM stock prediction",
                "reinforcement learning portfolio",
                "risk management trading",
            ]
            query = random.choice(queries)

            url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending"
            content = self._fetch_url(url, timeout=15)

            if content:
                # Parse XML (simple regex for this)
                titles = re.findall(r'<title>(.*?)</title>', content, re.DOTALL)
                links = re.findall(r'<id>(.*?)</id>', content)
                summaries = re.findall(r'<summary>(.*?)</summary>', content, re.DOTALL)

                for i, title in enumerate(titles[1:6]):  # Skip first (feed title)
                    title = title.strip().replace('\n', ' ')
                    if self._is_relevant(title):
                        ideas.append({
                            "source": "arxiv",
                            "title": title,
                            "url": links[i+1] if i+1 < len(links) else "",
                            "summary": summaries[i][:200] if i < len(summaries) else "",
                            "category": self._categorize(title),
                            "relevance": self._get_relevance_score(title),
                            "query": query,
                            "created": datetime.now(ET).isoformat(),
                        })
        except Exception as e:
            logger.debug(f"arXiv failed: {e}")

        self._save_discoveries("arxiv", ideas)
        return ideas

    # =========================================================================
    # STACKOVERFLOW - PYTHON TRADING CODE
    # =========================================================================
    def scrape_stackoverflow(self) -> List[Dict]:
        """Scrape StackOverflow for Python trading questions."""
        ideas = []

        queries = [
            "python backtesting",
            "pandas trading strategy",
            "vectorized backtesting",
            "python risk management",
            "LSTM stock prediction python",
        ]
        query = random.choice(queries)

        try:
            url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=votes&q={urllib.parse.quote(query)}&site=stackoverflow&pagesize=5"
            content = self._fetch_url(url)
            if content:
                data = json.loads(content)
                questions = data.get("items", [])
                for q in questions:
                    title = q.get("title", "")
                    if self._is_relevant(title):
                        ideas.append({
                            "source": "stackoverflow",
                            "title": title,
                            "score": q.get("score", 0),
                            "url": q.get("link", ""),
                            "answered": q.get("is_answered", False),
                            "category": self._categorize(title),
                            "relevance": self._get_relevance_score(title),
                            "query": query,
                            "created": datetime.now(ET).isoformat(),
                        })
        except Exception as e:
            logger.debug(f"StackOverflow failed: {e}")

        self._save_discoveries("stackoverflow", ideas)
        return ideas

    # =========================================================================
    # HACKERNEWS - AI/TECH NEWS
    # =========================================================================
    def scrape_hackernews(self) -> List[Dict]:
        """Scrape HackerNews for AI/trading/finance posts."""
        ideas = []

        try:
            url = "https://hacker-news.firebaseio.com/v0/topstories.json"
            content = self._fetch_url(url)
            if content:
                story_ids = json.loads(content)[:100]

                for story_id in story_ids[:30]:
                    try:
                        story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                        story_content = self._fetch_url(story_url)
                        if story_content:
                            story = json.loads(story_content)
                            title = story.get("title", "")

                            # Must be relevant to Kobe
                            if self._is_relevant(title):
                                ideas.append({
                                    "source": "hackernews",
                                    "title": title,
                                    "score": story.get("score", 0),
                                    "url": story.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                                    "category": self._categorize(title),
                                    "relevance": self._get_relevance_score(title),
                                    "created": datetime.now(ET).isoformat(),
                                })
                    except Exception:
                        continue
        except Exception as e:
            logger.debug(f"HackerNews failed: {e}")

        self._save_discoveries("hackernews", ideas)
        return ideas

    # =========================================================================
    # MAIN SCRAPING METHODS
    # =========================================================================
    def _save_discoveries(self, source: str, items: List[Dict]):
        """Save discoveries to file."""
        if not items:
            return

        file_path = self.state_dir / f"discoveries_{source}.json"

        # Load existing
        existing = []
        if file_path.exists():
            try:
                existing = json.loads(file_path.read_text())
            except Exception:
                pass

        # Add new (avoid duplicates by URL)
        existing_urls = {item.get("url") for item in existing}
        for item in items:
            if item.get("url") not in existing_urls:
                existing.append(item)

        # Keep last 100 per source
        existing = existing[-100:]

        file_path.write_text(json.dumps(existing, indent=2))

    def scrape_next_source(self) -> Dict[str, Any]:
        """Scrape next source in rotation."""
        source = self.get_next_source()

        scrapers = {
            "reddit_swing": self.scrape_reddit_swing,
            "reddit_quant": self.scrape_reddit_quant,
            "github_swing": self.scrape_github_swing,
            "github_ml": self.scrape_github_ml,
            "github_risk": self.scrape_github_risk,
            "arxiv_quant": self.scrape_arxiv_quant,
            "stackoverflow": self.scrape_stackoverflow,
            "hackernews": self.scrape_hackernews,
        }

        scraper = scrapers.get(source)
        if scraper:
            items = scraper()
            return {
                "status": "success",
                "source": source,
                "items_found": len(items),
                "items": items[:3],  # Return top 3 for logging
            }

        return {"status": "error", "source": source, "error": "Unknown source"}

    def scrape_all_sources(self) -> Dict[str, Any]:
        """Scrape ALL sources (for overnight/weekend deep research)."""
        results = {}

        for source in self.SOURCES:
            try:
                scrapers = {
                    "reddit_swing": self.scrape_reddit_swing,
                    "reddit_quant": self.scrape_reddit_quant,
                    "github_swing": self.scrape_github_swing,
                    "github_ml": self.scrape_github_ml,
                    "github_risk": self.scrape_github_risk,
                    "arxiv_quant": self.scrape_arxiv_quant,
                    "stackoverflow": self.scrape_stackoverflow,
                    "hackernews": self.scrape_hackernews,
                }

                scraper = scrapers.get(source)
                if scraper:
                    items = scraper()
                    results[source] = {
                        "status": "success",
                        "count": len(items),
                    }
            except Exception as e:
                results[source] = {"status": "error", "error": str(e)}

        return results

    def get_discovery_summary(self) -> Dict[str, int]:
        """Get count of discoveries per source."""
        summary = {}
        for source in self.SOURCES:
            file_path = self.state_dir / f"discoveries_{source.replace('_', '_')}.json"
            if file_path.exists():
                try:
                    data = json.loads(file_path.read_text())
                    summary[source] = len(data)
                except Exception:
                    summary[source] = 0
            else:
                summary[source] = 0
        return summary

    def get_best_discoveries(self, limit: int = 10) -> List[Dict]:
        """Get highest relevance discoveries across all sources."""
        all_items = []

        for source in self.SOURCES:
            file_path = self.state_dir / f"discoveries_{source.replace('_', '_')}.json"
            if file_path.exists():
                try:
                    data = json.loads(file_path.read_text())
                    all_items.extend(data)
                except Exception:
                    pass

        # Sort by relevance score
        all_items.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        return all_items[:limit]


# Backwards compatibility alias
UniversalScraper = KobeKnowledgeScraper
