"""
GitHub Strategy Scraper.

Fetches trading strategy repositories from GitHub using the public API.
NO authentication required for public repos.
Rate limit: 60 requests/hour (unauthenticated).

All strategies found are converted to ExternalIdea objects for testing
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
class GitHubRepo:
    """A GitHub repository containing trading strategy code."""
    repo_id: str
    name: str
    full_name: str  # owner/repo
    description: str
    url: str
    readme_content: str
    stars: int
    language: str
    topics: List[str]
    last_updated: str
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


class GitHubScraper:
    """
    Scrapes GitHub for trading strategy repositories.

    Uses GitHub's public search API (no auth required).
    Rate limited to 10 requests per minute to be respectful.
    """

    # Search queries for trading strategies
    SEARCH_QUERIES = [
        "algorithmic trading python",
        "trading strategy python",
        "quantitative trading",
        "mean reversion strategy",
        "momentum trading python",
        "backtest trading",
        "stock trading algorithm",
        "technical analysis trading",
    ]

    # Minimum stars to consider (filters out low-quality repos)
    MIN_STARS = 10

    # Rate limiting
    REQUEST_INTERVAL = 6.0  # seconds between requests (10/min)

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("state/autonomous/scrapers/github")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "KobeTraderBot-Research/1.0"
        })

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get_cache_path(self, query: str) -> Path:
        """Get cache file path for a query."""
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return self.cache_dir / f"search_{query_hash}.json"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cache is still valid."""
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < timedelta(hours=max_age_hours)

    def search_repos(self, query: str, max_results: int = 10) -> List[GitHubRepo]:
        """
        Search GitHub for repositories matching query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of GitHubRepo objects
        """
        cache_path = self._get_cache_path(query)

        # Check cache first
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                jlog("github_cache_hit", level="DEBUG", query=query)
                return [GitHubRepo(**r) for r in cached["repos"][:max_results]]
            except Exception as e:
                jlog("github_cache_error", level="WARNING", error=str(e))

        # Fetch from API
        self._rate_limit()

        try:
            url = "https://api.github.com/search/repositories"
            params = {
                "q": f"{query} language:python",
                "sort": "stars",
                "order": "desc",
                "per_page": min(max_results, 30)
            }

            resp = self._session.get(url, params=params, timeout=30)

            if resp.status_code == 403:
                jlog("github_rate_limited", level="WARNING")
                return []

            resp.raise_for_status()
            data = resp.json()

            repos = []
            for item in data.get("items", []):
                if item.get("stargazers_count", 0) < self.MIN_STARS:
                    continue

                # Fetch README content
                readme = self._fetch_readme(item["full_name"])

                repo = GitHubRepo(
                    repo_id=f"github:{item['full_name']}",
                    name=item["name"],
                    full_name=item["full_name"],
                    description=item.get("description", "") or "",
                    url=item["html_url"],
                    readme_content=readme,
                    stars=item["stargazers_count"],
                    language=item.get("language", "Python"),
                    topics=item.get("topics", []),
                    last_updated=item["updated_at"]
                )
                repos.append(repo)

            # Cache results
            with open(cache_path, "w") as f:
                json.dump({
                    "query": query,
                    "fetched_at": datetime.now().isoformat(),
                    "repos": [asdict(r) for r in repos]
                }, f, indent=2)

            jlog("github_search_complete", level="INFO",
                 query=query, repos_found=len(repos))

            return repos

        except requests.RequestException as e:
            jlog("github_search_error", level="ERROR",
                 query=query, error=str(e))
            return []

    def _fetch_readme(self, full_name: str) -> str:
        """Fetch README content for a repository."""
        self._rate_limit()

        try:
            url = f"https://api.github.com/repos/{full_name}/readme"
            resp = self._session.get(url, timeout=30)

            if resp.status_code != 200:
                return ""

            data = resp.json()

            # README is base64 encoded
            import base64
            content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")

            # Truncate if too long (keep first 5000 chars)
            if len(content) > 5000:
                content = content[:5000] + "\n\n[TRUNCATED]"

            return content

        except Exception as e:
            jlog("github_readme_error", level="DEBUG",
                 repo=full_name, error=str(e))
            return ""

    def scrape_all(self, max_repos_per_query: int = 5) -> List[GitHubRepo]:
        """
        Scrape all configured search queries.

        Args:
            max_repos_per_query: Max repos to fetch per query

        Returns:
            List of unique GitHubRepo objects
        """
        all_repos = {}

        for query in self.SEARCH_QUERIES:
            repos = self.search_repos(query, max_repos_per_query)
            for repo in repos:
                if repo.repo_id not in all_repos:
                    all_repos[repo.repo_id] = repo

        jlog("github_scrape_complete", level="INFO",
             total_repos=len(all_repos),
             queries_run=len(self.SEARCH_QUERIES))

        return list(all_repos.values())


def scrape_github_strategies() -> List[Dict[str, Any]]:
    """
    Handler function for autonomous brain task.

    Returns:
        List of strategy ideas extracted from GitHub repos
    """
    scraper = GitHubScraper()
    repos = scraper.scrape_all(max_repos_per_query=3)

    ideas = []
    for repo in repos:
        ideas.append({
            "source_type": "github",
            "source_id": repo.repo_id,
            "source_url": repo.url,
            "title": repo.name,
            "description": repo.description,
            "content": repo.readme_content,
            "metadata": {
                "stars": repo.stars,
                "language": repo.language,
                "topics": repo.topics,
                "last_updated": repo.last_updated
            },
            "fetched_at": datetime.now().isoformat()
        })

    return ideas
