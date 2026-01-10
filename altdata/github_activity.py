"""
GitHub Activity Monitor - Developer Sentiment Signal

Monitor GitHub commit frequency and developer chatter for tech companies.
A sudden halt in commits could signal trouble; increased activity may
indicate upcoming product launches.

Author: Kobe Trading System
Date: 2026-01-07
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests

logger = logging.getLogger(__name__)

# Cache directory
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache" / "github"


@dataclass
class RepoActivity:
    """Activity metrics for a GitHub repository."""
    repo_name: str
    owner: str
    commits_7d: int = 0
    commits_30d: int = 0
    open_issues: int = 0
    closed_issues_7d: int = 0
    stars: int = 0
    stars_growth_7d: int = 0
    forks: int = 0
    contributors_active: int = 0
    last_commit: Optional[str] = None
    sentiment_score: float = 0.0  # -1 to 1
    activity_trend: str = "stable"  # "increasing", "decreasing", "stable"
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CompanyGitHubProfile:
    """GitHub profile for a public company."""
    symbol: str
    company_name: str
    github_org: str
    key_repos: List[str] = field(default_factory=list)
    activity: Optional[RepoActivity] = None


# Symbol to GitHub org mapping
COMPANY_GITHUB_MAP = {
    "MSFT": CompanyGitHubProfile("MSFT", "Microsoft", "microsoft",
                                   ["vscode", "TypeScript", "terminal", "PowerToys"]),
    "GOOGL": CompanyGitHubProfile("GOOGL", "Google", "google",
                                    ["googletest", "material-design-icons", "gson"]),
    "META": CompanyGitHubProfile("META", "Meta", "facebook",
                                   ["react", "react-native", "pytorch", "jest"]),
    "AAPL": CompanyGitHubProfile("AAPL", "Apple", "apple",
                                   ["swift", "swift-evolution", "foundationdb"]),
    "NVDA": CompanyGitHubProfile("NVDA", "NVIDIA", "NVIDIA",
                                   ["cuda-samples", "TensorRT", "NeMo"]),
    "AMD": CompanyGitHubProfile("AMD", "AMD", "ROCm-Developer-Tools",
                                  ["ROCm", "hipBLAS"]),
    "AMZN": CompanyGitHubProfile("AMZN", "Amazon", "aws",
                                   ["aws-cdk", "aws-cli", "amazon-ecs-agent"]),
    "CRM": CompanyGitHubProfile("CRM", "Salesforce", "salesforce",
                                  ["lwc", "design-system"]),
    "ADBE": CompanyGitHubProfile("ADBE", "Adobe", "adobe",
                                   ["brackets", "spectrum-css"]),
    "ORCL": CompanyGitHubProfile("ORCL", "Oracle", "oracle",
                                   ["graal", "helidon"]),
}


class GitHubActivityMonitor:
    """
    Monitor GitHub activity for public tech companies.

    Tracks:
    - Commit frequency trends
    - Issue activity
    - Repository growth
    - Developer sentiment from commit messages
    """

    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize monitor.

        Args:
            api_token: Optional GitHub API token for higher rate limits
        """
        self.api_token = api_token
        self.base_url = "https://api.github.com"
        self._cache: Dict[str, RepoActivity] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _make_request(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make authenticated request to GitHub API."""
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.api_token:
            headers["Authorization"] = f"token {self.api_token}"

        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                logger.warning("GitHub API rate limit exceeded")
            else:
                logger.warning(f"GitHub API error: {response.status_code}")
        except Exception as e:
            logger.error(f"GitHub API request failed: {e}")

        return None

    def get_repo_activity(self, owner: str, repo: str) -> Optional[RepoActivity]:
        """
        Get activity metrics for a single repository.

        Args:
            owner: Repository owner/organization
            repo: Repository name

        Returns:
            RepoActivity or None if unavailable
        """
        cache_key = f"{owner}/{repo}"

        # Check cache (15 minute TTL)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached_time = datetime.fromisoformat(cached.fetched_at)
            if datetime.now() - cached_time < timedelta(minutes=15):
                return cached

        # Fetch repo info
        repo_data = self._make_request(f"/repos/{owner}/{repo}")
        if not repo_data:
            return None

        # Fetch recent commits
        commits_data = self._make_request(f"/repos/{owner}/{repo}/commits?per_page=100")
        commits_7d = 0
        commits_30d = 0
        last_commit = None
        commit_messages = []

        if commits_data:
            now = datetime.now()
            for commit in commits_data:
                commit_date_str = commit.get("commit", {}).get("author", {}).get("date", "")
                if commit_date_str:
                    try:
                        commit_date = datetime.fromisoformat(commit_date_str.replace("Z", "+00:00"))
                        commit_date = commit_date.replace(tzinfo=None)
                        days_ago = (now - commit_date).days

                        if days_ago <= 7:
                            commits_7d += 1
                        if days_ago <= 30:
                            commits_30d += 1

                        if last_commit is None:
                            last_commit = commit_date_str

                        # Collect commit messages for sentiment
                        msg = commit.get("commit", {}).get("message", "")
                        if msg:
                            commit_messages.append(msg[:200])
                    except Exception:
                        continue

        # Calculate sentiment from commit messages
        sentiment_score = self._analyze_commit_sentiment(commit_messages)

        # Determine activity trend
        if commits_7d > commits_30d / 4 * 1.5:
            activity_trend = "increasing"
        elif commits_7d < commits_30d / 4 * 0.5:
            activity_trend = "decreasing"
        else:
            activity_trend = "stable"

        activity = RepoActivity(
            repo_name=repo,
            owner=owner,
            commits_7d=commits_7d,
            commits_30d=commits_30d,
            open_issues=repo_data.get("open_issues_count", 0),
            stars=repo_data.get("stargazers_count", 0),
            forks=repo_data.get("forks_count", 0),
            last_commit=last_commit,
            sentiment_score=sentiment_score,
            activity_trend=activity_trend,
        )

        self._cache[cache_key] = activity
        return activity

    def _analyze_commit_sentiment(self, messages: List[str]) -> float:
        """
        Analyze sentiment from commit messages.

        Returns score from -1 (negative) to +1 (positive).
        """
        if not messages:
            return 0.0

        positive_words = {
            "fix", "improve", "add", "implement", "feature", "enhance",
            "optimize", "update", "support", "enable", "complete"
        }
        negative_words = {
            "bug", "error", "crash", "fail", "broken", "issue", "problem",
            "hack", "workaround", "revert", "disable", "remove"
        }

        positive_count = 0
        negative_count = 0

        for msg in messages:
            words = set(msg.lower().split())
            positive_count += len(words & positive_words)
            negative_count += len(words & negative_words)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def get_company_activity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get aggregated GitHub activity for a company.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with aggregated activity metrics
        """
        profile = COMPANY_GITHUB_MAP.get(symbol)
        if not profile:
            logger.warning(f"No GitHub mapping for {symbol}")
            return None

        activities = []
        for repo_name in profile.key_repos[:5]:  # Top 5 repos
            activity = self.get_repo_activity(profile.github_org, repo_name)
            if activity:
                activities.append(activity)

        if not activities:
            return None

        # Aggregate metrics
        total_commits_7d = sum(a.commits_7d for a in activities)
        total_commits_30d = sum(a.commits_30d for a in activities)
        avg_sentiment = sum(a.sentiment_score for a in activities) / len(activities)

        # Count trend types
        trends = [a.activity_trend for a in activities]
        dominant_trend = max(set(trends), key=trends.count)

        # Calculate activity score (0-100)
        activity_score = min(100, (total_commits_7d / max(total_commits_30d / 4, 1)) * 50 + 50)

        return {
            "symbol": symbol,
            "company": profile.company_name,
            "github_org": profile.github_org,
            "repos_tracked": len(activities),
            "commits_7d": total_commits_7d,
            "commits_30d": total_commits_30d,
            "avg_sentiment": round(avg_sentiment, 3),
            "dominant_trend": dominant_trend,
            "activity_score": round(activity_score, 1),
            "signal": self._generate_signal(activity_score, avg_sentiment, dominant_trend),
            "fetched_at": datetime.now().isoformat(),
            "repo_details": [
                {
                    "repo": a.repo_name,
                    "commits_7d": a.commits_7d,
                    "sentiment": a.sentiment_score,
                    "trend": a.activity_trend,
                }
                for a in activities
            ],
        }

    def _generate_signal(
        self,
        activity_score: float,
        sentiment: float,
        trend: str
    ) -> Dict[str, Any]:
        """Generate trading signal from GitHub activity."""
        # High activity + positive sentiment + increasing trend = bullish
        score = 0

        if activity_score > 70:
            score += 1
        elif activity_score < 30:
            score -= 1

        if sentiment > 0.2:
            score += 1
        elif sentiment < -0.2:
            score -= 1

        if trend == "increasing":
            score += 1
        elif trend == "decreasing":
            score -= 1

        if score >= 2:
            direction = "BULLISH"
            strength = min(1.0, score / 3)
        elif score <= -2:
            direction = "BEARISH"
            strength = min(1.0, abs(score) / 3)
        else:
            direction = "NEUTRAL"
            strength = 0.0

        return {
            "direction": direction,
            "strength": round(strength, 2),
            "confidence": 0.4,  # GitHub data has moderate confidence
        }

    def scan_all_companies(self) -> List[Dict[str, Any]]:
        """Scan all mapped companies and return results."""
        results = []
        for symbol in COMPANY_GITHUB_MAP:
            activity = self.get_company_activity(symbol)
            if activity:
                results.append(activity)
        return results


# Singleton instance
_monitor: Optional[GitHubActivityMonitor] = None


def get_github_monitor(api_token: Optional[str] = None) -> GitHubActivityMonitor:
    """Get or create singleton monitor."""
    global _monitor
    if _monitor is None:
        _monitor = GitHubActivityMonitor(api_token)
    return _monitor


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    monitor = get_github_monitor()

    print("Scanning GitHub activity for tech companies...")
    for symbol in ["MSFT", "NVDA", "META"]:
        result = monitor.get_company_activity(symbol)
        if result:
            print(f"\n{symbol} ({result['company']}):")
            print(f"  Commits (7d): {result['commits_7d']}")
            print(f"  Commits (30d): {result['commits_30d']}")
            print(f"  Sentiment: {result['avg_sentiment']:.2f}")
            print(f"  Trend: {result['dominant_trend']}")
            print(f"  Activity Score: {result['activity_score']}")
            print(f"  Signal: {result['signal']}")
