"""
External Research Scrapers for Autonomous Learning Brain.

This module contains scrapers for fetching trading strategy ideas from:
- GitHub: Open-source trading strategy repositories
- Reddit: r/algotrading, r/quant community discussions
- YouTube: Trading education video transcripts
- arXiv: Academic research papers on quantitative finance

ALL scraped data is used for HYPOTHESIS GENERATION only.
ALL strategies MUST be validated with REAL backtest data.
NO synthetic or fake data is EVER used.
"""

from .github_scraper import GitHubScraper, scrape_github_strategies
from .reddit_scraper import RedditScraper, scrape_reddit_ideas
from .youtube_scraper import YouTubeScraper, scrape_youtube_strategies
from .arxiv_scraper import ArxivScraper, scrape_arxiv_papers
from .source_manager import SourceManager, ExternalIdea

__all__ = [
    "GitHubScraper",
    "RedditScraper",
    "YouTubeScraper",
    "ArxivScraper",
    "SourceManager",
    "ExternalIdea",
    "scrape_github_strategies",
    "scrape_reddit_ideas",
    "scrape_youtube_strategies",
    "scrape_arxiv_papers",
]
