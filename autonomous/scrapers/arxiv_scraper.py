"""
arXiv Research Paper Scraper.

Fetches quantitative finance research papers from arXiv.
Uses arXiv API (FREE, no authentication required).

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
import xml.etree.ElementTree as ET

from core.structured_log import jlog


@dataclass
class ArxivPaper:
    """An arXiv paper about quantitative finance."""
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    url: str
    pdf_url: str
    categories: List[str]
    published: str
    updated: str
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ArxivScraper:
    """
    Scrapes arXiv for quantitative finance papers.

    Uses arXiv API (no auth required).
    Rate limited to respect arXiv guidelines.
    """

    # arXiv categories for quantitative finance
    CATEGORIES = [
        "q-fin.TR",   # Trading and Market Microstructure
        "q-fin.PM",   # Portfolio Management
        "q-fin.ST",   # Statistical Finance
        "q-fin.CP",   # Computational Finance
    ]

    # Search terms for trading strategies
    SEARCH_TERMS = [
        "algorithmic trading",
        "mean reversion",
        "momentum strategy",
        "trading signal",
        "technical analysis",
        "stock prediction",
        "backtesting"
    ]

    # arXiv API endpoint
    API_URL = "http://export.arxiv.org/api/query"

    # Rate limiting
    REQUEST_INTERVAL = 3.0  # seconds between requests

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("state/autonomous/scrapers/arxiv")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0
        self._session = requests.Session()

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get_cache_path(self, query_hash: str) -> Path:
        """Get cache file path for a query."""
        return self.cache_dir / f"search_{query_hash}.json"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """Check if cache is still valid."""
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < timedelta(hours=max_age_hours)

    def _parse_entry(self, entry: ET.Element, ns: dict) -> Optional[ArxivPaper]:
        """Parse a single arXiv entry from XML."""
        try:
            # Get paper ID from URL
            paper_id = entry.find("atom:id", ns)
            if paper_id is None or paper_id.text is None:
                return None

            arxiv_id = paper_id.text.split("/abs/")[-1]

            # Get title
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""

            # Get authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None and name.text:
                    authors.append(name.text)

            # Get abstract
            summary = entry.find("atom:summary", ns)
            abstract = summary.text.strip().replace("\n", " ") if summary is not None else ""

            # Get links
            url = paper_id.text
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            # Get categories
            categories = []
            for cat in entry.findall("atom:category", ns):
                term = cat.get("term", "")
                if term:
                    categories.append(term)

            # Get dates
            published = entry.find("atom:published", ns)
            updated = entry.find("atom:updated", ns)

            return ArxivPaper(
                paper_id=f"arxiv:{arxiv_id}",
                title=title,
                authors=authors[:5],  # Limit to first 5 authors
                abstract=abstract[:3000] if abstract else "",  # Truncate
                url=url,
                pdf_url=pdf_url,
                categories=categories,
                published=published.text if published is not None else "",
                updated=updated.text if updated is not None else ""
            )

        except Exception as e:
            jlog("arxiv_parse_error", level="DEBUG", error=str(e))
            return None

    def search_papers(self, query: str, max_results: int = 10) -> List[ArxivPaper]:
        """
        Search arXiv for papers matching query.

        Args:
            query: Search query string
            max_results: Maximum papers to return

        Returns:
            List of ArxivPaper objects
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        cache_path = self._get_cache_path(query_hash)

        # Check cache first
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                jlog("arxiv_cache_hit", level="DEBUG", query=query[:50])
                return [ArxivPaper(**p) for p in cached["papers"][:max_results]]
            except Exception as e:
                jlog("arxiv_cache_error", level="WARNING", error=str(e))

        # Fetch from API
        self._rate_limit()

        try:
            # Build search query
            # Search in q-fin categories
            cat_query = " OR ".join([f"cat:{cat}" for cat in self.CATEGORIES])
            full_query = f"({cat_query}) AND all:{query}"

            params = {
                "search_query": full_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }

            resp = self._session.get(self.API_URL, params=params, timeout=30)
            resp.raise_for_status()

            # Parse XML response
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.content)

            papers = []
            for entry in root.findall("atom:entry", ns):
                paper = self._parse_entry(entry, ns)
                if paper:
                    papers.append(paper)

            # Cache results
            with open(cache_path, "w") as f:
                json.dump({
                    "query": query,
                    "fetched_at": datetime.now().isoformat(),
                    "papers": [asdict(p) for p in papers]
                }, f, indent=2)

            jlog("arxiv_search_complete", level="INFO",
                 query=query[:50], papers_found=len(papers))

            return papers

        except requests.RequestException as e:
            jlog("arxiv_search_error", level="ERROR",
                 query=query[:50], error=str(e))
            return []

    def fetch_recent_papers(self, category: str = "q-fin.TR", max_results: int = 10) -> List[ArxivPaper]:
        """
        Fetch recent papers from a category.

        Args:
            category: arXiv category (e.g., q-fin.TR)
            max_results: Maximum papers to return

        Returns:
            List of ArxivPaper objects
        """
        cache_path = self._get_cache_path(f"recent_{category}")

        # Check cache first
        if self._is_cache_valid(cache_path, max_age_hours=12):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                jlog("arxiv_cache_hit", level="DEBUG", category=category)
                return [ArxivPaper(**p) for p in cached["papers"][:max_results]]
            except Exception as e:
                jlog("arxiv_cache_error", level="WARNING", error=str(e))

        # Fetch from API
        self._rate_limit()

        try:
            params = {
                "search_query": f"cat:{category}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }

            resp = self._session.get(self.API_URL, params=params, timeout=30)
            resp.raise_for_status()

            # Parse XML response
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.content)

            papers = []
            for entry in root.findall("atom:entry", ns):
                paper = self._parse_entry(entry, ns)
                if paper:
                    papers.append(paper)

            # Cache results
            with open(cache_path, "w") as f:
                json.dump({
                    "category": category,
                    "fetched_at": datetime.now().isoformat(),
                    "papers": [asdict(p) for p in papers]
                }, f, indent=2)

            jlog("arxiv_recent_fetched", level="INFO",
                 category=category, papers_found=len(papers))

            return papers

        except requests.RequestException as e:
            jlog("arxiv_fetch_error", level="ERROR",
                 category=category, error=str(e))
            return []

    def scrape_all(self, papers_per_query: int = 5) -> List[ArxivPaper]:
        """
        Scrape all configured search terms and categories.

        Args:
            papers_per_query: Max papers per search query

        Returns:
            List of unique ArxivPaper objects
        """
        all_papers = {}

        # Fetch recent papers from each category
        for category in self.CATEGORIES:
            papers = self.fetch_recent_papers(category, papers_per_query)
            for paper in papers:
                if paper.paper_id not in all_papers:
                    all_papers[paper.paper_id] = paper

        # Search for specific terms
        for term in self.SEARCH_TERMS:
            papers = self.search_papers(term, papers_per_query)
            for paper in papers:
                if paper.paper_id not in all_papers:
                    all_papers[paper.paper_id] = paper

        jlog("arxiv_scrape_complete", level="INFO",
             total_papers=len(all_papers),
             categories=len(self.CATEGORIES),
             search_terms=len(self.SEARCH_TERMS))

        return list(all_papers.values())


def scrape_arxiv_papers() -> List[Dict[str, Any]]:
    """
    Handler function for autonomous brain task.

    Returns:
        List of strategy ideas extracted from arXiv papers
    """
    scraper = ArxivScraper()
    papers = scraper.scrape_all(papers_per_query=3)

    ideas = []
    for paper in papers:
        ideas.append({
            "source_type": "arxiv",
            "source_id": paper.paper_id,
            "source_url": paper.url,
            "title": paper.title,
            "description": f"Authors: {', '.join(paper.authors[:3])} | Categories: {', '.join(paper.categories[:3])}",
            "content": paper.abstract,
            "metadata": {
                "authors": paper.authors,
                "categories": paper.categories,
                "published": paper.published,
                "pdf_url": paper.pdf_url
            },
            "fetched_at": datetime.now().isoformat()
        })

    return ideas
