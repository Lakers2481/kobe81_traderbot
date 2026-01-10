"""
Firecrawl Unified Web Ingestion Adapter
=======================================

Single API for all web sources -> LLM-ready markdown.

Replaces fragmented scrapers with unified pipeline:
- Anti-bot bypass built-in
- JavaScript rendering
- Structured data extraction with schemas
- PDF/document parsing
- Batch processing with rate limiting

USAGE:
    from autonomous.scrapers.firecrawl_adapter import FirecrawlAdapter, get_firecrawl

    adapter = get_firecrawl()

    # Simple scrape
    result = adapter.scrape_url("https://github.com/user/repo")

    # Structured extraction
    strategy = adapter.extract_strategy("https://github.com/user/trading-bot")

    # Batch scrape
    results = adapter.batch_scrape(["url1", "url2", "url3"])

Created: 2026-01-07
Based on: Firecrawl API (firecrawl.dev)
"""

from __future__ import annotations

import os
import logging
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import Firecrawl (paid API)
try:
    from firecrawl import FirecrawlApp
    HAS_FIRECRAWL = True
except ImportError:
    HAS_FIRECRAWL = False
    FirecrawlApp = None

# Try to import Trafilatura (FREE - no API key needed)
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    trafilatura = None
    logger.warning("Trafilatura not installed. Run: pip install trafilatura")


class ContentFormat(Enum):
    """Output formats for scraped content."""
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    EXTRACT = "extract"  # Structured extraction


@dataclass
class ScrapeResult:
    """Result from a single scrape operation."""
    url: str
    success: bool
    content: str = ""
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_data: Optional[Dict] = None
    error: Optional[str] = None
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'success': self.success,
            'content': self.content[:1000] + '...' if len(self.content) > 1000 else self.content,
            'title': self.title,
            'metadata': self.metadata,
            'extracted_data': self.extracted_data,
            'error': self.error,
            'scraped_at': self.scraped_at,
        }


@dataclass
class StrategyExtraction:
    """Extracted trading strategy from web content."""
    source_url: str
    strategy_name: str = ""
    strategy_type: str = ""  # momentum, mean_reversion, breakout, etc.
    entry_rules: List[str] = field(default_factory=list)
    exit_rules: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    timeframe: str = ""
    backtest_results: Optional[Dict] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_content: str = ""
    confidence: float = 0.0
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_url': self.source_url,
            'strategy_name': self.strategy_name,
            'strategy_type': self.strategy_type,
            'entry_rules': self.entry_rules,
            'exit_rules': self.exit_rules,
            'indicators': self.indicators,
            'timeframe': self.timeframe,
            'backtest_results': self.backtest_results,
            'parameters': self.parameters,
            'confidence': self.confidence,
            'extracted_at': self.extracted_at,
        }


# Schema for structured strategy extraction
STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy_name": {
            "type": "string",
            "description": "Name of the trading strategy"
        },
        "strategy_type": {
            "type": "string",
            "enum": ["momentum", "mean_reversion", "breakout", "trend_following", "arbitrage", "market_making", "other"],
            "description": "Category of the strategy"
        },
        "entry_rules": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of entry conditions/rules"
        },
        "exit_rules": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of exit conditions/rules"
        },
        "indicators": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Technical indicators used (RSI, MACD, etc.)"
        },
        "timeframe": {
            "type": "string",
            "description": "Trading timeframe (1m, 5m, 1h, 1d, etc.)"
        },
        "parameters": {
            "type": "object",
            "description": "Strategy parameters and their values"
        },
        "backtest_results": {
            "type": "object",
            "properties": {
                "win_rate": {"type": "number"},
                "profit_factor": {"type": "number"},
                "sharpe_ratio": {"type": "number"},
                "max_drawdown": {"type": "number"},
                "total_return": {"type": "number"},
            },
            "description": "Backtest performance metrics if mentioned"
        }
    },
    "required": ["strategy_name"]
}


class FirecrawlAdapter:
    """
    Unified web ingestion using Firecrawl API.

    Features:
    - Single API for any web source
    - Anti-bot bypass
    - JavaScript rendering
    - Structured extraction with schemas
    - Caching to avoid redundant scrapes
    - Rate limiting
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize Firecrawl adapter.

        Args:
            api_key: Firecrawl API key (or from FIRECRAWL_API_KEY env var)
            cache_dir: Directory for caching scraped content
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.api_key = api_key or os.environ.get('FIRECRAWL_API_KEY', '')
        self.cache_dir = cache_dir or Path("state/firecrawl_cache")
        self.cache_ttl_hours = cache_ttl_hours

        self._client: Optional[FirecrawlApp] = None
        self._scrape_count = 0

        if self.api_key:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("FirecrawlAdapter initialized with API key")
        else:
            logger.warning("No FIRECRAWL_API_KEY found - using fallback mode")

    @property
    def client(self) -> Optional[FirecrawlApp]:
        """Lazy-load Firecrawl client."""
        if self._client is None and HAS_FIRECRAWL and self.api_key:
            self._client = FirecrawlApp(api_key=self.api_key)
        return self._client

    def scrape_url(
        self,
        url: str,
        formats: List[str] = ["markdown"],
        use_cache: bool = True,
    ) -> ScrapeResult:
        """
        Scrape a single URL and return LLM-ready content.

        Args:
            url: URL to scrape
            formats: Output formats (markdown, html, extract)
            use_cache: Whether to use cached results

        Returns:
            ScrapeResult with content and metadata
        """
        # Check cache
        if use_cache:
            cached = self._get_cached(url)
            if cached:
                logger.debug(f"Cache hit for {url}")
                return cached

        # If no Firecrawl client, use fallback
        if not self.client:
            return self._fallback_scrape(url)

        try:
            result = self.client.scrape_url(url, params={
                'formats': formats,
                'onlyMainContent': True,
            })

            scrape_result = ScrapeResult(
                url=url,
                success=True,
                content=result.get('markdown', result.get('content', '')),
                title=result.get('metadata', {}).get('title', ''),
                metadata=result.get('metadata', {}),
            )

            self._scrape_count += 1

            # Cache result
            if use_cache:
                self._cache_result(url, scrape_result)

            logger.info(f"Scraped: {url[:50]}...")
            return scrape_result

        except Exception as e:
            logger.error(f"Firecrawl scrape failed: {e}")
            return ScrapeResult(
                url=url,
                success=False,
                error=str(e),
            )

    def extract_strategy(
        self,
        url: str,
        custom_schema: Optional[Dict] = None,
    ) -> StrategyExtraction:
        """
        Extract structured trading strategy from a URL.

        Uses Firecrawl's extraction mode to pull out:
        - Strategy name and type
        - Entry/exit rules
        - Indicators used
        - Parameters
        - Backtest results (if present)

        Args:
            url: URL containing strategy information
            custom_schema: Custom extraction schema (optional)

        Returns:
            StrategyExtraction with structured data
        """
        schema = custom_schema or STRATEGY_SCHEMA

        if not self.client:
            # Fallback: scrape and return raw for LLM extraction
            result = self._fallback_scrape(url)
            return StrategyExtraction(
                source_url=url,
                raw_content=result.content,
                confidence=0.3,
            )

        try:
            result = self.client.scrape_url(url, params={
                'formats': ['markdown', 'extract'],
                'extract': {
                    'schema': schema,
                    'prompt': "Extract trading strategy details from this content. Focus on entry rules, exit rules, indicators, and any backtest results mentioned."
                }
            })

            extracted = result.get('extract', {})

            return StrategyExtraction(
                source_url=url,
                strategy_name=extracted.get('strategy_name', ''),
                strategy_type=extracted.get('strategy_type', 'other'),
                entry_rules=extracted.get('entry_rules', []),
                exit_rules=extracted.get('exit_rules', []),
                indicators=extracted.get('indicators', []),
                timeframe=extracted.get('timeframe', ''),
                backtest_results=extracted.get('backtest_results'),
                parameters=extracted.get('parameters', {}),
                raw_content=result.get('markdown', ''),
                confidence=0.8 if extracted.get('strategy_name') else 0.3,
            )

        except Exception as e:
            logger.error(f"Strategy extraction failed: {e}")
            return StrategyExtraction(
                source_url=url,
                confidence=0.0,
            )

    def batch_scrape(
        self,
        urls: List[str],
        formats: List[str] = ["markdown"],
        max_concurrent: int = 5,
    ) -> List[ScrapeResult]:
        """
        Batch scrape multiple URLs.

        Args:
            urls: List of URLs to scrape
            formats: Output formats
            max_concurrent: Max concurrent requests (rate limiting)

        Returns:
            List of ScrapeResults
        """
        results = []

        for i, url in enumerate(urls):
            result = self.scrape_url(url, formats, use_cache=True)
            results.append(result)

            # Simple rate limiting
            if i > 0 and i % max_concurrent == 0:
                import time
                time.sleep(1)

        logger.info(f"Batch scraped {len(urls)} URLs, {sum(1 for r in results if r.success)} successful")
        return results

    def crawl_site(
        self,
        start_url: str,
        max_pages: int = 10,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[ScrapeResult]:
        """
        Crawl a website starting from a URL.

        Args:
            start_url: Starting URL
            max_pages: Maximum pages to crawl
            include_patterns: URL patterns to include (regex)
            exclude_patterns: URL patterns to exclude (regex)

        Returns:
            List of ScrapeResults for crawled pages
        """
        if not self.client:
            logger.warning("Crawl requires Firecrawl API - using single page fallback")
            return [self._fallback_scrape(start_url)]

        try:
            crawl_params = {
                'limit': max_pages,
                'scrapeOptions': {
                    'formats': ['markdown'],
                    'onlyMainContent': True,
                }
            }

            if include_patterns:
                crawl_params['includePaths'] = include_patterns
            if exclude_patterns:
                crawl_params['excludePaths'] = exclude_patterns

            # Start crawl job
            crawl_result = self.client.crawl_url(start_url, params=crawl_params)

            results = []
            for page in crawl_result.get('data', []):
                results.append(ScrapeResult(
                    url=page.get('metadata', {}).get('sourceURL', start_url),
                    success=True,
                    content=page.get('markdown', ''),
                    title=page.get('metadata', {}).get('title', ''),
                    metadata=page.get('metadata', {}),
                ))

            logger.info(f"Crawled {len(results)} pages from {start_url}")
            return results

        except Exception as e:
            logger.error(f"Crawl failed: {e}")
            return [ScrapeResult(url=start_url, success=False, error=str(e))]

    def scrape_github_readme(self, repo_url: str) -> ScrapeResult:
        """
        Specialized scraper for GitHub repository READMEs.

        Args:
            repo_url: GitHub repo URL (e.g., https://github.com/user/repo)

        Returns:
            ScrapeResult with README content
        """
        # Normalize to raw README URL
        if 'github.com' in repo_url:
            parts = repo_url.rstrip('/').split('/')
            if len(parts) >= 5:
                user = parts[3]
                repo = parts[4]
                raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/main/README.md"

                # Try main first, then master
                result = self.scrape_url(raw_url)
                if not result.success:
                    raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/master/README.md"
                    result = self.scrape_url(raw_url)

                result.metadata['github_repo'] = f"{user}/{repo}"
                return result

        # Fallback to direct scrape
        return self.scrape_url(repo_url)

    def scrape_arxiv_paper(self, arxiv_url: str) -> ScrapeResult:
        """
        Specialized scraper for arXiv papers.

        Args:
            arxiv_url: arXiv URL or paper ID

        Returns:
            ScrapeResult with paper abstract and content
        """
        # Normalize URL
        if 'arxiv.org' in arxiv_url:
            # Convert to abstract page for easier parsing
            arxiv_url = arxiv_url.replace('/pdf/', '/abs/')

        result = self.scrape_url(arxiv_url)
        result.metadata['source_type'] = 'arxiv'
        return result

    # ========== PRIVATE METHODS ==========

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cached(self, url: str) -> Optional[ScrapeResult]:
        """Get cached result if valid."""
        cache_key = self._get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)

                # Check TTL
                scraped_at = datetime.fromisoformat(data.get('scraped_at', '2000-01-01'))
                age_hours = (datetime.now() - scraped_at).total_seconds() / 3600

                if age_hours < self.cache_ttl_hours:
                    return ScrapeResult(**data)
            except Exception:
                pass

        return None

    def _cache_result(self, url: str, result: ScrapeResult):
        """Cache a scrape result."""
        cache_key = self._get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'url': result.url,
                    'success': result.success,
                    'content': result.content,
                    'title': result.title,
                    'metadata': result.metadata,
                    'extracted_data': result.extracted_data,
                    'error': result.error,
                    'scraped_at': result.scraped_at,
                }, f)
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    def _fallback_scrape(self, url: str) -> ScrapeResult:
        """
        FREE fallback scraping using Trafilatura (no API key needed).

        Trafilatura provides:
        - Clean text extraction from any web page
        - Automatic boilerplate removal (ads, navs, footers)
        - Metadata extraction (title, author, date)
        - Works on most websites without JS rendering

        Falls back to requests+BeautifulSoup if trafilatura unavailable.
        """
        # Try Trafilatura first (best free option)
        if HAS_TRAFILATURA:
            try:
                # Download and extract in one step
                downloaded = trafilatura.fetch_url(url)

                if downloaded:
                    # Extract text content (clean, LLM-ready)
                    text_content = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=True,
                        no_fallback=False,
                        favor_precision=True,
                    )

                    # Get metadata
                    metadata_result = trafilatura.extract_metadata(downloaded)

                    title = ""
                    metadata = {'scraper': 'trafilatura', 'free_mode': True}

                    if metadata_result:
                        title = metadata_result.title or ""
                        metadata.update({
                            'author': metadata_result.author,
                            'date': metadata_result.date,
                            'sitename': metadata_result.sitename,
                            'description': metadata_result.description,
                        })

                    if text_content:
                        return ScrapeResult(
                            url=url,
                            success=True,
                            content=text_content,
                            title=title,
                            metadata=metadata,
                        )

            except Exception as e:
                logger.debug(f"Trafilatura failed, trying BeautifulSoup: {e}")

        # Fallback to requests + BeautifulSoup
        try:
            import requests

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            content = response.text

            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')

                # Remove scripts, styles, nav, footer
                for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header']):
                    element.decompose()

                title = soup.title.string if soup.title else ''
                text_content = soup.get_text(separator='\n', strip=True)

                return ScrapeResult(
                    url=url,
                    success=True,
                    content=text_content,
                    title=title,
                    metadata={'scraper': 'beautifulsoup', 'free_mode': True},
                )
            except ImportError:
                return ScrapeResult(
                    url=url,
                    success=True,
                    content=content,
                    metadata={'scraper': 'raw', 'free_mode': True, 'raw_html': True},
                )

        except Exception as e:
            return ScrapeResult(
                url=url,
                success=False,
                error=f"Fallback scrape failed: {e}",
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        cache_files = list(self.cache_dir.glob("*.json")) if self.cache_dir.exists() else []
        return {
            'has_api_key': bool(self.api_key),
            'has_firecrawl': HAS_FIRECRAWL,
            'has_trafilatura': HAS_TRAFILATURA,  # FREE fallback
            'free_mode': not bool(self.api_key),  # Using free scrapers
            'scrape_count': self._scrape_count,
            'cache_size': len(cache_files),
        }


# Singleton instance
_firecrawl_instance: Optional[FirecrawlAdapter] = None


def get_firecrawl(api_key: Optional[str] = None) -> FirecrawlAdapter:
    """
    Get the singleton Firecrawl adapter instance.

    Args:
        api_key: Optional API key override

    Returns:
        FirecrawlAdapter instance
    """
    global _firecrawl_instance
    if _firecrawl_instance is None:
        _firecrawl_instance = FirecrawlAdapter(api_key=api_key)
    return _firecrawl_instance


def scrape_for_strategies(urls: List[str]) -> List[StrategyExtraction]:
    """
    Convenience function to extract strategies from multiple URLs.

    Args:
        urls: List of URLs to analyze

    Returns:
        List of StrategyExtraction results
    """
    adapter = get_firecrawl()
    return [adapter.extract_strategy(url) for url in urls]


if __name__ == '__main__':
    # Test the adapter
    print(f"Firecrawl library available: {HAS_FIRECRAWL}")

    adapter = FirecrawlAdapter()
    stats = adapter.get_stats()
    print(f"Adapter stats: {stats}")

    # Test fallback scrape
    result = adapter.scrape_url("https://example.com")
    print(f"\nTest scrape success: {result.success}")
    print(f"Content length: {len(result.content)}")
