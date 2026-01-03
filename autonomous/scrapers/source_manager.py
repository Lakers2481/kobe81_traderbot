"""
Source Manager - Coordinates all external research scrapers.

This module orchestrates fetching ideas from all external sources:
- GitHub strategy repositories
- Reddit trading discussions
- YouTube educational videos
- arXiv research papers

All ideas are validated with REAL backtest data - NO synthetic data ever.
"""

import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json

from core.structured_log import jlog

from .github_scraper import GitHubScraper, scrape_github_strategies
from .reddit_scraper import RedditScraper, scrape_reddit_ideas
from .youtube_scraper import YouTubeScraper, scrape_youtube_strategies
from .arxiv_scraper import ArxivScraper, scrape_arxiv_papers


@dataclass
class ExternalIdea:
    """
    A trading strategy idea from an external source.

    This represents raw content that needs LLM extraction and
    validation with REAL backtest data before becoming actionable.
    """
    idea_id: str
    source_type: str  # github, reddit, youtube, arxiv
    source_id: str    # Unique ID within source
    source_url: str   # Link to original
    title: str
    description: str
    content: str      # Full content for LLM extraction
    metadata: Dict[str, Any] = field(default_factory=dict)
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Processing status
    extracted: bool = False
    extraction_result: Optional[Dict] = None
    validated: bool = False
    validation_result: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ExternalIdea":
        """Create from dictionary."""
        return cls(**data)


class SourceManager:
    """
    Manages external research source scraping and idea collection.

    Coordinates:
    - Parallel scraping from multiple sources
    - Deduplication of ideas
    - Idea queue management
    - Source rotation and rate limiting
    """

    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or Path("state/autonomous/external_ideas")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.ideas_file = self.state_dir / "ideas_queue.json"
        self.processed_file = self.state_dir / "processed_ideas.json"

        # Initialize scrapers
        self.github_scraper = GitHubScraper()
        self.reddit_scraper = RedditScraper()
        self.youtube_scraper = YouTubeScraper()
        self.arxiv_scraper = ArxivScraper()

        # Load existing ideas
        self.ideas_queue: List[ExternalIdea] = self._load_ideas()
        self.processed_ids: set = self._load_processed_ids()

    def _load_ideas(self) -> List[ExternalIdea]:
        """Load ideas queue from disk."""
        if self.ideas_file.exists():
            try:
                with open(self.ideas_file) as f:
                    data = json.load(f)
                return [ExternalIdea.from_dict(d) for d in data]
            except Exception as e:
                jlog("source_manager_load_error", level="WARNING", error=str(e))
        return []

    def _save_ideas(self):
        """Save ideas queue to disk."""
        with open(self.ideas_file, "w") as f:
            json.dump([i.to_dict() for i in self.ideas_queue], f, indent=2)

    def _load_processed_ids(self) -> set:
        """Load set of processed idea IDs."""
        if self.processed_file.exists():
            try:
                with open(self.processed_file) as f:
                    return set(json.load(f))
            except Exception as e:
                jlog("source_manager_load_error", level="WARNING", error=str(e))
        return set()

    def _save_processed_ids(self):
        """Save processed IDs to disk."""
        with open(self.processed_file, "w") as f:
            json.dump(list(self.processed_ids), f)

    def _raw_to_idea(self, raw: Dict[str, Any]) -> ExternalIdea:
        """Convert raw scraper output to ExternalIdea."""
        import hashlib

        # Generate unique idea ID
        content_hash = hashlib.md5(
            f"{raw['source_id']}:{raw['title']}".encode()
        ).hexdigest()[:8]

        idea_id = f"idea_{raw['source_type']}_{content_hash}"

        return ExternalIdea(
            idea_id=idea_id,
            source_type=raw["source_type"],
            source_id=raw["source_id"],
            source_url=raw["source_url"],
            title=raw["title"],
            description=raw.get("description", ""),
            content=raw.get("content", ""),
            metadata=raw.get("metadata", {}),
            fetched_at=raw.get("fetched_at", datetime.now().isoformat())
        )

    def scrape_github(self) -> List[ExternalIdea]:
        """Scrape GitHub for strategy ideas."""
        jlog("source_manager_scraping", level="INFO", source="github")

        raw_ideas = scrape_github_strategies()
        ideas = []

        for raw in raw_ideas:
            idea = self._raw_to_idea(raw)
            if idea.idea_id not in self.processed_ids:
                ideas.append(idea)

        jlog("source_manager_scraped", level="INFO",
             source="github", new_ideas=len(ideas))

        return ideas

    def scrape_reddit(self) -> List[ExternalIdea]:
        """Scrape Reddit for strategy ideas."""
        jlog("source_manager_scraping", level="INFO", source="reddit")

        raw_ideas = scrape_reddit_ideas()
        ideas = []

        for raw in raw_ideas:
            idea = self._raw_to_idea(raw)
            if idea.idea_id not in self.processed_ids:
                ideas.append(idea)

        jlog("source_manager_scraped", level="INFO",
             source="reddit", new_ideas=len(ideas))

        return ideas

    def scrape_youtube(self) -> List[ExternalIdea]:
        """Scrape YouTube for strategy ideas."""
        jlog("source_manager_scraping", level="INFO", source="youtube")

        raw_ideas = scrape_youtube_strategies()
        ideas = []

        for raw in raw_ideas:
            idea = self._raw_to_idea(raw)
            if idea.idea_id not in self.processed_ids:
                ideas.append(idea)

        jlog("source_manager_scraped", level="INFO",
             source="youtube", new_ideas=len(ideas))

        return ideas

    def scrape_arxiv(self) -> List[ExternalIdea]:
        """Scrape arXiv for strategy ideas."""
        jlog("source_manager_scraping", level="INFO", source="arxiv")

        raw_ideas = scrape_arxiv_papers()
        ideas = []

        for raw in raw_ideas:
            idea = self._raw_to_idea(raw)
            if idea.idea_id not in self.processed_ids:
                ideas.append(idea)

        jlog("source_manager_scraped", level="INFO",
             source="arxiv", new_ideas=len(ideas))

        return ideas

    def scrape_all_sources(self) -> List[ExternalIdea]:
        """
        Scrape all configured sources for new ideas.

        Returns:
            List of new ExternalIdea objects (not previously processed)
        """
        jlog("source_manager_scrape_all_start", level="INFO")

        all_ideas = []

        # Scrape each source with delays between
        for scrape_func, source_name in [
            (self.scrape_github, "github"),
            (self.scrape_reddit, "reddit"),
            (self.scrape_youtube, "youtube"),
            (self.scrape_arxiv, "arxiv"),
        ]:
            try:
                ideas = scrape_func()
                all_ideas.extend(ideas)
            except Exception as e:
                jlog("source_manager_scrape_error", level="ERROR",
                     source=source_name, error=str(e))

            time.sleep(2)  # Delay between sources

        # Add new ideas to queue
        for idea in all_ideas:
            if idea.idea_id not in {i.idea_id for i in self.ideas_queue}:
                self.ideas_queue.append(idea)

        # Save updated queue
        self._save_ideas()

        jlog("source_manager_scrape_all_complete", level="INFO",
             total_new_ideas=len(all_ideas),
             queue_size=len(self.ideas_queue))

        return all_ideas

    def get_pending_ideas(self, limit: int = 10) -> List[ExternalIdea]:
        """
        Get ideas that haven't been extracted yet.

        Args:
            limit: Maximum number of ideas to return

        Returns:
            List of unprocessed ExternalIdea objects
        """
        pending = [i for i in self.ideas_queue if not i.extracted]
        return pending[:limit]

    def mark_extracted(self, idea_id: str, extraction_result: Dict):
        """Mark an idea as extracted with results."""
        for idea in self.ideas_queue:
            if idea.idea_id == idea_id:
                idea.extracted = True
                idea.extraction_result = extraction_result
                break

        self._save_ideas()

    def mark_validated(self, idea_id: str, validation_result: Dict):
        """Mark an idea as validated with results."""
        for idea in self.ideas_queue:
            if idea.idea_id == idea_id:
                idea.validated = True
                idea.validation_result = validation_result
                break

        # Move to processed
        self.processed_ids.add(idea_id)
        self._save_processed_ids()
        self._save_ideas()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected ideas."""
        by_source = {}
        extracted_count = 0
        validated_count = 0
        successful_count = 0

        for idea in self.ideas_queue:
            source = idea.source_type
            by_source[source] = by_source.get(source, 0) + 1

            if idea.extracted:
                extracted_count += 1
            if idea.validated:
                validated_count += 1
                if idea.validation_result and idea.validation_result.get("success"):
                    successful_count += 1

        return {
            "total_ideas": len(self.ideas_queue),
            "processed_ideas": len(self.processed_ids),
            "by_source": by_source,
            "extracted": extracted_count,
            "validated": validated_count,
            "successful": successful_count,
            "pending_extraction": len(self.ideas_queue) - extracted_count
        }
