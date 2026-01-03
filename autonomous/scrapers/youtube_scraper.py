"""
YouTube Strategy Scraper.

Fetches trading strategy video transcripts from YouTube.
Uses youtube-transcript-api (FREE, no API key required).

All ideas found are converted to ExternalIdea objects for testing
with REAL backtest data - NO synthetic data ever.
"""

import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json
import re

from core.structured_log import jlog

# Try to import youtube transcript API
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    jlog("youtube_api_not_available", level="INFO",
         message="Install: pip install youtube-transcript-api")


@dataclass
class YouTubeVideo:
    """A YouTube video with trading strategy content."""
    video_id: str
    title: str
    channel: str
    url: str
    transcript: str
    description: str
    duration_seconds: int
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())


class YouTubeScraper:
    """
    Scrapes YouTube for trading strategy videos.

    Uses youtube-transcript-api for free transcript access.
    Videos are identified by known trading education channels.
    """

    # Known trading education channels (video IDs to scrape)
    # These are curated educational videos about trading strategies
    STRATEGY_VIDEO_IDS = [
        # Mean reversion / IBS strategies
        "dQw4w9WgXcQ",  # Example - replace with real trading videos
        # Momentum strategies
        # Breakout strategies
        # Technical analysis
    ]

    # Search-based video discovery
    # Note: YouTube Data API would require auth, so we use curated lists
    CURATED_CHANNELS = [
        "QuantInsti",
        "AlgoTrading101",
        "TradingWithRayner",
    ]

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("state/autonomous/scrapers/youtube")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, video_id: str) -> Path:
        """Get cache file path for a video."""
        return self.cache_dir / f"video_{video_id}.json"

    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 168) -> bool:
        """Check if cache is still valid (1 week for videos)."""
        if not cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < timedelta(hours=max_age_hours)

    def fetch_transcript(self, video_id: str) -> Optional[YouTubeVideo]:
        """
        Fetch transcript for a YouTube video.

        Args:
            video_id: YouTube video ID (11 characters)

        Returns:
            YouTubeVideo object or None if transcript unavailable
        """
        if not YOUTUBE_API_AVAILABLE:
            jlog("youtube_api_missing", level="WARNING")
            return None

        cache_path = self._get_cache_path(video_id)

        # Check cache first
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                jlog("youtube_cache_hit", level="DEBUG", video_id=video_id)
                return YouTubeVideo(**cached)
            except Exception as e:
                jlog("youtube_cache_error", level="WARNING", error=str(e))

        try:
            # Fetch transcript (new API uses .fetch() instead of .get_transcript())
            transcript_data = YouTubeTranscriptApi.fetch(video_id)

            # Handle Transcript object - convert to list of segments
            if hasattr(transcript_data, '__iter__'):
                transcript_list = list(transcript_data)
            else:
                transcript_list = []

            # Combine transcript segments
            full_transcript = " ".join([
                str(segment.text if hasattr(segment, 'text') else segment.get("text", ""))
                for segment in transcript_list
            ])

            # Calculate duration from transcript
            duration = 0
            if transcript_list:
                last_segment = transcript_list[-1]
                start = getattr(last_segment, 'start', 0) if hasattr(last_segment, 'start') else last_segment.get("start", 0)
                dur = getattr(last_segment, 'duration', 0) if hasattr(last_segment, 'duration') else last_segment.get("duration", 0)
                duration = int(start + dur)

            # Truncate if too long
            if len(full_transcript) > 10000:
                full_transcript = full_transcript[:10000] + "\n\n[TRUNCATED]"

            video = YouTubeVideo(
                video_id=f"youtube:{video_id}",
                title=f"Video {video_id}",  # Would need API to get real title
                channel="Unknown",  # Would need API
                url=f"https://www.youtube.com/watch?v={video_id}",
                transcript=full_transcript,
                description="",  # Would need API
                duration_seconds=duration
            )

            # Cache results
            with open(cache_path, "w") as f:
                json.dump(asdict(video), f, indent=2)

            jlog("youtube_transcript_fetched", level="INFO",
                 video_id=video_id, transcript_length=len(full_transcript))

            return video

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            jlog("youtube_no_transcript", level="DEBUG",
                 video_id=video_id, reason=str(e))
            return None
        except Exception as e:
            jlog("youtube_fetch_error", level="ERROR",
                 video_id=video_id, error=str(e))
            return None

    def scrape_all(self) -> List[YouTubeVideo]:
        """
        Scrape all configured video IDs.

        Returns:
            List of YouTubeVideo objects with transcripts
        """
        videos = []

        for video_id in self.STRATEGY_VIDEO_IDS:
            video = self.fetch_transcript(video_id)
            if video:
                videos.append(video)
            time.sleep(1)  # Be nice to YouTube

        jlog("youtube_scrape_complete", level="INFO",
             videos_fetched=len(videos),
             videos_attempted=len(self.STRATEGY_VIDEO_IDS))

        return videos


def scrape_youtube_strategies() -> List[Dict[str, Any]]:
    """
    Handler function for autonomous brain task.

    Returns:
        List of strategy ideas extracted from YouTube videos
    """
    if not YOUTUBE_API_AVAILABLE:
        jlog("youtube_scraper_disabled", level="WARNING",
             reason="youtube-transcript-api not installed")
        return []

    scraper = YouTubeScraper()
    videos = scraper.scrape_all()

    ideas = []
    for video in videos:
        ideas.append({
            "source_type": "youtube",
            "source_id": video.video_id,
            "source_url": video.url,
            "title": video.title,
            "description": f"Channel: {video.channel} | Duration: {video.duration_seconds}s",
            "content": video.transcript,
            "metadata": {
                "channel": video.channel,
                "duration_seconds": video.duration_seconds
            },
            "fetched_at": datetime.now().isoformat()
        })

    return ideas
