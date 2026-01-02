"""
FinBERT Sentiment Analysis Module.

Provides financial sentiment analysis using pre-trained FinBERT model.
FinBERT is a BERT model fine-tuned on financial text, achieving state-of-the-art
performance on financial sentiment classification.

Sources:
- ProsusAI/finbert (HuggingFace)
- "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"
  (Araci, 2019)

Usage:
    from ml_features.sentiment import SentimentAnalyzer

    analyzer = SentimentAnalyzer()

    # Single headline
    result = analyzer.analyze("Apple beats earnings expectations")
    # Returns: {'label': 'positive', 'score': 0.95, 'positive': 0.95, 'negative': 0.02, 'neutral': 0.03}

    # Batch processing
    headlines = ["Apple beats earnings", "Tesla misses revenue targets"]
    results = analyzer.analyze_batch(headlines)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.structured_log import jlog

# Check for transformers availability
try:
    from transformers import (
        BertTokenizer,
        BertForSequenceClassification,
        pipeline,
        AutoModelForSequenceClassification,
        AutoTokenizer
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    jlog("transformers_not_available", level="WARNING",
         message="Install transformers: pip install transformers torch")


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""

    # Model configuration
    model_name: str = "ProsusAI/finbert"
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    batch_size: int = 16
    max_length: int = 512

    # Caching
    cache_results: bool = True
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "state" / "sentiment_cache")

    # Processing
    return_all_scores: bool = True  # Return all class probabilities
    confidence_threshold: float = 0.5  # Minimum confidence to consider valid

    # Rate limiting (for API-based fallbacks)
    max_requests_per_minute: int = 60


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT.

    FinBERT is trained on financial news and achieves >90% accuracy
    on financial sentiment classification tasks.

    Labels:
    - positive: Bullish sentiment (buy signal support)
    - negative: Bearish sentiment (sell signal support)
    - neutral: No strong sentiment (no signal modification)
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._cache: Dict[str, dict] = {}
        self._initialized = False

        # Create cache directory
        if self.config.cache_results:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _initialize(self) -> bool:
        """Lazy initialization of the model."""
        if self._initialized:
            return True

        if not TRANSFORMERS_AVAILABLE:
            jlog("sentiment_init_failed", level="ERROR",
                 message="transformers library not available")
            return False

        try:
            jlog("sentiment_loading_model", level="INFO",
                 model=self.config.model_name)

            # Determine device
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    device = 0  # GPU
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = -1  # CPU
            elif self.config.device == "cpu":
                device = -1
            elif self.config.device == "cuda":
                device = 0
            else:
                device = self.config.device

            # Load model using pipeline for efficiency
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.config.model_name,
                tokenizer=self.config.model_name,
                device=device,
                return_all_scores=self.config.return_all_scores,
                truncation=True,
                max_length=self.config.max_length
            )

            self._initialized = True
            jlog("sentiment_model_loaded", level="INFO",
                 model=self.config.model_name,
                 device=str(device))

            return True

        except Exception as e:
            jlog("sentiment_init_error", level="ERROR", error=str(e))
            return False

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _load_from_cache(self, text: str) -> Optional[dict]:
        """Load result from cache."""
        if not self.config.cache_results:
            return None

        key = self._get_cache_key(text)

        # Check memory cache first
        if key in self._cache:
            return self._cache[key]

        # Check disk cache
        cache_file = self.config.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    result = json.load(f)
                    self._cache[key] = result
                    return result
            except Exception:
                pass

        return None

    def _save_to_cache(self, text: str, result: dict) -> None:
        """Save result to cache."""
        if not self.config.cache_results:
            return

        key = self._get_cache_key(text)
        self._cache[key] = result

        # Save to disk
        cache_file = self.config.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except Exception:
            pass

    def analyze(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Financial text (headline, news, tweet)

        Returns:
            dict with:
            - label: 'positive', 'negative', or 'neutral'
            - score: confidence (0-1)
            - positive: probability of positive
            - negative: probability of negative
            - neutral: probability of neutral
        """
        if not text or not text.strip():
            return {
                'label': 'neutral',
                'score': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'valid': False
            }

        # Check cache
        cached = self._load_from_cache(text)
        if cached:
            return cached

        # Initialize model if needed
        if not self._initialize():
            return {
                'label': 'neutral',
                'score': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'valid': False,
                'error': 'Model not available'
            }

        try:
            # Run inference
            raw_result = self._pipeline(text)

            # Parse results
            if self.config.return_all_scores:
                # raw_result is list of list of dicts
                scores = {item['label'].lower(): item['score'] for item in raw_result[0]}
            else:
                # raw_result is list of dicts
                top_label = raw_result[0]['label'].lower()
                top_score = raw_result[0]['score']
                scores = {top_label: top_score}

            # Get all probabilities
            positive = scores.get('positive', 0.0)
            negative = scores.get('negative', 0.0)
            neutral = scores.get('neutral', 0.0)

            # Determine label
            max_score = max(positive, negative, neutral)
            if positive == max_score:
                label = 'positive'
            elif negative == max_score:
                label = 'negative'
            else:
                label = 'neutral'

            result = {
                'label': label,
                'score': max_score,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'valid': max_score >= self.config.confidence_threshold
            }

            # Cache result
            self._save_to_cache(text, result)

            return result

        except Exception as e:
            jlog("sentiment_analysis_error", level="WARNING",
                 error=str(e), text_preview=text[:100])
            return {
                'label': 'neutral',
                'score': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'valid': False,
                'error': str(e)
            }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Analyze sentiment of multiple texts.

        More efficient than calling analyze() repeatedly.

        Args:
            texts: List of financial texts

        Returns:
            List of sentiment results
        """
        if not texts:
            return []

        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._load_from_cache(text)
            if cached:
                results.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Process uncached texts
        if uncached_texts:
            if not self._initialize():
                for idx in uncached_indices:
                    results.append((idx, {
                        'label': 'neutral',
                        'score': 0.0,
                        'positive': 0.0,
                        'negative': 0.0,
                        'neutral': 1.0,
                        'valid': False,
                        'error': 'Model not available'
                    }))
            else:
                try:
                    # Batch inference
                    raw_results = self._pipeline(
                        uncached_texts,
                        batch_size=self.config.batch_size
                    )

                    for idx, raw_result, text in zip(uncached_indices, raw_results, uncached_texts):
                        if self.config.return_all_scores:
                            scores = {item['label'].lower(): item['score'] for item in raw_result}
                        else:
                            top_label = raw_result['label'].lower()
                            top_score = raw_result['score']
                            scores = {top_label: top_score}

                        positive = scores.get('positive', 0.0)
                        negative = scores.get('negative', 0.0)
                        neutral = scores.get('neutral', 0.0)

                        max_score = max(positive, negative, neutral)
                        if positive == max_score:
                            label = 'positive'
                        elif negative == max_score:
                            label = 'negative'
                        else:
                            label = 'neutral'

                        result = {
                            'label': label,
                            'score': max_score,
                            'positive': positive,
                            'negative': negative,
                            'neutral': neutral,
                            'valid': max_score >= self.config.confidence_threshold
                        }

                        self._save_to_cache(text, result)
                        results.append((idx, result))

                except Exception as e:
                    jlog("sentiment_batch_error", level="WARNING", error=str(e))
                    for idx in uncached_indices:
                        results.append((idx, {
                            'label': 'neutral',
                            'score': 0.0,
                            'positive': 0.0,
                            'negative': 0.0,
                            'neutral': 1.0,
                            'valid': False,
                            'error': str(e)
                        }))

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def get_sentiment_signal(self, texts: List[str]) -> float:
        """
        Get aggregated sentiment signal from multiple texts.

        Returns a score from -1 (very bearish) to +1 (very bullish).

        Args:
            texts: List of financial texts (headlines, tweets, etc.)

        Returns:
            Aggregated sentiment score (-1 to +1)
        """
        if not texts:
            return 0.0

        results = self.analyze_batch(texts)

        # Calculate net sentiment
        # positive - negative, weighted by confidence
        scores = []
        for r in results:
            if r.get('valid', False):
                # Net score: positive probability - negative probability
                net = r.get('positive', 0) - r.get('negative', 0)
                confidence = r.get('score', 0)
                scores.append(net * confidence)

        if not scores:
            return 0.0

        return np.mean(scores)

    def analyze_for_symbol(
        self,
        symbol: str,
        headlines: List[str],
        weights: Optional[List[float]] = None
    ) -> Dict[str, Union[float, str, List[dict]]]:
        """
        Analyze sentiment for a specific stock symbol.

        Args:
            symbol: Stock ticker
            headlines: List of headlines mentioning the symbol
            weights: Optional weights (e.g., by recency or source quality)

        Returns:
            Aggregated sentiment with details
        """
        if not headlines:
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'num_headlines': 0,
                'confidence': 0.0,
                'details': []
            }

        results = self.analyze_batch(headlines)

        if weights is None:
            weights = [1.0] * len(headlines)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Weighted aggregation
        weighted_positive = sum(r['positive'] * w for r, w in zip(results, weights))
        weighted_negative = sum(r['negative'] * w for r, w in zip(results, weights))
        weighted_neutral = sum(r['neutral'] * w for r, w in zip(results, weights))

        # Net sentiment score (-1 to +1)
        sentiment_score = weighted_positive - weighted_negative

        # Determine label
        if sentiment_score > 0.1:
            sentiment_label = 'bullish'
        elif sentiment_score < -0.1:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'neutral'

        # Confidence is how much sentiment is non-neutral
        confidence = 1.0 - weighted_neutral

        return {
            'symbol': symbol,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'num_headlines': len(headlines),
            'confidence': confidence,
            'weighted_positive': weighted_positive,
            'weighted_negative': weighted_negative,
            'weighted_neutral': weighted_neutral,
            'details': results
        }

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        if self.config.cache_results and self.config.cache_dir.exists():
            for f in self.config.cache_dir.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass
        jlog("sentiment_cache_cleared", level="DEBUG")


# =============================================================================
# ALTERNATIVE: Simple rule-based sentiment (fallback when no transformers)
# =============================================================================

class RuleBasedSentiment:
    """
    Simple rule-based sentiment analyzer.

    Fallback when transformers/FinBERT is not available.
    Uses financial keyword matching with polarity scores.
    """

    # Financial sentiment lexicon
    POSITIVE_WORDS = {
        'beat', 'beats', 'exceeds', 'exceeded', 'surge', 'surges', 'soar', 'soars',
        'gain', 'gains', 'profit', 'profits', 'bullish', 'rally', 'rallies',
        'upgrade', 'upgraded', 'buy', 'outperform', 'growth', 'strong', 'record',
        'breakthrough', 'innovation', 'success', 'successful', 'positive',
        'optimistic', 'upside', 'opportunity', 'expansion', 'momentum'
    }

    NEGATIVE_WORDS = {
        'miss', 'misses', 'missed', 'fall', 'falls', 'drop', 'drops', 'plunge',
        'plunges', 'decline', 'declines', 'loss', 'losses', 'bearish', 'crash',
        'downgrade', 'downgraded', 'sell', 'underperform', 'weak', 'warning',
        'crisis', 'failure', 'failed', 'negative', 'pessimistic', 'downside',
        'risk', 'concern', 'worried', 'layoffs', 'bankruptcy', 'lawsuit'
    }

    def analyze(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment using keyword matching."""
        if not text:
            return {
                'label': 'neutral',
                'score': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'method': 'rule_based'
            }

        words = text.lower().split()

        positive_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return {
                'label': 'neutral',
                'score': 1.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'method': 'rule_based'
            }

        positive = positive_count / len(words)
        negative = negative_count / len(words)
        neutral = 1.0 - (positive + negative)

        if positive > negative:
            label = 'positive'
            score = positive
        elif negative > positive:
            label = 'negative'
            score = negative
        else:
            label = 'neutral'
            score = 0.5

        return {
            'label': label,
            'score': score,
            'positive': positive,
            'negative': negative,
            'neutral': max(0, neutral),
            'method': 'rule_based'
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts."""
        return [self.analyze(t) for t in texts]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_analyzer: Optional[SentimentAnalyzer] = None

def get_sentiment(text: str) -> Dict[str, Union[str, float]]:
    """
    Get sentiment for a single text using global analyzer.

    Convenience function for quick sentiment analysis.

    Args:
        text: Financial text to analyze

    Returns:
        Sentiment result dict
    """
    global _global_analyzer

    if _global_analyzer is None:
        if TRANSFORMERS_AVAILABLE:
            _global_analyzer = SentimentAnalyzer()
        else:
            _global_analyzer = RuleBasedSentiment()

    return _global_analyzer.analyze(text)


def get_sentiment_batch(texts: List[str]) -> List[Dict]:
    """
    Get sentiment for multiple texts using global analyzer.

    Args:
        texts: List of financial texts

    Returns:
        List of sentiment results
    """
    global _global_analyzer

    if _global_analyzer is None:
        if TRANSFORMERS_AVAILABLE:
            _global_analyzer = SentimentAnalyzer()
        else:
            _global_analyzer = RuleBasedSentiment()

    return _global_analyzer.analyze_batch(texts)


def get_aggregated_sentiment(texts: List[str]) -> float:
    """
    Get aggregated sentiment score from multiple texts.

    Returns score from -1 (bearish) to +1 (bullish).

    Args:
        texts: List of financial texts

    Returns:
        Aggregated sentiment score
    """
    global _global_analyzer

    if _global_analyzer is None:
        if TRANSFORMERS_AVAILABLE:
            _global_analyzer = SentimentAnalyzer()
        else:
            _global_analyzer = RuleBasedSentiment()

    if isinstance(_global_analyzer, SentimentAnalyzer):
        return _global_analyzer.get_sentiment_signal(texts)
    else:
        # Rule-based fallback
        results = _global_analyzer.analyze_batch(texts)
        scores = [r['positive'] - r['negative'] for r in results]
        return np.mean(scores) if scores else 0.0
