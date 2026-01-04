"""
FAISS Vector Memory for Episodic Memory Similarity Search

Fast similarity search for finding similar past trading situations.
Uses sentence-transformers for embedding and FAISS for indexing.

Features:
- Sub-10ms similarity search
- Persistent index storage
- Automatic reindexing on updates
- Win/loss tracking for memory quality

Author: Kobe Trading System
Created: 2026-01-04
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from core.structured_log import get_logger

logger = get_logger(__name__)

# Lazy imports for optional dependencies
_faiss = None
_sentence_transformer = None


def _get_faiss():
    """Lazy import FAISS."""
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            logger.warning("FAISS not installed. Install with: pip install faiss-cpu")
            raise ImportError("FAISS required. Install with: pip install faiss-cpu")
    return _faiss


def _get_sentence_transformer():
    """Lazy import sentence-transformers."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer = SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
    return _sentence_transformer


class VectorMemory:
    """
    FAISS-based vector memory for fast similarity search.

    Usage:
        memory = VectorMemory()
        memory.add_episode(episode_dict)
        similar = memory.find_similar("AAPL showing IBS < 0.1 with RSI bounce", k=5)
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Fast, 384-dim embeddings
    INDEX_FILE = "faiss_index.bin"
    METADATA_FILE = "faiss_metadata.json"
    STATE_DIR = Path("state/cognitive/vector_memory")

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_dim: int = 384,
        state_dir: Optional[Path] = None
    ):
        """
        Initialize vector memory.

        Args:
            model_name: Sentence transformer model name
            embedding_dim: Embedding dimension (must match model)
            state_dir: Directory for index storage
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.state_dir = Path(state_dir) if state_dir else self.STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._encoder = None  # Lazy load
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}

        # Load existing index if available
        self._load_index()

    def _get_encoder(self):
        """Get or create sentence transformer encoder."""
        if self._encoder is None:
            SentenceTransformer = _get_sentence_transformer()
            self._encoder = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence transformer: {self.model_name}")
        return self._encoder

    def _create_index(self):
        """Create new FAISS index."""
        faiss = _get_faiss()
        # Use L2 distance with IVF for scalability
        self._index = faiss.IndexFlatL2(self.embedding_dim)
        logger.info(f"Created FAISS index with dim={self.embedding_dim}")

    def _load_index(self):
        """Load existing index from disk."""
        index_path = self.state_dir / self.INDEX_FILE
        metadata_path = self.state_dir / self.METADATA_FILE

        if index_path.exists() and metadata_path.exists():
            try:
                faiss = _get_faiss()
                self._index = faiss.read_index(str(index_path))

                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self._metadata = data.get('episodes', [])
                    self._id_to_idx = {ep['id']: i for i, ep in enumerate(self._metadata)}

                logger.info(f"Loaded FAISS index with {len(self._metadata)} episodes")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self._create_index()
        else:
            self._create_index()

    def _save_index(self):
        """Save index to disk."""
        if self._index is None:
            return

        try:
            faiss = _get_faiss()
            index_path = self.state_dir / self.INDEX_FILE
            metadata_path = self.state_dir / self.METADATA_FILE

            faiss.write_index(self._index, str(index_path))

            with open(metadata_path, 'w') as f:
                json.dump({
                    'episodes': self._metadata,
                    'updated_at': datetime.now().isoformat(),
                    'count': len(self._metadata)
                }, f, indent=2)

            logger.debug(f"Saved FAISS index with {len(self._metadata)} episodes")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def _episode_to_text(self, episode: Dict[str, Any]) -> str:
        """
        Convert episode dict to text for embedding.

        Args:
            episode: Episode dictionary with context, reasoning, outcome

        Returns:
            Text representation for embedding
        """
        parts = []

        # Context
        if 'context' in episode:
            ctx = episode['context']
            if isinstance(ctx, dict):
                if 'symbol' in ctx:
                    parts.append(f"Symbol: {ctx['symbol']}")
                if 'strategy' in ctx:
                    parts.append(f"Strategy: {ctx['strategy']}")
                if 'indicators' in ctx:
                    parts.append(f"Indicators: {ctx['indicators']}")
                if 'market_regime' in ctx:
                    parts.append(f"Regime: {ctx['market_regime']}")
                if 'signal_type' in ctx:
                    parts.append(f"Signal: {ctx['signal_type']}")
            else:
                parts.append(str(ctx))

        # Reasoning
        if 'reasoning' in episode:
            reasoning = episode['reasoning']
            if isinstance(reasoning, list):
                parts.append("Reasoning: " + "; ".join(reasoning[:3]))
            else:
                parts.append(f"Reasoning: {reasoning}")

        # Outcome (for learning)
        if 'outcome' in episode:
            outcome = episode['outcome']
            if isinstance(outcome, dict):
                if 'success' in outcome:
                    parts.append(f"Outcome: {'WIN' if outcome['success'] else 'LOSS'}")
                if 'pnl' in outcome:
                    parts.append(f"P&L: {outcome['pnl']}")
            else:
                parts.append(f"Outcome: {outcome}")

        return " | ".join(parts)

    def _generate_id(self, episode: Dict[str, Any]) -> str:
        """Generate unique ID for episode."""
        content = json.dumps(episode, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def add_episode(
        self,
        episode: Dict[str, Any],
        episode_id: Optional[str] = None
    ) -> str:
        """
        Add episode to vector memory.

        Args:
            episode: Episode dict with context, reasoning, outcome
            episode_id: Optional custom ID

        Returns:
            Episode ID
        """
        if self._index is None:
            self._create_index()

        episode_id = episode_id or self._generate_id(episode)

        # Check for duplicate
        if episode_id in self._id_to_idx:
            logger.debug(f"Episode {episode_id} already exists, skipping")
            return episode_id

        # Convert to text and embed
        text = self._episode_to_text(episode)
        encoder = self._get_encoder()
        embedding = encoder.encode([text], convert_to_numpy=True)

        # Add to index
        self._index.add(embedding.astype(np.float32))

        # Store metadata
        metadata = {
            'id': episode_id,
            'text': text[:500],  # Truncate for storage
            'added_at': datetime.now().isoformat(),
            'outcome': episode.get('outcome', {}),
            'context': episode.get('context', {}),
        }
        self._metadata.append(metadata)
        self._id_to_idx[episode_id] = len(self._metadata) - 1

        # Save periodically (every 10 episodes)
        if len(self._metadata) % 10 == 0:
            self._save_index()

        logger.debug(f"Added episode {episode_id} to vector memory")
        return episode_id

    def find_similar(
        self,
        query: str,
        k: int = 5,
        filter_wins: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar episodes.

        Args:
            query: Query text or episode description
            k: Number of results to return
            filter_wins: If True, only return winning episodes

        Returns:
            List of similar episodes with distances
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Embed query
        encoder = self._get_encoder()
        query_embedding = encoder.encode([query], convert_to_numpy=True)

        # Search
        distances, indices = self._index.search(
            query_embedding.astype(np.float32),
            min(k * 2, self._index.ntotal)  # Fetch extra for filtering
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            metadata = self._metadata[idx]

            # Apply filter
            if filter_wins is not None:
                outcome = metadata.get('outcome', {})
                is_win = outcome.get('success', False) or outcome.get('pnl', 0) > 0
                if filter_wins and not is_win:
                    continue
                if not filter_wins and is_win:
                    continue

            results.append({
                'id': metadata['id'],
                'distance': float(dist),
                'similarity': float(1 / (1 + dist)),  # Convert to similarity score
                'text': metadata.get('text', ''),
                'context': metadata.get('context', {}),
                'outcome': metadata.get('outcome', {}),
                'added_at': metadata.get('added_at'),
            })

            if len(results) >= k:
                break

        return results

    def find_similar_episode(
        self,
        episode: Dict[str, Any],
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find episodes similar to another episode.

        Args:
            episode: Episode dict to compare
            k: Number of results

        Returns:
            List of similar episodes
        """
        text = self._episode_to_text(episode)
        return self.find_similar(text, k)

    def get_win_rate_for_similar(
        self,
        query: str,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Get win rate statistics for similar past episodes.

        Args:
            query: Query text
            k: Number of similar episodes to analyze

        Returns:
            Win rate statistics
        """
        similar = self.find_similar(query, k)

        if not similar:
            return {
                'win_rate': None,
                'sample_size': 0,
                'avg_pnl': None,
                'recommendation': 'UNKNOWN'
            }

        wins = 0
        total_pnl = 0.0
        valid_outcomes = 0

        for ep in similar:
            outcome = ep.get('outcome', {})
            if 'success' in outcome:
                if outcome['success']:
                    wins += 1
                valid_outcomes += 1
            if 'pnl' in outcome:
                total_pnl += float(outcome['pnl'])

        if valid_outcomes == 0:
            return {
                'win_rate': None,
                'sample_size': 0,
                'avg_pnl': None,
                'recommendation': 'UNKNOWN'
            }

        win_rate = wins / valid_outcomes
        avg_pnl = total_pnl / len(similar) if similar else 0

        # Recommendation based on win rate
        if win_rate >= 0.7 and len(similar) >= 5:
            recommendation = 'BULLISH'
        elif win_rate <= 0.3 and len(similar) >= 5:
            recommendation = 'AVOID'
        else:
            recommendation = 'NEUTRAL'

        return {
            'win_rate': win_rate,
            'sample_size': valid_outcomes,
            'avg_pnl': avg_pnl,
            'avg_similarity': np.mean([s['similarity'] for s in similar]),
            'recommendation': recommendation,
            'similar_episodes': similar[:3]  # Top 3 for reference
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if self._index is None:
            return {'total_episodes': 0, 'index_size': 0}

        wins = sum(1 for m in self._metadata if m.get('outcome', {}).get('success', False))

        return {
            'total_episodes': len(self._metadata),
            'index_size': self._index.ntotal,
            'win_episodes': wins,
            'loss_episodes': len(self._metadata) - wins,
            'win_rate': wins / len(self._metadata) if self._metadata else 0,
        }

    def clear(self):
        """Clear all episodes."""
        self._create_index()
        self._metadata = []
        self._id_to_idx = {}
        self._save_index()
        logger.info("Vector memory cleared")

    def save(self):
        """Force save to disk."""
        self._save_index()


# Singleton instance
_memory: Optional[VectorMemory] = None


def get_vector_memory() -> VectorMemory:
    """Get or create singleton vector memory."""
    global _memory
    if _memory is None:
        _memory = VectorMemory()
    return _memory


# Convenience functions
def add_trade_episode(episode: Dict[str, Any]) -> str:
    """Add trading episode to vector memory."""
    return get_vector_memory().add_episode(episode)


def find_similar_trades(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Find similar past trades."""
    return get_vector_memory().find_similar(query, k)


def get_similar_trade_stats(query: str) -> Dict[str, Any]:
    """Get win rate for similar past trades."""
    return get_vector_memory().get_win_rate_for_similar(query)


if __name__ == "__main__":
    # Demo usage
    print("=== FAISS Vector Memory Demo ===\n")

    memory = VectorMemory()

    # Add some sample episodes
    episodes = [
        {
            'context': {
                'symbol': 'AAPL',
                'strategy': 'IBS_RSI',
                'indicators': 'IBS=0.05, RSI=4',
                'market_regime': 'BULLISH'
            },
            'reasoning': ['Low IBS indicates oversold', 'RSI below 5', 'Above SMA200'],
            'outcome': {'success': True, 'pnl': 150.0}
        },
        {
            'context': {
                'symbol': 'MSFT',
                'strategy': 'IBS_RSI',
                'indicators': 'IBS=0.08, RSI=8',
                'market_regime': 'NEUTRAL'
            },
            'reasoning': ['Borderline IBS', 'RSI approaching oversold'],
            'outcome': {'success': False, 'pnl': -75.0}
        },
        {
            'context': {
                'symbol': 'TSLA',
                'strategy': 'TURTLE_SOUP',
                'indicators': 'Sweep=0.5ATR, 20-day low',
                'market_regime': 'VOLATILE'
            },
            'reasoning': ['Strong sweep below 20-day low', 'Quick reversal candle'],
            'outcome': {'success': True, 'pnl': 300.0}
        },
    ]

    print("Adding sample episodes...")
    for ep in episodes:
        ep_id = memory.add_episode(ep)
        print(f"  Added: {ep_id}")

    print(f"\nMemory stats: {memory.get_stats()}")

    # Query
    print("\nFinding similar to 'AAPL with low IBS and RSI oversold':")
    similar = memory.find_similar("AAPL with low IBS and RSI oversold", k=3)
    for s in similar:
        print(f"  - {s['text'][:60]}... (similarity: {s['similarity']:.3f})")

    # Win rate analysis
    print("\nWin rate for similar IBS trades:")
    stats = memory.get_win_rate_for_similar("IBS strategy with oversold RSI")
    print(f"  Win Rate: {stats.get('win_rate', 0):.1%}")
    print(f"  Sample Size: {stats.get('sample_size', 0)}")
    print(f"  Recommendation: {stats.get('recommendation', 'UNKNOWN')}")

    memory.save()
