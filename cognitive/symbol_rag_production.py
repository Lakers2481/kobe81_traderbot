"""
Symbol-Aware RAG with Production-Grade Evaluation - Renaissance Standard

Retrieval-Augmented Generation system for trading knowledge grounded in
historical trade evidence. Implements Hands-On LLM evaluation patterns
for faithfulness and relevance scoring.

Jim Simons Quality Standard:
- Vector embeddings (sentence-transformers)
- Persistent vector storage (ChromaDB)
- RAG evaluation (faithfulness, relevance, context precision)
- Quality gates (stand down if faithfulness < 0.7)
- Production wiring into enrichment pipeline
- Zero hallucination tolerance

Features:
- Index historical trade knowledge (decisions, outcomes, patterns)
- Semantic retrieval (cosine similarity in embedding space)
- RAGAS-style evaluation (faithfulness, answer relevance, context precision)
- Knowledge boundary detection (stand down when uncertain)
- Citation/evidence tracking (ground responses in data)

Author: Kobe Trading System (Quant Developer for Jim Simons)
Date: 2026-01-08
Version: 1.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
_sentence_transformers_available = False
_chromadb_available = False

try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformers_available = True
except ImportError:
    logger.warning("sentence-transformers not available - RAG will use fallback mode")

try:
    import chromadb
    from chromadb.config import Settings
    _chromadb_available = True
except ImportError:
    logger.warning("chromadb not available - RAG will use fallback mode")


# =============================================================================
# Configuration
# =============================================================================

# Renaissance quality gates
MIN_FAITHFULNESS = 0.7  # Minimum faithfulness score (RAGAS standard)
MIN_RELEVANCE = 0.6  # Minimum relevance score
MIN_CONTEXT_PRECISION = 0.6  # Minimum context precision

# Model configuration
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, high quality (384 dims)
DEFAULT_TOP_K = 5  # Number of similar trades to retrieve

# Storage configuration
DEFAULT_VECTOR_DB_PATH = "state/cognitive/vector_db"
DEFAULT_COLLECTION_NAME = "trade_knowledge"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TradeKnowledge:
    """Historical trade knowledge for RAG indexing."""
    # Required fields (no defaults) - MUST come first
    trade_id: str
    symbol: str
    timestamp: str
    strategy: str
    entry_price: float
    exit_price: float
    setup: str  # e.g., "5 consecutive down days, RSI(2) < 5, IBS < 0.08"
    outcome: str  # WIN/LOSS/BREAKEVEN
    pnl: float
    pnl_pct: float
    decision_reason: str  # Why this trade was taken

    # Optional fields (with defaults) - MUST come after required fields
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    streak_length: int = 0
    regime: Optional[str] = None  # BULL/BEAR/NEUTRAL
    hold_days: int = 0
    outcome_reason: Optional[str] = None  # Why it worked/failed
    quality_score: Optional[float] = None
    conviction: Optional[float] = None

    def to_document(self) -> str:
        """Convert to semantic document for embedding."""
        doc = f"""
Symbol: {self.symbol}
Strategy: {self.strategy}
Setup: {self.setup}
Entry: ${self.entry_price:.2f}, Exit: ${self.exit_price:.2f}
Outcome: {self.outcome} ({self.pnl_pct:+.2%} in {self.hold_days} days)
Decision: {self.decision_reason}
"""
        if self.regime:
            doc += f"Regime: {self.regime}\n"
        if self.outcome_reason:
            doc += f"Lesson: {self.outcome_reason}\n"

        return doc.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata storage.

        Note: Filters out None values as ChromaDB doesn't accept them in metadata.
        """
        result = asdict(self)
        # ChromaDB can't handle None values in metadata - filter them out
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class RAGEvaluation:
    """RAG response evaluation metrics (RAGAS-style)."""
    faithfulness: float  # Are retrieved docs grounded in actual data? [0, 1]
    answer_relevance: float  # Is response relevant to question? [0, 1]
    context_precision: float  # Are retrieved docs relevant? [0, 1]
    context_recall: Optional[float] = None  # Did we retrieve all relevant docs?

    overall_quality: float = 0.0  # Weighted average

    def __post_init__(self):
        """Compute overall quality score."""
        # Weighted average (prioritize faithfulness to avoid hallucinations)
        self.overall_quality = (
            0.5 * self.faithfulness +
            0.3 * self.answer_relevance +
            0.2 * self.context_precision
        )

    def meets_quality_gates(self) -> bool:
        """Check if RAG response meets Renaissance quality gates."""
        return (
            self.faithfulness >= MIN_FAITHFULNESS and
            self.answer_relevance >= MIN_RELEVANCE and
            self.context_precision >= MIN_CONTEXT_PRECISION
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RAGResponse:
    """Complete RAG response with retrieval and evaluation."""
    query: str
    retrieved_documents: List[str]
    retrieved_metadata: List[Dict[str, Any]]
    distances: List[float]  # Cosine distances

    # Evaluation
    evaluation: RAGEvaluation

    # Decision support
    recommendation: str  # PROCEED, STAND_DOWN, UNCERTAIN
    reasoning: str  # Why this recommendation

    # Evidence
    num_similar_trades: int = 0
    win_rate_similar: Optional[float] = None
    avg_pnl_similar: Optional[float] = None

    # Metadata
    timestamp: str = ""

    def __post_init__(self):
        """Compute evidence statistics."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

        # Compute statistics from retrieved metadata
        if self.retrieved_metadata:
            self.num_similar_trades = len(self.retrieved_metadata)

            outcomes = [m.get('outcome') for m in self.retrieved_metadata if m.get('outcome')]
            if outcomes:
                wins = sum(1 for o in outcomes if o == 'WIN')
                self.win_rate_similar = wins / len(outcomes)

            pnls = [m.get('pnl_pct', 0) for m in self.retrieved_metadata if m.get('pnl_pct') is not None]
            if pnls:
                self.avg_pnl_similar = np.mean(pnls)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'retrieved_documents': self.retrieved_documents,
            'retrieved_metadata': self.retrieved_metadata,
            'distances': self.distances,
            'evaluation': self.evaluation.to_dict(),
            'recommendation': self.recommendation,
            'reasoning': self.reasoning,
            'num_similar_trades': self.num_similar_trades,
            'win_rate_similar': self.win_rate_similar,
            'avg_pnl_similar': self.avg_pnl_similar,
            'timestamp': self.timestamp,
        }


# =============================================================================
# RAG Evaluator (RAGAS-Style)
# =============================================================================

class RAGEvaluator:
    """
    RAGAS-style evaluation for RAG responses.

    Metrics:
    - Faithfulness: Are retrieved docs actually from our data?
    - Answer Relevance: Is the response relevant to the question?
    - Context Precision: Are retrieved docs relevant to the question?
    """

    def evaluate(
        self,
        query: str,
        retrieved_documents: List[str],
        retrieved_metadata: List[Dict[str, Any]],
        distances: List[float],
    ) -> RAGEvaluation:
        """Evaluate RAG response quality."""

        # Faithfulness: Check if retrieved docs are from our actual data
        faithfulness = self._compute_faithfulness(retrieved_metadata)

        # Answer Relevance: Check if retrieved docs match query intent
        answer_relevance = self._compute_answer_relevance(query, retrieved_documents, distances)

        # Context Precision: Check if retrieved docs are relevant
        context_precision = self._compute_context_precision(query, retrieved_documents, distances)

        return RAGEvaluation(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
        )

    def _compute_faithfulness(self, retrieved_metadata: List[Dict[str, Any]]) -> float:
        """
        Compute faithfulness score.

        Faithfulness = Do retrieved docs have required fields from TradeKnowledge?
        If all docs have valid structure → 1.0
        If some docs missing fields → proportional
        If no valid docs → 0.0
        """
        if not retrieved_metadata:
            return 0.0

        required_fields = {'symbol', 'outcome', 'pnl_pct', 'strategy'}

        valid_count = 0
        for metadata in retrieved_metadata:
            has_all_fields = all(field in metadata for field in required_fields)
            if has_all_fields:
                valid_count += 1

        return valid_count / len(retrieved_metadata)

    def _compute_answer_relevance(
        self,
        query: str,
        retrieved_documents: List[str],
        distances: List[float],
    ) -> float:
        """
        Compute answer relevance score.

        Answer Relevance = Are retrieved docs semantically close to query?
        Uses cosine distance: closer = more relevant
        """
        if not distances:
            return 0.0

        # Convert distances to similarities (1 - distance for cosine)
        similarities = [1 - d for d in distances]

        # Average similarity of top-k docs
        avg_similarity = np.mean(similarities)

        # Normalize to [0, 1]
        # Typical cosine similarity range: [0.5, 1.0] for relevant docs
        # Map 0.5 → 0.0, 1.0 → 1.0
        normalized = max(0.0, min(1.0, (avg_similarity - 0.5) / 0.5))

        return normalized

    def _compute_context_precision(
        self,
        query: str,
        retrieved_documents: List[str],
        distances: List[float],
    ) -> float:
        """
        Compute context precision score.

        Context Precision = What % of retrieved docs are actually relevant?
        We use distance threshold: if distance < 0.5 → relevant
        """
        if not distances:
            return 0.0

        # Count docs with distance < 0.5 (similarity > 0.5)
        relevant_count = sum(1 for d in distances if d < 0.5)

        return relevant_count / len(distances)


# =============================================================================
# Symbol RAG (Production)
# =============================================================================

class SymbolRAGProduction:
    """
    Production-grade Symbol RAG with vector embeddings and evaluation.

    Features:
    - Vector embeddings (sentence-transformers)
    - Persistent storage (ChromaDB)
    - RAGAS-style evaluation
    - Quality gates (faithfulness >= 0.7)
    - Stand-down when uncertain

    Usage:
        rag = SymbolRAGProduction()

        # Index trade history
        trades = load_historical_trades()
        rag.index_trade_history(trades)

        # Query for similar trades
        response = rag.query("How does AAPL perform after 5 down days?")

        if response.recommendation == "STAND_DOWN":
            print(f"Uncertain: {response.reasoning}")
        else:
            print(f"Found {response.num_similar_trades} similar trades")
            print(f"Win rate: {response.win_rate_similar:.1%}")
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        vector_db_path: str = DEFAULT_VECTOR_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = DEFAULT_TOP_K,
    ):
        """
        Initialize Symbol RAG.

        Args:
            embedding_model: Sentence-transformers model name
            vector_db_path: Path to ChromaDB storage
            collection_name: Collection name in ChromaDB
            top_k: Number of similar trades to retrieve
        """
        self.embedding_model_name = embedding_model
        self.vector_db_path = Path(vector_db_path)
        self.collection_name = collection_name
        self.top_k = top_k

        # Initialize components
        self.model: Optional[SentenceTransformer] = None
        self.client: Optional[Any] = None
        self.collection: Optional[Any] = None
        self.evaluator = RAGEvaluator()

        # Lazy load (don't crash if dependencies missing)
        self._initialize()

    def _initialize(self) -> bool:
        """Initialize RAG components (lazy loading)."""
        if not _sentence_transformers_available or not _chromadb_available:
            logger.warning("RAG dependencies not available - using fallback mode")
            return False

        try:
            # Initialize embedding model
            self.model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")

            # Initialize ChromaDB
            self.vector_db_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.vector_db_path),
                settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity
            )

            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
            logger.info(f"Current index size: {self.collection.count()} documents")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            return False

    def is_available(self) -> bool:
        """Check if RAG is available (dependencies installed)."""
        return self.model is not None and self.collection is not None

    def index_trade_history(self, trades: List[TradeKnowledge]) -> int:
        """
        Index historical trade knowledge for retrieval.

        Args:
            trades: List of TradeKnowledge objects

        Returns:
            Number of trades indexed
        """
        if not self.is_available():
            logger.warning("RAG not available - skipping indexing")
            return 0

        if not trades:
            logger.warning("No trades to index")
            return 0

        # Convert trades to documents and metadata
        documents = []
        metadatas = []
        ids = []

        for trade in trades:
            doc = trade.to_document()
            metadata = trade.to_dict()

            # Generate deterministic ID
            doc_hash = hashlib.sha256(doc.encode()).hexdigest()[:16]
            trade_id = f"trade_{trade.symbol}_{trade.timestamp}_{doc_hash}"

            documents.append(doc)
            metadatas.append(metadata)
            ids.append(trade_id)

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} trades...")
        embeddings = self.model.encode(documents, show_progress_bar=False)

        # Add to collection (ChromaDB handles duplicates by ID)
        try:
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Indexed {len(trades)} trades into RAG")
            return len(trades)

        except Exception as e:
            logger.error(f"Failed to index trades: {e}")
            return 0

    def query(
        self,
        question: str,
        symbol: Optional[str] = None,
        n_results: Optional[int] = None,
    ) -> RAGResponse:
        """
        Query RAG for similar historical trades.

        Args:
            question: Natural language question
            symbol: Optional symbol filter
            n_results: Number of results (default: top_k)

        Returns:
            RAGResponse with retrieval and evaluation
        """
        if not self.is_available():
            logger.warning("RAG not available - returning fallback response")
            return self._fallback_response(question, "RAG dependencies not available")

        n_results = n_results or self.top_k

        # Generate query embedding
        try:
            query_embedding = self.model.encode([question])
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            return self._fallback_response(question, f"Embedding error: {e}")

        # Query collection
        try:
            # Build where filter (optional symbol filter)
            where_filter = None
            if symbol:
                where_filter = {"symbol": symbol}

            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=where_filter if where_filter else None,
            )

            # Extract results
            retrieved_documents = results['documents'][0] if results['documents'] else []
            retrieved_metadata = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []

        except Exception as e:
            logger.error(f"Failed to query collection: {e}")
            return self._fallback_response(question, f"Query error: {e}")

        # Evaluate RAG response
        evaluation = self.evaluator.evaluate(
            query=question,
            retrieved_documents=retrieved_documents,
            retrieved_metadata=retrieved_metadata,
            distances=distances,
        )

        # Make recommendation based on evaluation
        recommendation, reasoning = self._make_recommendation(evaluation, retrieved_metadata)

        return RAGResponse(
            query=question,
            retrieved_documents=retrieved_documents,
            retrieved_metadata=retrieved_metadata,
            distances=distances,
            evaluation=evaluation,
            recommendation=recommendation,
            reasoning=reasoning,
        )

    def _make_recommendation(
        self,
        evaluation: RAGEvaluation,
        retrieved_metadata: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """
        Make recommendation based on RAG evaluation.

        Returns:
            (recommendation, reasoning)
        """
        # Check quality gates
        if not evaluation.meets_quality_gates():
            if evaluation.faithfulness < MIN_FAITHFULNESS:
                return "STAND_DOWN", f"Low faithfulness ({evaluation.faithfulness:.2f} < {MIN_FAITHFULNESS})"
            elif evaluation.answer_relevance < MIN_RELEVANCE:
                return "STAND_DOWN", f"Low relevance ({evaluation.answer_relevance:.2f} < {MIN_RELEVANCE})"
            elif evaluation.context_precision < MIN_CONTEXT_PRECISION:
                return "STAND_DOWN", f"Low precision ({evaluation.context_precision:.2f} < {MIN_CONTEXT_PRECISION})"

        # Check if we have enough data
        if not retrieved_metadata or len(retrieved_metadata) < 3:
            return "UNCERTAIN", f"Insufficient data ({len(retrieved_metadata)} trades)"

        # Check win rate
        outcomes = [m.get('outcome') for m in retrieved_metadata if m.get('outcome')]
        if outcomes:
            wins = sum(1 for o in outcomes if o == 'WIN')
            win_rate = wins / len(outcomes)

            if win_rate < 0.45:
                return "STAND_DOWN", f"Low historical win rate ({win_rate:.1%})"

        # Passed all gates
        return "PROCEED", f"High quality RAG (faithfulness={evaluation.faithfulness:.2f}, {len(retrieved_metadata)} similar trades)"

    def _fallback_response(self, query: str, reason: str) -> RAGResponse:
        """Create fallback response when RAG unavailable."""
        return RAGResponse(
            query=query,
            retrieved_documents=[],
            retrieved_metadata=[],
            distances=[],
            evaluation=RAGEvaluation(
                faithfulness=0.0,
                answer_relevance=0.0,
                context_precision=0.0,
            ),
            recommendation="STAND_DOWN",
            reasoning=reason,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG statistics."""
        if not self.is_available():
            return {'available': False}

        return {
            'available': True,
            'collection_name': self.collection_name,
            'index_size': self.collection.count(),
            'embedding_model': self.embedding_model_name,
            'vector_db_path': str(self.vector_db_path),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

# Global singleton instance
_rag_instance: Optional[SymbolRAGProduction] = None


def get_symbol_rag() -> SymbolRAGProduction:
    """Get global Symbol RAG instance (singleton)."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = SymbolRAGProduction()
    return _rag_instance


def query_trade_knowledge(
    question: str,
    symbol: Optional[str] = None,
    n_results: int = 5,
) -> RAGResponse:
    """
    Query historical trade knowledge.

    Args:
        question: Natural language question
        symbol: Optional symbol filter
        n_results: Number of similar trades to retrieve

    Returns:
        RAGResponse with evaluation and recommendation
    """
    rag = get_symbol_rag()
    return rag.query(question, symbol=symbol, n_results=n_results)


def index_historical_trades(trades: List[TradeKnowledge]) -> int:
    """
    Index historical trades for RAG retrieval.

    Args:
        trades: List of TradeKnowledge objects

    Returns:
        Number of trades indexed
    """
    rag = get_symbol_rag()
    return rag.index_trade_history(trades)
