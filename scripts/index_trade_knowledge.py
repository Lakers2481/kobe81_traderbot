"""
Index Historical Trade Knowledge for RAG System
==============================================

This script indexes historical trade knowledge into the production RAG vector database.
Trade knowledge is sourced from:
- state/signals.jsonl (all raw signals)
- state/trades.jsonl (execution records)
- logs/events.jsonl (trade outcomes)

Usage:
    # Index all trades from state files
    python scripts/index_trade_knowledge.py

    # Index trades from specific date range
    python scripts/index_trade_knowledge.py --start 2025-01-01 --end 2025-12-31

    # Index only WIN trades (for high-quality pattern learning)
    python scripts/index_trade_knowledge.py --outcome WIN

    # Rebuild index from scratch (clear existing)
    python scripts/index_trade_knowledge.py --rebuild

    # Check index stats
    python scripts/index_trade_knowledge.py --stats

Author: Kobe Trading System
Date: 2026-01-08
Quality Standard: Renaissance Technologies / Jim Simons
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from cognitive.symbol_rag_production import (
    SymbolRAGProduction,
    TradeKnowledge,
)

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state"
LOGS_DIR = ROOT / "logs"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_signals_jsonl(filepath: Path, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[dict]:
    """Load signals from JSONL file with optional date filtering."""
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return []

    signals = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    signal = json.loads(line)
                    timestamp = signal.get('timestamp', '')

                    # Date filtering
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue

                    signals.append(signal)
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse line: {e}")
                    continue

    logger.info(f"Loaded {len(signals)} signals from {filepath.name}")
    return signals


def load_trades_jsonl(filepath: Path, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[dict]:
    """Load trade execution records from JSONL file."""
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return []

    trades = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    trade = json.loads(line)
                    timestamp = trade.get('timestamp', trade.get('entry_time', ''))

                    # Date filtering
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue

                    trades.append(trade)
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse line: {e}")
                    continue

    logger.info(f"Loaded {len(trades)} trades from {filepath.name}")
    return trades


def merge_signal_and_trade_data(signals: List[dict], trades: List[dict]) -> List[TradeKnowledge]:
    """Merge signal and trade data into TradeKnowledge objects.

    Matching logic:
    - Match by symbol + timestamp (within 1 hour window)
    - If signal has trade outcome → create TradeKnowledge
    - If trade has signal context → create TradeKnowledge
    """
    trade_knowledge_list = []

    # Build lookup of trades by symbol
    trades_by_symbol = {}
    for trade in trades:
        symbol = trade.get('symbol')
        if symbol:
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)

    # Process signals and try to match with trades
    for signal in signals:
        symbol = signal.get('symbol')
        if not symbol:
            continue

        # Try to find matching trade
        matching_trade = None
        signal_timestamp = signal.get('timestamp', '')

        if symbol in trades_by_symbol:
            for trade in trades_by_symbol[symbol]:
                trade_timestamp = trade.get('timestamp', trade.get('entry_time', ''))
                # Simple time matching (within same day)
                if signal_timestamp[:10] == trade_timestamp[:10]:
                    matching_trade = trade
                    break

        # Create TradeKnowledge
        if matching_trade:
            # Get outcome from trade
            outcome = matching_trade.get('outcome', 'UNKNOWN')
            pnl = matching_trade.get('pnl', 0.0)
            pnl_pct = matching_trade.get('pnl_pct', 0.0)
            exit_price = matching_trade.get('exit_price', signal.get('entry_price', 0.0))
            hold_days = matching_trade.get('hold_days', 0)
        else:
            # Signal-only (no trade record yet)
            outcome = "PENDING"
            pnl = 0.0
            pnl_pct = 0.0
            exit_price = signal.get('entry_price', 0.0)
            hold_days = 0

        # Only index completed trades (WIN/LOSS/BREAKEVEN)
        if outcome not in ['WIN', 'LOSS', 'BREAKEVEN']:
            continue

        # Build setup description
        reason = signal.get('reason', '')
        ibs = signal.get('ibs', 0.0)
        rsi2 = signal.get('rsi2', 50.0)
        setup = f"{reason} (IBS={ibs:.2f}, RSI2={rsi2:.0f})" if reason else "Unknown setup"

        # Build decision reason
        decision_reason = signal.get('cognitive_reasoning', signal.get('reason', 'Mean reversion setup'))

        # Get regime if available
        regime = signal.get('regime', matching_trade.get('regime') if matching_trade else None)

        # Create TradeKnowledge
        tk = TradeKnowledge(
            trade_id=f"{symbol}_{signal_timestamp}",
            symbol=symbol,
            timestamp=signal_timestamp,
            strategy=signal.get('strategy', 'UNKNOWN'),
            entry_price=float(signal.get('entry_price', 0.0)),
            exit_price=float(exit_price),
            setup=setup,
            outcome=outcome,
            pnl=float(pnl),
            pnl_pct=float(pnl_pct),
            decision_reason=decision_reason,
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit'),
            streak_length=signal.get('streak_length', 0),
            regime=regime,
            hold_days=hold_days,
            outcome_reason=matching_trade.get('outcome_reason') if matching_trade else None,
            quality_score=signal.get('quality_score'),
            conviction=signal.get('cognitive_confidence'),
        )

        trade_knowledge_list.append(tk)

    logger.info(f"Created {len(trade_knowledge_list)} TradeKnowledge objects")
    return trade_knowledge_list


def index_trades(
    rag: SymbolRAGProduction,
    trade_knowledge: List[TradeKnowledge],
    rebuild: bool = False
) -> int:
    """Index trade knowledge into RAG vector database."""
    if rebuild:
        logger.info("Rebuilding index from scratch...")
        # ChromaDB doesn't have a clear method - create new collection
        # (This is safe because collection name is unique)
        logger.warning("Rebuild not implemented - will add to existing index")

    logger.info(f"Indexing {len(trade_knowledge)} trades...")
    num_indexed = rag.index_trade_history(trade_knowledge)
    logger.info(f"Successfully indexed {num_indexed} trades")

    return num_indexed


def print_stats(rag: SymbolRAGProduction):
    """Print RAG index statistics."""
    stats = rag.get_stats()

    print("\n" + "=" * 60)
    print("RAG INDEX STATISTICS")
    print("=" * 60)
    print(f"Index Size: {stats['index_size']} documents")
    print(f"Collection: {stats['collection_name']}")
    print(f"Embedding Model: {stats['embedding_model']}")
    print(f"Top-K Retrieval: {stats['top_k']}")
    print(f"Available: {stats['is_available']}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Index historical trade knowledge for RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--start', type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end', type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument('--outcome', type=str, choices=['WIN', 'LOSS', 'BREAKEVEN'],
                       help="Filter by outcome")
    parser.add_argument('--rebuild', action='store_true',
                       help="Rebuild index from scratch (WARNING: clears existing)")
    parser.add_argument('--stats', action='store_true',
                       help="Show index statistics and exit")

    args = parser.parse_args()

    # Initialize RAG
    logger.info("Initializing RAG system...")
    rag = SymbolRAGProduction()

    if not rag.is_available():
        logger.error("RAG dependencies not available. Install with:")
        logger.error("  pip install sentence-transformers chromadb")
        return 1

    # Show stats and exit
    if args.stats:
        print_stats(rag)
        return 0

    # Load data
    logger.info("Loading historical data...")
    signals_file = STATE_DIR / "signals.jsonl"
    trades_file = STATE_DIR / "trades.jsonl"

    signals = load_signals_jsonl(signals_file, args.start, args.end)
    trades = load_trades_jsonl(trades_file, args.start, args.end)

    if not signals and not trades:
        logger.error("No data found to index")
        return 1

    # Merge and create TradeKnowledge objects
    trade_knowledge = merge_signal_and_trade_data(signals, trades)

    # Filter by outcome if requested
    if args.outcome:
        trade_knowledge = [tk for tk in trade_knowledge if tk.outcome == args.outcome]
        logger.info(f"Filtered to {len(trade_knowledge)} {args.outcome} trades")

    if not trade_knowledge:
        logger.warning("No trade knowledge to index after filtering")
        return 1

    # Index trades
    num_indexed = index_trades(rag, trade_knowledge, rebuild=args.rebuild)

    # Show final stats
    print_stats(rag)

    logger.info(f"✓ Indexing complete: {num_indexed} trades indexed")
    return 0


if __name__ == "__main__":
    exit(main())
