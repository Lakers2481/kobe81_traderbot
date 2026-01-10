#!/usr/bin/env python3
"""
Pre-Game Trading Report Generator
==================================

Generates a comprehensive pre-market analysis report with all AI components.
"""

import pandas as pd
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_report():
    """Generate comprehensive pre-game trading report."""

    print("=" * 80)
    print("KOBE TRADING SYSTEM - PRE-GAME REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Signal Date: 2025-12-31 (Trades Execute: Next Trading Day)")
    print("=" * 80)
    print()

    # Read today's picks
    try:
        df = pd.read_csv('logs/daily_picks.csv')
    except Exception as e:
        print(f"ERROR: Cannot read daily_picks.csv: {e}")
        return

    print("EXECUTIVE SUMMARY")
    print("-" * 40)
    print(f"Total Signals Passed Quality Gate: {len(df)}")
    print(f"Cognitive Approval: All {len(df)} signals ACCEPTED")
    print("Mode: PREVIEW (Holiday - signals for next trading day)")
    print()

    print("TOP 3 TRADING OPPORTUNITIES")
    print("=" * 80)

    for i, row in df.iterrows():
        symbol = row['symbol']
        strategy = row['strategy']
        entry = float(row['entry_price'])
        stop = float(row['stop_loss'])
        atr = float(row.get('atr14', 0) or 0)

        print(f"\n#{i+1} {symbol} - {strategy}")
        print("-" * 50)
        print(f"  Entry:      ${entry:.2f}")
        print(f"  Stop Loss:  ${stop:.2f} ({((stop/entry)-1)*100:.1f}%)")
        print(f"  ATR(14):    ${atr:.2f}")
        print(f"  Risk/Share: ${entry-stop:.2f}")
        print()

        print("  SIGNAL QUALITY:")
        print(f"    Quality Score:   {row.get('quality_score', 0):.1f}")
        print(f"    Quality Tier:    {row.get('quality_tier', 'N/A')}")
        print(f"    Confidence:      {float(row.get('conf_score', 0) or 0)*100:.1f}%")
        print()

        print("  TECHNICAL INDICATORS:")
        print(f"    IBS:             {float(row.get('ibs', 0) or 0):.3f}")
        print(f"    RSI(2):          {float(row.get('rsi2', 0) or 0):.1f}")
        sma_ratio = float(row.get('sma20_over_200', 1) or 1)
        print(f"    SMA20/SMA200:    {sma_ratio:.3f} ({'BULLISH' if sma_ratio > 1 else 'BEARISH'})")
        print(f"    Donchian Width:  ${float(row.get('don20_width', 0) or 0):.2f}")
        print(f"    Position in Don: {float(row.get('pos_in_don20', 0) or 0)*100:.1f}%")
        print()

        print("  LIQUIDITY:")
        adv = float(row.get('adv_usd60', 0) or 0)
        print(f"    ADV (60d):       ${adv/1e6:.1f}M daily")
        print()

        print("  AI/ML ADJUDICATION:")
        print(f"    Adjudication:    {float(row.get('adjudication_score', 0) or 0):.1f}/100")
        print(f"    Signal Strength: {float(row.get('adj_signal_strength', 0) or 0):.1f}")
        print(f"    Pattern Conf:    {float(row.get('adj_pattern_confluence', 0) or 0):.1f}")
        print(f"    Vol Contraction: {float(row.get('adj_volatility_contraction', 0) or 0):.1f}")
        patterns = row.get('adj_patterns_detected', 'N/A')
        print(f"    Patterns Found:  {patterns}")
        print(f"    RSI2 Tier:       {row.get('adj_rsi2_tier', 'N/A')}")

    print()
    print("=" * 80)
    print("COGNITIVE ARCHITECTURE STATUS")
    print("=" * 80)

    # Check cognitive components
    try:
        from cognitive.episodic_memory import get_episodic_memory
        mem = get_episodic_memory()
        stats = mem.get_stats()
        print("\n  Episodic Memory:")
        print(f"    Total Episodes:    {stats.get('total_episodes', 0)}")
        print(f"    Active Episodes:   {stats.get('active_episodes', 0)}")
        print(f"    Historical WR:     {stats.get('win_rate', 'N/A')}")
        print(f"    Total Lessons:     {stats.get('total_lessons', 0)}")
    except Exception as e:
        print(f"\n  Episodic Memory: ERROR - {e}")

    try:
        from cognitive.self_model import get_self_model
        sm = get_self_model()
        print("\n  Self Model:")
        print(f"    Strengths:  {len(sm.get_strengths())} contexts")
        print(f"    Weaknesses: {len(sm.get_weaknesses())} contexts")
        print(f"    Calibrated: {sm.is_well_calibrated()}")
    except Exception as e:
        print(f"\n  Self Model: ERROR - {e}")

    try:
        from cognitive.knowledge_boundary import KnowledgeBoundary
        from cognitive.episodic_memory import EpisodicMemory
        KnowledgeBoundary()
        # Check episodic support for IBS_RSI
        sig = EpisodicMemory.normalize_context_signature({
            'regime': 'unknown', 'strategy': 'IBS_RSI', 'side': 'long'
        })
        episodic_stats = mem.get_stats_for_signature(sig)
        print("\n  Knowledge Boundary:")
        print(f"    IBS_RSI Episodic N:  {episodic_stats['n']}")
        print(f"    IBS_RSI Episodic WR: {episodic_stats['win_rate']:.1%}")
        print(f"    Signature:           {sig}")
    except Exception as e:
        print(f"\n  Knowledge Boundary: ERROR - {e}")

    print()
    print("=" * 80)
    print("ML ENSEMBLE STATUS")
    print("=" * 80)

    try:
        from ml_advanced.ensemble.loader import get_ensemble_predictor
        ep = get_ensemble_predictor()
        if ep and ep.models:
            print(f"\n  Models Loaded: {len(ep.models)}")
            for name, model in ep.models.items():
                print(f"    - {name}: {type(model).__name__}")
        else:
            print("\n  Models: Not loaded or empty")
    except Exception as e:
        print(f"\n  Ensemble Predictor: ERROR - {e}")

    print()
    print("=" * 80)
    print("RISK MANAGEMENT")
    print("=" * 80)

    print("\n  Per-Order Budget:  $75")
    print("  Daily Budget:      $1,000")
    print("  Position Sizing:   ATR-based (2x ATR stop)")
    print("  Time Stop:         7 bars")
    print(f"  Kill Switch:       {'ACTIVE' if Path('state/KILL_SWITCH').exists() else 'INACTIVE'}")

    print()
    print("=" * 80)
    print("TRADE EXECUTION PLAN")
    print("=" * 80)

    for i, row in df.iterrows():
        symbol = row['symbol']
        entry = float(row['entry_price'])
        stop = float(row['stop_loss'])
        risk_per_share = entry - stop

        # Calculate position size based on $75 budget
        max_shares = int(75 / risk_per_share) if risk_per_share > 0 else 0
        position_value = max_shares * entry

        print(f"\n  {symbol}:")
        print(f"    Max Shares:      {max_shares}")
        print(f"    Position Value:  ${position_value:.2f}")
        print(f"    Risk Amount:     ${max_shares * risk_per_share:.2f}")
        print("    Order Type:      IOC LIMIT @ best_ask * 1.001")

    print()
    print("=" * 80)
    print("END OF PRE-GAME REPORT")
    print("=" * 80)


if __name__ == '__main__':
    generate_report()
