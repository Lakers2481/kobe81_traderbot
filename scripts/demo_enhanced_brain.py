#!/usr/bin/env python3
"""
Demo Script for Enhanced Brain Integration.

Demonstrates all new capabilities:
- EnhancedResearchEngine (VectorBT, Alphalens)
- EnhancedAutonomousBrain (unified discovery system)
- KobeBrainGraph (LangGraph state machine)
- RAGEvaluator (LLM quality tracking)

Usage:
    python scripts/demo_enhanced_brain.py
    python scripts/demo_enhanced_brain.py --component research
    python scripts/demo_enhanced_brain.py --component brain
    python scripts/demo_enhanced_brain.py --component all

Created: 2026-01-07
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def demo_enhanced_research():
    """Demo EnhancedResearchEngine capabilities."""
    print("\n" + "=" * 60)
    print("DEMO: EnhancedResearchEngine")
    print("=" * 60)

    try:
        from autonomous.enhanced_research import EnhancedResearchEngine

        engine = EnhancedResearchEngine()

        # 1. Get summary
        print("\n1. Research Summary:")
        summary = engine.get_enhanced_research_summary()
        print(f"   Base Experiments: {summary.get('total_experiments', 0)}")
        print(f"   Alpha Discoveries: {summary['alpha_mining']['total_discoveries']}")
        print(f"   Validated: {summary['alpha_mining']['validated']}")
        print(f"   Top Sharpe: {summary['alpha_mining']['top_sharpe']:.2f}")

        # 2. Run VectorBT alpha sweep (if available)
        print("\n2. VectorBT Alpha Mining:")
        result = engine.run_vectorbt_alpha_sweep(min_sharpe=0.5)
        print(f"   Status: {result.get('status')}")
        if result.get('status') == 'success':
            print(f"   Alphas Discovered: {result.get('alphas_discovered', 0)}")
            print(f"   Top Sharpe: {result.get('top_sharpe', 0):.2f}")
        else:
            print(f"   Error: {result.get('error')}")
            print(f"   Recommendation: {result.get('recommendation', 'N/A')}")

        # 3. Get top alphas
        print("\n3. Top Alphas:")
        top = engine.get_top_alphas(n=5)
        if top:
            for i, alpha in enumerate(top, 1):
                print(f"   {i}. {alpha.alpha_name}")
                print(f"      Sharpe: {alpha.sharpe_ratio:.2f}")
                print(f"      Win Rate: {alpha.win_rate:.1%}")
                print(f"      Validated: {alpha.validated}")
        else:
            print("   No alphas discovered yet")

        # 4. Submit hypotheses to CuriosityEngine
        print("\n4. CuriosityEngine Integration:")
        submitted = engine.submit_alpha_hypotheses_to_curiosity_engine()
        print(f"   Hypotheses Submitted: {submitted}")

        print("\n" + "=" * 60)
        print("EnhancedResearchEngine demo complete!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Install dependencies: pip install vectorbt alphalens-reloaded")


def demo_enhanced_brain():
    """Demo EnhancedAutonomousBrain capabilities."""
    print("\n" + "=" * 60)
    print("DEMO: EnhancedAutonomousBrain")
    print("=" * 60)

    try:
        from autonomous.enhanced_brain import EnhancedAutonomousBrain

        # Initialize brain (no LangGraph for demo)
        brain = EnhancedAutonomousBrain(use_langgraph=False)

        # 1. Get status
        print("\n1. Brain Status:")
        status = brain.get_status()
        print(f"   Version: {status['enhanced_version']}")
        print(f"   Uptime Hours: {status['uptime_hours']:.1f}")
        print(f"   Cycles Completed: {status['cycles_completed']}")

        print("\n2. Awareness:")
        awareness = status['awareness']
        print(f"   Phase: {awareness['phase']}")
        print(f"   Work Mode: {awareness['work_mode']}")
        print(f"   Market Open: {awareness['market_open']}")

        print("\n3. Alpha Mining:")
        alpha = status['alpha_mining']
        print(f"   Total Discoveries: {alpha['total_discoveries']}")
        print(f"   Validated: {alpha['validated']}")
        print(f"   Promoted: {alpha['promoted']}")
        print(f"   Top Sharpe: {alpha['top_sharpe']:.2f}")

        print("\n4. LangGraph:")
        langgraph = status['langgraph']
        print(f"   Enabled: {langgraph['enabled']}")
        print(f"   Available: {langgraph['available']}")

        print("\n5. RAG Evaluator:")
        rag = status['rag_evaluator']
        print(f"   Enabled: {rag['enabled']}")
        print(f"   Evaluations: {rag['evaluations_count']}")

        # 2. Run single cycle
        print("\n6. Running Single Brain Cycle...")
        result = brain.run_single_cycle()
        print(f"   Timestamp: {result['timestamp']}")
        print(f"   Phase: {result['phase']}")
        print(f"   Work Mode: {result['work_mode']}")
        if result.get('task'):
            print(f"   Task Executed: {result['task']}")

        # 3. Check discoveries
        print("\n7. Checking for Discoveries...")
        discoveries = brain._check_for_discoveries()
        print(f"   Discoveries Found: {len(discoveries)}")
        for disc in discoveries[:3]:  # Show first 3
            print(f"   - {disc.discovery_type}: {disc.description[:60]}...")

        print("\n" + "=" * 60)
        print("EnhancedAutonomousBrain demo complete!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Enhanced brain not available")


def demo_brain_graph():
    """Demo KobeBrainGraph capabilities."""
    print("\n" + "=" * 60)
    print("DEMO: KobeBrainGraph (LangGraph State Machine)")
    print("=" * 60)

    try:
        from cognitive.brain_graph import get_brain_graph, HAS_LANGGRAPH

        if not HAS_LANGGRAPH:
            print("\nLangGraph not installed.")
            print("Install: pip install langgraph langchain-core")
            return

        brain_graph = get_brain_graph()
        if brain_graph is None:
            print("\nFailed to initialize KobeBrainGraph")
            return

        # 1. Show graph structure
        print("\n1. State Graph Structure:")
        print("   Nodes:")
        print("   - observe: Gather market context")
        print("   - analyze: Run scanner, compute signals")
        print("   - decide: Make trading decision")
        print("   - execute: Place orders")
        print("   - reflect: Learn from outcomes")
        print("   - research: Alpha mining")
        print("   - standby: Wait state")

        # 2. Visualize graph
        print("\n2. Graph Visualization (Mermaid):")
        try:
            mermaid = brain_graph.visualize()
            print(mermaid[:500] + "..." if len(mermaid) > 500 else mermaid)
        except Exception as e:
            print(f"   Visualization failed: {e}")

        # 3. Run a cycle
        print("\n3. Running State Machine Cycle...")
        from cognitive.states import create_initial_brain_state

        initial_state = create_initial_brain_state()
        result = brain_graph.run_cycle(initial_state)

        print(f"   Trading Phase: {result.get('trading_phase')}")
        print(f"   Decision: {result.get('decision')}")
        print(f"   Iteration: {result.get('iteration')}")

        print("\n" + "=" * 60)
        print("KobeBrainGraph demo complete!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Install: pip install langgraph langchain-core")


def demo_rag_evaluator():
    """Demo RAGEvaluator capabilities."""
    print("\n" + "=" * 60)
    print("DEMO: RAGEvaluator (LLM Quality Tracking)")
    print("=" * 60)

    try:
        from cognitive.rag_evaluator import get_rag_evaluator, RetrieverType

        evaluator = get_rag_evaluator()

        # 1. Generate explanations
        print("\n1. Generating Trade Explanations...")
        trade_context = {
            "symbol": "AAPL",
            "side": "long",
            "score": 75,
            "entry_price": 185.50,
        }

        explanations = evaluator.generate_explanations(
            trade_context,
            retriever_types=[
                RetrieverType.EPISODIC,
                RetrieverType.SEMANTIC,
                RetrieverType.PATTERN,
            ],
        )

        print(f"   Generated {len(explanations)} explanations")

        # 2. Show explanations
        print("\n2. Explanation Examples:")
        for exp in explanations[:2]:  # Show first 2
            print(f"\n   Retriever: {exp.retriever_type.value}")
            print(f"   Confidence: {exp.confidence:.1%}")
            text_preview = exp.explanation_text[:150]
            print(f"   Preview: {text_preview}...")

        # 3. Evaluate explanations
        print("\n3. Evaluating Explanation Quality...")
        for exp in explanations:
            eval_result = evaluator.evaluate_explanation(exp)
            print(f"\n   {exp.retriever_type.value}:")
            print(f"      Overall Score: {eval_result.overall_score:.1f}")
            print(f"      Quality Tier: {eval_result.quality_tier.value}")
            print(f"      Completeness: {eval_result.completeness_score:.1f}")
            print(f"      Coherence: {eval_result.coherence_score:.1f}")

        # 4. Get retriever stats
        print("\n4. Retriever Performance Stats:")
        stats = evaluator.get_retriever_stats()
        for retriever_type, rt_stats in stats.items():
            print(f"\n   {retriever_type.value}:")
            print(f"      Samples: {rt_stats['n_samples']}")
            print(f"      Avg Score: {rt_stats['avg_score']:.1f}")

        print("\n" + "=" * 60)
        print("RAGEvaluator demo complete!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nERROR: {e}")
        print("RAGEvaluator not available")


def demo_all():
    """Run all demos."""
    demo_enhanced_research()
    demo_enhanced_brain()
    demo_brain_graph()
    demo_rag_evaluator()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Run integration tests: pytest tests/test_enhanced_brain_integration.py -v")
    print("2. Start enhanced brain: python -m autonomous.enhanced_brain --cycle 60")
    print("3. Enable LangGraph: python -m autonomous.enhanced_brain --cycle 60 --langgraph")


def main():
    parser = argparse.ArgumentParser(
        description="Demo Enhanced Brain Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--component",
        choices=["research", "brain", "langgraph", "rag", "all"],
        default="all",
        help="Component to demo (default: all)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ENHANCED BRAIN INTEGRATION DEMO")
    print("=" * 60)
    print("Demonstrating new alpha mining + LangGraph integration")
    print()

    if args.component == "research":
        demo_enhanced_research()
    elif args.component == "brain":
        demo_enhanced_brain()
    elif args.component == "langgraph":
        demo_brain_graph()
    elif args.component == "rag":
        demo_rag_evaluator()
    else:
        demo_all()


if __name__ == "__main__":
    main()
