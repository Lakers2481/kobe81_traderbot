"""
Comprehensive Cognitive & ML System Integration Test
Tests all major components and their interactions for paper trading readiness.
"""

import sys
import traceback
from datetime import datetime

def _run_component_test(name, test_fn):
    """Test a component and return (success, message)."""
    try:
        result = test_fn()
        return True, f"{name}: OK {result if result else ''}"
    except Exception as e:
        return False, f"{name}: FAILED - {str(e)}\n{traceback.format_exc()}"

def main():
    print("=" * 80)
    print("COGNITIVE & ML SYSTEM INTEGRATION TEST")
    print("=" * 80)
    print()

    results = []

    # 1. CognitiveBrain
    def test_brain():
        from cognitive.cognitive_brain import get_cognitive_brain
        brain = get_cognitive_brain()
        status = brain.get_status()
        return f"(components: {len(status.get('components', {}))})"
    results.append(_run_component_test("1. CognitiveBrain", test_brain))

    # 2. MetacognitiveGovernor
    def test_governor():
        from cognitive.metacognitive_governor import MetacognitiveGovernor
        gov = MetacognitiveGovernor()
        stats = gov.get_routing_stats()
        return f"(decisions: {stats.get('total_decisions', 0)})"
    results.append(_run_component_test("2. MetacognitiveGovernor", test_governor))

    # 3. ReflectionEngine
    def test_reflection():
        from cognitive.reflection_engine import ReflectionEngine
        engine = ReflectionEngine()
        return "(initialized)"
    results.append(_run_component_test("3. ReflectionEngine", test_reflection))

    # 4. EpisodicMemory
    def test_episodic():
        from cognitive.episodic_memory import get_episodic_memory
        mem = get_episodic_memory()
        stats = mem.get_stats()
        return f"(episodes: {stats.get('total_episodes', 0)})"
    results.append(_run_component_test("4. EpisodicMemory", test_episodic))

    # 5. SemanticMemory
    def test_semantic():
        from cognitive.semantic_memory import get_semantic_memory
        mem = get_semantic_memory()
        stats = mem.get_stats()
        return f"(rules: {stats.get('total_rules', 0)})"
    results.append(_run_component_test("5. SemanticMemory", test_semantic))

    # 6. GameBriefings
    def test_briefings():
        from cognitive.game_briefings import GameBriefingEngine
        engine = GameBriefingEngine()
        return "(initialized)"
    results.append(_run_component_test("6. GameBriefings", test_briefings))

    # 7. CuriosityEngine
    def test_curiosity():
        from cognitive.curiosity_engine import get_curiosity_engine
        engine = get_curiosity_engine()
        stats = engine.get_stats()
        return f"(hypotheses: {stats.get('total_hypotheses', 0)}, edges: {stats.get('total_edges', 0)})"
    results.append(_run_component_test("7. CuriosityEngine", test_curiosity))

    # 8. KnowledgeBoundary
    def test_boundary():
        from cognitive.knowledge_boundary import KnowledgeBoundary
        kb = KnowledgeBoundary()
        return "(initialized)"
    results.append(_run_component_test("8. KnowledgeBoundary", test_boundary))

    # 9. SelfModel
    def test_selfmodel():
        from cognitive.self_model import get_self_model
        sm = get_self_model()
        status = sm.get_status()
        return f"(records: {status.get('performance_records', 0)})"
    results.append(_run_component_test("9. SelfModel", test_selfmodel))

    # 10. HMM Regime Detector
    def test_hmm():
        from ml_advanced.hmm_regime_detector import HMMRegimeDetector
        detector = HMMRegimeDetector()
        return "(initialized)"
    results.append(_run_component_test("10. HMM Regime Detector", test_hmm))

    # 11. Confidence Integrator
    def test_confidence():
        from ml_features.confidence_integrator import ConfidenceIntegrator
        ci = ConfidenceIntegrator()
        return "(initialized)"
    results.append(_run_component_test("11. Confidence Integrator", test_confidence))

    # 12. LLM Narrative Analyzer
    def test_llm():
        from cognitive.llm_narrative_analyzer import get_llm_analyzer
        analyzer = get_llm_analyzer()
        return "(initialized)"
    results.append(_run_component_test("12. LLM Narrative Analyzer", test_llm))

    # 13. Symbolic Reasoner
    def test_symbolic():
        from cognitive.symbolic_reasoner import get_symbolic_reasoner
        sr = get_symbolic_reasoner()
        rules = sr.get_rules()
        return f"(rules: {len(rules)})"
    results.append(_run_component_test("13. Symbolic Reasoner", test_symbolic))

    # 14. Dynamic Policy Generator
    def test_policy():
        from cognitive.dynamic_policy_generator import get_policy_generator
        pg = get_policy_generator()
        policies = pg.get_all_policies()
        return f"(policies: {len(policies)})"
    results.append(_run_component_test("14. Dynamic Policy Generator", test_policy))

    # 15. Global Workspace
    def test_workspace():
        from cognitive.global_workspace import get_workspace
        ws = get_workspace()
        stats = ws.get_stats()
        return f"(capacity: {stats.get('working_memory_capacity', 0)})"
    results.append(_run_component_test("15. Global Workspace", test_workspace))

    # 16. Integration Test: Full Deliberation
    def test_integration():
        from cognitive.cognitive_brain import get_cognitive_brain
        brain = get_cognitive_brain()

        # Test signal and context
        signal = {
            'symbol': 'AAPL',
            'strategy': 'ibs_rsi',
            'side': 'long',
            'entry_price': 150.00,
            'stop_loss': 148.50,
        }

        context = {
            'regime': 'BULLISH',
            'regime_confidence': 0.85,
            'vix': 18.5,
            'market_sentiment': {'compound': 0.3},
            'is_extreme_mood': False,
            'market_mood_score': 0.2,
            'market_mood_state': 'Cautiously Optimistic',
            'data_timestamp': datetime.now().isoformat(),
        }

        # Run deliberation
        decision = brain.deliberate(signal, context, fast_confidence=0.65)

        return f"(decision: {decision.decision_type.value}, confidence: {decision.confidence:.2f})"
    results.append(_run_component_test("16. INTEGRATION: Full Deliberation", test_integration))

    # Print results
    print()
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for success, message in results:
        status = "PASS" if success else "FAIL"
        print(f"[{status}] {message}")
        if success:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(results)} tests")
    print("=" * 80)

    if failed == 0:
        print()
        print("ALL COGNITIVE & ML COMPONENTS READY FOR PAPER TRADING")
        return 0
    else:
        print()
        print(f"WARNING: {failed} COMPONENTS FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(main())
