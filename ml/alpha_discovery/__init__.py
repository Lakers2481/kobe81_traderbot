"""
ML Alpha Discovery System
=========================

Comprehensive AI/ML pattern discovery for the Kobe trading robot.

Components:
1. Pattern Miner - Discover patterns via clustering (KMeans/DBSCAN)
2. Pattern Narrator - LLM explanations via Claude Sonnet 4
3. Feature Discovery - SHAP/permutation importance analysis
4. RL Agent - Reinforcement learning for timing optimization
5. Hybrid Pipeline - Orchestrated discovery-to-deployment workflow

Usage:
    from ml.alpha_discovery import (
        PatternClusteringEngine,
        PatternNarrator,
        FeatureImportanceAnalyzer,
        RLTradingAgent,
        HybridPatternPipeline,
    )

    # Run full discovery
    pipeline = HybridPatternPipeline()
    result = pipeline.run_discovery(trades_df, price_data)
"""

from .pattern_miner import (
    PatternCluster,
    PatternClusteringEngine,
    PatternLibrary,
)
from .pattern_narrator import (
    PatternNarrative,
    PatternPlaybook,
    PatternNarrator,
)
from .feature_discovery import (
    FeatureImportanceReport,
    FeatureImportanceAnalyzer,
)
from ml.alpha_discovery.rl_agent.trading_env import TradingEnv
from ml.alpha_discovery.rl_agent.agent import RLTradingAgent, RLAgentConfig
from ml.alpha_discovery.hybrid_pipeline.orchestrator import DiscoveryResult, HybridPatternPipeline

__all__ = [
    # Pattern Miner
    'PatternCluster',
    'PatternClusteringEngine',
    'PatternLibrary',
    # Pattern Narrator
    'PatternNarrative',
    'PatternPlaybook',
    'PatternNarrator',
    # Feature Discovery
    'FeatureImportanceReport',
    'FeatureImportanceAnalyzer',
    # RL Agent
    'TradingEnv',
    'RLTradingAgent',
    'RLAgentConfig',
    # Hybrid Pipeline
    'DiscoveryResult',
    'HybridPatternPipeline',
]
