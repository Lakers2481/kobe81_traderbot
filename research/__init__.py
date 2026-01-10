from __future__ import annotations

"""
Research package for feature discovery and alpha screening.

This package provides utilities for:
- Alpha mining (100+ alpha factors via AlphaLibrary)
- Fast backtesting (10,000+ variants via VectorBT)
- Factor validation (IC analysis via Alphalens)
- Brain integration (CuriosityEngine, ResearchEngine)

Components:
- AlphaLibrary: 100+ alpha factors organized by category
- AlphaMiner: VectorBT-based fast parameter sweep
- FactorValidator: Alphalens IC and quantile analysis
- AlphaResearchIntegration: Wires everything to Kobe's brain
"""

# Legacy alphas (3 basic alphas)
from research.alphas import compute_alphas, ALPHA_REGISTRY

# New alpha mining infrastructure (2026-01-07)
try:
    from research.alpha_library import AlphaLibrary, get_alpha_library
    HAS_ALPHA_LIBRARY = True
except ImportError:
    HAS_ALPHA_LIBRARY = False
    AlphaLibrary = None
    get_alpha_library = None

try:
    from research.vectorbt_miner import AlphaMiner, HAS_VBT
except ImportError:
    HAS_VBT = False
    AlphaMiner = None

try:
    from research.factor_validator import FactorValidator, FactorReport, HAS_ALPHALENS
except ImportError:
    HAS_ALPHALENS = False
    FactorValidator = None
    FactorReport = None

try:
    from research.alpha_research_integration import (
        AlphaResearchIntegration,
        AlphaDiscovery,
        get_alpha_research_integration,
        run_alpha_mining,
        get_alpha_library_alphas,
        get_research_summary,
    )
    HAS_INTEGRATION = True
except ImportError:
    HAS_INTEGRATION = False
    AlphaResearchIntegration = None
    AlphaDiscovery = None
    get_alpha_research_integration = None
    run_alpha_mining = None
    get_alpha_library_alphas = None
    get_research_summary = None


__all__ = [
    # Legacy
    'compute_alphas',
    'ALPHA_REGISTRY',

    # New alpha library
    'AlphaLibrary',
    'get_alpha_library',
    'HAS_ALPHA_LIBRARY',

    # VectorBT mining
    'AlphaMiner',
    'HAS_VBT',

    # Alphalens validation
    'FactorValidator',
    'FactorReport',
    'HAS_ALPHALENS',

    # Integration
    'AlphaResearchIntegration',
    'AlphaDiscovery',
    'get_alpha_research_integration',
    'run_alpha_mining',
    'get_alpha_library_alphas',
    'get_research_summary',
    'HAS_INTEGRATION',
]
