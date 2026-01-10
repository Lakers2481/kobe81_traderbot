"""
Analytics Module for Edge Analysis, Strategy Monitoring, and High-Performance Queries.

Provides tools for understanding and monitoring trading edge:
- Edge Decomposition: Stratify performance by multiple dimensions
- Factor Attribution: Identify which signal components drive profits
- Auto Stand-down: Automatic position reduction on edge degradation
- DuckDB Engine: 10-100x faster aggregations than pandas
"""

from analytics.edge_decomposition import (
    EdgeDecomposition,
    DecompositionResult,
    DimensionStats,
)
from analytics.factor_attribution import (
    FactorAttribution,
    AttributionResult,
    FactorContribution,
)
from analytics.auto_standdown import (
    AutoStanddown,
    StanddownRecommendation,
    StanddownSeverity,
)
from analytics.duckdb_engine import (
    DuckDBEngine,
    QueryResult,
    get_engine,
    query_trades,
    query_performance,
    query_positions,
    analyze_wf_results,
)

# Pyfolio integration (optional - requires pyfolio-reloaded)
try:
    from analytics.pyfolio_integration import (
        load_returns_from_backtest,
        load_transactions_from_backtest,
        fetch_benchmark_returns,
        generate_simple_tearsheet,
        generate_full_tearsheet,
        generate_wf_tearsheets,
        extract_pyfolio_metrics,
        HAS_PYFOLIO,
    )
except ImportError:
    HAS_PYFOLIO = False

__all__ = [
    # Edge Decomposition
    "EdgeDecomposition",
    "DecompositionResult",
    "DimensionStats",
    # Factor Attribution
    "FactorAttribution",
    "AttributionResult",
    "FactorContribution",
    # Auto Stand-down
    "AutoStanddown",
    "StanddownRecommendation",
    "StanddownSeverity",
    # DuckDB High-Performance Analytics
    "DuckDBEngine",
    "QueryResult",
    "get_engine",
    "query_trades",
    "query_performance",
    "query_positions",
    "analyze_wf_results",
]
