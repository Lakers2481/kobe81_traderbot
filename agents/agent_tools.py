"""
Agent Tools - Safe Operations for Agents
=========================================

Tools that agents can use for:
- File reading (safe paths only)
- Data fetching (cached only)
- Backtest execution
- Experiment registry operations

SAFETY CONSTRAINTS:
- Read-only for most operations
- Writes go to drafts/ directory only
- No network calls without caching
- No live trading actions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from llm import ToolDefinition
from agents.base_agent import ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# Safe Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
SAFE_READ_DIRS = [
    PROJECT_ROOT / "strategies",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "config",
    PROJECT_ROOT / "backtest",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "wf_outputs",
    PROJECT_ROOT / "experiments",
    PROJECT_ROOT / "docs",
]
DRAFTS_DIR = PROJECT_ROOT / "drafts"


def _is_safe_path(path: Path) -> bool:
    """Check if path is in a safe directory for reading."""
    path = path.resolve()
    for safe_dir in SAFE_READ_DIRS:
        try:
            path.relative_to(safe_dir.resolve())
            return True
        except ValueError:
            continue
    return False


# =============================================================================
# File Tools
# =============================================================================

def read_file(file_path: str, max_lines: int = 500) -> ToolResult:
    """
    Read a file from safe directories.

    Args:
        file_path: Path to file (relative or absolute)
        max_lines: Maximum lines to read

    Returns:
        ToolResult with file contents
    """
    try:
        path = Path(file_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not _is_safe_path(path):
            return ToolResult(
                success=False,
                output="",
                error=f"Path not in safe directories: {path}",
            )

        if not path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {path}",
            )

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"\n... (truncated at {max_lines} lines)")
                    break
                lines.append(line)

        return ToolResult(
            success=True,
            output="".join(lines),
            data={"path": str(path), "lines_read": len(lines)},
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=str(e),
        )


def list_files(directory: str, pattern: str = "*") -> ToolResult:
    """
    List files in a safe directory.

    Args:
        directory: Directory path
        pattern: Glob pattern (e.g., "*.py", "*.csv")

    Returns:
        ToolResult with file list
    """
    try:
        path = Path(directory)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not _is_safe_path(path) and path != PROJECT_ROOT:
            return ToolResult(
                success=False,
                output="",
                error=f"Path not in safe directories: {path}",
            )

        if not path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Directory not found: {path}",
            )

        files = sorted([str(f.relative_to(PROJECT_ROOT)) for f in path.glob(pattern)])

        return ToolResult(
            success=True,
            output="\n".join(files) if files else "(no files found)",
            data={"count": len(files), "files": files},
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=str(e),
        )


def write_draft(filename: str, content: str) -> ToolResult:
    """
    Write content to drafts directory.

    Args:
        filename: Filename (will be placed in drafts/)
        content: File content

    Returns:
        ToolResult with path written
    """
    try:
        DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_name = "".join(c for c in filename if c.isalnum() or c in "._-")
        if not safe_name:
            safe_name = f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        path = DRAFTS_DIR / safe_name

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return ToolResult(
            success=True,
            output=f"Written to: {path}",
            data={"path": str(path), "size": len(content)},
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=str(e),
        )


# =============================================================================
# Data Tools
# =============================================================================

def get_universe_symbols(
    universe_file: str = "data/universe/optionable_liquid_800.csv",
    limit: int = 100,
) -> ToolResult:
    """
    Get symbols from universe file.

    Args:
        universe_file: Path to universe CSV
        limit: Max symbols to return

    Returns:
        ToolResult with symbol list
    """
    try:
        import pandas as pd

        path = PROJECT_ROOT / universe_file
        if not path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Universe file not found: {path}",
            )

        df = pd.read_csv(path)
        col = "symbol" if "symbol" in df.columns else df.columns[0]
        symbols = df[col].tolist()[:limit]

        return ToolResult(
            success=True,
            output=", ".join(symbols[:20]) + (f"... ({len(symbols)} total)" if len(symbols) > 20 else ""),
            data={"symbols": symbols, "count": len(symbols)},
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=str(e),
        )


def get_cached_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> ToolResult:
    """
    Get cached OHLCV data for a symbol.

    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        ToolResult with data summary
    """
    try:
        import pandas as pd

        # Check cache directory
        cache_dir = PROJECT_ROOT / "cache" / "polygon"
        cache_file = cache_dir / f"{symbol.upper()}.csv"

        if not cache_file.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"No cached data for {symbol}. Data must be prefetched.",
            )

        df = pd.read_csv(cache_file, parse_dates=["date"])

        # Filter by date if specified
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        if df.empty:
            return ToolResult(
                success=False,
                output="",
                error=f"No data in date range for {symbol}",
            )

        summary = {
            "symbol": symbol,
            "rows": len(df),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "columns": list(df.columns),
        }

        return ToolResult(
            success=True,
            output=json.dumps(summary, indent=2),
            data={"summary": summary, "df": df},
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=str(e),
        )


# =============================================================================
# Backtest Tools
# =============================================================================

def run_mini_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str,
    strategy: str = "dual",
) -> ToolResult:
    """
    Run a mini backtest on specified symbols.

    Args:
        symbols: List of symbols (max 10)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        strategy: Strategy name (dual, ibs_rsi, turtle_soup)

    Returns:
        ToolResult with backtest summary
    """
    try:
        # Limit symbols for safety
        if len(symbols) > 10:
            symbols = symbols[:10]

        # Import strategy
        from strategies.registry import get_production_scanner

        scanner = get_production_scanner()

        # This would run actual backtest - simplified for now
        result = {
            "symbols": symbols,
            "start": start_date,
            "end": end_date,
            "strategy": strategy,
            "status": "would_run_backtest",
            "note": "Full implementation connects to backtest.engine",
        }

        return ToolResult(
            success=True,
            output=json.dumps(result, indent=2),
            data=result,
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=str(e),
        )


# =============================================================================
# Experiment Registry Tools
# =============================================================================

def get_experiment_status(experiment_id: str) -> ToolResult:
    """
    Get status of an experiment.

    Args:
        experiment_id: Experiment ID

    Returns:
        ToolResult with experiment status
    """
    try:
        from experiments.registry import get_experiment

        exp = get_experiment(experiment_id)
        if exp is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Experiment not found: {experiment_id}",
            )

        return ToolResult(
            success=True,
            output=json.dumps(exp, indent=2, default=str),
            data=exp,
        )

    except ImportError:
        return ToolResult(
            success=False,
            output="",
            error="experiments.registry not available",
        )
    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=str(e),
        )


def list_experiments(limit: int = 10) -> ToolResult:
    """
    List recent experiments.

    Args:
        limit: Max experiments to return

    Returns:
        ToolResult with experiment list
    """
    try:
        from experiments.registry import list_experiments as _list_experiments

        experiments = _list_experiments(limit=limit)

        return ToolResult(
            success=True,
            output=json.dumps(experiments, indent=2, default=str),
            data={"experiments": experiments, "count": len(experiments)},
        )

    except ImportError:
        return ToolResult(
            success=False,
            output="",
            error="experiments.registry not available",
        )
    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=str(e),
        )


# =============================================================================
# Tool Definitions
# =============================================================================

def get_file_tools() -> List[tuple]:
    """Get file operation tools."""
    return [
        (
            ToolDefinition(
                name="read_file",
                description="Read a file from the project (strategies, data, config, etc.)",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to file (relative to project root)",
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "Maximum lines to read (default 500)",
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            read_file,
        ),
        (
            ToolDefinition(
                name="list_files",
                description="List files in a directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "Directory path",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern (e.g., *.py)",
                        },
                    },
                    "required": ["directory"],
                },
            ),
            list_files,
        ),
        (
            ToolDefinition(
                name="write_draft",
                description="Write content to drafts directory (for review)",
                parameters={
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Filename for the draft",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                    },
                    "required": ["filename", "content"],
                },
            ),
            write_draft,
        ),
    ]


def get_data_tools() -> List[tuple]:
    """Get data access tools."""
    return [
        (
            ToolDefinition(
                name="get_universe_symbols",
                description="Get stock symbols from the 900-stock universe",
                parameters={
                    "type": "object",
                    "properties": {
                        "universe_file": {
                            "type": "string",
                            "description": "Universe file path",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max symbols to return",
                        },
                    },
                },
            ),
            get_universe_symbols,
        ),
        (
            ToolDefinition(
                name="get_cached_data",
                description="Get cached OHLCV data for a symbol",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)",
                        },
                    },
                    "required": ["symbol"],
                },
            ),
            get_cached_data,
        ),
    ]


def get_backtest_tools() -> List[tuple]:
    """Get backtest tools."""
    return [
        (
            ToolDefinition(
                name="run_mini_backtest",
                description="Run a quick backtest on up to 10 symbols",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of symbols (max 10)",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)",
                        },
                        "strategy": {
                            "type": "string",
                            "description": "Strategy (dual, ibs_rsi, turtle_soup)",
                        },
                    },
                    "required": ["symbols", "start_date", "end_date"],
                },
            ),
            run_mini_backtest,
        ),
    ]


def get_registry_tools() -> List[tuple]:
    """Get experiment registry tools."""
    return [
        (
            ToolDefinition(
                name="get_experiment_status",
                description="Get status of an experiment",
                parameters={
                    "type": "object",
                    "properties": {
                        "experiment_id": {
                            "type": "string",
                            "description": "Experiment ID",
                        },
                    },
                    "required": ["experiment_id"],
                },
            ),
            get_experiment_status,
        ),
        (
            ToolDefinition(
                name="list_experiments",
                description="List recent experiments",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Max experiments to return",
                        },
                    },
                },
            ),
            list_experiments,
        ),
    ]


def get_all_tools() -> List[tuple]:
    """Get all available tools."""
    return (
        get_file_tools() +
        get_data_tools() +
        get_backtest_tools() +
        get_registry_tools()
    )
