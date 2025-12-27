"""
Backtest Visualization Module
==============================

Interactive and static visualizations for backtest results.

Based on Backtesting.py visualization patterns with enhancements:
- Interactive HTML plots (Plotly)
- Static publication-ready charts (Matplotlib)
- Equity curves with drawdown overlay
- Trade markers on price charts
- Performance heatmaps
- Monthly returns table

Usage:
    from backtest.visualization import BacktestPlotter

    plotter = BacktestPlotter(results)
    plotter.plot_equity()
    plotter.plot_trades()
    plotter.save_report("backtest_report.html")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available, interactive plots disabled")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available, static plots disabled")


@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    theme: str = "dark"  # "dark" or "light"
    width: int = 1200
    height: int = 800
    colors: Dict[str, str] = None

    def __post_init__(self):
        if self.colors is None:
            if self.theme == "dark":
                self.colors = {
                    'background': '#1a1a2e',
                    'text': '#eaeaea',
                    'equity': '#00d4aa',
                    'drawdown': '#ff6b6b',
                    'buy': '#00ff88',
                    'sell': '#ff4444',
                    'grid': '#2a2a4a',
                    'positive': '#00d4aa',
                    'negative': '#ff6b6b',
                }
            else:
                self.colors = {
                    'background': '#ffffff',
                    'text': '#333333',
                    'equity': '#2ecc71',
                    'drawdown': '#e74c3c',
                    'buy': '#27ae60',
                    'sell': '#c0392b',
                    'grid': '#ecf0f1',
                    'positive': '#27ae60',
                    'negative': '#e74c3c',
                }


class BacktestPlotter:
    """
    Generates visualizations for backtest results.

    Supports:
    - Interactive Plotly charts
    - Static Matplotlib charts
    - HTML report generation
    """

    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        metrics: Dict[str, float],
        prices: Optional[pd.DataFrame] = None,
        config: Optional[PlotConfig] = None,
    ):
        """
        Initialize plotter with backtest results.

        Args:
            equity_curve: DataFrame with 'equity' column, timestamp index
            trades: DataFrame with trade records
            metrics: Dictionary of performance metrics
            prices: Optional OHLCV data for overlay plots
            config: Plot configuration
        """
        self.equity = equity_curve
        self.trades = trades
        self.metrics = metrics
        self.prices = prices
        self.config = config or PlotConfig()

    def plot_equity(
        self,
        show_drawdown: bool = True,
        benchmark: Optional[pd.Series] = None,
        interactive: bool = True,
    ) -> Union[go.Figure, plt.Figure, None]:
        """
        Plot equity curve with optional drawdown and benchmark.

        Args:
            show_drawdown: Show drawdown as secondary axis
            benchmark: Optional benchmark series (e.g., SPY returns)
            interactive: Use Plotly (True) or Matplotlib (False)

        Returns:
            Figure object
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_equity_plotly(show_drawdown, benchmark)
        elif MATPLOTLIB_AVAILABLE:
            return self._plot_equity_matplotlib(show_drawdown, benchmark)
        else:
            logger.warning("No visualization library available")
            return None

    def _plot_equity_plotly(
        self,
        show_drawdown: bool,
        benchmark: Optional[pd.Series],
    ) -> go.Figure:
        """Create interactive Plotly equity chart."""
        fig = make_subplots(
            rows=2 if show_drawdown else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3] if show_drawdown else [1.0],
            subplot_titles=["Portfolio Value", "Drawdown"] if show_drawdown else ["Portfolio Value"],
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=self.equity.index,
                y=self.equity['equity'],
                name='Portfolio',
                line=dict(color=self.config.colors['equity'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba({int(self.config.colors['equity'][1:3], 16)}, "
                          f"{int(self.config.colors['equity'][3:5], 16)}, "
                          f"{int(self.config.colors['equity'][5:7], 16)}, 0.1)",
            ),
            row=1, col=1
        )

        # Benchmark
        if benchmark is not None:
            # Normalize benchmark to start at same value as equity
            benchmark_normalized = benchmark / benchmark.iloc[0] * self.equity['equity'].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=benchmark_normalized,
                    name='Benchmark',
                    line=dict(color='#888888', width=1, dash='dash'),
                ),
                row=1, col=1
            )

        # Drawdown
        if show_drawdown:
            cummax = self.equity['equity'].cummax()
            drawdown = (self.equity['equity'] / cummax - 1) * 100

            fig.add_trace(
                go.Scatter(
                    x=self.equity.index,
                    y=drawdown,
                    name='Drawdown',
                    line=dict(color=self.config.colors['drawdown'], width=1),
                    fill='tozeroy',
                    fillcolor=f"rgba({int(self.config.colors['drawdown'][1:3], 16)}, "
                              f"{int(self.config.colors['drawdown'][3:5], 16)}, "
                              f"{int(self.config.colors['drawdown'][5:7], 16)}, 0.3)",
                ),
                row=2, col=1
            )

        # Layout
        fig.update_layout(
            title=self._generate_title(),
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
            paper_bgcolor=self.config.colors['background'],
            plot_bgcolor=self.config.colors['background'],
            font=dict(color=self.config.colors['text']),
            width=self.config.width,
            height=self.config.height,
            showlegend=True,
            legend=dict(x=0, y=1.1, orientation='h'),
            hovermode='x unified',
        )

        fig.update_xaxes(
            gridcolor=self.config.colors['grid'],
            showgrid=True,
        )
        fig.update_yaxes(
            gridcolor=self.config.colors['grid'],
            showgrid=True,
        )

        return fig

    def _plot_equity_matplotlib(
        self,
        show_drawdown: bool,
        benchmark: Optional[pd.Series],
    ) -> plt.Figure:
        """Create static Matplotlib equity chart."""
        if show_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
            ax2 = None

        # Set style based on theme
        if self.config.theme == 'dark':
            plt.style.use('dark_background')

        # Equity curve
        ax1.plot(self.equity.index, self.equity['equity'], color=self.config.colors['equity'], linewidth=2, label='Portfolio')
        ax1.fill_between(self.equity.index, 0, self.equity['equity'], alpha=0.1, color=self.config.colors['equity'])

        if benchmark is not None:
            benchmark_normalized = benchmark / benchmark.iloc[0] * self.equity['equity'].iloc[0]
            ax1.plot(benchmark.index, benchmark_normalized, color='gray', linestyle='--', linewidth=1, label='Benchmark')

        ax1.set_title(self._generate_title())
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        if show_drawdown and ax2 is not None:
            cummax = self.equity['equity'].cummax()
            drawdown = (self.equity['equity'] / cummax - 1) * 100
            ax2.fill_between(self.equity.index, 0, drawdown, color=self.config.colors['drawdown'], alpha=0.3)
            ax2.plot(self.equity.index, drawdown, color=self.config.colors['drawdown'], linewidth=1)
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _generate_title(self) -> str:
        """Generate chart title with key metrics."""
        parts = ["Kobe Backtest Results"]

        if 'total_return_pct' in self.metrics:
            ret = self.metrics['total_return_pct']
            parts.append(f"Return: {ret:+.1f}%")

        if 'sharpe_ratio' in self.metrics:
            sharpe = self.metrics['sharpe_ratio']
            parts.append(f"Sharpe: {sharpe:.2f}")

        if 'max_drawdown_pct' in self.metrics:
            dd = self.metrics['max_drawdown_pct']
            parts.append(f"MaxDD: {dd:.1f}%")

        if 'win_rate' in self.metrics:
            wr = self.metrics['win_rate']
            parts.append(f"Win Rate: {wr*100:.1f}%")

        return " | ".join(parts)

    def plot_trades(
        self,
        symbol: Optional[str] = None,
        interactive: bool = True,
    ) -> Union[go.Figure, plt.Figure, None]:
        """
        Plot price chart with trade markers.

        Args:
            symbol: Filter to specific symbol
            interactive: Use Plotly or Matplotlib
        """
        if self.prices is None or self.trades.empty:
            logger.warning("No price data or trades to plot")
            return None

        trades = self.trades
        if symbol:
            trades = trades[trades['symbol'] == symbol]
            prices = self.prices[self.prices['symbol'] == symbol] if 'symbol' in self.prices.columns else self.prices
        else:
            prices = self.prices

        if trades.empty or prices.empty:
            return None

        if interactive and PLOTLY_AVAILABLE:
            return self._plot_trades_plotly(prices, trades, symbol)
        elif MATPLOTLIB_AVAILABLE:
            return self._plot_trades_matplotlib(prices, trades, symbol)
        return None

    def _plot_trades_plotly(
        self,
        prices: pd.DataFrame,
        trades: pd.DataFrame,
        symbol: Optional[str],
    ) -> go.Figure:
        """Create interactive trade chart with Plotly."""
        fig = go.Figure()

        # Candlestick or line chart
        if all(col in prices.columns for col in ['open', 'high', 'low', 'close']):
            prices_indexed = prices.set_index('timestamp') if 'timestamp' in prices.columns else prices
            fig.add_trace(go.Candlestick(
                x=prices_indexed.index,
                open=prices_indexed['open'],
                high=prices_indexed['high'],
                low=prices_indexed['low'],
                close=prices_indexed['close'],
                name='Price',
            ))
        else:
            close_col = 'close' if 'close' in prices.columns else prices.columns[0]
            fig.add_trace(go.Scatter(
                x=prices.index if 'timestamp' not in prices.columns else prices['timestamp'],
                y=prices[close_col],
                name='Price',
                line=dict(color='#888888'),
            ))

        # Entry markers
        entries = trades[trades.get('entry_price', 0) > 0] if 'entry_price' in trades.columns else pd.DataFrame()
        if not entries.empty and 'entry_date' in entries.columns:
            fig.add_trace(go.Scatter(
                x=entries['entry_date'],
                y=entries['entry_price'],
                mode='markers',
                name='Entry',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color=self.config.colors['buy'],
                ),
            ))

        # Exit markers
        exits = trades[trades.get('exit_price', 0) > 0] if 'exit_price' in trades.columns else pd.DataFrame()
        if not exits.empty and 'exit_date' in exits.columns:
            fig.add_trace(go.Scatter(
                x=exits['exit_date'],
                y=exits['exit_price'],
                mode='markers',
                name='Exit',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color=self.config.colors['sell'],
                ),
            ))

        title = f"Trades{' - ' + symbol if symbol else ''}"
        fig.update_layout(
            title=title,
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
            xaxis_rangeslider_visible=False,
            width=self.config.width,
            height=self.config.height,
        )

        return fig

    def _plot_trades_matplotlib(
        self,
        prices: pd.DataFrame,
        trades: pd.DataFrame,
        symbol: Optional[str],
    ) -> plt.Figure:
        """Create static trade chart with Matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Price line
        close_col = 'close' if 'close' in prices.columns else prices.columns[0]
        x = prices.index if 'timestamp' not in prices.columns else prices['timestamp']
        ax.plot(x, prices[close_col], color='gray', linewidth=1, label='Price')

        # Entry markers
        if 'entry_date' in trades.columns and 'entry_price' in trades.columns:
            ax.scatter(trades['entry_date'], trades['entry_price'],
                      marker='^', color=self.config.colors['buy'], s=100, label='Entry', zorder=5)

        # Exit markers
        if 'exit_date' in trades.columns and 'exit_price' in trades.columns:
            ax.scatter(trades['exit_date'], trades['exit_price'],
                      marker='v', color=self.config.colors['sell'], s=100, label='Exit', zorder=5)

        ax.set_title(f"Trades{' - ' + symbol if symbol else ''}")
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_monthly_returns(self, interactive: bool = True) -> Union[go.Figure, plt.Figure, None]:
        """
        Plot monthly returns heatmap.
        """
        if self.equity.empty:
            return None

        # Calculate monthly returns
        equity = self.equity['equity'].copy()
        # Use 'ME' for pandas 2.2+, 'M' for older versions
        try:
            monthly = equity.resample('ME').last()
        except ValueError:
            monthly = equity.resample('M').last()
        monthly_returns = monthly.pct_change().dropna() * 100

        # Pivot to year x month
        df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values,
        })
        pivot = df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

        if interactive and PLOTLY_AVAILABLE:
            return self._plot_monthly_heatmap_plotly(pivot)
        elif MATPLOTLIB_AVAILABLE:
            return self._plot_monthly_heatmap_matplotlib(pivot)
        return None

    def _plot_monthly_heatmap_plotly(self, pivot: pd.DataFrame) -> go.Figure:
        """Create interactive monthly returns heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        ))

        fig.update_layout(
            title="Monthly Returns (%)",
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
            width=self.config.width,
            height=400,
        )

        return fig

    def _plot_monthly_heatmap_matplotlib(self, pivot: pd.DataFrame) -> plt.Figure:
        """Create static monthly returns heatmap."""
        fig, ax = plt.subplots(figsize=(12, 4))

        cmap = plt.cm.RdYlGn
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=-10, vmax=10)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 5 else 'black'
                    ax.text(j, i, f"{val:.1f}%", ha='center', va='center', color=color, fontsize=8)

        ax.set_title("Monthly Returns (%)")
        plt.colorbar(im, ax=ax, label='Return %')
        plt.tight_layout()

        return fig

    def generate_report(
        self,
        output_path: Union[str, Path],
        include_trades_table: bool = True,
    ) -> Path:
        """
        Generate comprehensive HTML report.

        Args:
            output_path: Path to save HTML report
            include_trades_table: Include detailed trades table

        Returns:
            Path to generated report
        """
        output_path = Path(output_path)

        # Generate all plots
        figs = []

        equity_fig = self.plot_equity(interactive=True)
        if equity_fig:
            figs.append(("Equity Curve", equity_fig))

        monthly_fig = self.plot_monthly_returns(interactive=True)
        if monthly_fig:
            figs.append(("Monthly Returns", monthly_fig))

        trades_fig = self.plot_trades(interactive=True)
        if trades_fig:
            figs.append(("Trades", trades_fig))

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Kobe Backtest Report</title>",
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eaeaea; }",
            ".metric { display: inline-block; margin: 10px 20px; padding: 15px; background: #2a2a4a; border-radius: 5px; }",
            ".metric-value { font-size: 24px; font-weight: bold; }",
            ".metric-label { font-size: 12px; color: #888; }",
            ".positive { color: #00d4aa; }",
            ".negative { color: #ff6b6b; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #444; padding: 8px; text-align: right; }",
            "th { background: #2a2a4a; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Kobe Trading System - Backtest Report</h1>",
            f"<p>Generated: {pd.Timestamp.now()}</p>",
            "<h2>Performance Summary</h2>",
            '<div class="metrics">',
        ]

        # Add metrics
        for key, value in self.metrics.items():
            label = key.replace('_', ' ').title()
            if isinstance(value, float):
                if 'pct' in key or 'rate' in key:
                    formatted = f"{value:.2f}%"
                else:
                    formatted = f"{value:.2f}"
                css_class = 'positive' if value > 0 else 'negative' if value < 0 else ''
            else:
                formatted = str(value)
                css_class = ''

            html_parts.append(
                f'<div class="metric">'
                f'<div class="metric-value {css_class}">{formatted}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>'
            )

        html_parts.append('</div>')

        # Add plots
        for title, fig in figs:
            html_parts.append(f"<h2>{title}</h2>")
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

        # Add trades table
        if include_trades_table and not self.trades.empty:
            html_parts.append("<h2>Trade Log</h2>")
            html_parts.append(self.trades.to_html(classes='trades-table'))

        html_parts.extend([
            "</body>",
            "</html>",
        ])

        output_path.write_text("\n".join(html_parts))
        logger.info(f"Report saved to: {output_path}")

        return output_path

    def save_plots(self, output_dir: Union[str, Path], format: str = 'png') -> List[Path]:
        """
        Save all plots to files.

        Args:
            output_dir: Directory to save plots
            format: 'png', 'svg', 'pdf', or 'html'

        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        plots = [
            ('equity_curve', lambda: self.plot_equity(interactive=format == 'html')),
            ('monthly_returns', lambda: self.plot_monthly_returns(interactive=format == 'html')),
            ('trades', lambda: self.plot_trades(interactive=format == 'html')),
        ]

        for name, plot_func in plots:
            try:
                fig = plot_func()
                if fig is None:
                    continue

                if format == 'html' and hasattr(fig, 'write_html'):
                    path = output_dir / f"{name}.html"
                    fig.write_html(str(path))
                elif hasattr(fig, 'write_image'):
                    path = output_dir / f"{name}.{format}"
                    fig.write_image(str(path))
                elif hasattr(fig, 'savefig'):
                    path = output_dir / f"{name}.{format}"
                    fig.savefig(str(path), dpi=150, bbox_inches='tight')
                    plt.close(fig)
                else:
                    continue

                saved.append(path)
                logger.info(f"Saved: {path}")

            except Exception as e:
                logger.warning(f"Could not save {name}: {e}")

        return saved
