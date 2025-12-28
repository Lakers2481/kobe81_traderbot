"""
Intelligent Trade Executor
===========================

Central orchestrator that wires together all advanced components:
1. AdaptiveStrategySelector - picks strategy based on HMM regime
2. Strategy - generates trading signals
3. CognitiveSignalProcessor - scores signals with AI confidence (now integrated)
4. PortfolioRiskManager - approves and sizes trades
5. OrderManager - executes orders intelligently (new)
6. TrailingStopManager - manages dynamic exits

This is THE integration layer that makes all our advanced components
work together in production.

Usage:
    from execution.intelligent_executor import IntelligentExecutor

    executor = IntelligentExecutor(equity=100000)

    # Run the full pipeline
    results = executor.execute_pipeline(
        universe_data=price_dict,
        spy_data=spy_df,
        vix_level=18.5,
        current_positions=positions
    )

    # Or execute a single signal with full intelligence
    result = executor.execute_signal_intelligently(
        signal={'symbol': 'AAPL', 'entry_price': 150, 'stop_loss': 145},
        price_data=aapl_df,
        spy_data=spy_df,
        vix_level=18.5,
        current_positions=positions
    )
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd

# Import the new OrderManager
from execution.order_manager import get_order_manager
from oms.order_state import OrderRecord, OrderStatus
from strategies.adaptive_selector import AdaptiveStrategySelector
from ml_features.confidence_integrator import get_confidence_integrator
from portfolio.risk_manager import get_risk_manager
from risk.policy_gate import PolicyGate
from risk.trailing_stops import get_trailing_stop_manager

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of intelligent execution attempt."""
    symbol: str
    signal: Dict[str, Any]
    approved: bool
    executed: bool
    shares: int
    position_size: float
    ml_confidence: float
    regime: str
    strategy_used: str
    rejection_reason: Optional[str] = None
    broker_order_id: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'approved': self.approved,
            'executed': self.executed,
            'shares': self.shares,
            'position_size': round(self.position_size, 2),
            'ml_confidence': round(self.ml_confidence, 4),
            'regime': self.regime,
            'strategy_used': self.strategy_used,
            'rejection_reason': self.rejection_reason,
            'broker_order_id': self.broker_order_id,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class PipelineResult:
    """Result of full pipeline execution."""
    signals_generated: int
    signals_approved: int
    signals_executed: int
    signals_rejected: int
    total_capital_deployed: float
    regime: str
    strategy_used: str
    execution_results: List[ExecutionResult]
    trailing_stop_updates: List[Dict]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signals_generated': self.signals_generated,
            'signals_approved': self.signals_approved,
            'signals_executed': self.signals_executed,
            'signals_rejected': self.signals_rejected,
            'total_capital_deployed': round(self.total_capital_deployed, 2),
            'regime': self.regime,
            'strategy_used': self.strategy_used,
            'execution_results': [r.to_dict() for r in self.execution_results],
            'trailing_stop_updates': self.trailing_stop_updates,
            'timestamp': self.timestamp.isoformat(),
        }


class IntelligentExecutor:
    """
    Central orchestrator for intelligent trade execution.

    Wires together:
    - AdaptiveStrategySelector (regime-based strategy selection)
    - ConfidenceIntegrator (ML confidence scoring)
    - PortfolioRiskManager (position sizing and risk checks)
    - OrderManager (intelligent order routing and execution)
    - TrailingStopManager (dynamic exit management)
    """

    def __init__(
        self,
        equity: float = 100000.0,
        paper_mode: bool = True,
        min_confidence: float = 0.5,
        max_signals_per_run: int = 5,
    ):
        self.equity = equity
        self.paper_mode = paper_mode
        self.min_confidence = min_confidence
        self.max_signals_per_run = max_signals_per_run

        # Lazy-loaded components
        self._strategy_selector = None
        self._confidence_integrator = None
        self._risk_manager = None
        self._trailing_stop_manager = None
        self._policy_gate = None
        self._order_manager = None # New lazy-loaded component

        logger.info(f"IntelligentExecutor initialized (paper_mode={paper_mode})")

    @property
    def strategy_selector(self):
        """Lazy load AdaptiveStrategySelector."""
        if self._strategy_selector is None:
            try:
                self._strategy_selector = AdaptiveStrategySelector()
            except Exception as e:
                logger.warning(f"AdaptiveStrategySelector not available: {e}")
        return self._strategy_selector

    @property
    def confidence_integrator(self):
        """Lazy load ConfidenceIntegrator."""
        if self._confidence_integrator is None:
            try:
                self._confidence_integrator = get_confidence_integrator()
            except Exception as e:
                logger.warning(f"ConfidenceIntegrator not available: {e}")
        return self._confidence_integrator

    @property
    def risk_manager(self):
        """Lazy load PortfolioRiskManager."""
        if self._risk_manager is None:
            try:
                self._risk_manager = get_risk_manager(equity=self.equity)
            except Exception as e:
                logger.warning(f"PortfolioRiskManager not available: {e}")
        return self._risk_manager

    @property
    def trailing_stop_manager(self):
        """Lazy load TrailingStopManager."""
        if self._trailing_stop_manager is None:
            try:
                self._trailing_stop_manager = get_trailing_stop_manager()
            except Exception as e:
                logger.warning(f"TrailingStopManager not available: {e}")
        return self._trailing_stop_manager

    @property
    def policy_gate(self):
        """Lazy load PolicyGate for budget enforcement."""
        if self._policy_gate is None:
            try:
                self._policy_gate = PolicyGate()
            except Exception as e:
                logger.warning(f"PolicyGate not available: {e}")
        return self._policy_gate

    @property
    def order_manager(self):
        """Lazy load OrderManager."""
        if self._order_manager is None:
            self._order_manager = get_order_manager()
        return self._order_manager

    def update_equity(self, new_equity: float):
        """Update equity across all components."""
        self.equity = new_equity
        if self._risk_manager:
            self._risk_manager.update_equity(new_equity)
        logger.info(f"Equity updated to ${new_equity:,.2f}")

    def execute_signal_intelligently(
        self,
        signal: Dict[str, Any],
        price_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        vix_level: Optional[float] = None,
        current_positions: Optional[List[Dict]] = None,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """
        Execute a single signal through the full intelligent pipeline.

        Steps:
        1. Calculate ML confidence
        2. Evaluate through PortfolioRiskManager
        3. Check PolicyGate budgets
        4. Submit order via OrderManager (if not dry_run)

        Args:
            signal: Trade signal dict
            price_data: Price history for the symbol
            spy_data: Optional SPY data for context
            vix_level: Optional VIX level
            current_positions: List of current positions
            dry_run: If True, don't actually execute

        Returns:
            ExecutionResult with full details
        """
        symbol = signal.get('symbol', 'UNKNOWN')
        strategy = signal.get('strategy', 'unknown')
        current_positions = current_positions or []

        # === Step 1: Calculate ML Confidence ===
        ml_confidence = 0.5  # Default neutral
        if self.confidence_integrator:
            try:
                ml_confidence = self.confidence_integrator.get_simple_confidence(
                    signal=signal,
                    price_data=price_data,
                    spy_data=spy_data,
                    vix_level=vix_level
                )
            except Exception as e:
                logger.warning(f"Confidence calculation failed for {symbol}: {e}")

        # === Step 2: Check Minimum Confidence ===
        if ml_confidence < self.min_confidence:
            return ExecutionResult(
                symbol=symbol,
                signal=signal,
                approved=False,
                executed=False,
                shares=0,
                position_size=0,
                ml_confidence=ml_confidence,
                regime="unknown", # This should be derived from context if available
                strategy_used=strategy,
                rejection_reason=f"ML confidence {ml_confidence:.2f} below threshold {self.min_confidence}",
            )

        # === Step 3: Evaluate through PortfolioRiskManager ===
        decision = None
        if self.risk_manager:
            try:
                decision = self.risk_manager.evaluate_trade(
                    signal=signal,
                    current_positions=current_positions,
                    price_data=price_data,
                    ml_confidence=ml_confidence
                )
            except Exception as e:
                logger.warning(f"Risk evaluation failed for {symbol}: {e}")

        if decision and not decision.approved:
            return ExecutionResult(
                symbol=symbol,
                signal=signal,
                approved=False,
                executed=False,
                shares=0,
                position_size=0,
                ml_confidence=ml_confidence,
                regime="unknown",
                strategy_used=strategy,
                rejection_reason=decision.rejection_reason,
                warnings=decision.warnings if decision.warnings else [],
            )

        # === Step 4: Check PolicyGate ===
        shares = decision.shares if decision else 0
        position_size = decision.position_size if decision else 0
        entry_price = signal.get('entry_price', 0)
        side = signal.get('side', 'long')

        if self.policy_gate and shares > 0:
            try:
                # PolicyGate.check() signature: (symbol, side, price, qty) -> (bool, str)
                allowed, reason = self.policy_gate.check(
                    symbol=symbol,
                    side=side,
                    price=entry_price,
                    qty=shares
                )
                if not allowed:
                    return ExecutionResult(
                        symbol=symbol,
                        signal=signal,
                        approved=False,
                        executed=False,
                        shares=0,
                        position_size=0,
                        ml_confidence=ml_confidence,
                        regime="unknown",
                        strategy_used=strategy,
                        rejection_reason=f"PolicyGate blocked: {reason}",
                    )
            except Exception as e:
                logger.warning(f"PolicyGate check failed for {symbol}: {e}")

        # === Step 5: Submit Order via OrderManager (if not dry_run) ===
        executed = False
        broker_order_id = None
        final_order_status = OrderStatus.PENDING # Default

        if not dry_run and shares > 0:
            try:
                # Construct OrderRecord for the OrderManager
                order_record = OrderRecord(
                    decision_id=f"DEC_{datetime.now().strftime('%Y%m%d%H%M%S')}_{symbol}_EXEC",
                    signal_id=signal.get('signal_id', 'unknown'),
                    symbol=symbol,
                    side=side.upper(),
                    qty=shares,
                    limit_price=entry_price, # Use entry price as initial limit for intelligent executor
                    tif="IOC", # IntelligentExecutor will determine actual TIF
                    order_type="LIMIT", # IntelligentExecutor will determine actual order type
                    idempotency_key=f"IDEMP_{uuid.uuid4().hex[:6].upper()}",
                    created_at=datetime.now(),
                    entry_price_decision=entry_price, # Benchmark for TCA
                    strategy_used=strategy,
                )
                # The OrderManager handles the actual submission and status updates
                execution_id = self.order_manager.submit_order(order_record)
                
                # After submit_order, the order_record object is updated by OrderManager
                final_order_status = order_record.status
                broker_order_id = order_record.broker_order_id
                executed = (order_record.status == OrderStatus.FILLED)

            except Exception as e:
                logger.error(f"OrderManager submission failed for {symbol}: {e}")
                final_order_status = OrderStatus.FAILED
                
        return ExecutionResult(
            symbol=symbol,
            signal=signal,
            approved=True,
            executed=executed,
            shares=shares,
            position_size=position_size,
            ml_confidence=ml_confidence,
            regime="unknown", # This should be derived from context if available
            strategy_used=strategy,
            broker_order_id=broker_order_id,
            warnings=decision.warnings if decision and decision.warnings else [],
            rejection_reason=None if executed else f"Order submission failed with status: {final_order_status.value}",
        )

    def execute_pipeline(
        self,
        universe_data: Dict[str, pd.DataFrame],
        spy_data: pd.DataFrame,
        vix_level: Optional[float] = None,
        current_positions: Optional[List[Dict]] = None,
        dry_run: bool = False,
    ) -> PipelineResult:
        """
        Execute the full intelligent trading pipeline.

        Steps:
        1. Detect regime and select strategy
        2. Generate signals from selected strategy
        3. Score each signal with ML confidence (via CognitiveSignalProcessor)
        4. Evaluate and size through risk manager
        5. Submit orders via OrderManager
        6. Update trailing stops on existing positions

        Args:
            universe_data: Dict of symbol -> price DataFrame
            spy_data: SPY price data for regime detection
            vix_level: Optional current VIX level
            current_positions: List of current position dicts
            dry_run: If True, don't actually execute

        Returns:
            PipelineResult with full details
        """
        current_positions = current_positions or []
        execution_results = []
        trailing_stop_updates = []

        # === Step 1: Detect Regime and Select Strategy ===
        regime = "unknown"
        strategy_name = "unknown"
        strategy = None
        config = None

        if self.strategy_selector:
            try:
                strategy, config = self.strategy_selector.get_strategy_for_regime(spy_data)
                regime = self.strategy_selector._current_regime.value
                strategy_name = config.strategy_name if config else "unknown"

                if config and config.skip_trading:
                    logger.info(f"Skipping trading in {regime} regime: {config.notes}")
                    return PipelineResult(
                        signals_generated=0,
                        signals_approved=0,
                        signals_executed=0,
                        signals_rejected=0,
                        total_capital_deployed=0,
                        regime=regime,
                        strategy_used=strategy_name,
                        execution_results=[],
                        trailing_stop_updates=[],
                    )
            except Exception as e:
                logger.warning(f"Strategy selection failed: {e}")

        # === Step 2: Generate Signals ===
        all_signals = []

        if strategy:
            for symbol, price_df in universe_data.items():
                try:
                    signals_df = strategy.generate_signals(price_df)
                    if signals_df is not None and not signals_df.empty:
                        for _, row in signals_df.iterrows():
                            signal = row.to_dict()
                            signal['symbol'] = symbol
                            signal['strategy'] = strategy_name
                            all_signals.append(signal)
                except Exception as e:
                    logger.warning(f"Signal generation failed for {symbol}: {e}")

        # === Step 3: Limit signals per run ===
        all_signals = all_signals[:self.max_signals_per_run]

        # === Step 4: Execute Each Signal ===
        signals_approved = 0
        signals_executed = 0
        signals_rejected = 0
        total_capital = 0

        for signal in all_signals:
            symbol = signal.get('symbol')
            price_data = universe_data.get(symbol)

            if price_data is None:
                continue

            result = self.execute_signal_intelligently(
                signal=signal,
                price_data=price_data,
                spy_data=spy_data,
                vix_level=vix_level,
                current_positions=current_positions,
                dry_run=dry_run,
            )

            result.regime = regime
            result.strategy_used = strategy_name
            execution_results.append(result)

            if result.approved:
                signals_approved += 1
                if result.executed:
                    signals_executed += 1
                    total_capital += result.position_size
            else:
                signals_rejected += 1

        # === Step 5: Update Trailing Stops on Existing Positions ===
        if self.trailing_stop_manager and current_positions:
            try:
                current_prices = {}
                for pos in current_positions:
                    symbol = pos.get('symbol')
                    if symbol in universe_data:
                        current_prices[symbol] = universe_data[symbol]['close'].iloc[-1]

                stop_updates = self.trailing_stop_manager.update_all_stops(
                    positions=current_positions,
                    current_prices=current_prices,
                    vix_level=vix_level
                )

                for update in stop_updates:
                    if update.should_update:
                        trailing_stop_updates.append({
                            'symbol': update.symbol,
                            'old_stop': update.old_stop,
                            'new_stop': update.new_stop,
                            'state': update.state.value,
                            'reason': update.reason,
                        })
            except Exception as e:
                logger.warning(f"Trailing stop update failed: {e}")

        return PipelineResult(
            signals_generated=len(all_signals),
            signals_approved=signals_approved,
            signals_executed=signals_executed,
            signals_rejected=signals_rejected,
            total_capital_deployed=total_capital,
            regime=regime,
            strategy_used=strategy_name,
            execution_results=execution_results,
            trailing_stop_updates=trailing_stop_updates,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            'equity': self.equity,
            'paper_mode': self.paper_mode,
            'min_confidence': self.min_confidence,
            'max_signals_per_run': self.max_signals_per_run,
            'components': {
                'strategy_selector': self._strategy_selector is not None,
                'confidence_integrator': self._confidence_integrator is not None,
                'risk_manager': self._risk_manager is not None,
                'trailing_stop_manager': self._trailing_stop_manager is not None,
                'policy_gate': self._policy_gate is not None,
                'order_manager': self._order_manager is not None,
            }
        }


# Singleton instance
_intelligent_executor: Optional[IntelligentExecutor] = None


def get_intelligent_executor(equity: float = 100000.0, paper_mode: bool = True) -> IntelligentExecutor:
    """Get or create singleton IntelligentExecutor."""
    global _intelligent_executor
    if _intelligent_executor is None:
        _intelligent_executor = IntelligentExecutor(equity=equity, paper_mode=paper_mode)
    return _intelligent_executor
