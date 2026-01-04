"""
Langfuse LLM Observability Tracer

Full LLM call observability:
- Trace all LLM calls with prompts/responses
- Log latencies and token usage
- Track costs per model
- Evaluate response quality
- Debug reasoning chains
- Agent action tracking

Author: Kobe Trading System
Created: 2026-01-04
"""

import os
import json
import time
import hashlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass, field
from functools import wraps

from core.structured_log import get_logger

logger = get_logger(__name__)

# Lazy import Langfuse
_langfuse_client = None


def _get_langfuse():
    """Lazy import and initialize Langfuse client."""
    global _langfuse_client
    if _langfuse_client is None:
        try:
            from langfuse import Langfuse

            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

            if public_key and secret_key:
                _langfuse_client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                logger.info("Langfuse client initialized")
            else:
                logger.warning("Langfuse keys not found, using local-only tracing")
                _langfuse_client = None
        except ImportError:
            logger.warning("Langfuse not installed. Install with: pip install langfuse")
            _langfuse_client = None
    return _langfuse_client


@dataclass
class TraceSpan:
    """Single span in a trace."""
    id: str
    name: str
    type: str  # 'llm', 'agent', 'tool', 'custom'
    start_time: float
    end_time: Optional[float] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    children: List['TraceSpan'] = field(default_factory=list)


@dataclass
class Trace:
    """Complete trace of an operation."""
    id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    spans: List[TraceSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def total_duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    def total_tokens(self) -> Dict[str, int]:
        total_in = sum(s.tokens_in for s in self.spans)
        total_out = sum(s.tokens_out for s in self.spans)
        return {'input': total_in, 'output': total_out, 'total': total_in + total_out}

    def total_cost(self) -> float:
        return sum(s.cost for s in self.spans)


class LLMTracer:
    """
    LLM observability tracer with Langfuse integration.

    Features:
    - Cloud sync with Langfuse (if configured)
    - Local trace storage for offline analysis
    - Token and cost tracking
    - Reasoning chain visualization
    """

    STATE_DIR = Path("state/observability")
    MAX_LOCAL_TRACES = 1000

    # Estimated costs per 1K tokens (USD)
    MODEL_COSTS = {
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        'claude-3-opus': {'input': 0.015, 'output': 0.075},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
        'local': {'input': 0.0, 'output': 0.0},
    }

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize tracer.

        Args:
            session_id: Optional session ID for grouping traces
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.STATE_DIR.mkdir(parents=True, exist_ok=True)

        self._current_trace: Optional[Trace] = None
        self._trace_stack: List[TraceSpan] = []
        self._traces: List[Trace] = []

        # Try to initialize Langfuse
        self._langfuse = _get_langfuse()

    def _generate_id(self) -> str:
        """Generate unique ID."""
        return hashlib.sha256(
            f"{datetime.now().isoformat()}_{id(self)}".encode()
        ).hexdigest()[:12]

    def _estimate_cost(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int
    ) -> float:
        """Estimate cost for token usage."""
        # Normalize model name
        model_lower = model.lower()
        costs = None

        for model_key, model_costs in self.MODEL_COSTS.items():
            if model_key in model_lower:
                costs = model_costs
                break

        if costs is None:
            costs = self.MODEL_COSTS.get('local', {'input': 0, 'output': 0})

        cost = (tokens_in / 1000 * costs['input']) + (tokens_out / 1000 * costs['output'])
        return round(cost, 6)

    @contextmanager
    def trace(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[Trace, None, None]:
        """
        Context manager for creating a trace.

        Usage:
            with tracer.trace("decision_pipeline") as t:
                # ... do work ...
        """
        trace = Trace(
            id=self._generate_id(),
            name=name,
            start_time=time.time(),
            metadata=metadata or {},
            session_id=self.session_id
        )
        self._current_trace = trace

        try:
            yield trace
        finally:
            trace.end_time = time.time()
            self._traces.append(trace)
            self._save_trace(trace)
            self._sync_to_langfuse(trace)
            self._current_trace = None

            # Cleanup old traces
            if len(self._traces) > self.MAX_LOCAL_TRACES:
                self._traces = self._traces[-self.MAX_LOCAL_TRACES:]

    @contextmanager
    def span(
        self,
        name: str,
        span_type: str = 'custom',
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Generator[TraceSpan, None, None]:
        """
        Context manager for creating a span within current trace.

        Usage:
            with tracer.span("llm_call", span_type="llm") as s:
                # ... make LLM call ...
                s.output_data = response
        """
        span = TraceSpan(
            id=self._generate_id(),
            name=name,
            type=span_type,
            start_time=time.time(),
            input_data=input_data,
            metadata=metadata or {}
        )

        # Add to current trace
        if self._current_trace:
            self._current_trace.spans.append(span)

        self._trace_stack.append(span)

        try:
            yield span
        except Exception as e:
            span.error = str(e)
            raise
        finally:
            span.end_time = time.time()
            self._trace_stack.pop()

    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """
        Log an LLM call.

        Args:
            model: Model name
            prompt: Input prompt
            response: Model response
            tokens_in: Input tokens
            tokens_out: Output tokens
            latency_ms: Latency in milliseconds
            metadata: Additional metadata
        """
        cost = self._estimate_cost(model, tokens_in, tokens_out)

        span = TraceSpan(
            id=self._generate_id(),
            name=f"llm:{model}",
            type='llm',
            start_time=time.time() - (latency_ms / 1000),
            end_time=time.time(),
            input_data={'prompt': prompt[:1000]},  # Truncate for storage
            output_data={'response': response[:2000]},
            metadata={
                'model': model,
                'latency_ms': latency_ms,
                **(metadata or {})
            },
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost
        )

        if self._current_trace:
            self._current_trace.spans.append(span)

        logger.debug(f"LLM call logged: {model}, {tokens_in}+{tokens_out} tokens, ${cost:.4f}")
        return span

    def log_agent_action(
        self,
        agent_name: str,
        action: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        reasoning: Optional[str] = None,
        success: bool = True
    ) -> TraceSpan:
        """
        Log an agent action.

        Args:
            agent_name: Name of agent
            action: Action taken
            input_data: Action input
            output_data: Action output
            reasoning: Agent's reasoning
            success: Whether action succeeded
        """
        span = TraceSpan(
            id=self._generate_id(),
            name=f"agent:{agent_name}:{action}",
            type='agent',
            start_time=time.time(),
            end_time=time.time(),
            input_data=input_data,
            output_data=output_data,
            metadata={
                'agent': agent_name,
                'action': action,
                'reasoning': reasoning,
                'success': success
            }
        )

        if self._current_trace:
            self._current_trace.spans.append(span)

        return span

    def log_tool_call(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        output_data: Any,
        latency_ms: float = 0,
        success: bool = True
    ) -> TraceSpan:
        """Log a tool call."""
        span = TraceSpan(
            id=self._generate_id(),
            name=f"tool:{tool_name}",
            type='tool',
            start_time=time.time() - (latency_ms / 1000),
            end_time=time.time(),
            input_data=input_data,
            output_data={'result': str(output_data)[:1000]},
            metadata={
                'tool': tool_name,
                'latency_ms': latency_ms,
                'success': success
            }
        )

        if self._current_trace:
            self._current_trace.spans.append(span)

        return span

    def _save_trace(self, trace: Trace) -> None:
        """Save trace to local storage."""
        try:
            trace_file = self.STATE_DIR / f"trace_{trace.id}.json"

            trace_data = {
                'id': trace.id,
                'name': trace.name,
                'session_id': trace.session_id,
                'start_time': trace.start_time,
                'end_time': trace.end_time,
                'duration_ms': trace.total_duration_ms(),
                'total_tokens': trace.total_tokens(),
                'total_cost': trace.total_cost(),
                'metadata': trace.metadata,
                'spans': [
                    {
                        'id': s.id,
                        'name': s.name,
                        'type': s.type,
                        'start_time': s.start_time,
                        'end_time': s.end_time,
                        'duration_ms': (s.end_time - s.start_time) * 1000 if s.end_time else 0,
                        'tokens_in': s.tokens_in,
                        'tokens_out': s.tokens_out,
                        'cost': s.cost,
                        'metadata': s.metadata,
                        'error': s.error,
                    }
                    for s in trace.spans
                ]
            }

            with open(trace_file, 'w') as f:
                json.dump(trace_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save trace: {e}")

    def _sync_to_langfuse(self, trace: Trace) -> None:
        """Sync trace to Langfuse cloud."""
        if self._langfuse is None:
            return

        try:
            # Create Langfuse trace
            lf_trace = self._langfuse.trace(
                id=trace.id,
                name=trace.name,
                session_id=trace.session_id,
                metadata=trace.metadata
            )

            # Add spans
            for span in trace.spans:
                if span.type == 'llm':
                    lf_trace.generation(
                        name=span.name,
                        model=span.metadata.get('model', 'unknown'),
                        input=span.input_data,
                        output=span.output_data,
                        usage={
                            'input': span.tokens_in,
                            'output': span.tokens_out,
                            'total': span.tokens_in + span.tokens_out
                        },
                        metadata=span.metadata
                    )
                else:
                    lf_trace.span(
                        name=span.name,
                        input=span.input_data,
                        output=span.output_data,
                        metadata=span.metadata
                    )

            self._langfuse.flush()
            logger.debug(f"Trace {trace.id} synced to Langfuse")

        except Exception as e:
            logger.warning(f"Failed to sync to Langfuse: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        total_tokens = {'input': 0, 'output': 0}
        total_cost = 0.0
        llm_calls = 0
        agent_actions = 0

        for trace in self._traces:
            for span in trace.spans:
                total_tokens['input'] += span.tokens_in
                total_tokens['output'] += span.tokens_out
                total_cost += span.cost

                if span.type == 'llm':
                    llm_calls += 1
                elif span.type == 'agent':
                    agent_actions += 1

        return {
            'session_id': self.session_id,
            'total_traces': len(self._traces),
            'total_tokens': total_tokens,
            'total_cost': round(total_cost, 4),
            'llm_calls': llm_calls,
            'agent_actions': agent_actions,
            'langfuse_connected': self._langfuse is not None
        }

    def get_recent_traces(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent traces summary."""
        return [
            {
                'id': t.id,
                'name': t.name,
                'duration_ms': t.total_duration_ms(),
                'spans': len(t.spans),
                'tokens': t.total_tokens()['total'],
                'cost': t.total_cost()
            }
            for t in self._traces[-n:]
        ]


# Singleton instance
_tracer: Optional[LLMTracer] = None


def get_llm_tracer() -> LLMTracer:
    """Get or create singleton tracer."""
    global _tracer
    if _tracer is None:
        _tracer = LLMTracer()
    return _tracer


# Convenience functions
def trace_llm_call(
    model: str,
    prompt: str,
    response: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: float = 0
) -> TraceSpan:
    """Log an LLM call (convenience function)."""
    return get_llm_tracer().log_llm_call(
        model, prompt, response, tokens_in, tokens_out, latency_ms
    )


def trace_agent_action(
    agent_name: str,
    action: str,
    reasoning: Optional[str] = None,
    success: bool = True
) -> TraceSpan:
    """Log an agent action (convenience function)."""
    return get_llm_tracer().log_agent_action(
        agent_name, action, reasoning=reasoning, success=success
    )


@contextmanager
def start_trace(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Start a trace context (convenience function)."""
    with get_llm_tracer().trace(name, metadata) as t:
        yield t


def end_trace():
    """End current trace (convenience function)."""
    # Traces end automatically via context manager
    pass


# Decorator for tracing functions
def traced(name: Optional[str] = None, span_type: str = 'custom'):
    """Decorator to trace a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_name = name or func.__name__
            tracer = get_llm_tracer()

            with tracer.span(trace_name, span_type=span_type) as span:
                span.input_data = {'args': str(args)[:200], 'kwargs': str(kwargs)[:200]}
                result = func(*args, **kwargs)
                span.output_data = {'result': str(result)[:500]}
                return result

        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo usage
    print("=== Langfuse LLM Tracer Demo ===\n")

    tracer = LLMTracer()

    with tracer.trace("trading_decision") as trace:
        # Simulate LLM call
        tracer.log_llm_call(
            model="gpt-4",
            prompt="Should I buy AAPL given current market conditions?",
            response="Based on the technical analysis, AAPL shows...",
            tokens_in=150,
            tokens_out=200,
            latency_ms=1500
        )

        # Simulate agent action
        tracer.log_agent_action(
            agent_name="TechnicalAnalyst",
            action="analyze_chart",
            reasoning="Checking RSI and MACD indicators",
            success=True
        )

        # Simulate tool call
        tracer.log_tool_call(
            tool_name="fetch_price_data",
            input_data={"symbol": "AAPL", "period": "1d"},
            output_data={"price": 178.50, "change": 0.5},
            latency_ms=200
        )

    print("Stats:")
    stats = tracer.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nRecent Traces:")
    for t in tracer.get_recent_traces():
        print(f"  {t['name']}: {t['duration_ms']:.0f}ms, {t['tokens']} tokens, ${t['cost']:.4f}")
