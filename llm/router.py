"""
LLM Provider Router
===================

Cost-aware routing across multiple providers:
1. Try local (Ollama) first - free
2. Fall back to vLLM if available - free
3. Fall back to Claude - paid with budget cap
4. Fall back to OpenAI - paid with budget cap

Enforces $50/day budget across all paid providers.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional, Type

from .provider_base import ProviderBase, LLMMessage, LLMResponse
from .provider_anthropic import AnthropicProvider
from .provider_openai import OpenAIProvider, VLLMProvider
from .provider_ollama import OllamaProvider
from .token_budget import TokenBudget, get_token_budget

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task types for provider selection."""
    SIMPLE = "simple"         # Fast, simple tasks (use local)
    REASONING = "reasoning"   # Complex reasoning (prefer Claude)
    CODING = "coding"         # Code generation (prefer local code models)
    ANALYSIS = "analysis"     # Data analysis (prefer balanced)
    TOOL_USE = "tool_use"     # Tool calling (require Claude/OpenAI)


# Provider priority by task type (lower = higher priority)
TASK_PROVIDER_PRIORITY: Dict[TaskType, List[str]] = {
    TaskType.SIMPLE: ["ollama", "vllm", "anthropic", "openai"],
    TaskType.REASONING: ["anthropic", "openai", "ollama", "vllm"],
    TaskType.CODING: ["ollama", "vllm", "anthropic", "openai"],
    TaskType.ANALYSIS: ["anthropic", "ollama", "openai", "vllm"],
    TaskType.TOOL_USE: ["anthropic", "openai"],  # Local models have limited tool support
}


class ProviderRouter:
    """
    Routes LLM requests to the most appropriate provider.

    Selection criteria:
    1. Availability (is provider up?)
    2. Cost (prefer free local providers)
    3. Task type (reasoning vs simple)
    4. Budget (enforce $50/day limit)
    """

    # Provider classes by name
    PROVIDER_CLASSES: Dict[str, Type[ProviderBase]] = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "vllm": VLLMProvider,
        "ollama": OllamaProvider,
    }

    def __init__(
        self,
        budget: Optional[TokenBudget] = None,
        prefer_local: bool = True,
    ):
        """
        Initialize router.

        Args:
            budget: Token budget for cost control
            prefer_local: Prefer free local providers when possible
        """
        self._budget = budget or get_token_budget()
        self._prefer_local = prefer_local
        self._providers: Dict[str, ProviderBase] = {}
        self._availability_cache: Dict[str, bool] = {}

    def _get_provider(self, name: str, **kwargs) -> Optional[ProviderBase]:
        """Get or create a provider instance."""
        cache_key = f"{name}:{kwargs.get('model', 'default')}"

        if cache_key not in self._providers:
            provider_class = self.PROVIDER_CLASSES.get(name)
            if provider_class is None:
                logger.warning(f"Unknown provider: {name}")
                return None

            try:
                self._providers[cache_key] = provider_class(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to create {name} provider: {e}")
                return None

        return self._providers.get(cache_key)

    def _check_availability(self, provider: ProviderBase) -> bool:
        """Check provider availability with caching."""
        cache_key = f"{provider.provider_name}:{provider.model}"

        if cache_key not in self._availability_cache:
            self._availability_cache[cache_key] = provider.is_available()

        return self._availability_cache[cache_key]

    def get_provider(
        self,
        task_type: TaskType = TaskType.SIMPLE,
        provider_type: Optional[str] = None,
        model: Optional[str] = None,
        require_tools: bool = False,
        **kwargs,
    ) -> Optional[ProviderBase]:
        """
        Get the best available provider for a task.

        Args:
            task_type: Type of task (affects provider selection)
            provider_type: Force specific provider (anthropic, openai, ollama, vllm)
            model: Specific model to use
            require_tools: Require tool calling support
            **kwargs: Additional provider kwargs

        Returns:
            Best available provider, or None if none available
        """
        # If specific provider requested, use it
        if provider_type:
            provider_kwargs = kwargs.copy()
            if model:
                provider_kwargs["model"] = model
            return self._get_provider(provider_type, **provider_kwargs)

        # If tools required, restrict to providers that support them well
        if require_tools:
            task_type = TaskType.TOOL_USE

        # Get priority list for task type
        priority = TASK_PROVIDER_PRIORITY.get(task_type, ["anthropic", "ollama"])

        # Check budget for paid providers
        budget_ok = self._budget.can_use(2000)  # Estimate 2K tokens

        # Try providers in priority order
        for provider_name in priority:
            # Skip paid providers if budget exceeded
            if provider_name in ["anthropic", "openai"] and not budget_ok:
                logger.warning(f"Skipping {provider_name} due to budget limit")
                continue

            provider_kwargs = kwargs.copy()
            if model and provider_name == priority[0]:
                # Only use custom model for first-priority provider
                provider_kwargs["model"] = model

            provider = self._get_provider(provider_name, **provider_kwargs)
            if provider and self._check_availability(provider):
                logger.debug(f"Selected provider: {provider_name} for {task_type.value}")
                return provider

        logger.warning(f"No provider available for task type: {task_type.value}")
        return None

    def chat(
        self,
        messages: List[LLMMessage],
        task_type: TaskType = TaskType.SIMPLE,
        **kwargs,
    ) -> LLMResponse:
        """
        Send chat request to best available provider.

        Args:
            messages: Conversation history
            task_type: Type of task
            **kwargs: Additional kwargs passed to provider.chat()

        Returns:
            LLMResponse from the selected provider
        """
        provider = self.get_provider(task_type=task_type)
        if provider is None:
            return LLMResponse(
                content="No LLM provider available",
                finish_reason="error",
            )

        response = provider.chat(messages, **kwargs)

        # Record usage for paid providers
        if provider.provider_name in ["anthropic", "openai"]:
            if response.total_tokens > 0:
                self._budget.record_usage(
                    tokens=response.total_tokens,
                    model=provider.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )

        return response

    def get_budget_status(self) -> Dict[str, any]:
        """Get current budget status."""
        return self._budget.get_status()

    def clear_availability_cache(self) -> None:
        """Clear availability cache to force re-checking."""
        self._availability_cache.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

_default_router: Optional[ProviderRouter] = None


def get_router() -> ProviderRouter:
    """Get or create default router."""
    global _default_router
    if _default_router is None:
        _default_router = ProviderRouter()
    return _default_router


def get_provider(
    provider_type: Optional[str] = None,
    model: Optional[str] = None,
    task_type: str = "simple",
    **kwargs,
) -> Optional[ProviderBase]:
    """
    Get a provider instance.

    Args:
        provider_type: Specific provider (anthropic, openai, ollama, vllm)
        model: Specific model
        task_type: Task type string (simple, reasoning, coding, analysis, tool_use)
        **kwargs: Additional provider kwargs

    Returns:
        Provider instance or None

    Example:
        # Get default provider
        provider = get_provider()

        # Get specific provider
        provider = get_provider(provider_type="ollama", model="llama3.2")

        # Get provider for reasoning
        provider = get_provider(task_type="reasoning")
    """
    router = get_router()

    # Convert string task type to enum
    try:
        task_enum = TaskType(task_type)
    except ValueError:
        task_enum = TaskType.SIMPLE

    return router.get_provider(
        task_type=task_enum,
        provider_type=provider_type,
        model=model,
        **kwargs,
    )


def quick_chat(
    prompt: str,
    system: Optional[str] = None,
    task_type: str = "simple",
) -> str:
    """
    Quick one-shot chat.

    Args:
        prompt: User prompt
        system: Optional system prompt
        task_type: Task type (simple, reasoning, coding, analysis)

    Returns:
        Response text or empty string on error
    """
    messages = []
    if system:
        messages.append(LLMMessage(role="system", content=system))
    messages.append(LLMMessage(role="user", content=prompt))

    try:
        task_enum = TaskType(task_type)
    except ValueError:
        task_enum = TaskType.SIMPLE

    router = get_router()
    response = router.chat(messages, task_type=task_enum)

    return response.content
