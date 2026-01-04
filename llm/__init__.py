"""
LLM Provider Layer - Multi-Provider Abstraction
===============================================

Provides a unified interface for multiple LLM providers:
- Anthropic (Claude)
- OpenAI-compatible (vLLM, OpenAI, local servers)
- Ollama (local models)

Usage:
    from llm import get_provider, ProviderRouter, LLMMessage

    # Get default provider
    provider = get_provider()
    response = provider.chat([LLMMessage(role="user", content="Hello")])

    # Get specific provider
    provider = get_provider(provider_type="ollama", model="llama3.2")

    # Use router for cost-aware selection
    router = ProviderRouter()
    provider = router.get_provider(task_type="reasoning")

    # Quick one-shot chat
    from llm import quick_chat
    answer = quick_chat("What is 2+2?", task_type="simple")

CRITICAL: This layer is for research/reasoning ONLY.
- NO live trading decisions
- NO price predictions
- Token budget enforced ($50/day max)
"""

from .provider_base import (
    ProviderBase,
    LLMMessage,
    LLMResponse,
    ToolDefinition,
    ToolCall,
    ResponseFormat,
)
from .token_budget import TokenBudget, calculate_cost_usd, LLM_PRICING, get_token_budget
from .router import ProviderRouter, get_provider, get_router, quick_chat, TaskType
from .provider_anthropic import AnthropicProvider
from .provider_openai import OpenAIProvider, VLLMProvider
from .provider_ollama import OllamaProvider

__all__ = [
    # Base classes
    "ProviderBase",
    "LLMMessage",
    "LLMResponse",
    "ToolDefinition",
    "ToolCall",
    "ResponseFormat",
    # Providers
    "AnthropicProvider",
    "OpenAIProvider",
    "VLLMProvider",
    "OllamaProvider",
    # Token budget
    "TokenBudget",
    "calculate_cost_usd",
    "get_token_budget",
    "LLM_PRICING",
    # Router
    "ProviderRouter",
    "get_provider",
    "get_router",
    "quick_chat",
    "TaskType",
]
