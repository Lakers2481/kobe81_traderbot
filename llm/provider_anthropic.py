"""
Anthropic Claude Provider
=========================

Production Claude integration with:
- API key validation
- Token usage tracking
- Tool calling support
- Cost estimation
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .provider_base import (
    ProviderBase,
    LLMMessage,
    LLMResponse,
    ToolDefinition,
    ToolCall,
    ResponseFormat,
)
from .token_budget import calculate_cost_usd

logger = logging.getLogger(__name__)


class AnthropicProvider(ProviderBase):
    """
    Claude provider via Anthropic API.

    Supports:
    - Claude 3 Haiku, Sonnet, Opus
    - Claude 3.5 Haiku, Sonnet
    - Claude Sonnet 4 (default)
    """

    # Default model for production
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    # Supported models with capabilities
    SUPPORTED_MODELS = {
        "claude-3-haiku-20240307": {"max_tokens": 4096, "tool_use": True},
        "claude-3-5-haiku-20241022": {"max_tokens": 8192, "tool_use": True},
        "claude-3-sonnet-20240229": {"max_tokens": 4096, "tool_use": True},
        "claude-3-5-sonnet-20241022": {"max_tokens": 8192, "tool_use": True},
        "claude-sonnet-4-20250514": {"max_tokens": 8192, "tool_use": True},
        "claude-3-opus-20240229": {"max_tokens": 4096, "tool_use": True},
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Anthropic provider.

        Args:
            model: Claude model to use
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum output tokens
            timeout: Request timeout in seconds
            api_key: Anthropic API key (defaults to env var)
        """
        super().__init__(model, temperature, max_tokens, timeout)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                if self._api_key:
                    self._client = anthropic.Anthropic(
                        api_key=self._api_key,
                        timeout=self.timeout,
                    )
                else:
                    logger.warning("ANTHROPIC_API_KEY not set")
            except ImportError:
                logger.warning("anthropic package not installed: pip install anthropic")
        return self._client

    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        if self._available is not None:
            return self._available

        client = self._get_client()
        if client is None:
            self._available = False
            return False

        # Try a minimal API call to verify connectivity
        try:
            # Just check if we have a valid client with API key
            self._available = self._api_key is not None and len(self._api_key) > 0
        except Exception as e:
            logger.warning(f"Anthropic availability check failed: {e}")
            self._available = False

        return self._available

    def chat(
        self,
        messages: List[LLMMessage],
        response_format: ResponseFormat = ResponseFormat.TEXT,
        tools: Optional[List[ToolDefinition]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send chat request to Claude.

        Args:
            messages: Conversation history
            response_format: Output format (TEXT, JSON, TOOL_CALL)
            tools: Available tools for function calling
            temperature: Override default temperature
            max_tokens: Override max tokens

        Returns:
            LLMResponse with content and metadata
        """
        client = self._get_client()
        if client is None:
            return LLMResponse(
                content="",
                model=self.model,
                finish_reason="error",
            )

        # Extract system message and convert format
        system_content = ""
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "tool":
                # Tool result message
                api_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Build request kwargs
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": api_messages,
        }

        if system_content:
            kwargs["system"] = system_content

        # Temperature
        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            kwargs["temperature"] = temp

        # Tools for function calling
        if tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools
            ]

        try:
            response = client.messages.create(**kwargs)

            # Extract content and tool calls
            content = ""
            tool_calls = []

            for block in response.content:
                if hasattr(block, "text"):
                    content = block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input,
                        )
                    )

            # Get usage info
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)

            # Calculate cost
            cost = calculate_cost_usd(self.model, input_tokens, output_tokens)

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                finish_reason=response.stop_reason or "stop",
                raw_response=response,
                cost_usd=cost,
            )

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                finish_reason="error",
            )

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        return calculate_cost_usd(self.model, input_tokens, output_tokens)
