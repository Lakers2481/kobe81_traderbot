"""
OpenAI-Compatible Provider
==========================

Supports OpenAI API and compatible endpoints:
- OpenAI (GPT-4, GPT-3.5)
- vLLM local servers
- LM Studio
- Any OpenAI-compatible API
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


class OpenAIProvider(ProviderBase):
    """
    OpenAI-compatible provider.

    Works with:
    - OpenAI API (gpt-4-turbo, gpt-4o, gpt-3.5-turbo)
    - vLLM served models (http://localhost:8000/v1)
    - LM Studio (http://localhost:1234/v1)
    - Any OpenAI-compatible endpoint
    """

    # Default model
    DEFAULT_MODEL = "gpt-4o"

    # Default to OpenAI, but can be overridden for local servers
    DEFAULT_BASE_URL = None  # Uses OpenAI default

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 60.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI-compatible provider.

        Args:
            model: Model name
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum output tokens
            timeout: Request timeout in seconds
            api_key: API key (defaults to env var)
            base_url: API base URL (for vLLM/local servers)
        """
        super().__init__(model, temperature, max_tokens, timeout)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                kwargs: Dict[str, Any] = {
                    "timeout": self.timeout,
                }

                if self._api_key:
                    kwargs["api_key"] = self._api_key
                elif self._base_url:
                    # Local servers often don't need API key
                    kwargs["api_key"] = "sk-no-key-required"

                if self._base_url:
                    kwargs["base_url"] = self._base_url

                self._client = OpenAI(**kwargs)

            except ImportError:
                logger.warning("openai package not installed: pip install openai")
        return self._client

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if self._available is not None:
            return self._available

        client = self._get_client()
        if client is None:
            self._available = False
            return False

        # For local servers, try a connection test
        if self._base_url:
            try:
                # Try listing models (works with most OpenAI-compatible servers)
                client.models.list()
                self._available = True
            except Exception as e:
                logger.warning(f"OpenAI-compatible server not available: {e}")
                self._available = False
        else:
            # For OpenAI, just check if we have an API key
            self._available = self._api_key is not None and len(self._api_key) > 0

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
        Send chat request to OpenAI-compatible API.

        Args:
            messages: Conversation history
            response_format: Output format
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

        # Convert messages to OpenAI format
        api_messages = []
        for msg in messages:
            if msg.role == "tool":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            else:
                api_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        # Build request kwargs
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        elif self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        # Temperature
        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            kwargs["temperature"] = temp

        # JSON mode
        if response_format == ResponseFormat.JSON:
            kwargs["response_format"] = {"type": "json_object"}

        # Tools for function calling
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        try:
            response = client.chat.completions.create(**kwargs)

            # Extract content and tool calls
            choice = response.choices[0]
            content = choice.message.content or ""
            tool_calls = []

            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=json.loads(tc.function.arguments),
                        )
                    )

            # Get usage info
            input_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0

            # Calculate cost
            cost = calculate_cost_usd(self.model, input_tokens, output_tokens)

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                finish_reason=choice.finish_reason or "stop",
                raw_response=response,
                cost_usd=cost,
            )

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                finish_reason="error",
            )

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        return calculate_cost_usd(self.model, input_tokens, output_tokens)


class VLLMProvider(OpenAIProvider):
    """
    vLLM local server provider.

    Convenience wrapper that defaults to localhost:8000.
    """

    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 60.0,
        base_url: str = DEFAULT_BASE_URL,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key="sk-no-key-required",  # vLLM doesn't need API key
            base_url=base_url,
        )

    @property
    def provider_name(self) -> str:
        return "vllm"
