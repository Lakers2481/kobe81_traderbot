"""
LLM Provider Base - Abstract Interface
======================================

Defines the abstract base class for all LLM providers.
All providers must implement: chat(), is_available(), get_cost_estimate()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ResponseFormat(Enum):
    """Output format for LLM responses."""
    TEXT = "text"
    JSON = "json"
    TOOL_CALL = "tool_call"


@dataclass
class LLMMessage:
    """A message in a conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None  # For tool messages
    tool_call_id: Optional[str] = None  # For tool result messages


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API calls."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolCall:
    """A tool call made by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    finish_reason: str = ""
    raw_response: Optional[Any] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class ProviderBase(ABC):
    """
    Abstract base class for all LLM providers.

    Subclasses must implement:
    - provider_name (property): String identifier
    - is_available(): Check if provider can be used
    - chat(): Main chat interface

    Optional overrides:
    - chat_json(): JSON-mode chat
    - tool_call(): Tool-calling chat
    - get_cost_estimate(): Cost calculation
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 60.0,
    ):
        """
        Initialize provider.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514")
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum output tokens
            timeout: Request timeout in seconds
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._available: Optional[bool] = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier (anthropic, openai, ollama)."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider is available.

        Returns True if:
        - API key is configured (for cloud providers)
        - Server is reachable (for local providers)
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[LLMMessage],
        response_format: ResponseFormat = ResponseFormat.TEXT,
        tools: Optional[List[ToolDefinition]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send a chat request to the LLM.

        Args:
            messages: Conversation history
            response_format: Desired output format
            tools: Available tools for function calling
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            LLMResponse with content and metadata
        """
        pass

    def chat_json(
        self,
        messages: List[LLMMessage],
        schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Chat with JSON response mode.

        Args:
            messages: Conversation history
            schema: Optional JSON schema for response validation
            temperature: Override default temperature

        Returns:
            LLMResponse with JSON content
        """
        # Add JSON instruction if not in system prompt
        json_instruction = "Respond with valid JSON only. No additional text."
        enhanced_messages = list(messages)

        if enhanced_messages and enhanced_messages[0].role == "system":
            enhanced_messages[0] = LLMMessage(
                role="system",
                content=f"{enhanced_messages[0].content}\n\n{json_instruction}",
            )
        else:
            enhanced_messages.insert(0, LLMMessage(role="system", content=json_instruction))

        return self.chat(
            enhanced_messages,
            response_format=ResponseFormat.JSON,
            temperature=temperature or 0.0,  # Low temp for structured output
        )

    def tool_call(
        self,
        messages: List[LLMMessage],
        tools: List[ToolDefinition],
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Chat with tool calling.

        Args:
            messages: Conversation history
            tools: Available tools
            temperature: Override default temperature

        Returns:
            LLMResponse with tool_calls if any
        """
        return self.chat(
            messages,
            response_format=ResponseFormat.TOOL_CALL,
            tools=tools,
            temperature=temperature,
        )

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost in USD for a request.

        Override in subclasses with provider-specific pricing.
        """
        return 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r}, available={self._available})"
