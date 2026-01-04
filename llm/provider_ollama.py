"""
Ollama Local Provider
=====================

Run local LLMs via Ollama:
- Llama 3.2
- Mistral
- CodeLlama
- DeepSeek
- Qwen
- Any Ollama-supported model

Zero cost, runs locally.
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

logger = logging.getLogger(__name__)


class OllamaProvider(ProviderBase):
    """
    Ollama local LLM provider.

    Runs models locally with zero cost.
    Supports most open-source models via Ollama.
    """

    # Default model
    DEFAULT_MODEL = "llama3.2"

    # Default Ollama server
    DEFAULT_HOST = "http://localhost:11434"

    # Recommended models by capability
    RECOMMENDED_MODELS = {
        "fast": "llama3.2",        # Fast, good for simple tasks
        "reasoning": "deepseek-r1",  # Good reasoning capability
        "coding": "codellama",     # Code-focused
        "balanced": "mistral",     # Good balance of speed/quality
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 120.0,  # Longer timeout for local
        host: Optional[str] = None,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum output tokens
            timeout: Request timeout in seconds (longer for local)
            host: Ollama server URL
        """
        super().__init__(model, temperature, max_tokens, timeout)
        self._host = host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)
        self._client = None

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _get_client(self):
        """Lazy-load Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self._host)
            except ImportError:
                logger.warning("ollama package not installed: pip install ollama")
        return self._client

    def is_available(self) -> bool:
        """Check if Ollama server is available and model is pulled."""
        if self._available is not None:
            return self._available

        client = self._get_client()
        if client is None:
            self._available = False
            return False

        try:
            # Check if server is running
            models = client.list()

            # Check if our model is available
            model_names = [m.get("name", "").split(":")[0] for m in models.get("models", [])]
            if self.model.split(":")[0] in model_names:
                self._available = True
            else:
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {model_names}. Run: ollama pull {self.model}"
                )
                self._available = False

        except Exception as e:
            logger.warning(f"Ollama server not available at {self._host}: {e}")
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
        Send chat request to Ollama.

        Args:
            messages: Conversation history
            response_format: Output format
            tools: Available tools (limited support)
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

        # Convert messages to Ollama format
        api_messages = []
        for msg in messages:
            api_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        # Build options
        options: Dict[str, Any] = {}
        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            options["temperature"] = temp
        if max_tokens or self.max_tokens:
            options["num_predict"] = max_tokens or self.max_tokens

        # JSON mode hint (Ollama uses format parameter)
        format_param = None
        if response_format == ResponseFormat.JSON:
            format_param = "json"

        try:
            response = client.chat(
                model=self.model,
                messages=api_messages,
                options=options if options else None,
                format=format_param,
            )

            # Extract content
            content = response.get("message", {}).get("content", "")

            # Estimate tokens (Ollama may provide these)
            prompt_eval = response.get("prompt_eval_count", 0)
            eval_count = response.get("eval_count", 0)

            # Tool calls are not well-supported in Ollama, but we handle them if present
            tool_calls = []
            if tools and response_format == ResponseFormat.TOOL_CALL:
                # Try to parse tool call from JSON response
                try:
                    parsed = json.loads(content)
                    if "tool_name" in parsed and "arguments" in parsed:
                        tool_calls.append(
                            ToolCall(
                                id=f"ollama_{hash(content) % 10000}",
                                name=parsed["tool_name"],
                                arguments=parsed["arguments"],
                            )
                        )
                except (json.JSONDecodeError, KeyError):
                    pass

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                input_tokens=prompt_eval,
                output_tokens=eval_count,
                model=self.model,
                finish_reason="stop",
                raw_response=response,
                cost_usd=0.0,  # Local = free
            )

        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return LLMResponse(
                content="",
                model=self.model,
                finish_reason="error",
            )

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Local models are free."""
        return 0.0

    def pull_model(self) -> bool:
        """Pull the model if not already available."""
        client = self._get_client()
        if client is None:
            return False

        try:
            logger.info(f"Pulling Ollama model: {self.model}")
            client.pull(self.model)
            self._available = True
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {self.model}: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        client = self._get_client()
        if client is None:
            return []

        try:
            models = client.list()
            return [m.get("name", "") for m in models.get("models", [])]
        except Exception:
            return []
