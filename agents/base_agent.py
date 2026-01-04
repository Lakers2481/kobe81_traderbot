"""
Base Agent - ReAct Pattern Implementation
==========================================

Implements the ReAct (Reasoning + Acting) pattern:
1. THINK: Analyze situation, plan next step
2. ACT: Execute tool or action
3. OBSERVE: Process result
4. REPEAT until task complete or max iterations

CRITICAL SAFETY:
- PAPER_ONLY = True (hardcoded)
- Cannot execute live trades
- Cannot auto-merge code
- Human approval required for promotions
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm import get_provider, LLMMessage, LLMResponse, ToolDefinition, ProviderBase

logger = logging.getLogger(__name__)


# =============================================================================
# Safety Constants (NEVER MODIFY)
# =============================================================================

PAPER_ONLY = True  # HARDCODED - agents cannot trade live
APPROVE_LIVE_ACTION = False  # HARDCODED - requires human approval
MAX_ITERATIONS = 20  # Safety limit on agent loops


class AgentStatus(Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Waiting for human approval


@dataclass
class AgentConfig:
    """Agent configuration."""
    name: str
    description: str
    max_iterations: int = MAX_ITERATIONS
    temperature: float = 0.3  # Low for consistent reasoning
    verbose: bool = True
    provider_type: Optional[str] = None  # None = auto-select
    model: Optional[str] = None


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    data: Optional[Any] = None


@dataclass
class AgentThought:
    """Agent's reasoning about the current state."""
    observation: str  # What the agent observes
    reasoning: str    # How the agent analyzes it
    plan: str         # What the agent plans to do
    criticism: str    # Self-critique
    confidence: float = 0.5  # 0-1 confidence in plan


@dataclass
class AgentAction:
    """Action the agent wants to take."""
    tool: str              # Tool name to call
    tool_input: Dict[str, Any]  # Tool parameters
    thought: str           # Why this action


@dataclass
class AgentResult:
    """Final result from agent execution."""
    success: bool
    output: str
    thoughts: List[AgentThought] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)
    iterations: int = 0
    duration_seconds: float = 0.0
    status: AgentStatus = AgentStatus.COMPLETED
    error: Optional[str] = None
    data: Optional[Any] = None


class BaseAgent(ABC):
    """
    Abstract base class for ReAct agents.

    Subclasses must implement:
    - get_system_prompt(): Agent-specific instructions
    - get_tools(): Available tools for the agent

    Optional overrides:
    - preprocess_task(): Modify task before execution
    - postprocess_result(): Modify result before returning
    """

    # Safety enforcement
    PAPER_ONLY = PAPER_ONLY  # Class-level, cannot be overridden

    def __init__(self, config: AgentConfig):
        """
        Initialize agent.

        Args:
            config: Agent configuration
        """
        if not self.PAPER_ONLY:
            raise RuntimeError("SAFETY VIOLATION: Agents must be paper-only")

        self.config = config
        self._provider: Optional[ProviderBase] = None
        self._tools: Dict[str, Tuple[ToolDefinition, Callable]] = {}
        self._history: List[LLMMessage] = []

    @property
    def provider(self) -> ProviderBase:
        """Get or create LLM provider."""
        if self._provider is None:
            self._provider = get_provider(
                provider_type=self.config.provider_type,
                model=self.config.model,
                task_type="reasoning",
            )
        return self._provider

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    def get_tools(self) -> List[Tuple[ToolDefinition, Callable]]:
        """
        Return available tools for this agent.

        Returns:
            List of (ToolDefinition, handler_function) tuples
        """
        pass

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
    ) -> None:
        """Register a tool for the agent."""
        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        )
        self._tools[name] = (tool_def, handler)

    def _setup_tools(self) -> None:
        """Initialize tools from get_tools()."""
        for tool_def, handler in self.get_tools():
            self._tools[tool_def.name] = (tool_def, handler)

    def _get_tool_definitions(self) -> List[ToolDefinition]:
        """Get tool definitions for LLM."""
        return [tool_def for tool_def, _ in self._tools.values()]

    def _execute_tool(self, name: str, input_data: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        if name not in self._tools:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {name}",
            )

        _, handler = self._tools[name]

        try:
            result = handler(**input_data)

            # Handle different return types
            if isinstance(result, ToolResult):
                return result
            elif isinstance(result, str):
                return ToolResult(success=True, output=result)
            elif isinstance(result, dict):
                return ToolResult(
                    success=True,
                    output=json.dumps(result, indent=2),
                    data=result,
                )
            else:
                return ToolResult(success=True, output=str(result))

        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def _build_react_prompt(self, task: str, observation: str = "") -> str:
        """Build the ReAct prompt for current iteration."""
        prompt = f"""TASK: {task}

You must follow the ReAct pattern:
1. THOUGHT: Analyze the current state and plan your next action
2. ACTION: Call a tool if needed, or respond with FINAL ANSWER

Available tools:
{self._format_tools()}

{f'OBSERVATION from last action: {observation}' if observation else ''}

Respond in this exact format:

THOUGHT: [Your reasoning about what to do next]

Then either:

ACTION:
```json
{{"tool": "tool_name", "input": {{"param1": "value1"}}}}
```

OR if you have the final answer:

FINAL ANSWER: [Your complete response to the task]
"""
        return prompt

    def _format_tools(self) -> str:
        """Format tool descriptions for prompt."""
        lines = []
        for name, (tool_def, _) in self._tools.items():
            params = json.dumps(tool_def.parameters.get("properties", {}), indent=2)
            lines.append(f"- {name}: {tool_def.description}\n  Parameters: {params}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> Tuple[Optional[AgentAction], Optional[str]]:
        """
        Parse LLM response into action or final answer.

        Returns:
            (AgentAction, None) if action parsed
            (None, final_answer) if final answer found
            (None, None) if parse failed
        """
        # Check for final answer
        if "FINAL ANSWER:" in response:
            parts = response.split("FINAL ANSWER:", 1)
            final_answer = parts[1].strip() if len(parts) > 1 else ""
            thought = ""
            if "THOUGHT:" in parts[0]:
                thought = parts[0].split("THOUGHT:", 1)[1].strip()
            return None, final_answer

        # Try to parse action
        if "ACTION:" in response:
            try:
                # Extract JSON from response
                action_part = response.split("ACTION:", 1)[1]
                # Find JSON block
                start = action_part.find("{")
                end = action_part.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = action_part[start:end]
                    action_data = json.loads(json_str)

                    # Extract thought
                    thought = ""
                    if "THOUGHT:" in response:
                        thought_part = response.split("THOUGHT:", 1)[1]
                        thought = thought_part.split("ACTION:", 1)[0].strip()

                    return AgentAction(
                        tool=action_data.get("tool", ""),
                        tool_input=action_data.get("input", {}),
                        thought=thought,
                    ), None
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"Failed to parse action: {e}")

        return None, None

    def preprocess_task(self, task: str) -> str:
        """Override to modify task before execution."""
        return task

    def postprocess_result(self, result: AgentResult) -> AgentResult:
        """Override to modify result before returning."""
        return result

    def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Run the agent on a task.

        Args:
            task: The task to perform
            context: Optional context data

        Returns:
            AgentResult with output and execution details
        """
        # Safety check
        if not self.PAPER_ONLY:
            return AgentResult(
                success=False,
                output="SAFETY VIOLATION: Agent not in paper-only mode",
                status=AgentStatus.FAILED,
            )

        start_time = time.time()

        # Setup
        self._setup_tools()
        task = self.preprocess_task(task)

        # Initialize history
        self._history = [
            LLMMessage(role="system", content=self.get_system_prompt()),
        ]

        if context:
            context_str = json.dumps(context, indent=2, default=str)
            self._history.append(
                LLMMessage(role="user", content=f"CONTEXT:\n{context_str}")
            )

        thoughts: List[AgentThought] = []
        actions: List[AgentAction] = []
        observation = ""

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                logger.info(f"[{self.config.name}] Iteration {iteration + 1}")

            # Build prompt
            prompt = self._build_react_prompt(task, observation)
            self._history.append(LLMMessage(role="user", content=prompt))

            # Get LLM response
            try:
                response = self.provider.chat(
                    messages=self._history,
                    temperature=self.config.temperature,
                )
                response_text = response.content
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return AgentResult(
                    success=False,
                    output=f"LLM error: {e}",
                    thoughts=thoughts,
                    actions=actions,
                    iterations=iteration + 1,
                    duration_seconds=time.time() - start_time,
                    status=AgentStatus.FAILED,
                    error=str(e),
                )

            self._history.append(LLMMessage(role="assistant", content=response_text))

            # Parse response
            action, final_answer = self._parse_response(response_text)

            if final_answer is not None:
                # Task complete
                result = AgentResult(
                    success=True,
                    output=final_answer,
                    thoughts=thoughts,
                    actions=actions,
                    iterations=iteration + 1,
                    duration_seconds=time.time() - start_time,
                    status=AgentStatus.COMPLETED,
                )
                return self.postprocess_result(result)

            if action is not None:
                actions.append(action)

                if self.config.verbose:
                    logger.info(f"[{self.config.name}] Action: {action.tool}")

                # Execute tool
                tool_result = self._execute_tool(action.tool, action.tool_input)

                if tool_result.success:
                    observation = f"Tool '{action.tool}' succeeded:\n{tool_result.output}"
                else:
                    observation = f"Tool '{action.tool}' failed: {tool_result.error}"

            else:
                # No action or final answer - ask to continue
                observation = "Could not parse your response. Please use the exact format shown."

        # Max iterations reached
        return AgentResult(
            success=False,
            output="Max iterations reached without completing task",
            thoughts=thoughts,
            actions=actions,
            iterations=self.config.max_iterations,
            duration_seconds=time.time() - start_time,
            status=AgentStatus.FAILED,
            error="max_iterations",
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name!r}, paper_only={self.PAPER_ONLY})"
