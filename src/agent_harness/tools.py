"""Tools: what the model may ask the harness to do.

Each tool is defined once: its function and the description the model sees live in
one Tool object. The registry turns tool calls from the model into function calls and
turns every failure into an error message the model can read, never a crash.
"""

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_RESULT_CHARS = 10_000  # keeps one huge tool result from flooding a small model's context


@dataclass(frozen=True)
class Tool:
    """One tool: the function to run plus what the model needs to know to call it."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments
    function: Callable[..., Any]  # plain or async function, returns something printable
    read_only: bool = True  # False = changes something (used for approvals in phase 6)

    def schema(self) -> dict[str, Any]:
        """The tool in the chat-completions "tools" format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """All tools the model can use, looked up by name."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools()]

    async def call(self, name: str, raw_arguments: str) -> str:
        """Run one tool call from the model. Always returns text; never raises."""
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'. Available tools: {', '.join(self.names())}"

        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as error:
            return f"error: the arguments are not valid JSON ({error.msg}). Send a JSON object."
        if not isinstance(arguments, dict):
            return 'error: the arguments must be a JSON object, like {"filename": "notes.txt"}'

        try:
            result = tool.function(**arguments)
            if inspect.isawaitable(result):  # async tools (MCP, phase 4) return a coroutine
                result = await result
        except Exception as error:  # a failing tool must not crash the harness
            return f"error: {name} failed: {type(error).__name__}: {error}"

        text = str(result)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"\n[truncated: {len(text)} characters in total]"
        return text
