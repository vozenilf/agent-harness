"""The harness: keeps the conversation, talks to the model and runs the agent loop."""

from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI

from agent_harness.config import ModelConfig
from agent_harness.tools import ToolRegistry

SYSTEM_PROMPT = """You are a helpful personal assistant. Be concise.

The user's workspace folder holds their personal files: notes, todo lists, ideas.
Use your tools to list, read and write them - never claim you cannot see the user's files.
If a tool returns an error, read it and try to fix the problem."""

MAX_STEPS = 10  # model calls per question; stops a model that keeps calling tools forever

# Called for each tool call ("tool_call", text) and its result ("tool_result", text),
# so a front end (the CLI now, a web page on Sunday) can show what the agent is doing.
EventHandler = Callable[[str, str], None]


class Harness:
    """Holds the conversation and runs the agent loop: model -> tools -> model -> ..."""

    def __init__(
        self,
        client: AsyncOpenAI,
        config: ModelConfig,
        tools: ToolRegistry | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = MAX_STEPS,
        on_event: EventHandler | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps
        self.on_event = on_event or (lambda kind, text: None)  # default: do nothing
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    async def ask(self, user_message: str) -> str:
        """Answer one user message, running as many tool rounds as the model needs."""
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(self.max_steps):
            message = await self._call_model()

            if not message.tool_calls:  # no tool requested: this is the final answer
                answer = message.content or ""
                self.messages.append({"role": "assistant", "content": answer})
                return answer

            # Record the model's request, then run each tool and record its result.
            self.messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [call.model_dump() for call in message.tool_calls],
                }
            )
            for call in message.tool_calls:
                result = await self._run_tool_call(call)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        stopped = f"[stopped: no final answer after {self.max_steps} steps]"
        self.messages.append({"role": "assistant", "content": stopped})
        return stopped

    async def _call_model(self) -> Any:
        """One request to the model with the whole conversation and the tool list."""
        extra: dict[str, Any] = {}
        if self.config.reasoning_effort is not None:
            extra["reasoning_effort"] = self.config.reasoning_effort
        if self.tools.names():  # some servers reject an empty tool list
            extra["tools"] = self.tools.schemas()

        response = await self.client.chat.completions.create(
            model=self.config.model, messages=self.messages, **extra
        )
        return response.choices[0].message

    async def _run_tool_call(self, call: Any) -> str:
        """Run one tool call and report it through on_event."""
        if call.type != "function":  # we only ever offer function tools
            return f"error: unsupported tool call type '{call.type}'"
        name, arguments = call.function.name, call.function.arguments
        self.on_event("tool_call", f"{name}({arguments})")
        result = await self.tools.call(name, arguments)
        self.on_event("tool_result", result)
        return result
