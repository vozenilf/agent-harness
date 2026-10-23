"""The harness: keeps the conversation and talks to the model.

Phase 1: one model call per question. The agent loop with tools arrives in phase 2.
"""

from typing import Any

from openai import AsyncOpenAI

from agent_harness.config import ModelConfig

SYSTEM_PROMPT = "You are a helpful personal assistant. Be concise."


class Harness:
    """Holds the conversation and sends it to the model.

    The client is passed in rather than created here, so tests can hand in a client
    that talks to a fake server instead of Ollama.
    """

    def __init__(
        self, client: AsyncOpenAI, config: ModelConfig, system_prompt: str = SYSTEM_PROMPT
    ) -> None:
        self.client = client
        self.config = config
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    async def ask(self, user_message: str) -> str:
        """Send the user's message (with the whole conversation so far) and return the answer."""
        self.messages.append({"role": "user", "content": user_message})

        # Only send reasoning_effort when it is set: not every server accepts it.
        extra = {}
        if self.config.reasoning_effort is not None:
            extra["reasoning_effort"] = self.config.reasoning_effort

        response = await self.client.chat.completions.create(
            model=self.config.model, messages=self.messages, **extra
        )
        answer = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": answer})
        return answer
