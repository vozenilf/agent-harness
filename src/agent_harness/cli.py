"""Command-line chat: read a line, ask the harness, print the answer."""

import asyncio

from openai import APIError, AsyncOpenAI

from agent_harness.config import load_model_config
from agent_harness.harness import Harness


async def repl(harness: Harness) -> None:
    """Read-eval-print loop. Ends on /quit; a failed model call is reported, not fatal."""
    while True:
        line = (await asyncio.to_thread(input, "\nyou: ")).strip()
        if not line:
            continue
        if line == "/quit":
            return
        try:
            answer = await harness.ask(line)
        except APIError as error:
            print(f"\n[error] the model call failed: {error}")
            continue
        print(f"\nassistant: {answer}")


def main() -> None:
    """Entry point of the `agent-harness` command."""
    config = load_model_config()
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    print(f"agent-harness ({config.model} at {config.base_url}) - /quit or Ctrl+C to exit")
    try:
        asyncio.run(repl(Harness(client, config)))
    except (EOFError, KeyboardInterrupt):
        pass
    print("\nbye!")
