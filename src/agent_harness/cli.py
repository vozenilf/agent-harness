"""Command-line chat: read a line, ask the harness, print the answer."""

import asyncio
from pathlib import Path

from openai import APIError, AsyncOpenAI

from agent_harness.config import load_model_config
from agent_harness.harness import Harness
from agent_harness.tools import ToolRegistry
from agent_harness.workspace import Workspace

WORKSPACE_DIR = Path("workspace")  # relative to the folder you start the harness from


def print_event(kind: str, text: str) -> None:
    """Show tool activity, shortened so long file contents don't fill the screen."""
    short = text if len(text) <= 200 else text[:200] + "..."
    print(f"  [{kind}] {short}")


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

    WORKSPACE_DIR.mkdir(exist_ok=True)
    tools = ToolRegistry(Workspace(WORKSPACE_DIR).tools())
    harness = Harness(client, config, tools=tools, on_event=print_event)

    print(f"agent-harness ({config.model} at {config.base_url})")
    print(f"tools: {', '.join(tools.names())} - /quit or Ctrl+C to exit")
    try:
        asyncio.run(repl(harness))
    except (EOFError, KeyboardInterrupt):
        pass
    print("\nbye!")
