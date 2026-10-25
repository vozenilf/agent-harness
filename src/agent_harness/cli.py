"""Command-line chat: read a line, ask the harness, print the answer."""

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

from openai import APIError, AsyncOpenAI

from agent_harness.config import load_model_config
from agent_harness.harness import SYSTEM_PROMPT, Harness
from agent_harness.mcp_client import connect_servers, load_server_configs
from agent_harness.memory import Memory, system_prompt_with_memory
from agent_harness.tools import ToolRegistry
from agent_harness.workspace import Workspace

# All relative to the folder you start the harness from.
WORKSPACE_DIR = Path("workspace")  # gitignored
MEMORY_FILE = Path("memory.md")  # gitignored
MCP_CONFIG = Path("mcp_servers.json")  # committed; secrets come from ${VARS}


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


async def run() -> None:
    """Set everything up, chat, and close the MCP servers at the end."""
    config = load_model_config()
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    WORKSPACE_DIR.mkdir(exist_ok=True)
    memory = Memory(MEMORY_FILE)

    async with AsyncExitStack() as stack:  # MCP servers live exactly as long as this block
        print("starting MCP servers...")
        mcp_tools = await connect_servers(load_server_configs(MCP_CONFIG), stack)
        tools = ToolRegistry(Workspace(WORKSPACE_DIR).tools() + memory.tools() + mcp_tools)
        harness = Harness(
            client,
            config,
            tools=tools,
            system_prompt=system_prompt_with_memory(SYSTEM_PROMPT, memory),
            on_event=print_event,
        )
        print(f"\nagent-harness ({config.model} at {config.base_url})")
        print(f"tools: {', '.join(tools.names())}")
        print(f"memory: {len(memory.facts())} saved fact(s) in {MEMORY_FILE}")
        print("/quit or Ctrl+C to exit")
        await repl(harness)


def main() -> None:
    """Entry point of the `agent-harness` command."""
    try:
        asyncio.run(run())
    except (EOFError, KeyboardInterrupt):
        pass
    print("\nbye!")
