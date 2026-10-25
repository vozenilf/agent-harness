"""MCP client: start the servers listed in mcp_servers.json and offer their tools.

Each server is a separate program we start and talk to over stdin/stdout. Its tools
become ordinary Tool objects, so the registry and the agent loop treat them exactly
like our own file tools.
"""

import json
import os
from collections.abc import Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool
from pydantic import BaseModel

from agent_harness.tools import Tool


class ServerConfig(BaseModel):
    """How to start one MCP server: the same format Claude Desktop and Claude Code use."""

    command: str
    args: list[str] = []
    env: dict[str, str] = {}  # values may use ${VAR}, filled in from our environment


def load_server_configs(path: Path) -> dict[str, ServerConfig]:
    """Read {"mcpServers": {name: {command, args, env}}}. No file means no servers."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: ServerConfig.model_validate(spec) for name, spec in data.get("mcpServers", {}).items()
    }


def expand_env(env: dict[str, str]) -> dict[str, str]:
    """Replace ${VAR} with the value from our environment, so secrets stay out of the file."""
    expanded = {key: os.path.expandvars(value) for key, value in env.items()}
    for key, value in expanded.items():
        if "$" in value:  # expandvars leaves unknown variables untouched
            raise ValueError(f"environment variable for {key} is not set ({env[key]})")
    return expanded


def result_to_text(result: Any) -> str:
    """Turn an MCP tool result into the plain text the model gets."""
    if not isinstance(result, CallToolResult):
        return "error: the server returned an unsupported kind of result"
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content)
    text = text or "(no text output)"
    return f"error: {text}" if result.is_error else text


def make_tool(server: str, session: ClientSession, tool: MCPTool) -> Tool:
    """Wrap one MCP tool as our Tool. Its function forwards the call to the server."""

    async def call(**arguments: Any) -> str:
        result = await session.call_tool(tool.name, arguments)
        return result_to_text(result)

    annotations = tool.annotations
    return Tool(
        name=f"{server}__{tool.name}",  # prefix: two servers may both have a "search" tool
        description=f"[{server}] {tool.description or ''}".strip(),
        parameters=tool.input_schema,
        function=call,
        read_only=bool(annotations and annotations.read_only_hint),  # unknown = may change things
    )


async def connect_servers(
    configs: dict[str, ServerConfig],
    stack: AsyncExitStack,
    log: Callable[[str], None] = print,
) -> list[Tool]:
    """Start every server and collect its tools. A server that fails is skipped, not fatal.

    The connections stay open until `stack` is closed (when the harness exits).
    """
    tools: list[Tool] = []
    for name, config in configs.items():
        server_stack = AsyncExitStack()  # this server's own cleanup, so a failure can't leak
        try:
            params = StdioServerParameters(
                command=config.command, args=config.args, env=expand_env(config.env)
            )
            read, write = await server_stack.enter_async_context(stdio_client(params))
            session = await server_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            server_tools = (await session.list_tools()).tools
        except Exception as error:
            await server_stack.aclose()
            log(f"  [mcp] '{name}' failed to start: {type(error).__name__}: {error}")
            continue

        stack.push_async_callback(server_stack.aclose)  # close it when the harness exits
        tools += [make_tool(name, session, tool) for tool in server_tools]
        log(f"  [mcp] connected '{name}': {', '.join(t.name for t in server_tools)}")
    return tools
