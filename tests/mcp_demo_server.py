"""A tiny MCP server for the tests, started over stdio exactly like a real one.

Not a test file itself (no test_ prefix): tests/test_mcp_client.py launches it.
"""

import os

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

server = MCPServer("demo")


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@server.tool()
def secret() -> str:
    """Return the DEMO_SECRET environment variable the server was started with."""
    return os.environ.get("DEMO_SECRET", "(not set)")


@server.tool()
def fail() -> str:
    """Always fails."""
    raise ToolError("this tool always fails")  # ToolError: the message reaches the client


if __name__ == "__main__":
    server.run()
