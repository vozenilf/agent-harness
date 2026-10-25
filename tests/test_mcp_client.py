"""MCP client: config loading, and real connections to a tiny demo server over stdio."""

import json
import sys
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import pytest

from agent_harness.mcp_client import (
    ServerConfig,
    connect_servers,
    expand_env,
    load_server_configs,
)
from agent_harness.tools import ToolRegistry

DEMO_SERVER = Path(__file__).parent / "mcp_demo_server.py"


def demo_config(**env: str) -> ServerConfig:
    """Start the demo server with the same Python that runs the tests."""
    return ServerConfig(command=sys.executable, args=[str(DEMO_SERVER)], env=env)


# --- config: no servers started --------------------------------------------


def test_load_server_configs(tmp_path):
    path = tmp_path / "mcp_servers.json"
    path.write_text(
        json.dumps({"mcpServers": {"time": {"command": "uvx", "args": ["mcp-server-time"]}}}),
        encoding="utf-8",
    )

    configs = load_server_configs(path)

    assert configs == {"time": ServerConfig(command="uvx", args=["mcp-server-time"])}


def test_missing_config_file_means_no_servers(tmp_path):
    assert load_server_configs(tmp_path / "nope.json") == {}


def test_expand_env_fills_in_variables(monkeypatch):
    monkeypatch.setenv("NOTION_TOKEN", "ntn_123")

    assert expand_env({"NOTION_TOKEN": "${NOTION_TOKEN}"}) == {"NOTION_TOKEN": "ntn_123"}


def test_expand_env_refuses_unset_variables(monkeypatch):
    monkeypatch.delenv("NOTION_TOKEN", raising=False)

    with pytest.raises(ValueError, match="NOTION_TOKEN"):
        expand_env({"NOTION_TOKEN": "${NOTION_TOKEN}"})


# --- real connections: the demo server runs as a separate process ----------


@asynccontextmanager
async def connected_demo() -> AsyncGenerator[ToolRegistry]:
    """Start the demo server, hand its tools over, stop it at the end of the block.

    A context manager rather than a fixture: the MCP connection must be opened and
    closed in the same task, and pytest runs fixture setup and teardown separately.
    """
    async with AsyncExitStack() as stack:
        tools = await connect_servers({"demo": demo_config(DEMO_SECRET="s3cret")}, stack)
        yield ToolRegistry(tools)


async def test_tools_get_the_server_prefix_and_annotations():
    async with connected_demo() as registry:
        tools = {tool.name: tool for tool in registry.tools()}

    assert sorted(tools) == ["demo__add", "demo__fail", "demo__secret"]
    assert tools["demo__add"].description == "[demo] Add two numbers."
    assert tools["demo__add"].read_only  # annotated read_only_hint=True
    assert not tools["demo__secret"].read_only  # no annotation: assume it may change things


async def test_calling_tools_goes_to_the_server():
    async with connected_demo() as registry:
        assert await registry.call("demo__add", '{"a": 2, "b": 3}') == "5"
        assert await registry.call("demo__secret", "{}") == "s3cret"  # env reached the server


async def test_a_failing_server_tool_returns_an_error():
    async with connected_demo() as registry:
        result = await registry.call("demo__fail", "{}")

    assert result.startswith("error:")
    assert "this tool always fails" in result


async def test_a_server_that_cannot_start_is_skipped():
    messages = []
    configs = {
        "broken": ServerConfig(command="this-command-does-not-exist"),
        "demo": demo_config(),
    }
    async with AsyncExitStack() as stack:
        tools = await connect_servers(configs, stack, log=messages.append)

    assert {tool.name for tool in tools} == {"demo__add", "demo__fail", "demo__secret"}
    assert messages[0].startswith("  [mcp] 'broken' failed to start")
    assert messages[1].startswith("  [mcp] connected 'demo'")


async def test_a_server_with_a_missing_variable_is_skipped(monkeypatch):
    monkeypatch.delenv("DEMO_SECRET", raising=False)
    messages = []
    async with AsyncExitStack() as stack:
        tools = await connect_servers(
            {"demo": demo_config(DEMO_SECRET="${DEMO_SECRET}")}, stack, log=messages.append
        )

    assert tools == []
    assert "DEMO_SECRET" in messages[0]
