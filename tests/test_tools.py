"""ToolRegistry: schemas for the model, and tool calls that never crash the harness."""

import pytest

from agent_harness.tools import MAX_RESULT_CHARS, Tool, ToolRegistry

NO_ARGS = {"type": "object", "properties": {}}


def make_tool(name="echo", function=lambda text: f"echo: {text}", read_only=True):
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    return Tool(name, "Repeat the text.", parameters, function, read_only)


def test_schemas_use_the_chat_completions_tool_format():
    registry = ToolRegistry([make_tool()])

    assert registry.schemas() == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Repeat the text.",
                "parameters": make_tool().parameters,
            },
        }
    ]


def test_duplicate_names_are_rejected():
    registry = ToolRegistry([make_tool()])

    with pytest.raises(ValueError, match="duplicate"):
        registry.add(make_tool())


async def test_call_runs_a_plain_function():
    registry = ToolRegistry([make_tool()])

    assert await registry.call("echo", '{"text": "hi"}') == "echo: hi"


async def test_call_awaits_an_async_function():
    async def slow_echo(text):
        return f"async: {text}"

    registry = ToolRegistry([make_tool(function=slow_echo)])

    assert await registry.call("echo", '{"text": "hi"}') == "async: hi"


async def test_empty_arguments_mean_no_arguments():
    registry = ToolRegistry([Tool("ping", "Ping.", NO_ARGS, lambda: "pong")])

    assert await registry.call("ping", "") == "pong"


async def test_unknown_tool_returns_an_error_listing_the_real_tools():
    registry = ToolRegistry([make_tool()])

    result = await registry.call("delete_everything", "{}")

    assert result.startswith("error: unknown tool 'delete_everything'")
    assert "echo" in result


@pytest.mark.parametrize("raw", ['{"text": "hi"', "not json at all"])
async def test_broken_json_returns_an_error(raw):
    registry = ToolRegistry([make_tool()])

    assert "not valid JSON" in await registry.call("echo", raw)


async def test_json_that_is_not_an_object_returns_an_error():
    registry = ToolRegistry([make_tool()])

    assert "must be a JSON object" in await registry.call("echo", '["hi"]')


async def test_wrong_argument_names_return_an_error():
    registry = ToolRegistry([make_tool()])

    result = await registry.call("echo", '{"txt": "hi"}')

    assert result.startswith("error: echo failed: TypeError")


async def test_an_exception_inside_the_tool_returns_an_error():
    def broken(text):
        raise RuntimeError("disk on fire")

    registry = ToolRegistry([make_tool(function=broken)])

    result = await registry.call("echo", '{"text": "hi"}')

    assert result == "error: echo failed: RuntimeError: disk on fire"


async def test_long_results_are_truncated():
    registry = ToolRegistry([make_tool(function=lambda text: "x" * (MAX_RESULT_CHARS + 500))])

    result = await registry.call("echo", '{"text": "hi"}')

    assert result.startswith("x" * MAX_RESULT_CHARS)
    assert result.endswith(f"[truncated: {MAX_RESULT_CHARS + 500} characters in total]")