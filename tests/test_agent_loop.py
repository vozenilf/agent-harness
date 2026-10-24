"""The agent loop: model asks for tools, harness runs them, model answers."""

import pytest

from agent_harness.harness import Harness
from agent_harness.tools import Tool, ToolRegistry


@pytest.fixture
def events() -> list[tuple[str, str]]:
    """Collects every (kind, text) event the harness reports."""
    return []


@pytest.fixture
def agent(fake_llm, config, workspace, events) -> Harness:
    """A harness with the workspace tools, reporting events into the events list."""
    return Harness(
        fake_llm.client(),
        config,
        tools=ToolRegistry(workspace.tools()),
        on_event=lambda kind, text: events.append((kind, text)),
    )


async def test_tool_list_is_sent_to_the_model(agent, fake_llm):
    fake_llm.say("Hi!")

    await agent.ask("Hello")

    sent_tools = [t["function"]["name"] for t in fake_llm.requests[0]["tools"]]
    assert sent_tools == ["list_files", "read_file", "write_file"]


async def test_no_tools_key_when_there_are_no_tools(harness, fake_llm):
    fake_llm.say("Hi!")

    await harness.ask("Hello")

    assert "tools" not in fake_llm.requests[0]


async def test_tool_call_then_answer(agent, fake_llm):
    fake_llm.call_tools(("read_file", {"filename": "notes.txt"}))
    fake_llm.say("Your note says: buy milk.")

    answer = await agent.ask("What is in my notes?")

    assert answer == "Your note says: buy milk."
    assert len(fake_llm.requests) == 2  # one question, two model calls

    second = fake_llm.requests[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    tool_request, tool_result = second[2], second[3]
    assert tool_request["tool_calls"][0]["function"]["name"] == "read_file"
    assert tool_result == {
        "role": "tool",
        "tool_call_id": tool_request["tool_calls"][0]["id"],  # links result to request
        "content": "buy milk",
    }


async def test_several_tool_calls_in_one_reply(agent, fake_llm, workspace):
    fake_llm.call_tools(
        ("write_file", {"filename": "a.txt", "content": "A"}),
        ("write_file", {"filename": "b.txt", "content": "B"}),
    )
    fake_llm.say("Done.")

    await agent.ask("Create a.txt and b.txt")

    messages = fake_llm.requests[1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool"]
    assert workspace.read_file("a.txt") == "A"
    assert workspace.read_file("b.txt") == "B"


async def test_tool_errors_go_back_to_the_model(agent, fake_llm):
    fake_llm.call_tools(("read_file", {"filename": "../secret.txt"}))
    fake_llm.say("Sorry, I can't read files outside your workspace.")

    answer = await agent.ask("Read ../secret.txt")

    tool_message = fake_llm.requests[1]["messages"][-1]
    assert tool_message["content"].startswith("error: read_file failed: PermissionError")
    assert answer == "Sorry, I can't read files outside your workspace."


async def test_broken_arguments_do_not_crash_the_loop(agent, fake_llm):
    fake_llm.call_tools(("read_file", '{"filename": "notes.txt"'))  # missing }
    fake_llm.call_tools(("read_file", {"filename": "notes.txt"}))  # the model retries
    fake_llm.say("Buy milk.")

    assert await agent.ask("Read my notes") == "Buy milk."
    first_result = fake_llm.requests[1]["messages"][-1]["content"]
    assert "not valid JSON" in first_result


async def test_loop_stops_after_max_steps(fake_llm, config):
    always = Tool("ping", "Ping.", {"type": "object", "properties": {}}, lambda: "pong")
    harness = Harness(fake_llm.client(), config, tools=ToolRegistry([always]), max_steps=3)
    for _ in range(3):
        fake_llm.call_tools(("ping", {}))  # a model that never stops calling tools

    answer = await harness.ask("Ping forever")

    assert answer == "[stopped: no final answer after 3 steps]"
    assert len(fake_llm.requests) == 3


async def test_events_report_each_call_and_result(agent, fake_llm, events):
    fake_llm.call_tools(("read_file", {"filename": "notes.txt"}))
    fake_llm.say("Buy milk.")

    await agent.ask("Read my notes")

    assert events == [
        ("tool_call", 'read_file({"filename": "notes.txt"})'),
        ("tool_result", "buy milk"),
    ]


async def test_history_keeps_the_tool_round_for_the_next_question(agent, fake_llm):
    fake_llm.call_tools(("read_file", {"filename": "notes.txt"}))
    fake_llm.say("Buy milk.")
    fake_llm.say("You asked about your notes.")

    await agent.ask("Read my notes")
    await agent.ask("What did I ask before?")

    roles = [m["role"] for m in fake_llm.requests[2]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
