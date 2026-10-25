"""Approval: tools that change something run only if the approver says yes."""

import pytest

from agent_harness.cli import ask_user, show_tools
from agent_harness.harness import DECLINED, Harness
from agent_harness.memory import Memory
from agent_harness.tools import ToolRegistry


class Approver:
    """A scripted approver that records what it was asked."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    async def __call__(self, tool, arguments: str) -> bool:
        self.asked.append((tool.name, arguments))
        return self.answer


def make_agent(fake_llm, config, workspace, approver, events=None):
    return Harness(
        fake_llm.client(),
        config,
        tools=ToolRegistry(workspace.tools()),
        approve=approver,
        on_event=(lambda kind, text: events.append((kind, text))) if events is not None else None,
    )


async def test_read_only_tools_run_without_asking(fake_llm, config, workspace):
    approver = Approver(answer=False)
    agent = make_agent(fake_llm, config, workspace, approver)
    fake_llm.call_tools(("read_file", {"filename": "notes.txt"}))
    fake_llm.say("Buy milk.")

    await agent.ask("Read my notes")

    assert approver.asked == []
    assert fake_llm.requests[1]["messages"][-1]["content"] == "buy milk"


async def test_an_approved_write_runs(fake_llm, config, workspace):
    approver = Approver(answer=True)
    agent = make_agent(fake_llm, config, workspace, approver)
    fake_llm.call_tools(("write_file", {"filename": "todo.md", "content": "- call mum"}))
    fake_llm.say("Done.")

    await agent.ask("Create todo.md")

    assert approver.asked == [("write_file", '{"filename": "todo.md", "content": "- call mum"}')]
    assert workspace.read_file("todo.md") == "- call mum"


async def test_a_declined_write_does_not_run(fake_llm, config, workspace):
    events = []
    agent = make_agent(fake_llm, config, workspace, Approver(answer=False), events)
    fake_llm.call_tools(("write_file", {"filename": "todo.md", "content": "- call mum"}))
    fake_llm.say("OK, I won't create it.")

    answer = await agent.ask("Create todo.md")

    assert not (workspace.root / "todo.md").exists()  # nothing was written
    assert fake_llm.requests[1]["messages"][-1]["content"] == DECLINED  # the model is told
    assert ("declined", "write_file") in events
    assert answer == "OK, I won't create it."


async def test_without_an_approver_everything_runs(fake_llm, config, workspace):
    agent = make_agent(fake_llm, config, workspace, approver=None)
    fake_llm.call_tools(("write_file", {"filename": "todo.md", "content": "x"}))
    fake_llm.say("Done.")

    await agent.ask("Create todo.md")

    assert workspace.read_file("todo.md") == "x"


# --- the CLI's approver and /tools -------------------------------------------


@pytest.mark.parametrize(
    ("typed", "approved"), [("y", True), ("YES", True), ("n", False), ("", False)]
)
async def test_ask_user_needs_an_explicit_yes(workspace, monkeypatch, typed, approved):
    write_file = next(t for t in workspace.tools() if t.name == "write_file")
    monkeypatch.setattr("builtins.input", lambda prompt="": typed)

    assert await ask_user(write_file, "{}") is approved


async def test_save_memory_is_approved_without_asking(tmp_path, monkeypatch):
    save_memory = Memory(tmp_path / "memory.md").tools()[0]
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("must not ask"))

    assert await ask_user(save_memory, '{"fact": "x"}') is True


def test_show_tools_marks_how_each_tool_runs(harness, workspace, capsys):
    harness.tools = ToolRegistry(workspace.tools())

    show_tools(harness)

    out = capsys.readouterr().out
    assert "read_file" in out and "read-only" in out
    assert "write_file" in out and "asks first" in out
