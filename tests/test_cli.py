"""The command-line loop, with input() replaced by a scripted list of lines."""

from agent_harness.cli import repl


def type_lines(monkeypatch, *lines):
    """Make input() return these lines one by one, as if the user typed them."""
    typed = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(typed))


async def test_repl_prints_answers_and_stops_on_quit(harness, fake_llm, monkeypatch, capsys):
    fake_llm.say("Hello!")
    type_lines(monkeypatch, "Hi", "", "/quit")  # the empty line is skipped

    await repl(harness)

    assert "assistant: Hello!" in capsys.readouterr().out
    assert len(fake_llm.requests) == 1


async def test_repl_survives_an_unreachable_model(harness, fake_llm, monkeypatch, capsys):
    fake_llm.down = True
    type_lines(monkeypatch, "Hi", "/quit")

    await repl(harness)  # must return normally, not crash

    assert "[error] the model call failed" in capsys.readouterr().out
