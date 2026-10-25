"""Long-term memory: a markdown file of facts, and the save_memory tool."""

import pytest

from agent_harness.harness import Harness
from agent_harness.memory import EMPTY, MAX_FACT_CHARS, Memory, system_prompt_with_memory
from agent_harness.tools import ToolRegistry


@pytest.fixture
def memory(tmp_path) -> Memory:
    return Memory(tmp_path / "memory.md")  # the file doesn't exist yet


def test_no_file_means_no_facts(memory):
    assert memory.facts() == []
    assert memory.as_text() == EMPTY


def test_saved_facts_are_written_as_a_markdown_list(memory):
    assert memory.save("The user's name is Jacob.") == "saved: The user's name is Jacob."
    memory.save("An international meeting on Thursday.")

    assert memory.path.read_text(encoding="utf-8") == (
        "- The user's name is Jacob.\n- An international meeting on Thursday.\n"
    )
    assert memory.facts() == ["The user's name is Jacob.", "An international meeting on Thursday."]


def test_memory_survives_a_restart(memory):
    memory.save("Likes short answers.")

    later = Memory(memory.path)  # a new session reading the same file

    assert later.facts() == ["Likes short answers."]


def test_duplicates_are_skipped_ignoring_case(memory):
    memory.save("Likes tea.")

    assert memory.save("likes TEA.") == "already known: likes TEA."
    assert memory.facts() == ["Likes tea."]


def test_a_fact_is_kept_on_one_line(memory):
    memory.save("Works on\n  two   projects")

    assert memory.facts() == ["Works on two projects"]


@pytest.mark.parametrize("fact", ["", "   \n  ", "x" * (MAX_FACT_CHARS + 1)])
def test_empty_or_overlong_facts_are_refused(memory, fact):
    with pytest.raises(ValueError):
        memory.save(fact)
    assert not memory.path.exists()


def test_system_prompt_contains_the_facts(memory):
    memory.save("The user's name is Jacob.")

    prompt = system_prompt_with_memory("Be helpful.", memory)

    assert prompt.startswith("Be helpful.")
    assert "- The user's name is Jacob." in prompt
    assert "save_memory" in prompt


async def test_the_model_can_save_a_fact_through_the_harness(fake_llm, config, memory):
    harness = Harness(fake_llm.client(), config, tools=ToolRegistry(memory.tools()))
    fake_llm.call_tools(("save_memory", {"fact": "The user's name is Jacob."}))
    fake_llm.say("Nice to meet you, Jacob!")

    await harness.ask("Hi, I'm Jacob.")

    assert memory.facts() == ["The user's name is Jacob."]
    assert fake_llm.requests[1]["messages"][-1]["content"] == "saved: The user's name is Jacob."
