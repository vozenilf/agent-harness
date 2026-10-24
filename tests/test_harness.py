"""Harness.ask: what we send to the model and what we do with the answer."""

from agent_harness.config import ModelConfig
from agent_harness.harness import Harness


async def test_ask_returns_the_model_answer(harness, fake_llm):
    fake_llm.say("Hello!")

    assert await harness.ask("Hi") == "Hello!"


async def test_ask_sends_model_system_prompt_and_turns_thinking_off(fake_llm, config):
    fake_llm.say("Hello!")
    harness = Harness(fake_llm.client(), config, system_prompt="Be brief.")

    await harness.ask("Hi")

    sent = fake_llm.requests[0]
    assert sent["model"] == "fake-model"
    assert sent["reasoning_effort"] == "none"
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]


async def test_whole_conversation_is_sent_on_the_next_turn(harness, fake_llm):
    fake_llm.say("Nice to meet you, Eddie.")
    fake_llm.say("Your name is Eddie.")

    await harness.ask("My name is Eddie.")
    await harness.ask("What is my name?")

    second = fake_llm.requests[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[2]["content"] == "Nice to meet you, Eddie."


async def test_reasoning_effort_is_left_out_when_not_set(fake_llm):
    fake_llm.say("Hello!")
    harness = Harness(fake_llm.client(), ModelConfig(model="m", reasoning_effort=None))

    await harness.ask("Hi")

    assert "reasoning_effort" not in fake_llm.requests[0]


async def test_empty_answer_becomes_empty_string(harness, fake_llm):
    fake_llm.say(None)

    assert await harness.ask("Hi") == ""
