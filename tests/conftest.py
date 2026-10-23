"""Shared fixtures: a fake model server built on httpx2.MockTransport.

The real openai client is used; only the network underneath it is replaced. So the
tests also check the JSON our code actually sends.
"""

import json
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from agent_harness.config import ModelConfig
from agent_harness.harness import Harness


class FakeLLM:
    """Plays the model server: answers with queued replies and records every request."""

    def __init__(self) -> None:
        self.replies: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.down = False  # True = behave like a server that isn't running

    def say(self, content: str | None) -> None:
        """Queue the next answer, as the JSON an OpenAI-compatible server returns."""
        self.replies.append(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "fake-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            }
        )

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        if self.down:
            raise httpx2.ConnectError("connection refused")
        self.requests.append(json.loads(request.content))
        return httpx2.Response(200, json=self.replies.pop(0))

    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url="http://fake-server/v1",
            api_key="test",
            max_retries=0,  # fail fast in tests instead of retrying
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler)),
        )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(model="fake-model")


@pytest.fixture
def harness(fake_llm, config) -> Harness:
    """A harness wired to the fake server instead of Ollama."""
    return Harness(fake_llm.client(), config)
