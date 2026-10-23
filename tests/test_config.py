"""Model config: local-Ollama defaults, overridable from the environment."""

import pytest

from agent_harness.config import ENV_VARS, load_model_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test with none of our variables set, whatever the real shell has."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_defaults_point_to_local_ollama():
    config = load_model_config()
    assert config.base_url == "http://localhost:11434/v1"
    assert config.model == "qwen3.5:4b"
    assert config.reasoning_effort == "none"


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("HARNESS_MODEL", "qwen3.5:2b")
    monkeypatch.setenv("HARNESS_BASE_URL", "http://host.docker.internal:11434/v1")

    config = load_model_config()

    assert config.model == "qwen3.5:2b"
    assert config.base_url == "http://host.docker.internal:11434/v1"
    assert config.api_key == "ollama"  # not overridden, keeps its default
