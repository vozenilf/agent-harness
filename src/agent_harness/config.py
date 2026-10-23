"""Model settings: where the model lives and which one to use.

Defaults point to a local Ollama server. Environment variables override them, so the
same code runs against another machine (or, later, Docker) without edits.
"""

import os

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """One model behind an OpenAI-compatible endpoint."""

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"  # Ollama ignores the key, but the openai library requires one
    model: str = "qwen3.5:4b"
    reasoning_effort: str | None = "none"  # "none" = no thinking; on a CPU thinking takes minutes


# environment variable -> ModelConfig field it overrides
ENV_VARS = {
    "HARNESS_BASE_URL": "base_url",
    "HARNESS_API_KEY": "api_key",
    "HARNESS_MODEL": "model",
}


def load_model_config() -> ModelConfig:
    """Build the config from defaults, overridden by any HARNESS_* variables that are set."""
    overrides = {field: os.environ[var] for var, field in ENV_VARS.items() if var in os.environ}
    return ModelConfig(**overrides)
