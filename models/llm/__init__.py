from .base import BaseLLM
from pathlib import Path

ADDITIONAL_MODEL_IDS = {"Qwen/Qwen3.5-2B"}


def get_llm(model_id: str | None = None, **kwargs) -> BaseLLM:
    """Create the LLM implementation registered for a model ID."""
    from .qwen import MODEL_ID, Qwen

    model_id = MODEL_ID if model_id is None else model_id
    if (
        model_id == MODEL_ID
        or model_id in ADDITIONAL_MODEL_IDS
        or Path(model_id).expanduser().is_dir()
    ):
        return Qwen(model_id=model_id, **kwargs)

    raise ValueError(f"LLM model {model_id!r} isn't supported.")

__all__ = ["BaseLLM", "get_llm"]
