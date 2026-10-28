from .base import BaseLLM

def get_llm(model_id: str = "Qwen/Qwen3.5-4B", **kwargs) -> BaseLLM:
    """Get LLM engine."""
    if model_id == "Qwen/Qwen3.5-4B":
        from .qwen import QwenLLM
        return QwenLLM(model_id=model_id, **kwargs)
    raise ValueError(f"LLM model_id {model_id} is not supported.")

__all__ = ["BaseLLM", "get_llm"]
