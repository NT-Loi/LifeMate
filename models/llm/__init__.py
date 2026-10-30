from .base import BaseLLM

def get_llm(engine: str = "qwen", **kwargs) -> BaseLLM:
  """Factory initialize LLM Engine."""
  
  if engine.lower() == "qwen":
    from .qwen import Qwen
    return Qwen(**kwargs)

  raise ValueError(f"LLM engine {engine} isn't supported.")


__all__ = ["BaseLLM", "get_llm"]
