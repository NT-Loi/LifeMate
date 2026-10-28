from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseLLM(ABC):
  """Abstract base class cho các mô hình ngôn ngữ (LLM)."""

  @abstractmethod
  def chat(
      self,
      messages: List[Dict[str, str]],
      max_new_tokens: int = 256,
      temperature: float = 0.0,
      enable_thinking: bool = False,
      **kwargs: Any,
  ) -> str:
    """Nhận danh sách message (system, user, assistant) và trả về text sinh ra."""
    pass

  @abstractmethod
  def generate(
      self,
      prompt: str,
      max_new_tokens: int = 256,
      temperature: float = 0.0,
      **kwargs: Any,
  ) -> str:
    """Nhận raw prompt string và trả về text sinh ra."""
    pass
