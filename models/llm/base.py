from abc import ABC, abstractmethod


class BaseLLM(ABC):

  def reset_cache(self):
    """Clear conversation state held by the backend, if any."""
    pass

  @abstractmethod
  def chat(
      self,
      messages: list[dict[str, str]],
      tools: list[dict] | None = None,
  ) -> str:
    """
    Generate a response from a list of chat messages.

    Args:
        messages: List of messages, e.g.
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
        tools: Optional function schemas exposed to the model.

    Returns:
        The assistant's response.
    """
    pass
