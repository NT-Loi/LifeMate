from abc import ABC, abstractmethod


class BaseLLM(ABC):

  @abstractmethod
  def chat(self, messages: list[dict[str, str]]) -> str:
    """
    Generate a response from a list of chat messages.

    Args:
        messages: List of messages, e.g.
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]

    Returns:
        The assistant's response.
    """
    pass
