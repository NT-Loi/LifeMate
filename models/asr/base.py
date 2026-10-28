from abc import ABC, abstractmethod


class BaseASR(ABC):

  @abstractmethod
  def transcribe(self, audio_path: str) -> str:
    """
    Transcribe audio file to text.
    Args:
        audio_path: Path to the audio file.
    Returns:
        Recognized text.
    """
    pass
