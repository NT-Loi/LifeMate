from .base import BaseASR

def get_asr(engine: str = "gipformer", **kwargs) -> BaseASR:
  """Factory initialize ASR Engine."""
  if not engine or engine.lower() == "none":
    return None

  if engine.lower() == "gipformer":
    from .gipformer import GipformerASR
    return GipformerASR(**kwargs)

  if engine.lower() in ("phoasr_whisper"):
    from .phoasr_whisper import PhoASRWhisper
    return PhoASRWhisper(**kwargs)

  raise ValueError(f"ASR Engine {engine} isn't supported.")

__all__ = ["BaseASR", "get_asr"]
