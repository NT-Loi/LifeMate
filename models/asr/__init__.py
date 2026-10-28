from .base import BaseASR

def get_asr(engine: str = None, **kwargs):
    """Get ASR engine."""
    if not engine or str(engine).lower() == "none":
      return None

    if engine == "gipformer":
      from .gipformer import GipformerASR
      kwargs.pop("device", None)
      return GipformerASR(**kwargs)

    if engine == "phoasr_whisper":
      from .phoasr_whisper import PhoASRWhisper
      return PhoASRWhisper(**kwargs)

    raise ValueError(f"ASR engine {engine} is not supported.")

__all__ = ["BaseASR", "get_asr"]
