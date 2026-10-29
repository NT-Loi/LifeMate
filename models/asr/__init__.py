from .base import BaseASR


def get_asr(engine: str = "gipformer", **kwargs) -> BaseASR:
  """Factory khởi tạo ASR engine theo cấu hình."""
  if not engine or engine.lower() == "none":
    return None

  if engine.lower() == "gipformer":
    from .gipformer import GipformerASR

    return GipformerASR(**kwargs)

  if engine.lower() in ("phoasr", "phoasr-whisper", "phoasr_whisper", "whisper"):
    from .phoasr_whisper import PhoASRWhisper

    return PhoASRWhisper(**kwargs)

  raise ValueError(f"ASR engine không hỗ trợ: {engine}")


__all__ = ["BaseASR", "get_asr"]
