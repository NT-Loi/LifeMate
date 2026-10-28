import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from typing import Optional
from .base import BaseASR

MODEL_ID = "Qualcomm-AI-Research/PhoASR-whisper-small"

class PhoASRWhisper(BaseASR):

  def __init__(
    self,
    model_id: str = MODEL_ID,
    device: Optional[str] = None
  ):
    self.device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    self.model_id = model_id

    self.processor = AutoProcessor.from_pretrained(model_id)
    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    ).to(self.device)
    self.model.eval()

    # just to ignore warning
    self.model.generation_config.forced_decoder_ids = None

  def transcribe(self, audio_path: str) -> str:
      audio, sr = sf.read(audio_path, dtype="float32")

      if audio.ndim > 1:
          audio = audio.mean(axis=1)

      inputs = self.processor(
          audio,
          sampling_rate=sr,
          return_tensors="pt",
          return_attention_mask=True,
      )

      input_features = inputs.input_features.to(self.device)
      attention_mask = inputs.get("attention_mask")
      attention_mask = attention_mask.to(self.device)

      with torch.inference_mode():
          ids = self.model.generate(
              input_features=input_features,
              attention_mask=attention_mask,
              language="vi",
              task="transcribe",
          )
      return self.processor.batch_decode(
          ids, skip_special_tokens=True
      )[0]


if __name__ == "__main__":
  # Test on a sample audio file
  audio_file = "data/train+public_test/train+public_test/train/audio/train_0001/train_0001.wav"  # Replace with your audio file
  model = PhoASRWhisper()

  import time
  import soundfile as sf
  start = time.time()
  text = model.transcribe(audio_file)
  elapsed = time.time() - start

  audio_info = sf.info(audio_file)
  duration = audio_info.duration
  rtf = elapsed / duration if duration > 0 else 0

  print(f"\n  File: {audio_file}")
  print(f"  Text: {text}")
  print(f"  Time: {elapsed:.2f}s | Audio: {duration:.2f}s | RTF: {rtf:.3f}")