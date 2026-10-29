import os
import json
import logging
from typing import Any, Dict, Iterator, List, Optional, Union
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, USER, ContentItem, Message
from .base import BaseLLM

# Tránh phân mảnh bộ nhớ CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)
MODEL_ID = "Qwen/Qwen3.5-4B"


def clear_all_gpu_cache():
  """Giải phóng cache trên tất cả GPU đang có trong hệ thống."""
  if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
      with torch.cuda.device(i):
        torch.cuda.empty_cache()


class QwenLLM(BaseFnCallModel, BaseLLM):
  """Qwen LLM engine tương thích hoàn toàn với framework Qwen-Agent và BaseLLM.

  Hỗ trợ:
    - Mô hình Qwen3.5-4B (4.54B params < 5B quy chuẩn ZAC 2026), Qwen2.5-3B, Qwen2.5-1.5B.
    - Function calling tự nhiên qua ChatML và Qwen-Agent prompt templates.
    - Chạy trên Single-GPU hoặc Multi-GPU (Kaggle 2x T4: balanced_low_0) hoặc CPU.
  """

  def __init__(
      self,
      model_id: str = MODEL_ID,
      device: Optional[str] = None,
      torch_dtype: Optional[str] = None,
      load_in_4bit: bool = True,
      max_context_length: int = 16384,
      cfg: Optional[Dict] = None,
  ):
    cfg = cfg or {}
    cfg.setdefault("fncall_prompt_type", "nous")
    super().__init__(cfg)

    self.model_id = model_id
    self.model = model_id  # Qwen-Agent kiểm tra self.llm.model
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    self.max_context_length = min(max_context_length, 16384)

    # Tự động chọn kiểu dữ liệu tối ưu:
    if torch_dtype is None:
      if self.device.startswith("cuda"):
        torch_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
      else:
        torch_dtype = "bfloat16"

    dtype = (
        torch.bfloat16
        if torch_dtype == "bfloat16"
        else (torch.float16 if torch_dtype == "float16" else torch.float32)
    )

    # Cấu hình 4-bit NF4 Quantization trên CUDA để chống OOM (đặc biệt trên GPU 15GB như Tesla T4)
    quantization_config = None
    if self.device.startswith("cuda") and load_in_4bit:
      try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"⚡ Bật 4-bit NF4 Quantization (BitsAndBytes) cho {model_id} (giảm VRAM từ ~9GB xuống ~2.8GB)!")
      except Exception as e:
        logger.warning(f"⚠️ Không thể khởi tạo BitsAndBytesConfig ({e}), nạp mô hình không lượng tử.")
        quantization_config = None

    logger.info(
        f"🤖 Đang nạp {model_id} (device={self.device}, dtype={torch_dtype}, 4bit={quantization_config is not None})..."
    )
    self.config = AutoConfig.from_pretrained(model_id)
    arch = getattr(self.config, "architectures", [""])[0]

    if hasattr(transformers, "Qwen3_5ForConditionalGeneration") and "Qwen3_5" in arch:
      model_cls = transformers.Qwen3_5ForConditionalGeneration
    else:
      model_cls = AutoModelForCausalLM

    if self.device.startswith("cuda"):
      gpu_count = torch.cuda.device_count()
      # Nếu có từ 2 GPU trở lên (ví dụ Kaggle 2x T4), sử dụng "balanced_low_0" để ép Accelerate
      # chia đều trọng số mô hình qua cả 2 GPU và chừa headroom trên GPU 0 cho activations
      dev_map = "balanced_low_0" if gpu_count > 1 else "auto"

      load_kwargs = {"device_map": dev_map}
      if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
      else:
        load_kwargs["torch_dtype"] = dtype

      logger.info(f"🚀 Phân bổ mô hình lên {gpu_count} GPU bằng device_map='{dev_map}'...")
      self.hf_model = model_cls.from_pretrained(
          model_id,
          **load_kwargs,
      )

      # In xác thực danh sách GPU mà mô hình đang chiếm giữ
      hf_dev_map = getattr(self.hf_model, "hf_device_map", {})
      if hf_dev_map:
        active_gpus = sorted(list(set(str(v) for v in hf_dev_map.values())))
        logger.info(f"✅ Xác nhận: Mô hình đã được phân bổ thành công trên {len(active_gpus)} thiết bị: {active_gpus}")
    else:
      self.hf_model = model_cls.from_pretrained(
          model_id,
          torch_dtype=dtype,
      ).to("cpu")
      torch.set_num_threads(16)
    self.hf_model.eval()

    self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    num_params = sum(p.numel() for p in self.hf_model.parameters())
    logger.info(
        f"✅ {model_id} đã sẵn sàng. Tổng tham số: {num_params / 1e9:.2f}B (< 5B)"
    )

  def _get_plain_messages(self, messages: List[Message]) -> List[dict]:
    """Chuyển đổi danh sách Qwen-Agent Message thành định dạng chuẩn cho apply_chat_template."""
    plain = []
    for m in messages:
      role = "tool" if m.role == FUNCTION else m.role

      # Lấy text content
      if isinstance(m.content, list):
        content_text = "".join(
            item.text
            if hasattr(item, "text")
            else (
                item.get("text", "")
                if isinstance(item, dict)
                else str(item)
            )
            for item in m.content
        )
      else:
        content_text = str(m.content or "")

      # Nếu assistant message có lệnh gọi hàm function_call
      if getattr(m, "function_call", None):
        fn = m.function_call
        fn_args = (
            fn.arguments
            if isinstance(fn.arguments, dict)
            else json.loads(fn.arguments or "{}")
        )
        call_dict = {"name": fn.name, "arguments": fn_args}
        call_str = f"<tool_call>\n{json.dumps(call_dict, ensure_ascii=False)}\n</tool_call>"
        content_text = f"{content_text}\n{call_str}".strip()

      plain.append({"role": role, "content": content_text})
    return plain

  def _chat_stream(
      self,
      messages: List[Message],
      delta_stream: bool,
      generate_cfg: dict,
  ) -> Iterator[List[Message]]:
    """Thực thi inference cho Qwen-Agent và stream kết quả."""
    plain_messages = self._get_plain_messages(messages)

    try:
      prompt = self.tokenizer.apply_chat_template(
          plain_messages,
          tokenize=False,
          add_generation_prompt=True,
          enable_thinking=generate_cfg.get("enable_thinking", False),
      )
    except TypeError:
      prompt = self.tokenizer.apply_chat_template(
          plain_messages,
          tokenize=False,
          add_generation_prompt=True,
      )

    max_new_tokens = generate_cfg.get("max_new_tokens", 512)
    try:
      model_device = next(self.hf_model.parameters()).device
    except Exception:
      model_device = getattr(self.hf_model, "device", self.device)

    inputs = self.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=self.max_context_length - max_new_tokens,
    ).to(model_device)

    temperature = generate_cfg.get("temperature", 0.0)
    do_sample = temperature > 0.0

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": self.tokenizer.eos_token_id,
    }
    if do_sample:
      gen_kwargs["temperature"] = temperature

    clear_all_gpu_cache()

    with torch.inference_mode():
      output_ids = self.hf_model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0, input_len:]
    raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

    del inputs
    del output_ids
    clear_all_gpu_cache()

    for stop_token in ["<|im_end|>", "<|endoftext|>"]:
      raw_output = raw_output.replace(stop_token, "")

    yield [Message(role=ASSISTANT, content=raw_output.strip())]

  def _chat_no_stream(
      self,
      messages: List[Message],
      generate_cfg: dict,
  ) -> List[Message]:
    out = []
    for out in self._chat_stream(
        messages, delta_stream=False, generate_cfg=generate_cfg
    ):
      pass
    return out

  def chat(
      self,
      messages: Any,
      max_new_tokens: int = 512,
      temperature: float = 0.0,
      enable_thinking: bool = False,
      **kwargs: Any,
  ) -> Any:
    """Hỗ trợ cả giao diện gọi từ Qwen-Agent lẫn giao diện chuẩn BaseLLM."""
    # Nếu được gọi từ Qwen-Agent (có functions hoặc stream)
    if "functions" in kwargs or kwargs.get("stream", False):
      return BaseFnCallModel.chat(self, messages, **kwargs)

    # Nếu được gọi trực tiếp theo chuẩn BaseLLM
    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
    }
    gen_cfg.update(kwargs)
    qwen_msgs = [
        Message(role=m.get("role", "user"), content=m.get("content", ""))
        for m in messages
    ]
    res_list = self._chat_no_stream(qwen_msgs, gen_cfg)
    if res_list:
      return res_list[0].content
    return ""

  def generate(
      self,
      prompt: str,
      max_new_tokens: int = 512,
      temperature: float = 0.0,
      **kwargs: Any,
  ) -> str:
    """Sinh text trực tiếp từ raw prompt."""
    return self.chat(
        [{"role": "user", "content": prompt}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs,
    )


__all__ = ["QwenLLM", "MODEL_ID"]