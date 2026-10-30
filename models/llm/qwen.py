from .base import BaseLLM
import re
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

import logging
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISIONS = {
    MODEL_ID: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}

class Qwen(BaseLLM):
    def __init__(self, model_id: str = MODEL_ID, max_context_length: int = 16384, 
                is_quantized: bool = True, dtype: str = None, device: str = None,
                max_new_tokens: int | None = None):
        self.max_context_length = min(max_context_length, 16384)
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


        if dtype is None:
            if self.device.startswith("cuda") or (self.device == "auto" and torch.cuda.is_available()):
                dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                dtype = "bfloat16"

        dtype = (torch.bfloat16 if dtype == "bfloat16" else 
                    (torch.float16 if dtype == "float16" else torch.float32)
            )

        # 4-bit NF4 Quantization
        quantization_config = None
        if is_quantized and self.device.startswith("cuda"):
            try:
                from transformers import BitsAndBytesConfig 
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info(f"⚡ Using 4-bit NF4 Quantization.")
            except Exception as e:
                logger.warning(f"⚠️ Could't initialize BitsAndBytesConfig ({e}).")
                quantization_config = None

        logger.info(f"🤖 Loading {model_id} (device={self.device}, dtype={dtype}, is_quantized={is_quantized})...")
        
        try:
            revision = MODEL_REVISIONS.get(model_id)
            load_kwargs = {"local_files_only": True}
            if revision:
                load_kwargs["revision"] = revision

            self.processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)

            load_kwargs["dtype"] = dtype

            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
                if self.device and (self.device.startswith("cuda:") or self.device == "cpu"):
                    load_kwargs["device_map"] = {"": self.device}
                else:
                    load_kwargs["device_map"] = "auto"
            else:
                if self.device == "auto":
                    load_kwargs["device_map"] = "auto"
                else:
                    load_kwargs["device_map"] = self.device

            self.model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
            self.model.eval()

            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"✅ {model_id} loaded successfully. Total parameters: {total_params / 1e9:.2f}B")

        except Exception:
            logger.exception(f"❌ Failed to load model {model_id}.")
            raise

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )

            # Move inputs to the model's device
            inputs = {
                key: value.to(self.model.device)
                for key, value in inputs.items()
            }

            input_length = inputs["input_ids"].shape[-1]
            if input_length >= self.max_context_length:
                raise ValueError(
                    f"Input length ({input_length}) exceeds the context limit "
                    f"({self.max_context_length})."
                )
            token_budget = self.max_context_length - input_length
            if self.max_new_tokens is not None:
                token_budget = min(token_budget, self.max_new_tokens)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=token_budget,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    **kwargs
                )

            # Decode only newly generated tokens
            generated_tokens = outputs[0, input_length:]
            response = self.processor.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()

            return response.strip()

        except Exception:
            logger.exception("Failed to generate response.")
            raise

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )

    model = Qwen()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer concisely.",
        },
        {
            "role": "user",
            "content": "Explain what a Python decorator is with a simple example.",
        },
    ]

    response = model.chat(messages)

    print("\n" + "=" * 50)
    print("User:", messages[-1]["content"])
    print("Assistant:", response)
    print("=" * 50)
