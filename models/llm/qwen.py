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
                max_new_tokens: int | None = None, enable_thinking: bool = False):
        self.max_context_length = min(max_context_length, 16384)
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.reset_cache()


        if dtype is None:
            uses_cuda = torch.cuda.is_available() and (
                self.device.startswith("cuda") or self.device == "auto"
            )
            if uses_cuda:
                device = self.device if self.device.startswith("cuda:") else None
                native_bf16 = torch.cuda.get_device_capability(device)[0] >= 8
                dtype = "bfloat16" if native_bf16 else "float16"
            else:
                dtype = "bfloat16"

        dtype = (torch.bfloat16 if dtype == "bfloat16" else 
                    (torch.float16 if dtype == "float16" else torch.float32)
            )

        # 4-bit NF4 Quantization
        quantization_config = None
        if is_quantized and torch.cuda.is_available() and (
            self.device.startswith("cuda") or self.device == "auto"
        ):
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

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> str:
        try:
            continuing = self._can_continue(messages, tools)
            past_key_values = self._past_key_values if continuing else None

            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
                "enable_thinking": self.enable_thinking,
            }
            if continuing:
                # The previous assistant turn is already represented by the live cache.
                # Feed its final uncached token, then only the newly appended user message.
                inputs = self.processor.apply_chat_template(
                    [messages[-1]],
                    **template_kwargs,
                )
                inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
                suffix_ids = inputs["input_ids"]
                pending_ids = self._pending_tokens.to(self.model.device)
                inputs["input_ids"] = torch.cat((pending_ids, suffix_ids), dim=-1)
                inputs["attention_mask"] = torch.ones(
                    (1, past_key_values.get_seq_length() + inputs["input_ids"].shape[-1]),
                    dtype=inputs["attention_mask"].dtype,
                    device=self.model.device,
                )
                if "mm_token_type_ids" in inputs:
                    inputs["mm_token_type_ids"] = torch.zeros_like(inputs["input_ids"])
                logger.info(
                    "Reusing KV cache (%d tokens); encoding only %d new prompt tokens.",
                    past_key_values.get_seq_length(),
                    inputs["input_ids"].shape[-1],
                )
            else:
                if self._past_key_values is not None:
                    logger.info("Conversation changed; starting a fresh KV cache.")
                    self.reset_cache()
                if tools is not None:
                    template_kwargs["tools"] = tools
                inputs = self.processor.apply_chat_template(messages, **template_kwargs)
                inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

            input_length = inputs["input_ids"].shape[-1]
            context_length = input_length + (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            if context_length >= self.max_context_length:
                raise ValueError(
                    f"Input length ({context_length}) exceeds the context limit "
                    f"({self.max_context_length})."
                )
            token_budget = self.max_context_length - context_length
            if self.max_new_tokens is not None:
                token_budget = min(token_budget, self.max_new_tokens)
            logger.info(
                "Generation input: prompt_tokens=%d cached_tokens=%d context_tokens=%d max_context=%d max_new_tokens=%d",
                input_length,
                past_key_values.get_seq_length() if past_key_values is not None else 0,
                context_length,
                self.max_context_length,
                token_budget,
            )

            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            kwargs["return_dict_in_generate"] = True
            kwargs["use_cache"] = True

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=token_budget,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    **kwargs
                )

            # Decode only newly generated tokens
            sequences = outputs.sequences
            self._past_key_values = outputs.past_key_values
            self._pending_tokens = sequences[:, -1:].detach()
            self._message_snapshot = [dict(message) for message in messages]
            self._cached_tools = tools
            generated_tokens = sequences[0, input_length:]
            self.last_raw_response = self.processor.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=False,
            ).strip()
            response = self.processor.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()

            response = response.strip()
            self._last_response = response
            return response

        except Exception:
            logger.exception("Failed to generate response.")
            raise

    def reset_cache(self):
        """Prevent a new sample from reusing another sample's conversation cache."""
        self._past_key_values = None
        self._pending_tokens = None
        self._message_snapshot = None
        self._cached_tools = None
        self._last_response = None
        self.last_raw_response = None

    def _can_continue(self, messages, tools) -> bool:
        if (
            self._past_key_values is None
            or self._pending_tokens is None
            or self._message_snapshot is None
            or self._cached_tools != tools
            or len(messages) != len(self._message_snapshot) + 2
        ):
            return False

        previous_turn = messages[-2]
        return (
            messages[:-2] == self._message_snapshot
            and previous_turn.get("role") == "assistant"
            and previous_turn.get("content") == self._last_response
            and messages[-1].get("role") == "user"
        )

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
