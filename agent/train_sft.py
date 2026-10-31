"""Supervised QLoRA fine-tuning for LifeMate using the labeled train split."""

import argparse
import json
import logging
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from agent.prompt import LifeMatePromptBuilder
from config import MODE
from models.llm.qwen import MODEL_ID, MODEL_REVISIONS

logger = logging.getLogger("train_sft")
END_OF_MESSAGE = "<|im_end|>"


def _render_completion(processor, messages, tools, call=None):
    prompt = processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if call is None:
        assistant_message = {"role": "assistant", "content": "[]"}
    else:
        arguments = {
            key: ("true" if value is True else "false" if value is False else value)
            for key, value in call.get("arguments", {}).items()
        }
        assistant_message = {
            "role": "assistant",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": arguments,
                },
            }],
        }

    full = processor.apply_chat_template(
        [*messages, assistant_message],
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not full.startswith(prompt):
        raise ValueError("Qwen chat template did not preserve its generation prefix")
    completion = full[len(prompt):]
    if END_OF_MESSAGE not in completion:
        raise ValueError("Qwen chat template omitted the assistant end token")
    return prompt, completion


def _build_examples(samples, builder, processor, max_length):
    examples = []
    overlong = 0
    for sample in samples:
        audio_text = sample.get("audio_text")
        if not isinstance(audio_text, str) or not audio_text.strip():
            raise ValueError(f"{sample.get('id')}: train row has no audio_text")

        messages = builder.create_prompt(sample, audio_text)
        tools = builder.build_tools(sample.get("context", {}).get("installed_apps", {}))
        available_tools = {tool["name"] for tool in tools}
        label = sample.get("label", {})
        calls = label.get("tool_calls", [])
        results = label.get("tool_results", [])
        if len(calls) != len(results):
            raise ValueError(f"{sample.get('id')}: tool_calls/tool_results length mismatch")

        steps = [*zip(calls, results)]
        if not steps:
            steps = [(None, None)]

        for call, result in steps:
            if call is not None and call.get("name") not in available_tools:
                raise ValueError(
                    f"{sample.get('id')}: gold tool {call.get('name')!r} is not installed"
                )
            prompt, completion = _render_completion(processor, messages, tools, call)
            token_count = len(processor.tokenizer(
                prompt + completion,
                add_special_tokens=False,
            )["input_ids"])
            if token_count <= max_length:
                examples.append({"prompt": prompt, "completion": completion})
            else:
                overlong += 1

            if call is None or call["name"] == "sys.ask_missing_info":
                break

            response = completion.rsplit(END_OF_MESSAGE, 1)[0]
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": (
                    "Kết quả thực thi công cụ ở lượt vừa rồi:\n"
                    + json.dumps([result], ensure_ascii=False, indent=2)
                ),
            })

        if calls and not any(call["name"] == "sys.ask_missing_info" for call in calls):
            prompt, completion = _render_completion(processor, messages, tools)
            token_count = len(processor.tokenizer(
                prompt + completion,
                add_special_tokens=False,
            )["input_ids"])
            if token_count <= max_length:
                examples.append({"prompt": prompt, "completion": completion})
            else:
                overlong += 1

    return examples, overlong


def _load_samples(path, max_samples=None):
    with Path(path).open(encoding="utf-8") as source:
        samples = [json.loads(line) for line in source if line.strip()]
    if max_samples is not None:
        samples = samples[:max_samples]
    if len(samples) < 2:
        raise ValueError("Need at least two train samples to make a train/validation split")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5 for LifeMate with SFT + QLoRA")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--data-path", type=Path, default=MODE["train"]["data_path"])
    parser.add_argument("--catalog-path", type=Path, default=MODE["train"]["catalog_path"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lifemate-qwen35-4b-sft"))
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-samples", type=int, default=None, help="Use the first N rows for a short run")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("QLoRA training requires a CUDA GPU; run this on the GCP L4 VM.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    samples = _load_samples(args.data_path, args.max_samples)
    random.Random(args.seed).shuffle(samples)
    validation_count = max(1, round(len(samples) * 0.1))
    validation_samples = samples[:validation_count]
    train_samples = samples[validation_count:]

    revision = MODEL_REVISIONS.get(args.model_id)
    processor_kwargs = {"local_files_only": False}
    if revision:
        processor_kwargs["revision"] = revision
    processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs)
    builder = LifeMatePromptBuilder(args.catalog_path)

    train_rows, train_overlong = _build_examples(
        train_samples, builder, processor, args.max_length
    )
    validation_rows, validation_overlong = _build_examples(
        validation_samples, builder, processor, args.max_length
    )
    if not train_rows or not validation_rows:
        raise ValueError("No usable training or validation examples remain after length filtering")

    logger.info(
        "Prepared %d train samples / %d SFT turns and %d validation samples / %d SFT turns; "
        "skipped %d overlength turns (train=%d, validation=%d).",
        len(train_samples), len(train_rows), len(validation_samples), len(validation_rows),
        train_overlong + validation_overlong, train_overlong, validation_overlong,
    )

    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model_kwargs = {
        "revision": revision,
        "dtype": compute_dtype,
        "quantization_config": quantization,
        "device_map": {"": 0},
    }
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}
    model = AutoModelForMultimodalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        exclude_modules=r".*visual.*",
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows),
        processing_class=processor.tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(args.output_dir)
    (args.output_dir / "validation_ids.json").write_text(
        json.dumps(sorted(sample["id"] for sample in validation_samples), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved QLoRA adapter and processor to %s", args.output_dir)


if __name__ == "__main__":
    main()
