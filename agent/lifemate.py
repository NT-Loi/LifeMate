from agent.tools import get_tool_env
from config import MODE
from models import get_asr, get_llm
from agent.prompt import LifeMatePromptBuilder

import json
import random
import re
import numpy as np
import torch

import logging
logger = logging.getLogger(__name__)

from pathlib import Path


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LifeMateAgent:
    """LifeMate Agentic Assistant."""

    def __init__(
        self,
        asr: str | None = "Gipformer",
        model_id: str | None = None,
        mode: str = "train",
        device: str | None = None,
        data_dir: str | Path | None = None,
        tools_dir: str | Path | None = None,
        catalog_path: str | Path | None = None,
        **llm_kwargs
    ):
        seed_everything()
        self.mode = mode

        # Load config
        cfg = MODE.get(self.mode, {})
        self.tools_dir = Path(tools_dir) if tools_dir else cfg.get("tools_dir")
        self.catalog_path = Path(catalog_path) if catalog_path else cfg.get("catalog_path")
        self.data_dir = Path(data_dir) if data_dir else cfg.get("data_dir")

        # Init ASR
        if asr and str(asr).lower() != "none":
            logger.info(f"Initialize ASR: {asr}")
            self.asr = get_asr(asr)
        else:
            self.asr = None

        # Init LLM
        if device:
            llm_kwargs["device"] = device
        logger.info("Initialize LLM: model_id=%s (device=%s)", model_id or "default", device)
        self.llm = get_llm(model_id=model_id, **llm_kwargs)

        # Load ToolEnv 
        self.ToolEnv = get_tool_env(self.mode)

        # Build Prompt
        self.prompt_builder = LifeMatePromptBuilder(self.catalog_path)

        logger.info(f"LifeMate Agent [{self.mode.upper()}] is ready!")

    @staticmethod
    def parse_json_calls(text: str, tool_schemas: dict[str, dict] | None = None):
        """Parse Qwen native function calls, with JSON-list compatibility."""
        text = text.strip()
        if "<think>" in text or "</think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()

        native_calls = []
        call_pattern = re.compile(
            r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>",
            re.DOTALL,
        )
        parameter_pattern = re.compile(
            r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL
        )
        for call_match in call_pattern.finditer(text):
            name = call_match.group(1).strip()
            schema = (tool_schemas or {}).get(name, {})
            properties = schema.get("parameters", {}).get("properties", {})
            arguments = {}
            for parameter_match in parameter_pattern.finditer(call_match.group(2)):
                key = parameter_match.group(1).strip()
                raw_value = parameter_match.group(2).strip()
                value_schema = properties.get(key, {})
                try:
                    value = json.loads(raw_value)
                except json.JSONDecodeError:
                    value = raw_value

                expected_type = value_schema.get("type")
                if expected_type == "string" and not isinstance(value, str):
                    value = raw_value
                elif expected_type == "array" and not isinstance(value, list):
                    value = raw_value
                elif expected_type == "object" and not isinstance(value, dict):
                    value = raw_value
                arguments[key] = value
            native_calls.append({"name": name, "arguments": arguments})
        if native_calls:
            return native_calls

        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
            else:
                text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    calls = []
                    for call in parsed:
                        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                            continue
                        arguments = call.get("arguments", {})
                        if arguments is None:
                            arguments = {}
                        if isinstance(arguments, dict):
                            calls.append({"name": call["name"], "arguments": arguments})
                    return calls
            except Exception:
                pass
        return []

    def predict(self, sample, max_turns: int = 10):
        self.llm.reset_cache()

        # Create ToolEnv session
        id = sample.get("id")
        env = self.ToolEnv(self.mode, id)

        # Transcribe audio if provided or referenced in sample
        if "audio" in sample and self.asr is not None:
            audio_file = self.data_dir / sample["audio"]
            audio_text = self.asr.transcribe(audio_file)
        else:
            audio_text = sample.get("audio_text", None)
            assert audio_text is not None, f"IN {self.mode.upper()} MODE, PLEASE INITIALIZE AN ASR ENGINE FIRST!" 

        # Build Prompt messages
        messages = self.prompt_builder.create_prompt(sample, audio_text)
        tools = self.prompt_builder.build_tools(
            sample.get("context", {}).get("installed_apps", {})
        )

        all_tool_calls = []
        all_tool_results = []
        full_responses = []

        executed_sigs = set()
        executed_summary = []
        prev_turn_sigs = None

        # ReAct loop
        for turn in range(max_turns):
            response = self.llm.chat(messages, tools=tools)
            full_responses.append(response)
            tool_calls = self.parse_json_calls(
                response, self.prompt_builder.tools_by_name
            )
            
            if not tool_calls:
                messages.append({"role": "assistant", "content": response})
                break  

            # Sanitize tool arguments: strip empty strings/None and redundant parameters
            sanitized_calls = []
            for call in tool_calls:
                c_name = call.get("name", "")
                c_args = call.get("arguments", {})
                cleaned_args = {k: v for k, v in c_args.items() if v not in ("", None)}
                if c_name == "sms.send" and "contact_id" in cleaned_args and "phone_number" in cleaned_args:
                    cleaned_args.pop("phone_number", None)
                sanitized_calls.append({"name": c_name, "arguments": cleaned_args})
            tool_calls = sanitized_calls

            turn_sigs = [(c.get("name", ""), json.dumps(c.get("arguments", {}), sort_keys=True)) for c in tool_calls]
            if prev_turn_sigs is not None and turn_sigs == prev_turn_sigs:
                logger.warning(f"Phát hiện vòng lặp lặp lại chính xác ở lượt {turn + 1}. Dừng ReAct loop.")
                messages.append({"role": "assistant", "content": response})
                break

            # Validate tool calls with env.validate first
            validation_errors = []
            for call in tool_calls:
                name = call.get("name", "")
                args = call.get("arguments", {})
                val = env.validate(name, args)
                if not val.get("ok"):
                    validation_errors.append(
                        f"- Tool '{name}': {val.get('error')}"
                    )

            # If model returns invalid tool call, feed error back and try again
            if validation_errors:
                logger.warning(f"Lỗi kiểm tra tham số tool ở lượt {turn + 1}: {validation_errors}")
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": (
                        "Phát hiện lỗi cú pháp công cụ:\n"
                        + "\n".join(validation_errors)
                        + "\nVui lòng sửa function call theo schema công cụ đã cung cấp:"
                    ),
                })
                continue

            # Execute valid tool calls in env (skip duplicate calls)
            turn_results = []
            new_calls_executed = 0
            for call in tool_calls:
                name = call.get("name", "")
                args = call.get("arguments", {})
                sig = (name, json.dumps(args, sort_keys=True))
                kind = env.kind(name)

                if sig in executed_sigs:
                    logger.info(f"Bỏ qua công cụ đã thực thi trước đó: {name} {args}")
                    continue

                res = env.call(name, args)
                turn_results.append(res)
                executed_sigs.add(sig)
                executed_summary.append(f"- {name} {args} (loại: {kind})")
                all_tool_calls.append(call)
                all_tool_results.append(res)
                new_calls_executed += 1

            if new_calls_executed == 0:
                logger.info("Không có hành động mới nào cần thực thi trong lượt này. Dừng ReAct loop.")
                messages.append({"role": "assistant", "content": response})
                break

            if any(call["name"] == "sys.ask_missing_info" for call in tool_calls):
                break

            prev_turn_sigs = turn_sigs
            messages.append({"role": "assistant", "content": response})

            feedback_content = (
                f"Kết quả thực thi công cụ ở lượt vừa rồi:\n"
                f"{json.dumps(turn_results, ensure_ascii=False, indent=2)}\n\n"
                f"Các công cụ ĐÃ THỰC THI từ đầu đến giờ:\n"
                + "\n".join(executed_summary)
                + f"\n\nYêu cầu ban đầu của người dùng: \"{audio_text}\"\n\n"
                f"HƯỚNG DẪN XỬ LÝ LƯỢT TIẾP THEO:\n"
                f"1. Phân tích kết quả tra cứu vừa nhận được (kiểm tra điều kiện logic, ID đối tượng, trạng thái hiện tại).\n"
                f"2. Đánh giá logic điều kiện (nếu yêu cầu có dạng 'Nếu... thì...'):\n"
                f"   - Chỉ thực hiện hành động thuộc nhánh khớp với kết quả; không chạy nhánh đối lập. Điều kiện sai vẫn có thể có nhánh 'còn nếu/không thì'. Nếu nhánh đúng nói 'thôi/giữ nguyên' hoặc không có hành động, trả về [] ngay; tuyệt đối không gọi các write phụ thuộc điều kiện.\n"
                f"   - Nếu còn thiếu đối tượng/ID cho nhánh được chọn, gọi read tool tương ứng rồi mới write. Với yêu cầu 'dời/đổi/chuyển/hoãn' mục có sẵn, tìm mục hiện tại rồi update, không create. Với x-ref, dùng ID trả về, không dùng tên thay ID.\n"
                f"3. Phân rã yêu cầu thành các hành động còn thiếu, gồm các hành động vô điều kiện; đối chiếu với 'Các công cụ ĐÃ THỰC THI' để không lặp lại hành động đã xong. Gửi riêng từng hành động được yêu cầu; nhắn tin không thay thế cho cập nhật trạng thái app. Bỏ các tham số tùy chọn không cần và giữ nội dung tin nhắn ngắn, sát yêu cầu.\n"
                f"4. Khi cập nhật (update/set/modify), chỉ truyền ID và các trường giá trị mới cần thay đổi; khi đổi giờ, giữ nguyên thời lượng nếu không được yêu cầu đổi.\n"
                f"5. Khi các hành động của nhánh được chọn và các hành động vô điều kiện đã xong, trả về []. Không lặp lại cùng một hành động đã chạy thành công, kể cả khi đổi câu chữ ở tham số fuzzy."
            )
            messages.append({"role": "user", "content": feedback_content})

        return {
            "id": id,
            "transcribed_text": audio_text,
            "full_response": "\n".join(full_responses),
            "tool_calls": env.get_tool_calls(),
            "tool_results": all_tool_results,
            "state_hash": env.get_state_hash(),
            "messages": messages,
        }

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )

    mode = "train"
    parser = argparse.ArgumentParser(description="Run LifeMate on selected training samples")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--ids",
        nargs="+",
        default=[
            "train_1036",
            "train_0977",
            "train_1061",
            "train_0004",
            "train_0018",
            "train_0026",
        ],
    )
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    data_path = MODE[mode]["data_path"]
    wanted_ids = set(args.ids)
    samples = {}
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("id") in wanted_ids:
                    samples[item["id"]] = item
                if wanted_ids.issubset(samples):
                    break

    missing_ids = wanted_ids - samples.keys()
    if missing_ids:
        raise SystemExit(f"Samples not found in {data_path}: {', '.join(sorted(missing_ids))}")

    agent = LifeMateAgent(
        asr=None,
        model_id=args.model_id,
        mode=mode,
        max_new_tokens=args.max_new_tokens,
    )
    for sample_id in args.ids:
        sample = samples[sample_id]
        print(f"\nRunning {sample_id} with {args.model_id}...", flush=True)
        result = agent.predict(sample=sample, max_turns=args.max_turns)

        gold_calls = sample.get("label", {}).get("tool_calls", [])
        gold_env = agent.ToolEnv(mode, sample_id)
        for call in gold_calls:
            gold_env.call(call.get("name", ""), call.get("arguments", {}))
        gold_hash = gold_env.get_state_hash()

        print(f"\n{'=' * 80}\nSAMPLE {sample_id} | MODEL {args.model_id}")
        print(f"Request: {sample.get('audio_text', '')}")
        print("Generated responses:")
        print(result["full_response"])
        print("Predicted tool calls:")
        print(json.dumps(result["tool_calls"], ensure_ascii=False, indent=2))
        print("Gold tool calls:")
        print(json.dumps(gold_calls, ensure_ascii=False, indent=2))
        print(f"State hash match: {result['state_hash'] == gold_hash}", flush=True)
