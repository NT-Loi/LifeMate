from agent.tools import get_tool_env
from config import MODE
from models import get_asr, get_llm
from agent.prompt import LifeMatePromptBuilder

import json
import re

import logging
logger = logging.getLogger(__name__)

from typing import Any
from pathlib import Path

class LifeMateAgent:
    """LifeMate Agentic Assistant."""

    def __init__(
        self,
        asr: str | None = "Gipformer",
        llm: str | Any = "Qwen",
        mode: str = "train",
        device: str | None = None,
        data_dir: str | Path | None = None,
        tools_dir: str | Path | None = None,
        catalog_path: str | Path | None = None,
        **llm_kwargs
    ):
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
        if isinstance(llm, str):
            logger.info(f"Initialize LLM: {llm} (device={device})")
            if device:
                llm_kwargs["device"] = device
            self.llm = get_llm(engine=llm, **llm_kwargs)
        else:
            self.llm = llm

        # Load ToolEnv 
        self.ToolEnv = get_tool_env(self.mode)

        # Build Prompt
        self.prompt_builder = LifeMatePromptBuilder(self.catalog_path)

        logger.info(f"LifeMate Agent [{self.mode.upper()}] is ready!")

    @staticmethod
    def parse_json_calls(text: str):
        """Extract JSON tool calls list from model output text."""
        text = text.strip()
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
                    return [c for c in parsed if isinstance(c, dict) and "name" in c]
            except Exception:
                pass
        return []

    def predict(self, sample, max_turns: int = 10):
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

        all_tool_calls = []
        all_tool_results = []
        full_responses = []

        executed_sigs = set()
        executed_summary = []
        prev_turn_sigs = None

        # ReAct loop
        for turn in range(max_turns):
            response = self.llm.chat(messages)
            full_responses.append(response)
            tool_calls = self.parse_json_calls(response)
            
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
                        + "\nVui lòng sửa lại tên công cụ hoặc tham số và trả về mảng JSON hợp lệ:"
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

            prev_turn_sigs = turn_sigs
            messages.append({"role": "assistant", "content": response})

            feedback_content = (
                f"Kết quả thực thi công cụ ở lượt vừa rồi:\n"
                f"{json.dumps(turn_results, ensure_ascii=False, indent=2)}\n\n"
                f"Các công cụ ĐÃ THỰC THI từ đầu đến giờ:\n"
                + "\n".join(executed_summary)
                + f"\n\nYêu cầu ban đầu của người dùng: \"{audio_text}\"\n\n"
                f"HƯỚNG DẪN BƯỚC TIẾP THEO:\n"
                f"1. Phân tích kết quả các công cụ vừa nhận được ở trên (các ID thu được như event_id, contact_id, điều kiện thời tiết rain=true/false, v.v.).\n"
                f"2. Nếu điều kiện thỏa mãn và còn hành động chưa thực hiện: Hãy gọi đầy đủ các công cụ hành động đó (sử dụng các ID và thông tin đã thu thập được ở trên).\n"
                f"3. Đối chiếu danh sách các công cụ ĐÃ THỰC THI ở trên với yêu cầu ban đầu của người dùng: \"{audio_text}\". Nếu TẤT CẢ các hành động theo yêu cầu ĐÃ ĐƯỢC THỰC HIỆN XONG (hoặc điều kiện không xảy ra): BẮT BUỘC trả về mảng rỗng: [] để kết thúc.\n"
                f"4. TUYỆT ĐỐI KHÔNG thực hiện lại các hành động hoặc gọi lại các công cụ đã có trong danh sách ĐÃ THỰC THI ở trên."
            )
            messages.append({"role": "user", "content": feedback_content})

        return {
            "id": id,
            "transcribed_text": audio_text,
            "full_response": "\n".join(full_responses),
            "tool_calls": env.get_tool_calls(),
            "tool_results": all_tool_results,
            "final_state": env.get_state(),
            "state_hash": env.get_state_hash(),
            "messages": messages,
        }

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )

    mode = "train"
    data_path = MODE[mode]["data_path"]
    sample_id = "train_0001"
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("id") == sample_id:
                    sample = item
                    break

    # Initialize LifeMate agent 
    agent = LifeMateAgent(asr=None, llm="Qwen", mode="train")
    result = agent.predict(sample=sample)

    # Output multi-turn chat corresponding to messages
    print("=" * 80)
    print(f"LIFEMATE AGENT TEST CASE: {sample_id}")
    print("=" * 80)
    print("\n[TRANSCRIBE COMPARISON]")
    print(f"Agent Transcribe:\n{result['transcribed_text']}")
    print(f"\nSample Audio Text:\n{sample.get('audio_text', '')}")

    print("\n" + "=" * 80)
    print("MULTI-TURN CHAT CONVERSATION (MESSAGES)")
    print("=" * 80)

    turn_count = 0
    for idx, msg in enumerate(result.get("messages", [])):
        role = msg.get("role", "").lower()
        content = msg.get("content", "")

        if role == "system":
            print("\n" + "-" * 80)
            print("⚙️ [SYSTEM SPECIFICATIONS]")
            print("-" * 80)
            print(content)
        elif role == "user":
            if idx == 1:
                turn_count += 1
                print("\n" + "-" * 80)
                print(f"👤 [TURN {turn_count}] USER:")
                print("-" * 80)
                print(content)
            else:
                print("\n" + "-" * 80)
                print(f"🔧 [TOOL EXECUTION RESULT]:")
                print("-" * 80)
                print(content)
        elif role == "assistant":
            print("\n" + "-" * 80)
            print(f"🤖 [TURN {turn_count}] ASSISTANT:")
            print("-" * 80)
            print(content)
            turn_count += 1

    print("\n" + "=" * 80)
    print("[FINAL RESULTS]")
    print("=" * 80)
    print("Tool Calls:")
    print(json.dumps(result.get("tool_calls", []), ensure_ascii=False, indent=2))
    print("\nTool Results:")
    print(json.dumps(result.get("tool_results", []), ensure_ascii=False, indent=2))
    # print(f"\nFinal State:")
    # print(json.dumps(result.get('final_state'), ensure_ascii=False, indent=2))
    print(f"\nState Hash: {result.get('state_hash')}")
    print("=" * 80)