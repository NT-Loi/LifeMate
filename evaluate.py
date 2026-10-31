import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import gc
import json
import os
from pathlib import Path
from queue import Empty
import re
import shutil
import sys
import time
from typing import Any
import unicodedata

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from agent.lifemate import LifeMateAgent
from config import MODE

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


class _Tee:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    @property
    def encoding(self):
        return getattr(self.terminal, "encoding", "utf-8")

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


@contextmanager
def _capture_output(log_path: Path | str):
    """Capture output while keeping Python log records away from tqdm's terminal stream."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        file_handler = logging.StreamHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s"
        ))
        root_logger = logging.getLogger()
        old_handlers = root_logger.handlers[:]
        for handler in old_handlers:
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)
        sys.stdout = _Tee(old_stdout, log_file)
        sys.stderr = _Tee(old_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            root_logger.removeHandler(file_handler)
            for handler in old_handlers:
                root_logger.addHandler(handler)
            file_handler.close()


def _error_result(sample: dict, error: Exception) -> dict:
    result = {
        "id": sample.get("id"),
        "audio_text": sample.get("audio_text", ""),
        "score": 0.0,
        "state_match": False,
        "p": 0.0,
        "r": 0.0,
        "f1": 0.0,
        "sub_calls": [],
        "gold_calls": sample.get("label", {}).get("tool_calls", []),
        "sub_hash": "",
        "gold_hash": "",
        "messages": [],
        "tool_results": [],
        "error": f"{type(error).__name__}: {error}",
    }
    if hasattr(error, "elapsed_seconds"):
        result["elapsed_seconds"] = error.elapsed_seconds
    return result


def _recover_after_error(
    agent: LifeMateAgent | None,
    error: Exception,
    device: str | None = None,
):
    if agent is not None:
        agent.llm.reset_cache()
    if not (isinstance(error, MemoryError) or "out of memory" in str(error).lower()):
        return

    gc.collect()
    if torch.cuda.is_available():
        device = device or getattr(getattr(agent, "llm", None), "device", None)
        if device == "auto" or device is None:
            gpu_ids = range(torch.cuda.device_count())
        elif str(device).startswith("cuda:"):
            gpu_ids = [int(str(device).split(":", 1)[1])]
        elif str(device).startswith("cuda"):
            gpu_ids = [torch.cuda.current_device()]
        else:
            gpu_ids = []

        for gpu_id in gpu_ids:
            try:
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
                    free, total = torch.cuda.mem_get_info(gpu_id)
                    logger.error(
                        "GPU %d memory after OOM recovery: allocated=%.2f GiB, reserved=%.2f GiB, free=%.2f/%.2f GiB",
                        gpu_id,
                        torch.cuda.memory_allocated(gpu_id) / 1024**3,
                        torch.cuda.memory_reserved(gpu_id) / 1024**3,
                        free / 1024**3,
                        total / 1024**3,
                    )
            except Exception:
                logger.exception("Could not recover or read CUDA memory stats for GPU %d.", gpu_id)


def get_fuzzy_props(catalog_path: str | Path) -> dict[str, set[str]]:
    """Trích xuất danh sách các thuộc tính fuzzy (x-match: fuzzy) từ catalog."""
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    fuzzy = {}
    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            props = tool.get("parameters", {}).get("properties", {})
            f_keys = {k for k, v in props.items() if v.get("x-match") == "fuzzy"}
            if f_keys:
                fuzzy[tool["name"]] = f_keys
    return fuzzy


def token_f1(s1: Any, s2: Any) -> float:
    """Tính Token-F1 sau khi Unicode NFC và lowercased."""
    t1 = re.findall(r"\w+", unicodedata.normalize("NFC", str(s1 or "")).lower())
    t2 = re.findall(r"\w+", unicodedata.normalize("NFC", str(s2 or "")).lower())
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    overlap = sum((Counter(t1) & Counter(t2)).values())
    return (2.0 * overlap / (len(t1) + len(t2))) if overlap else 0.0


def match_call(c1: dict, c2: dict, fuzzy_map: dict[str, set[str]]) -> bool:
    """Kiểm tra 2 tool calls có khớp nhau theo quy tắc ZAC 2026."""
    if c1.get("name") != c2.get("name"):
        return False
    name = c1.get("name", "")
    f_keys = fuzzy_map.get(name, set())
    a1 = {k: v for k, v in (c1.get("arguments") or {}).items() if v is not None}
    a2 = {k: v for k, v in (c2.get("arguments") or {}).items() if v is not None}

    for k in set(a1) | set(a2):
        if k in f_keys:
            if token_f1(a1.get(k), a2.get(k)) <= 0.6:
                return False
        elif a1.get(k) != a2.get(k):
            return False
    return True


def evaluate_sample(sub_calls: list, sub_hash: str, gold_calls: list, gold_hash: str, fuzzy_map: dict) -> dict:
    """Đánh giá 1 mẫu dự đoán theo đúng thể lệ và công thức ZAC 2026."""
    state_match = (sub_hash == gold_hash)
    ns, ng = len(sub_calls), len(gold_calls)

    # Trường hợp mẫu out-of-scope (|G| == 0)
    if ng == 0:
        score = 1.0 if ns == 0 else 0.0
        return {"score": score, "state_match": state_match, "p": score, "r": score, "f1": score}

    # Tìm Maximum Bipartite Matching (Kuhn)
    adj = [[j for j, gc in enumerate(gold_calls) if match_call(sc, gc, fuzzy_map)] for sc in sub_calls]
    match = [-1] * ng

    def dfs(u: int, vis: list[bool]) -> bool:
        for v in adj[u]:
            if not vis[v]:
                vis[v] = True
                if match[v] < 0 or dfs(match[v], vis):
                    match[v] = u
                    return True
        return False

    M = sum(dfs(u, [False] * ng) for u in range(ns))
    p = M / ns if ns else 0.0
    r = M / ng
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    score = f1 if state_match else 0.0

    return {"score": score, "state_match": state_match, "p": p, "r": r, "f1": f1, "M": M, "G": ng, "S": ns}


def format_conversation(ev: dict) -> str:
    """Định dạng toàn bộ lịch sử hội thoại của 1 sample thành dạng văn bản dễ đọc."""
    lines = []
    sid = ev.get("id", "UNKNOWN")
    audio_text = ev.get("audio_text", "")
    score = ev.get("score", 0.0)
    state_match = ev.get("state_match", False)

    lines.append("=" * 80)
    lines.append(f"SAMPLE ID: {sid} | SCORE: {score:.4f} | STATE MATCH: {'✅' if state_match else '❌'}")
    lines.append(f"Yêu cầu người dùng (Audio Text): \"{audio_text}\"")
    lines.append("=" * 80)

    messages = ev.get("messages", [])
    if not messages:
        if ev.get("error"):
            lines.append(f"\n❌ LỖI TRONG QUÁ TRÌNH THỰC THI: {ev.get('error')}")
        else:
            lines.append("\nKhông có lịch sử hội thoại.")
    else:
        turn_count = 0
        for idx, msg in enumerate(messages):
            role = msg.get("role", "").lower()
            content = msg.get("content", "")

            if role == "system":
                lines.append("\n" + "-" * 80)
                lines.append("⚙️ [SYSTEM PROMPT]")
                lines.append("-" * 80)
                lines.append(content)
            elif role == "user":
                if idx == 1:
                    turn_count += 1
                    lines.append("\n" + "-" * 80)
                    lines.append(f"👤 [TURN {turn_count}] USER (User Prompt):")
                    lines.append("-" * 80)
                    lines.append(content)
                else:
                    lines.append("\n" + "-" * 80)
                    lines.append("🔧 [TOOL EXECUTION FEEDBACK]:")
                    lines.append("-" * 80)
                    lines.append(content)
            elif role == "assistant":
                lines.append("\n" + "-" * 80)
                lines.append(f"🤖 [TURN {turn_count}] ASSISTANT (Model Response):")
                lines.append("-" * 80)
                lines.append(content)
                turn_count += 1

    lines.append("\n" + "-" * 80)
    lines.append("📊 [KẾT QUẢ ĐÁNH GIÁ (EVALUATION SUMMARY)]")
    lines.append("-" * 80)
    lines.append(f"• State Match:          {'✅' if state_match else '❌'}")
    lines.append(f"• Predicted State Hash: {ev.get('sub_hash', '')}")
    lines.append(f"• Gold State Hash:      {ev.get('gold_hash', '')}")
    lines.append(f"• Tool Precision:       {ev.get('p', 0.0):.4f}")
    lines.append(f"• Tool Recall:          {ev.get('r', 0.0):.4f}")
    lines.append(f"• Tool F1:              {ev.get('f1', 0.0):.4f}")
    lines.append(f"• Final Score:          {score:.4f}")
    lines.append("\nPredicted Tool Calls:")
    lines.append(json.dumps(ev.get("sub_calls", []), ensure_ascii=False, indent=2))
    lines.append("\nGold Tool Calls:")
    lines.append(json.dumps(ev.get("gold_calls", []), ensure_ascii=False, indent=2))
    lines.append("\n" + "=" * 80 + "\n\n")

    return "\n".join(lines)


def _process_single_sample(agent: LifeMateAgent, sample: dict, fuzzy_map: dict, mode: str) -> dict:
    """Xử lý và chấm điểm 1 sample, lưu trữ cả thông tin chạy, lịch sử hội thoại và kết quả điểm."""
    sid = sample["id"]
    gold_calls = sample.get("label", {}).get("tool_calls", [])

    # Replay gold calls trên ToolEnv để lấy gold_state_hash chuẩn
    gold_env = agent.ToolEnv(mode, sid)
    for gc in gold_calls:
        gold_env.call(gc.get("name", ""), gc.get("arguments", {}))
    gold_hash = gold_env.get_state_hash()

    # Chạy agent dự đoán qua audio_text
    res = agent.predict(sample)
    sub_calls = res.get("tool_calls", [])
    sub_hash = res.get("state_hash", "")

    ev = evaluate_sample(sub_calls, sub_hash, gold_calls, gold_hash, fuzzy_map)
    ev["id"] = sid
    ev["audio_text"] = sample.get("audio_text", "")
    ev["sub_calls"] = sub_calls
    ev["gold_calls"] = gold_calls
    ev["sub_hash"] = sub_hash
    ev["gold_hash"] = gold_hash
    ev["messages"] = res.get("messages", [])
    ev["tool_results"] = res.get("tool_results", [])
    ev["full_response"] = res.get("full_response", "")
    return ev


def process_single_sample(agent: LifeMateAgent, sample: dict, fuzzy_map: dict, mode: str) -> dict:
    started = time.perf_counter()
    try:
        result = _process_single_sample(agent, sample, fuzzy_map, mode)
    except Exception as error:
        elapsed = round(time.perf_counter() - started, 3)
        error.elapsed_seconds = elapsed
        logger.info("Sample %s failed after %.2f seconds.", sample.get("id"), elapsed)
        raise

    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    logger.info("Sample %s completed in %.2f seconds.", sample.get("id"), result["elapsed_seconds"])
    return result


def _worker_process(
    gpu_id: int | str,
    sub_samples: list[dict],
    mode: str,
    model_id: str | None,
    llm_kwargs: dict,
    out_queue: Any,
    log_path: str,
):
    """Worker tiến trình chạy trên 1 GPU riêng biệt."""
    device = f"cuda:{gpu_id}" if str(gpu_id).isdigit() else str(gpu_id)
    with _capture_output(log_path):
        logger.info("Starting worker on %s for %d samples.", device, len(sub_samples))
        try:
            if device.startswith("cuda:") and torch.cuda.is_available():
                torch.cuda.set_device(int(gpu_id))
            agent = LifeMateAgent(asr=None, model_id=model_id, mode=mode, device=device, **llm_kwargs)
            fuzzy_map = get_fuzzy_props(agent.catalog_path)
        except Exception as e:
            logger.exception("Worker %s failed while initializing its model.", device)
            _recover_after_error(None, e, device=device)
            for sample in sub_samples:
                out_queue.put(_error_result(sample, e))
            return

        for sample in sub_samples:
            try:
                ev = process_single_sample(agent, sample, fuzzy_map, mode)
                out_queue.put(ev)
            except Exception as e:
                logger.exception("Worker %s failed on sample %s.", device, sample.get("id"))
                _recover_after_error(agent, e, device=device)
                out_queue.put(_error_result(sample, e))


def save_evaluation_results(
    results: list[dict],
    output_dir: Path | str,
    mode: str,
    model_id: str | None = None,
    gpus: Any = None,
) -> dict:
    """Lưu kết quả dự đoán (submission format), lịch sử hội thoại và chi tiết đánh giá."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lưu file submission format (predictions.jsonl)
    pred_path = output_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in results:
            pred_item = {
                "id": r["id"],
                "tool_calls": r.get("sub_calls", []),
                "state_hash": r.get("sub_hash", ""),
            }
            f.write(json.dumps(pred_item, ensure_ascii=False) + "\n")

    # 2. Lưu chi tiết đánh giá từng sample (eval_details.jsonl) kèm lịch sử messages
    details_path = output_dir / "eval_details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 3. Lưu toàn bộ lịch sử hội thoại từng sample dạng text dễ đọc (conversations.txt)
    conv_path = output_dir / "conversations.txt"
    with open(conv_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(format_conversation(r))

    # 4. Lưu tổng kết chỉ số (eval_summary.json)
    n = len(results)
    failed_sample_ids = [r.get("id") for r in results if r.get("error")]
    sample_times = [r["elapsed_seconds"] for r in results if "elapsed_seconds" in r]
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "num_samples": n,
        "error_count": len(failed_sample_ids),
        "failed_sample_ids": failed_sample_ids,
        "mean_elapsed_seconds": round(sum(sample_times) / len(sample_times), 3) if sample_times else None,
        "max_elapsed_seconds": round(max(sample_times), 3) if sample_times else None,
        "samples_over_30_seconds": sum(seconds > 30 for seconds in sample_times),
        "model_id": model_id or "default",
        "gpus": gpus,
        "state_match_acc": round(sum(1.0 for r in results if r.get("state_match")) / n, 4) if n else 0.0,
        "tool_precision": round(sum(r.get("p", 0.0) for r in results) / n, 4) if n else 0.0,
        "tool_recall": round(sum(r.get("r", 0.0) for r in results) / n, 4) if n else 0.0,
        "tool_f1": round(sum(r.get("f1", 0.0) for r in results) / n, 4) if n else 0.0,
        "mean_score": round(sum(r.get("score", 0.0) for r in results) / n, 4) if n else 0.0,
        "output_files": {
            "predictions": str(pred_path),
            "eval_details": str(details_path),
            "conversations": str(conv_path),
            "eval_summary": str(output_dir / "eval_summary.json"),
            "run_log": str(output_dir / "run.log"),
            "worker_logs": [str(path) for path in sorted(output_dir.glob("worker_*.log"))],
        }
    }
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 5. Sao chép bản copy vào eval_results/latest để tiện tham chiếu
    try:
        latest_dir = output_dir.parent / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pred_path, latest_dir / "predictions.jsonl")
        shutil.copyfile(details_path, latest_dir / "eval_details.jsonl")
        shutil.copyfile(conv_path, latest_dir / "conversations.txt")
        shutil.copyfile(summary_path, latest_dir / "eval_summary.json")
        for stale_log in latest_dir.glob("worker_*.log"):
            stale_log.unlink()
        for worker_log in output_dir.glob("worker_*.log"):
            shutil.copyfile(worker_log, latest_dir / worker_log.name)
    except Exception:
        logger.exception("Could not refresh eval_results/latest.")

    return summary


def print_summary(results: list[dict], output_dir: Path | str | None = None):
    """In bảng tổng kết kết quả đánh giá và đường dẫn file đã lưu."""
    n = len(results)
    if n == 0:
        print("Không có kết quả đánh giá nào.")
        return

    avg_score = sum(r["score"] for r in results) / n
    avg_sm = sum(1.0 for r in results if r["state_match"]) / n
    avg_p = sum(r["p"] for r in results) / n
    avg_r = sum(r["r"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n

    print("\n" + "=" * 60)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN':^60}")
    print("=" * 60)
    print(f"Số lượng mẫu:        {n}")
    print(f"State Match Acc:     {avg_sm * 100:.2f}%")
    print(f"Tool Precision:      {avg_p * 100:.2f}%")
    print(f"Tool Recall:         {avg_r * 100:.2f}%")
    print(f"Tool F1:             {avg_f1 * 100:.2f}%")
    print(f"⭐ MEAN SCORE:       {avg_score * 100:.2f}%")
    print("=" * 60)

    if output_dir:
        out_p = Path(output_dir)
        print("💾 Kết quả chạy & đánh giá đã được lưu tại:")
        print(f"   📂 {out_p}")
        print(f"   ├── 📄 predictions.jsonl   (Dự đoán format nộp bài ZAC 2026)")
        print(f"   ├── 📄 eval_details.jsonl  (Chi tiết từng sample: tool calls, hash, điểm, messages)")
        print(f"   ├── 💬 conversations.txt   (Lịch sử toàn bộ hội thoại từng sample - dễ đọc)")
        print(f"   └── 📊 eval_summary.json   (Chỉ số tổng kết)")
        print("=" * 60)
        if n == 1:
            print(f"💡 Xem chi tiết hội thoại: {out_p / 'conversations.txt'}")
            print("=" * 60)


def _run_evaluation(
    samples: list[dict],
    mode: str = "train",
    gpus: list[int] | str | None = None,
    output_dir: Path | str | None = None,
    model_id: str | None = None,
    **llm_kwargs,
) -> list[dict]:
    """Hàm chạy đánh giá đa GPU hoặc đơn GPU, kèm progress bar và lưu file."""
    if not samples:
        print("Không có mẫu nào để đánh giá.")
        return []

    # Khởi tạo thư mục output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("eval_results") / f"{mode}_{timestamp}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tự động xác định GPUs nếu không chỉ định
    if gpus is None:
        if torch.cuda.is_available():
            num_cuda = torch.cuda.device_count()
            gpus = list(range(num_cuda)) if num_cuda > 1 else [0]
        else:
            gpus = ["cpu"]
    elif isinstance(gpus, str):
        if gpus.lower() == "auto":
            gpus = "auto"
        else:
            gpus = [int(x.strip()) for x in gpus.split(",") if x.strip().isdigit()]

    # Mở file ghi tăng dần (real-time stream) phòng trường hợp tiến trình bị gián đoạn
    raw_pred_file = open(output_dir / "predictions.jsonl", "w", encoding="utf-8")
    raw_details_file = open(output_dir / "eval_details.jsonl", "w", encoding="utf-8")
    raw_conv_file = open(output_dir / "conversations.txt", "w", encoding="utf-8")

    def _record_sample(ev: dict):
        if ev.get("error"):
            logger.error("Sample %s failed: %s", ev.get("id"), ev["error"])
        pred_item = {
            "id": ev["id"],
            "tool_calls": ev.get("sub_calls", []),
            "state_hash": ev.get("sub_hash", ""),
        }
        raw_pred_file.write(json.dumps(pred_item, ensure_ascii=False) + "\n")
        raw_pred_file.flush()
        raw_details_file.write(json.dumps(ev, ensure_ascii=False) + "\n")
        raw_details_file.flush()
        raw_conv_file.write(format_conversation(ev))
        raw_conv_file.flush()

    # 1. Chế độ Model-Parallelism sharded: 1 model chia đều các GPU với device_map="auto"
    if gpus == "auto":
        print(f"\n🚀 Đánh giá {len(samples)} mẫu với 1 Agent sharded trên toàn bộ GPU (device_map='auto')...")
        agent = LifeMateAgent(asr=None, model_id=model_id, mode=mode, device="auto", **llm_kwargs)
        fuzzy_map = get_fuzzy_props(agent.catalog_path)
        results = []

        with tqdm(total=len(samples), desc="Evaluating [AUTO SHARD]", unit="sample") as pbar:
            for sample in samples:
                try:
                    ev = process_single_sample(agent, sample, fuzzy_map, mode)
                except Exception as e:
                    logger.exception("Failed to evaluate sample %s.", sample.get("id"))
                    _recover_after_error(agent, e)
                    ev = _error_result(sample, e)
                results.append(ev)
                _record_sample(ev)

                run_avg = sum(r["score"] for r in results) / len(results)
                run_sm = sum(1.0 for r in results if r.get("state_match")) / len(results)
                pbar.set_postfix({"ID": ev["id"], "State": "✅" if ev.get("state_match") else "❌", "Score": f"{ev.get('score', 0):.2f}", "Avg": f"{run_avg:.2f}", "SM_Acc": f"{run_sm*100:.0f}%"})
                pbar.update(1)

        raw_pred_file.close()
        raw_details_file.close()
        raw_conv_file.close()
        save_evaluation_results(results, output_dir, mode, model_id, gpus)
        print_summary(results, output_dir)
        return results

    # 2. Chế độ Data-Parallelism đa GPU: Mỗi GPU chạy 1 worker độc lập (Tối đa thông lượng trên T4x2)
    if isinstance(gpus, list) and len(gpus) > 1 and len(samples) > 1:
        print(f"\n🚀 Đánh giá song song {len(samples)} mẫu trên {len(gpus)} GPUs: {gpus}...")
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = []

        for w_idx, gpu_id in enumerate(gpus):
            sub_samples = [samples[i] for i in range(w_idx, len(samples), len(gpus))]
            if not sub_samples:
                continue
            p = ctx.Process(
                target=_worker_process,
                args=(
                    gpu_id,
                    sub_samples,
                    mode,
                    model_id,
                    llm_kwargs,
                    queue,
                    str(output_dir / f"worker_{gpu_id}.log"),
                ),
            )
            p.start()
            processes.append(p)

        results = []
        with tqdm(total=len(samples), desc=f"Evaluating (GPUs {gpus})", unit="sample") as pbar:
            while len(results) < len(samples):
                try:
                    ev = queue.get(timeout=5)
                except Empty:
                    if all(not process.is_alive() for process in processes):
                        completed_ids = {result.get("id") for result in results}
                        missing_ids = [sample["id"] for sample in samples if sample["id"] not in completed_ids]
                        exit_codes = [process.exitcode for process in processes]
                        logger.error(
                            "All GPU workers exited before returning %d samples; missing=%s exit_codes=%s",
                            len(missing_ids),
                            missing_ids,
                            exit_codes,
                        )
                        raise RuntimeError(
                            f"GPU worker exited before returning samples {missing_ids}; exit codes: {exit_codes}"
                        )
                    continue
                results.append(ev)
                _record_sample(ev)

                run_avg = sum(r["score"] for r in results) / len(results)
                run_sm = sum(1.0 for r in results if r.get("state_match")) / len(results)
                pbar.set_postfix({
                    "ID": ev["id"],
                    "State": "✅" if ev.get("state_match") else "❌",
                    "Score": f"{ev.get('score', 0):.2f}",
                    "Avg": f"{run_avg:.2f}",
                    "SM": f"{run_sm*100:.0f}%",
                })
                pbar.update(1)

        for p in processes:
            p.join()
        failed_workers = [p.exitcode for p in processes if p.exitcode not in (0, None)]
        if failed_workers:
            logger.error("GPU workers exited with non-zero codes after returning results: %s", failed_workers)

        raw_pred_file.close()
        raw_details_file.close()
        raw_conv_file.close()

        # Sắp xếp lại kết quả theo thứ tự ID ban đầu
        sample_id_order = {s["id"]: idx for idx, s in enumerate(samples)}
        results.sort(key=lambda r: sample_id_order.get(r["id"], 999999))

        save_evaluation_results(results, output_dir, mode, model_id, gpus)
        print_summary(results, output_dir)
        return results

    # 3. Chế độ đơn GPU hoặc CPU
    dev_str = f"cuda:{gpus[0]}" if isinstance(gpus, list) and str(gpus[0]).isdigit() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Đánh giá {len(samples)} mẫu trên thiết bị [{dev_str.upper()}]...")
    agent = LifeMateAgent(asr=None, model_id=model_id, mode=mode, device=dev_str, **llm_kwargs)
    fuzzy_map = get_fuzzy_props(agent.catalog_path)
    results = []

    with tqdm(total=len(samples), desc=f"Evaluating [{dev_str.upper()}]", unit="sample") as pbar:
        for sample in samples:
            try:
                ev = process_single_sample(agent, sample, fuzzy_map, mode)
            except Exception as e:
                logger.exception("Failed to evaluate sample %s.", sample.get("id"))
                _recover_after_error(agent, e)
                ev = _error_result(sample, e)
            results.append(ev)
            _record_sample(ev)

            run_avg = sum(r["score"] for r in results) / len(results)
            run_sm = sum(1.0 for r in results if r.get("state_match")) / len(results)
            pbar.set_postfix({"ID": ev["id"], "State": "✅" if ev.get("state_match") else "❌", "Score": f"{ev.get('score', 0):.2f}", "Avg": f"{run_avg:.2f}", "SM": f"{run_sm*100:.0f}%"})
            pbar.update(1)

    raw_pred_file.close()
    raw_details_file.close()
    raw_conv_file.close()

    save_evaluation_results(results, output_dir, mode, model_id, gpus)
    print_summary(results, output_dir)
    return results


def run_evaluation(
    samples: list[dict],
    mode: str = "train",
    gpus: list[int] | str | None = None,
    output_dir: Path | str | None = None,
    model_id: str | None = None,
    **llm_kwargs,
) -> list[dict]:
    if not samples:
        print("Không có mẫu nào để đánh giá.")
        return []

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("eval_results") / f"{mode}_{timestamp}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if llm_kwargs.get("max_new_tokens") is None:
        llm_kwargs["max_new_tokens"] = 512

    run_log = output_dir / "run.log"
    try:
        with _capture_output(run_log):
            logger.info(
                "Evaluation started: mode=%s samples=%d model=%s max_new_tokens=%s",
                mode,
                len(samples),
                model_id or "default",
                llm_kwargs["max_new_tokens"],
            )
            return _run_evaluation(
                samples=samples,
                mode=mode,
                gpus=gpus,
                output_dir=output_dir,
                model_id=model_id,
                **llm_kwargs,
            )
    finally:
        if run_log.is_file():
            latest_dir = output_dir.parent / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_log, latest_dir / "run.log")


def main():
    parser = argparse.ArgumentParser(description="Đánh giá LifeMate Agent (ZAC 2026)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "public_test"], help="Tập dữ liệu đánh giá")
    parser.add_argument("--num-samples", type=int, default=10, help="Số lượng mẫu (-1 để chạy hết, mặc định: 10)")
    parser.add_argument("--id", type=str, default=None, help="Đánh giá duy nhất 1 sample ID (ví dụ: train_0001)")
    parser.add_argument("--gpus", type=str, default=None, help="Danh sách GPU (ví dụ: '0,1' hoặc 'auto' hoặc '0')")
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục lưu kết quả (mặc định: eval_results/<mode>_<timestamp>)")
    parser.add_argument("--model-id", type=str, default=None, help="Hugging Face Model ID tùy chọn")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Giới hạn token mới mỗi lượt để giảm VRAM (mặc định: 512)")
    args = parser.parse_args()

    mode = args.mode
    data_path = MODE[mode]["data_path"]
    assert Path(data_path).is_file(), f"Không tìm thấy file dữ liệu: {data_path}"

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if args.id and item.get("id") != args.id:
                    continue
                samples.append(item)
                if not args.id and args.num_samples > 0 and len(samples) >= args.num_samples:
                    break

    run_evaluation(
        samples=samples,
        mode=mode,
        gpus=args.gpus,
        output_dir=args.output_dir,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
