import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import Any
import unicodedata

import torch
import torch.multiprocessing as mp

from agent.lifemate import LifeMateAgent
from config import MODE


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


def process_single_sample(agent: LifeMateAgent, sample: dict, fuzzy_map: dict, mode: str) -> dict:
    """Xử lý và chấm điểm 1 sample."""
    sid = sample["id"]
    gold_calls = sample.get("label", {}).get("tool_calls", [])

    # Replay gold calls trên ToolEnv để lấy gold_state_hash chuẩn
    gold_env = agent.ToolEnv(mode, sid)
    for gc in gold_calls:
        gold_env.call(gc.get("name", ""), gc.get("arguments", {}))
    gold_hash = gold_env.get_state_hash()

    # Chạy agent dự đoán qua audio_text
    res = agent.predict(sample)
    sub_calls = res["tool_calls"]
    sub_hash = res["state_hash"]

    ev = evaluate_sample(sub_calls, sub_hash, gold_calls, gold_hash, fuzzy_map)
    ev["id"] = sid
    return ev


def _worker_process(
    gpu_id: int | str,
    sub_samples: list[dict],
    mode: str,
    llm_name: str,
    llm_kwargs: dict,
    out_queue: Any,
):
    """Worker tiến trình chạy trên 1 GPU riêng biệt."""
    device = f"cuda:{gpu_id}" if str(gpu_id).isdigit() else str(gpu_id)
    if device.startswith("cuda:") and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))

    agent = LifeMateAgent(asr=None, llm=llm_name, mode=mode, device=device, **llm_kwargs)
    fuzzy_map = get_fuzzy_props(agent.catalog_path)

    for i, sample in enumerate(sub_samples, 1):
        try:
            ev = process_single_sample(agent, sample, fuzzy_map, mode)
            status = "✅" if ev["state_match"] else "❌"
            print(
                f"[{device.upper()}] [{i:02d}/{len(sub_samples)}] {ev['id']} | "
                f"State: {status} | P: {ev['p']:.2f} | R: {ev['r']:.2f} | F1: {ev['f1']:.2f} | Score: {ev['score']:.2f}",
                flush=True,
            )
            out_queue.put(ev)
        except Exception as e:
            print(f"[{device.upper()}] Lỗi xử lý {sample.get('id')}: {e}", flush=True)
            out_queue.put({
                "id": sample.get("id"),
                "score": 0.0,
                "state_match": False,
                "p": 0.0,
                "r": 0.0,
                "f1": 0.0,
                "error": str(e),
            })


def print_summary(results: list[dict]):
    """In bảng tổng kết kết quả đánh giá."""
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


def run_evaluation(
    samples: list[dict],
    mode: str = "train",
    gpus: list[int] | str | None = None,
    llm_name: str = "Qwen",
    **llm_kwargs,
) -> list[dict]:
    """Hàm chạy đánh giá đa GPU hoặc đơn GPU."""
    if not samples:
        print("Không có mẫu nào để đánh giá.")
        return []

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

    # Chế độ Model-Parallelism sharded: 1 model chia đều các GPU với device_map="auto"
    if gpus == "auto":
        print(f"\n🚀 Đánh giá {len(samples)} mẫu với 1 Agent sharded trên toàn bộ GPU (device_map='auto')...")
        agent = LifeMateAgent(asr=None, llm=llm_name, mode=mode, device="auto", **llm_kwargs)
        fuzzy_map = get_fuzzy_props(agent.catalog_path)
        results = []
        for i, sample in enumerate(samples, 1):
            ev = process_single_sample(agent, sample, fuzzy_map, mode)
            results.append(ev)
            status = "✅" if ev["state_match"] else "❌"
            print(f"[{i:02d}/{len(samples)}] {ev['id']} | State: {status} | P: {ev['p']:.2f} | R: {ev['r']:.2f} | F1: {ev['f1']:.2f} | Score: {ev['score']:.2f}")
        print_summary(results)
        return results

    # Chế độ Data-Parallelism đa GPU: Mỗi GPU chạy 1 worker độc lập (Tối đa thông lượng trên T4x2)
    if isinstance(gpus, list) and len(gpus) > 1 and len(samples) > 1:
        print(f"\n🚀 Đánh giá {len(samples)} mẫu song song trên {len(gpus)} GPUs: {gpus}...")
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = []

        for w_idx, gpu_id in enumerate(gpus):
            sub_samples = [samples[i] for i in range(w_idx, len(samples), len(gpus))]
            if not sub_samples:
                continue
            p = ctx.Process(
                target=_worker_process,
                args=(gpu_id, sub_samples, mode, llm_name, llm_kwargs, queue),
            )
            p.start()
            processes.append(p)

        results = []
        for _ in range(len(samples)):
            results.append(queue.get())

        for p in processes:
            p.join()

        # Sắp xếp lại kết quả theo thứ tự ID ban đầu
        sample_id_order = {s["id"]: idx for idx, s in enumerate(samples)}
        results.sort(key=lambda r: sample_id_order.get(r["id"], 999999))
        print_summary(results)
        return results

    # Chế độ đơn GPU hoặc CPU
    dev_str = f"cuda:{gpus[0]}" if isinstance(gpus, list) and str(gpus[0]).isdigit() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Đánh giá {len(samples)} mẫu trên thiết bị [{dev_str.upper()}]...")
    agent = LifeMateAgent(asr=None, llm=llm_name, mode=mode, device=dev_str, **llm_kwargs)
    fuzzy_map = get_fuzzy_props(agent.catalog_path)
    results = []

    for i, sample in enumerate(samples, 1):
        ev = process_single_sample(agent, sample, fuzzy_map, mode)
        results.append(ev)
        status = "✅" if ev["state_match"] else "❌"
        print(f"[{i:02d}/{len(samples)}] {ev['id']} | State: {status} | P: {ev['p']:.2f} | R: {ev['r']:.2f} | F1: {ev['f1']:.2f} | Score: {ev['score']:.2f}")

    print_summary(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Đánh giá LifeMate Agent (ZAC 2026)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "public_test"], help="Tập dữ liệu đánh giá")
    parser.add_argument("--num-samples", type=int, default=10, help="Số lượng mẫu (-1 để chạy hết, mặc định: 10)")
    parser.add_argument("--id", type=str, default=None, help="Đánh giá duy nhất 1 sample ID (ví dụ: train_0001)")
    parser.add_argument("--gpus", type=str, default=None, help="Danh sách GPU (ví dụ: '0,1' hoặc 'auto' hoặc '0')")
    parser.add_argument("--model-id", type=str, default=None, help="Hugging Face Model ID tùy chọn")
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

    llm_kwargs = {}
    if args.model_id:
        llm_kwargs["model_id"] = args.model_id

    run_evaluation(
        samples=samples,
        mode=mode,
        gpus=args.gpus,
        llm_name="Qwen",
        **llm_kwargs,
    )


if __name__ == "__main__":
    main()
