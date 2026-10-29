#!/usr/bin/env python3
"""LifeMate CLI - Multi-modal Agentic Voice Assistant.

Track LifeMate - Zalo AI Challenge 2026.
Hỗ trợ chạy suy luận và đánh giá trên tập 'train' hoặc 'public_test'
sử dụng bộ công cụ tương ứng trong agent/tools/<mode>.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from agent.lifemate import LifeMateAgent
from config import TRAIN_DATA_DIR, PUBLIC_TEST_DATA_DIR, MODE, get_mode_config
from evaluate.metrics import evaluate_sample, load_fuzzy_props_map


def parse_args():
  parser = argparse.ArgumentParser(
      description="LifeMate Agent Flow Runner (Train / Public Test)"
  )
  parser.add_argument(
      "--mode",
      type=str,
      default="public_test",
      choices=["train", "public_test"],
      help="Chế độ chạy: 'train' hoặc 'public_test' (mặc định: %(default)s)",
  )
  parser.add_argument(
      "--num-samples",
      type=int,
      default=-1,
      help="Số lượng mẫu chạy (-1 để chạy toàn bộ tập dữ liệu, mặc định: -1)",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed khi lấy mẫu ngẫu nhiên (mặc định: 42)",
  )
  parser.add_argument(
      "--asr",
      type=str,
      default="gipformer",
      choices=["gipformer", "phoasr", "none"],
      help="Mô hình nhận dạng giọng nói ASR (mặc định: gipformer)",
  )
  parser.add_argument(
      "--llm",
      type=str,
      default="Qwen/Qwen3.5-4B",
      help="Mô hình ngôn ngữ lớn LLM (mặc định: Qwen/Qwen3.5-4B)",
  )
  parser.add_argument(
      "--tools-dir",
      type=str,
      default=None,
      help="Đường dẫn thư mục tools tùy chỉnh (mặc định: agent/tools/<mode>)",
  )
  parser.add_argument(
      "--data-dir",
      type=str,
      default=None,
      help="Đường dẫn trực tiếp tới thư mục dữ liệu mode (ví dụ trên Drive: /content/drive/MyDrive/.../public_test)",
  )
  parser.add_argument(
      "--data-root",
      type=str,
      default=None,
      help="Đường dẫn thư mục gốc dữ liệu (mặc định: data/train+public_test/train+public_test)",
  )
  parser.add_argument(
      "--output",
      type=str,
      default="submission.jsonl",
      help="File jsonl lưu kết quả đầu ra (mặc định: submission.jsonl)",
  )
  parser.add_argument(
      "--eval",
      action="store_true",
      help="Đánh giá điểm số so với nhãn vàng (chỉ áp dụng khi mode='train')",
  )
  parser.add_argument(
      "--device",
      type=str,
      default=None,
      help="Thiết bị chạy LLM ('cpu', 'cuda', hoặc None để auto)",
  )
  parser.add_argument(
      "--load-in-4bit",
      action="store_true",
      default=True,
      help="Sử dụng 4-bit NF4 quantization để chống tràn VRAM trên GPU (mặc định: True)",
  )
  parser.add_argument(
      "--no-4bit",
      dest="load_in_4bit",
      action="store_false",
      help="Tắt 4-bit quantization, nạp mô hình ở độ chính xác float16/bfloat16",
  )
  parser.add_argument(
      "--num-shards",
      type=int,
      default=1,
      help="Tổng số shard để chia dữ liệu khi chạy song song đa GPU (mặc định: 1)",
  )
  parser.add_argument(
      "--shard-id",
      type=int,
      default=0,
      help="ID của shard hiện tại (0 <= shard-id < num-shards, mặc định: 0)",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  mode = args.mode.lower()

  # Xác định thư mục dữ liệu và file jsonl
  if args.data_dir:
    split_dir = Path(args.data_dir).resolve()
  elif args.data_root:
    root_p = Path(args.data_root).resolve()
    split_dir = root_p / mode if (root_p / mode).exists() else root_p
  else:
    cfg_dir = Path(TRAIN_DATA_DIR if mode == "train" else PUBLIC_TEST_DATA_DIR)
    if not cfg_dir.is_absolute():
      cfg_dir = ROOT / cfg_dir
    if cfg_dir.exists():
      split_dir = cfg_dir
    else:
      default_root = ROOT / "data" / "train+public_test" / "train+public_test"
      split_dir = default_root / mode

  # Cập nhật cấu hình MODE để LifeMateAgent và zaic_tools tự động sử dụng đúng thư mục
  if split_dir.exists() and str(split_dir) != MODE[mode]["data_dir"]:
    MODE[mode] = get_mode_config(mode, str(split_dir))

  jsonl_path = split_dir / f"{mode}.jsonl"
  if not jsonl_path.exists():
    cand = list(split_dir.glob("*.jsonl"))
    if cand:
      jsonl_path = cand[0]

  if not jsonl_path.exists():
    print(f"❌ Không tìm thấy file dữ liệu {jsonl_path} trong {split_dir}!")
    sys.exit(1)

  print(f"\n{'='*75}")
  print(f"🌟 LIFEMATE AGENT FLOW - ZALO AI CHALLENGE 2026")
  print(f"{'='*75}")
  print(f"   • Chế độ (Mode):       {mode.upper()}")
  print(f"   • Tools Runtime:       agent/tools/{mode}")
  print(f"   • ASR Model:           {args.asr}")
  print(f"   • LLM Model:           {args.llm}")
  if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"   • Thiết bị tăng tốc:   {num_gpus}x CUDA GPU")
    for i in range(num_gpus):
      p = torch.cuda.get_device_properties(i)
      print(f"     + GPU {i}: {p.name} ({p.total_memory / (1024**3):.1f} GB VRAM)")
  else:
    print(f"   • Thiết bị:            CPU")
  print(f"   • 4-bit Quantization:  {'Bật' if args.load_in_4bit else 'Tắt'}")
  print(f"   • Dữ liệu đầu vào:     {jsonl_path}")
  print(f"   • File xuất kết quả:   {args.output}")
  print(f"{'='*75}\n")

  # Nạp danh sách mẫu
  samples = []
  with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
      if line.strip():
        samples.append(json.loads(line))

  total_available = len(samples)
  if args.num_samples > 0 and args.num_samples < total_available:
    random.seed(args.seed)
    eval_samples = random.sample(samples, args.num_samples)
    print(f"🎯 Lấy {len(eval_samples)}/{total_available} mẫu ngẫu nhiên (seed={args.seed}).")
  else:
    eval_samples = samples
    print(f"🎯 Chạy trên toàn bộ {len(eval_samples)} mẫu.")

  if args.num_shards > 1:
    eval_samples = [s for i, s in enumerate(eval_samples) if i % args.num_shards == args.shard_id]
    print(f"🧩 Phân vùng dữ liệu: Shard {args.shard_id}/{args.num_shards} ({len(eval_samples)} mẫu)")

  # Khởi tạo LifeMate Agent
  agent = LifeMateAgent(
      mode=mode,
      data_dir=str(split_dir),
      asr=args.asr,
      llm=args.llm,
      device=args.device,
      load_in_4bit=args.load_in_4bit,
      max_tokens=256,
      enable_thinking=False,
  )

  hf_map = getattr(getattr(agent.llm, "hf_model", None), "hf_device_map", None)
  if hf_map:
    unique_gpus = sorted(list(set(str(v) for v in hf_map.values())))
    print(f"🎯 Phân bổ mô hình: {len(unique_gpus)} thiết bị ({unique_gpus})")

  fuzzy_props_map = {}
  if args.eval and mode == "train":
    fuzzy_props_map = load_fuzzy_props_map(agent.catalog_path)

  # Mở file output để ghi kết quả theo format submission (hỗ trợ resume)
  out_path = Path(args.output).resolve()
  out_path.parent.mkdir(parents=True, exist_ok=True)

  done_ids = set()
  if out_path.exists():
    with open(out_path, "r", encoding="utf-8") as rf:
      for line in rf:
        line = line.strip()
        if line:
          try:
            done_ids.add(json.loads(line).get("id"))
          except Exception:
            pass
  if done_ids:
    print(f"🔄 Đã tìm thấy {len(done_ids)} mẫu đã xử lý trong {out_path.name}. Tự động resume tiếp!")

  out_fp = open(out_path, "a", encoding="utf-8")

  scores, state_matches, precisions, recalls, f1s, latencies = [], [], [], [], [], []
  start_wall = time.time()

  print(f"\n🚀 Bắt đầu thực thi pipeline ({len(eval_samples)} mẫu)...")
  try:
    for idx, sample in enumerate(eval_samples, 1):
      sid = sample["id"]
      if sid in done_ids:
        continue
      res = agent.step(sample)
      latencies.append(res["total_latency"])

      # Ghi dòng submission
      sub_row = {
          "id": sid,
          "tool_calls": res["tool_calls"],
          "state_hash": res["state_hash"],
      }
      out_fp.write(json.dumps(sub_row, ensure_ascii=False) + "\n")
      out_fp.flush()

      # Đánh giá nếu có nhãn vàng (train)
      eval_str = ""
      if args.eval and mode == "train":
        gold_label = sample.get("label", {})
        gold_calls = gold_label.get("tool_calls", [])

        # Tính gold_state_hash
        gold_env = agent.ToolEnv("train", sid, root=data_root)
        for gc in gold_calls:
          gold_env.call(gc.get("name", ""), gc.get("arguments", {}))
        gold_state_hash = gold_env.get_state_hash()

        eval_res = evaluate_sample(
            sub_calls=res["tool_calls"],
            sub_state_hash=res["state_hash"],
            gold_calls=gold_calls,
            gold_state_hash=gold_state_hash,
            fuzzy_props_map=fuzzy_props_map,
        )

        scores.append(eval_res["score"])
        state_matches.append(1.0 if eval_res["state_match"] else 0.0)
        precisions.append(eval_res["tool_precision"])
        recalls.append(eval_res["tool_recall"])
        f1s.append(eval_res["tool_f1"])

        match_icon = "✅" if eval_res["state_match"] else "❌"
        eval_str = f" | Score: {eval_res['score']:.2f} (State {match_icon}, F1: {eval_res['tool_f1']:.2f})"

      print(
          f"  [{idx:03d}/{len(eval_samples):03d}] Mẫu: {sid} "
          f"| Latency: {res['total_latency']:.2f}s "
          f"(ASR: {res['asr_latency']:.2f}s, LLM: {res['llm_latency']:.2f}s)"
          f"{eval_str}"
      )
      if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
          with torch.cuda.device(i):
            torch.cuda.empty_cache()
  finally:
    out_fp.close()

  total_wall_time = time.time() - start_wall

  print(f"\n{'='*75}")
  print(f"📊 TỔNG KẾT HOÀN THÀNH PIPELINE ({mode.upper()})")
  print(f"{'='*75}")
  print(f"⏱️ Tổng thời gian chạy:   {total_wall_time:.2f}s")
  print(f"⚡ Độ trễ TB mỗi mẫu:     {sum(latencies)/len(latencies):.2f}s")
  print(f"💾 File nộp bài / kết quả: {out_path}")

  if args.eval and mode == "train" and scores:
    print(f"\n🏆 KẾT QUẢ ĐÁNH GIÁ (ZAC 2026 METRICS):")
    print(f"   • Điểm tổng hợp (Mean Score):      {sum(scores)/len(scores)*100:.2f}%")
    print(f"   • Khớp trạng thái (State Match):    {sum(state_matches)/len(state_matches)*100:.2f}%")
    print(f"   • Tool Precision:                  {sum(precisions)/len(precisions)*100:.2f}%")
    print(f"   • Tool Recall:                     {sum(recalls)/len(recalls)*100:.2f}%")
    print(f"   • Tool F1-Score:                   {sum(f1s)/len(f1s)*100:.2f}%")
  print(f"{'='*75}\n")


if __name__ == "__main__":
  main()
