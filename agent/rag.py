"""RAG / Dynamic Few-Shot Example Retriever for LifeMate Agent.

Truy xuất các mẫu tương đồng cao nhất từ tập 'train' dựa trên thuật toán BM25 offline
(hoàn toàn không cần API bên ngoài, tuân thủ 100% quy chế offline của ZAC 2026).
"""

from __future__ import annotations

from collections import Counter
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
  """Tokenize văn bản tiếng Việt đơn giản (chữ thường, tách từ)."""
  if not text:
    return []
  return re.findall(r"\w+", text.lower())


class TrainExampleRetriever:
  """Bộ tìm kiếm BM25 ngữ cảnh các ví dụ vàng từ tập train.jsonl."""

  def __init__(
      self,
      train_path: Optional[Union[str, Path]] = None,
      k1: float = 1.5,
      b: float = 0.75,
  ):
    self.k1 = k1
    self.b = b
    self.train_samples: List[dict] = []
    self.corpus: List[List[str]] = []
    self.avgdl: float = 0.0
    self.idf: Dict[str, float] = {}

    self.load(train_path)

  def find_train_file(self, hint_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Tự động tìm kiếm file train.jsonl trong các thư mục khả dĩ."""
    candidates = []
    if hint_path:
      hp = Path(hint_path)
      if hp.is_file() and hp.name == "train.jsonl":
        candidates.append(hp)
      elif hp.is_dir():
        candidates.append(hp / "train.jsonl")
        candidates.append(hp / "train" / "train.jsonl")
        candidates.append(hp.parent / "train" / "train.jsonl")

    # Local workspace default
    root = Path(__file__).resolve().parent.parent
    candidates.extend([
        root / "data" / "train+public_test" / "train+public_test" / "train" / "train.jsonl",
        Path("/kaggle/input/zaic2026-public-test/train/train.jsonl"),
        Path("/kaggle/input/zaic2026/train/train.jsonl"),
        Path("/kaggle/input/train/train.jsonl"),
    ])

    # Quét toàn bộ /kaggle/input nếu đang chạy trên Kaggle
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
      for p in kaggle_input.rglob("train.jsonl"):
        candidates.append(p)

    for cand in candidates:
      if cand and cand.exists() and cand.is_file():
        return cand.resolve()
    return None

  def load(self, train_path: Optional[Union[str, Path]] = None) -> bool:
    """Nạp và lập chỉ mục BM25 toàn bộ mẫu tập train."""
    actual_path = self.find_train_file(train_path)
    if not actual_path or not actual_path.exists():
      logger.warning("⚠️ Không tìm thấy file train.jsonl để khởi tạo RAG Few-Shot Retriever.")
      return False

    try:
      with open(actual_path, "r", encoding="utf-8") as f:
        self.train_samples = [json.loads(line) for line in f if line.strip()]

      self.corpus = [tokenize(d.get("audio_text", "")) for d in self.train_samples]
      n_docs = len(self.corpus)
      if n_docs == 0:
        return False

      self.avgdl = sum(len(doc) for doc in self.corpus) / n_docs

      # Tính Document Frequency (DF)
      df = Counter()
      for doc in self.corpus:
        for word in set(doc):
          df[word] += 1

      # Tính Inverse Document Frequency (IDF) theo chuẩn BM25
      self.idf = {
          word: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
          for word, freq in df.items()
      }

      logger.info(
          f"📚 [RAG] Đã nạp và lập chỉ mục BM25 cho {n_docs} mẫu tập train thành công từ: {actual_path.name}"
      )
      return True
    except Exception as e:
      logger.warning(f"⚠️ Lỗi khi nạp train.jsonl cho RAG: {e}")
      return False

  def retrieve(
      self,
      query: str,
      top_k: int = 2,
      available_tools: Optional[set] = None,
  ) -> List[dict]:
    """Tìm kiếm top_k mẫu train tương đồng nhất với query, ưu tiên các mẫu có tools khả dụng."""
    if not self.corpus or not query:
      return []

    q_tokens = tokenize(query)
    if not q_tokens:
      return []

    scores = []
    for i, doc in enumerate(self.corpus):
      doc_len = len(doc)
      doc_counts = Counter(doc)
      score = 0.0
      for token in q_tokens:
        if token in doc_counts:
          tf = doc_counts[token]
          w_idf = self.idf.get(token, 0.0)
          denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
          score += w_idf * ((tf * (self.k1 + 1.0)) / denom)
      if score > 0:
        scores.append((score, i))

    scores.sort(key=lambda x: x[0], reverse=True)

    if not available_tools:
      return [self.train_samples[i] for _, i in scores[:top_k]]

    # Lọc mẫu: Chỉ chọn các mẫu mà tất cả tool được gọi đều có trong available_tools
    allowed_tools = set(available_tools) | {"sys.ask_missing_info"}
    filtered = []
    for _, i in scores:
      s = self.train_samples[i]
      gold_calls = s.get("label", {}).get("tool_calls", [])
      tool_names = {c.get("name") for c in gold_calls if c.get("name")}
      if tool_names.issubset(allowed_tools):
        filtered.append(s)
      if len(filtered) >= top_k:
        break

    # Nếu không đủ mẫu khớp hoàn toàn tool, bù thêm từ top điểm cao nhất
    if len(filtered) < top_k:
      for _, i in scores:
        s = self.train_samples[i]
        if s not in filtered:
          filtered.append(s)
        if len(filtered) >= top_k:
          break

    return filtered

  def format_few_shot_prompt(
      self,
      query: str,
      top_k: int = 2,
      available_tools: Optional[set] = None,
  ) -> str:
    """Định dạng các mẫu tìm được thành khối ví dụ minh họa Few-Shot cho System Prompt."""
    top_samples = self.retrieve(query, top_k=top_k, available_tools=available_tools)
    if not top_samples:
      return ""

    lines = [
        "### CÁC VÍ DỤ MẪU THAM KHẢO CHUẨN (FEW-SHOT TỪ TẬP HUẤN LUYỆN):",
        "(Lưu ý: Các ví dụ dưới đây chỉ dùng để tham khảo cú pháp và quy trình suy luận. Bạn CHỈ ĐƯỢC PHÉP gọi các công cụ có trong danh sách công cụ của mẫu hiện tại.)",
    ]
    for idx, s in enumerate(top_samples, 1):
      user_cmd = s.get("audio_text", "").strip()
      gold_calls = s.get("label", {}).get("tool_calls", [])
      lines.append(f"• Ví dụ {idx}:")
      lines.append(f'  - Người dùng nói: "{user_cmd}"')
      lines.append(f"  - Lệnh gọi tool chính xác: {json.dumps(gold_calls, ensure_ascii=False)}")
    lines.append("")
    return "\n".join(lines)
