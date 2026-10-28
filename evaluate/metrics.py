"""Official Evaluation Metrics for Zalo AI Challenge 2026 - LifeMate Track.

Công thức tính điểm chính thức:
  - State match: sub_state_hash == gold_state_hash. Nếu khác nhau, điểm mẫu = 0.
  - Tool-call match: Tên tool trùng khớp, mọi đối số non-fuzzy phải giống hệt,
    mọi đối số fuzzy (x-match: fuzzy) phải có token-F1 > 0.6 (NFC, chữ thường).
  - So khớp thứ tự độc lập bằng Maximum Bipartite Matching:
      P = M / (M + U) = M / |S|
      R = M / |G|
      F1 = 2PR / (P + R) = 2M / (|G| + |S|)
  - Mẫu out-of-scope (|G| == 0): Điểm = 1.0 nếu |S| == 0, ngược lại 0.0.
  - Sample Score = State_Match * F1.
"""

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Set
import unicodedata


def normalize_text_for_eval(text: Any) -> str:
  """Chuẩn hóa chuỗi văn bản cho đánh giá fuzzy:

  - Ép kiểu về string.
  - Unicode NFC.
  - Chuyển thành chữ thường.
  - Tách từ.
  """
  if text is None:
    return ""
  s = str(text)
  s = unicodedata.normalize("NFC", s).lower()
  # Thay dấu câu bằng khoảng trắng
  s = re.sub(r"[^\w\s]", " ", s)
  return re.sub(r"\s+", " ", s).strip()


def compute_token_f1(text1: Any, text2: Any) -> float:
  """Tính Token-F1 score giữa 2 chuỗi (NFC + lowercased)."""
  t1 = normalize_text_for_eval(text1).split()
  t2 = normalize_text_for_eval(text2).split()
  if not t1 and not t2:
    return 1.0
  if not t1 or not t2:
    return 0.0
  common = Counter(t1) & Counter(t2)
  same = sum(common.values())
  if same == 0:
    return 0.0
  precision = same / len(t1)
  recall = same / len(t2)
  return 2 * precision * recall / (precision + recall)


def load_fuzzy_props_map(catalog_or_path: Any) -> Dict[str, Set[str]]:
  """Nạp danh sách các thuộc tính fuzzy ('x-match': 'fuzzy') từ tools catalog."""
  if isinstance(catalog_or_path, (str, Path)):
    path = Path(catalog_or_path)
    if path.is_dir():
      # Thử tìm file *_tools_catalog.json trong thư mục
      candidates = list(path.glob("*tools_catalog.json"))
      if candidates:
        path = candidates[0]
      else:
        return {}
    if not path.exists():
      return {}
    with open(path, "r", encoding="utf-8") as fp:
      catalog = json.load(fp)
  elif isinstance(catalog_or_path, dict):
    catalog = catalog_or_path
  else:
    return {}

  fuzzy_map: Dict[str, Set[str]] = {}
  for app in catalog.get("apps", []):
    for tool in app.get("tools", []):
      tname = tool.get("name")
      if not tname:
        continue
      props = tool.get("parameters", {}).get("properties", {})
      fuzzy_keys = {k for k, v in props.items() if v.get("x-match") == "fuzzy"}
      if fuzzy_keys:
        fuzzy_map[tname] = fuzzy_keys

  # Tool hệ thống nếu có
  for tool in catalog.get("system_tools", []):
    tname = tool.get("name")
    if tname:
      props = tool.get("parameters", {}).get("properties", {})
      fuzzy_keys = {k for k, v in props.items() if v.get("x-match") == "fuzzy"}
      if fuzzy_keys:
        fuzzy_map[tname] = fuzzy_keys

  return fuzzy_map


def match_single_call(
    sub_call: dict,
    gold_call: dict,
    fuzzy_props_map: Dict[str, Set[str]],
) -> bool:
  """So khớp 1 tool call dự đoán với 1 gold tool call."""
  if sub_call.get("name") != gold_call.get("name"):
    return False

  tname = sub_call.get("name")
  fuzzy_keys = fuzzy_props_map.get(tname, set())
  sub_args = sub_call.get("arguments") or {}
  gold_args = gold_call.get("arguments") or {}

  all_keys = set(sub_args.keys()) | set(gold_args.keys())
  for k in all_keys:
    if k in fuzzy_keys:
      v_sub = sub_args.get(k, "")
      v_gold = gold_args.get(k, "")
      if compute_token_f1(v_sub, v_gold) <= 0.6:
        return False
    else:
      if sub_args.get(k) != gold_args.get(k):
        return False
  return True


def max_bipartite_matching(
    adj: List[List[int]], n_sub: int, n_gold: int
) -> int:
  """Tìm Maximum Bipartite Matching bằng thuật toán Kuhn."""
  match = [-1] * n_gold

  def dfs(u: int, visited: List[bool]) -> bool:
    for v in adj[u]:
      if not visited[v]:
        visited[v] = True
        if match[v] < 0 or dfs(match[v], visited):
          match[v] = u
          return True
    return False

  matched_count = 0
  for u in range(n_sub):
    visited = [False] * n_gold
    if dfs(u, visited):
      matched_count += 1
  return matched_count


def evaluate_sample(
    sub_calls: list,
    sub_state_hash: str,
    gold_calls: list,
    gold_state_hash: str,
    fuzzy_props_map: Dict[str, Set[str]],
) -> dict:
  """Đánh giá 1 mẫu dự đoán theo công thức chính thức ZAC 2026."""
  state_matched = sub_state_hash == gold_state_hash
  n_gold = len(gold_calls)
  n_sub = len(sub_calls)

  # Trường hợp out-of-scope (|G| == 0)
  if n_gold == 0:
    score = 1.0 if n_sub == 0 else 0.0
    return {
        "state_match": state_matched,
        "tool_precision": 1.0 if n_sub == 0 else 0.0,
        "tool_recall": 1.0 if n_sub == 0 else 0.0,
        "tool_f1": 1.0 if n_sub == 0 else 0.0,
        "score": score,
        "matched_calls": 0,
        "gold_calls": 0,
        "submitted_calls": n_sub,
    }

  # Nếu state_hash không khớp -> điểm mẫu = 0
  if not state_matched:
    return {
        "state_match": False,
        "tool_precision": 0.0,
        "tool_recall": 0.0,
        "tool_f1": 0.0,
        "score": 0.0,
        "matched_calls": 0,
        "gold_calls": n_gold,
        "submitted_calls": n_sub,
    }

  # Tạo đồ thị 2 phía
  adj = [[] for _ in range(n_sub)]
  for i, sc in enumerate(sub_calls):
    for j, gc in enumerate(gold_calls):
      if match_single_call(sc, gc, fuzzy_props_map):
        adj[i].append(j)

  M = max_bipartite_matching(adj, n_sub, n_gold)
  U = n_sub - M

  if M == 0:
    P, R, F1 = 0.0, 0.0, 0.0
  else:
    P = M / (M + U)
    R = M / n_gold
    F1 = 2 * P * R / (P + R)

  return {
      "state_match": True,
      "tool_precision": P,
      "tool_recall": R,
      "tool_f1": F1,
      "score": F1,
      "matched_calls": M,
      "gold_calls": n_gold,
      "submitted_calls": n_sub,
  }
