from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from qwen_agent.agents import FnCallAgent
from qwen_agent.tools import BaseTool

from config import MODE
from models import get_asr, get_llm
from agent.tools import get_tool_env
from agent.rag import TrainExampleRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_asr_text(text: str) -> str:
  """Khắc phục các lỗi nghe nhầm ngữ âm kinh điển của mô hình ASR."""
  if not text:
    return ""
  t = text
  # Lỗi vàng SJC -> vàng sec / s j c
  t = re.sub(r"\bvàng\s+sec\b", "vàng SJC", t, flags=re.IGNORECASE)
  t = re.sub(r"\bs\s*j\s*c\b", "SJC", t, flags=re.IGNORECASE)
  # Lỗi thuốc Omeprazole -> thuốc omezon
  t = re.sub(r"\bomezon\b", "Omeprazole", t, flags=re.IGNORECASE)
  return t


class ZaicTool(BaseTool):
  """Wrapper chuyển đổi tool từ catalog của ZAC 2026 thành BaseTool của Qwen-Agent."""

  def __init__(
      self,
      name: str,
      description: str,
      parameters: dict,
      env: Any = None,
  ):
    self.name = name
    self.description = description or f"Tool {name}"

    props = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    req = parameters.get("required", []) if isinstance(parameters, dict) else []
    # Đảm bảo schema OpenAI hợp lệ tuyệt đối theo chuẩn is_tool_schema của Qwen-Agent
    self.parameters = {
        "type": "object",
        "properties": props,
        "required": [r for r in req if r in props],
    }
    self.env = env
    super().__init__()

  def call(self, params: Union[str, dict], **kwargs: Any) -> str:
    """Thực thi tool trực tiếp trên phiên môi trường (Session) của sample hiện tại."""
    if isinstance(params, str):
      try:
        params = json.loads(params)
      except Exception:
        params = {}
    if not isinstance(params, dict):
      params = {}

    if self.env is not None:
      try:
        result = self.env.call(self.name, params)
        return json.dumps(result, ensure_ascii=False)
      except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

    return json.dumps({"ok": True, "executed": self.name}, ensure_ascii=False)


class LifeMateAgent:
  """LifeMate Agentic Assistant sử dụng framework Qwen-Agent.

  Kiến trúc:
    - Framework: Qwen-Agent (FnCallAgent)
    - ASR Engine: Gipformer 68M / PhoASR-Whisper
    - LLM Engine: Qwen/Qwen3.5-4B (4.54B params < 5B ZAC 2026 rules)
    - Tool Runtime: zaic_tools môi trường cô lập cho train / public_test
  """

  def __init__(
      self,
      asr: Optional[str] = "gipformer",
      llm: str = "Qwen/Qwen3.5-4B",
      mode: str = "train",
      device: Optional[str] = None,
      max_tokens: int = 1024,
      enable_thinking: bool = False,
      data_dir: Optional[str] = None,
      load_in_4bit: bool = True,
      **kwargs: Any,
  ):
    self.mode = (mode or "train").lower()
    self.max_tokens = max_tokens
    self.enable_thinking = enable_thinking

    # 1. Nạp cấu hình từ config.MODE hoặc data_dir tùy chỉnh
    if data_dir:
      from config import get_mode_config
      cfg = get_mode_config(self.mode, data_dir)
    else:
      cfg = MODE.get(self.mode)
    assert cfg, f"Mode {self.mode} không tồn tại trong config.MODE"

    self.tools_dir = Path(cfg["tools_dir"]).resolve()
    self.catalog_path = Path(cfg["catalog_path"]).resolve()
    self.data_dir = Path(cfg["data_dir"]).resolve()
    self.data_root = self.data_dir.parent

    # 2. Khởi tạo ASR
    if asr and asr.lower() != "none":
      logger.info(f"🎙️ Khởi tạo ASR: {asr}")
      self.asr = get_asr(asr, device=device)
    else:
      logger.info("🎙️ Chế độ ASR: Tắt (sử dụng audio_text trực tiếp)")
      self.asr = None

    # 3. Khởi tạo LLM tương thích Qwen-Agent
    logger.info(f"🧠 Khởi tạo Qwen-Agent LLM: {llm} (device={device or 'auto'}, 4bit={load_in_4bit}, max_tokens={max_tokens})...")
    llm_cfg = kwargs.pop("cfg", {}) or {}
    llm_cfg.setdefault("generate_cfg", {})["max_new_tokens"] = max_tokens
    self.llm = get_llm(model_id=llm, device=device, load_in_4bit=load_in_4bit, cfg=llm_cfg, **kwargs)

    # 4. Nạp ToolEnv factory từ thư mục data
    self.ToolEnv = get_tool_env(self.mode, tools_dir=self.tools_dir)

    # 5. Đọc và lập chỉ mục catalog tool schema
    self.tool_schemas: Dict[str, dict] = {}
    self._load_catalog_schemas()

    # 6. Khởi tạo RAG Few-Shot Retriever từ tập train
    self.retriever = TrainExampleRetriever(self.data_dir)

    logger.info(f"LifeMate Agent (Framework: Qwen-Agent, Mode: {self.mode.upper()}) is ready!")

  def _load_catalog_schemas(self):
    """Lập chỉ mục toàn bộ schema công cụ từ file catalog.json."""
    if self.catalog_path.exists():
      with open(self.catalog_path, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)
      for app in catalog_data.get("apps", []):
        for tool in app.get("tools", []):
          tname = tool.get("name")
          if tname:
            self.tool_schemas[tname] = tool

    # Đảm bảo có tool hệ thống sys.ask_missing_info
    if "sys.ask_missing_info" not in self.tool_schemas:
      self.tool_schemas["sys.ask_missing_info"] = {
          "name": "sys.ask_missing_info",
          "description": "Yêu cầu người dùng bổ sung thông tin cần thiết khi chưa rõ ràng",
          "parameters": {
              "type": "object",
              "properties": {
                  "tool_name": {"type": "string", "description": "Tên công cụ dự định gọi"},
                  "params": {"type": "array", "description": "Danh sách tham số còn thiếu", "items": {"type": "string"}},
              },
              "required": ["tool_name", "params"],
          },
      }

  def build_sample_tools(self, sample: dict, env: Any) -> List[ZaicTool]:
    """Tạo danh sách ZaicTool tương ứng với các ứng dụng đã cài đặt trên mẫu dữ liệu."""
    context = sample.get("context", {}) or {}
    installed_apps = context.get("installed_apps", {}) or {}

    tool_names = set()
    if installed_apps:
      for app_data in installed_apps.values():
        if isinstance(app_data, dict):
          for tname in app_data.get("tools", []):
            tool_names.add(tname)
    else:
      # Nếu mẫu không giới hạn installed_apps, lấy tất cả tools trong catalog
      tool_names = set(self.tool_schemas.keys())

    # Luôn thêm sys.ask_missing_info
    tool_names.add("sys.ask_missing_info")

    tools = []
    for tname in sorted(tool_names):
      schema = self.tool_schemas.get(tname)
      if schema:
        tools.append(
            ZaicTool(
                name=tname,
                description=schema.get("description", ""),
                parameters=schema.get("parameters", {}),
                env=env,
            )
        )
    return tools

  def build_system_message(self, sample: dict, user_text: str = "") -> str:
    """Tạo system message mang ngữ cảnh thời gian, vị trí, hồ sơ người dùng, quy tắc ZAC và Few-Shot RAG."""
    context = sample.get("context", {}) or {}
    current_time = context.get("current_time", "")
    timezone = context.get("timezone", "Asia/Ho_Chi_Minh")
    user_profile = context.get("user_profile", {}) or {}
    location = context.get("location", {}) or {}

    city = user_profile.get("city", "") or "Việt Nam"

    # Tập hợp các tool khả dụng trên mẫu dữ liệu hiện tại
    installed_apps = context.get("installed_apps", {}) or {}
    available_tools = set()
    if installed_apps:
      for app_data in installed_apps.values():
        if isinstance(app_data, dict):
          for tname in app_data.get("tools", []):
            available_tools.add(tname)
    else:
      available_tools = set(self.tool_schemas.keys())
    available_tools.add("sys.ask_missing_info")

    # Lấy các ví dụ mẫu tương đồng từ tập train qua BM25 (chỉ chọn các mẫu có tools khả dụng)
    few_shot_doc = ""
    if hasattr(self, "retriever") and self.retriever and user_text:
      few_shot_doc = self.retriever.format_few_shot_prompt(
          user_text, top_k=2, available_tools=available_tools
      )

    sys_msg = (
        "Bạn là LifeMate, trợ lý giọng nói thông minh cho gia đình và cuộc sống cá nhân.\n"
        f"• Thời gian hiện tại: {current_time} ({timezone})\n"
        f"• Vị trí: {city} (Lat: {location.get('lat')}, Lng: {location.get('lng')})\n"
        f"• Hồ sơ người dùng: {json.dumps(user_profile, ensure_ascii=False)}\n\n"
        "QUY TẮC BẮT BUỘC KHI GỌI CÔNG CỤ (ZALO AI CHALLENGE 2026):\n"
        "1. Dựa trên yêu cầu của người dùng, BẮT BUỘC gọi công cụ phù hợp với tham số chính xác. TUYỆT ĐỐI không trả lời bằng văn bản trò chuyện thông thường nếu người dùng yêu cầu hành động.\n"
        "2. Nhà thông minh (smart_home): Luôn gọi 'smart_home.get_status' trước để lấy danh sách thiết bị và trạng thái hiện tại trước khi gọi lệnh điều khiển ('light_set', 'fan_set', 'climate_set'...). Mọi enum trạng thái ('power', 'mode'...) phải viết chữ thường (ví dụ: 'on', 'off', 'cool').\n"
        "3. Danh bạ & Liên lạc: Luôn gọi 'contacts.search' trước với tên người liên hệ để lấy 'contact_id' trước khi gửi tin nhắn ('sms.send') hoặc gọi điện ('phone.call') hoặc chuyển tiền.\n"
        "4. Lịch trình (calendar): Luôn gọi 'calendar.list_events' trước để kiểm tra sự kiện trước khi dời, sửa hoặc hủy lịch ('calendar.cancel_event').\n"
        "5. Giá vàng (gold_price): 'gold_price.get_price' dùng tham số 'type' là 'sjc' (vàng SJC / vàng miếng), 'ring' (vàng nhẫn) hoặc 'world'.\n"
        "6. Nếu thiếu thông tin bắt buộc mà không thể suy luận được, BẮT BUỘC gọi 'sys.ask_missing_info' với 'tool_name' và 'params' (mảng danh sách tham số thiếu).\n"
        "7. Nếu yêu cầu là câu hỏi bình thường, chào hỏi hoặc ngoài phạm vi công cụ cài đặt, trả lời ngắn gọn và không gọi công cụ.\n\n"
        f"{few_shot_doc}"
    )
    return sys_msg

  def perceive(
      self, sample: dict, audio_path_override: Optional[str] = None
  ) -> Tuple[str, float]:
    """Bước 1: Chuyển đổi giọng nói thành văn bản bằng ASR."""
    t0 = time.time()
    user_text = ""

    if self.asr is not None:
      audio_rel = audio_path_override or sample.get("audio")
      if audio_rel:
        possible_paths = [
            self.data_dir / audio_rel,
            self.data_root / self.mode / audio_rel,
            Path(audio_rel),
        ]
        audio_full_path = next((p for p in possible_paths if p.exists()), None)
        if audio_full_path:
          try:
            user_text = self.asr.transcribe(str(audio_full_path))
          except Exception as e:
            logger.warning(f"⚠️ ASR Error tại {sample.get('id')}: {e}")
            user_text = sample.get("audio_text", "")
        else:
          logger.warning(f"⚠️ Không tìm thấy file âm thanh {audio_rel}, fallback sang audio_text.")
          user_text = sample.get("audio_text", "")
      else:
        user_text = sample.get("audio_text", "")
    else:
      user_text = sample.get("audio_text", "")

    asr_latency = time.time() - t0
    return user_text, asr_latency

  def step(self, sample: dict) -> dict:
    """Xử lý end-to-end 1 mẫu dữ liệu: Perceive -> Qwen-Agent ReAct -> ToolEnv -> Kết quả."""
    t_start = time.time()
    sid = sample["id"]

    # 1. Perceive: Giọng nói -> Văn bản
    user_text, asr_lat = self.perceive(sample)
    user_text = normalize_asr_text(user_text)

    # 2. Khởi tạo môi trường ToolEnv cho sample
    env = self.ToolEnv(self.mode, sid)

    # 3. Tạo danh sách ZaicTool và System Message cho mẫu
    sample_tools = self.build_sample_tools(sample, env)
    system_message = self.build_system_message(sample, user_text=user_text)

    # 4. Khởi tạo FnCallAgent từ Qwen-Agent
    t_agent = time.time()
    agent = FnCallAgent(
        function_list=sample_tools,
        llm=self.llm,
        system_message=system_message,
    )

    # 5. Thực thi vòng lặp Agent
    messages = [{"role": "user", "content": user_text}]
    response = []
    try:
      for rsp in agent.run(messages):
        response = rsp
    except Exception as e:
      logger.error(f"❌ Qwen-Agent Runtime Error tại {sid}: {e}")

    agent_lat = time.time() - t_agent
    total_lat = time.time() - t_start

    # 6. Thu thập kết quả chuẩn format nộp bài ZAC 2026
    executed_calls = env.get_tool_calls()
    final_state_hash = env.get_state_hash()

    return {
        "id": sid,
        "tool_calls": executed_calls,
        "state_hash": final_state_hash,
        "user_text": user_text,
        "response": response,
        "asr_latency": asr_lat,
        "agent_latency": agent_lat,
        "llm_latency": agent_lat,
        "total_latency": total_lat,
    }

  def batch_predict(
      self,
      samples: List[dict],
      output_file: Optional[str | Path] = None,
      verbose: bool = True,
  ) -> List[dict]:
    """Chạy batch trên danh sách mẫu và xuất file submission.jsonl."""
    out_fp = None
    if output_file is not None:
      out_path = Path(output_file)
      out_path.parent.mkdir(parents=True, exist_ok=True)
      out_fp = open(out_path, "w", encoding="utf-8")

    results = []
    total = len(samples)
    if verbose:
      print(f"🚀 Bắt đầu batch predict với Qwen-Agent trên {total} mẫu ({self.mode})...")

    try:
      for idx, sample in enumerate(samples, 1):
        res = self.step(sample)
        results.append(res)

        if out_fp is not None:
          sub_row = {
              "id": res["id"],
              "tool_calls": res["tool_calls"],
              "state_hash": res["state_hash"],
          }
          out_fp.write(json.dumps(sub_row, ensure_ascii=False) + "\n")
          out_fp.flush()

        if verbose and (idx % 10 == 0 or idx == total):
          print(f"  [{idx}/{total}] Mẫu {res['id']} hoàn thành trong {res['total_latency']:.2f}s | Calls: {len(res['tool_calls'])}")
    finally:
      if out_fp is not None:
        out_fp.close()

    return results

  def __call__(self, sample: dict) -> dict:
    return self.step(sample)


if __name__ == "__main__":
  agent = LifeMateAgent(asr="none")
  print("Agent initialized successfully!")