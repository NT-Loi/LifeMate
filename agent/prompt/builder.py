import json
from pathlib import Path
from typing import Any, Dict, List


class LifeMatePromptBuilder:
  """Xây dựng prompt tối ưu cho LLM thực hiện Function / Tool Calling."""

  def __init__(self, catalog_or_path: Any):
    self.tools_by_name: Dict[str, dict] = {}
    self._load_catalog(catalog_or_path)

  def _load_catalog(self, catalog_or_path: Any):
    catalog = {}
    if isinstance(catalog_or_path, (str, Path)):
      path = Path(catalog_or_path)
      if path.is_dir():
        candidates = list(path.glob("*tools_catalog.json"))
        if candidates:
          path = candidates[0]
      if path.exists():
        with open(path, "r", encoding="utf-8") as fp:
          catalog = json.load(fp)
    elif isinstance(catalog_or_path, dict):
      catalog = catalog_or_path

    for app in catalog.get("apps", []):
      for tool in app.get("tools", []):
        tname = tool.get("name")
        if tname:
          self.tools_by_name[tname] = tool

    # Thêm sys.ask_missing_info nếu chưa có
    if "sys.ask_missing_info" not in self.tools_by_name:
      self.tools_by_name["sys.ask_missing_info"] = {
          "name": "sys.ask_missing_info",
          "description": "Yêu cầu người dùng bổ sung thông tin cần thiết",
          "parameters": {
              "type": "object",
              "properties": {
                  "tool_name": {
                      "type": "string",
                      "description": "Tên công cụ dự định gọi",
                  },
                  "params": {
                      "type": "array",
                      "description": "Danh sách các tham số còn thiếu",
                      "items": {"type": "string"},
                  },
              },
              "required": ["tool_name", "params"],
          },
      }

  def build_tool_specs(self, installed_apps: Dict[str, dict]) -> str:
    """Tạo tài liệu mô tả các tool được cài đặt trên thiết bị của user."""
    lines = []
    for app_name, app_info in installed_apps.items():
      tool_list = app_info.get("tools", [])
      lines.append(f"📦 Ứng dụng: {app_name}")
      for tname in tool_list:
        schema = self.tools_by_name.get(tname)
        if schema:
          desc = schema.get("description", "")
          params = schema.get("parameters", {}).get("properties", {})
          required = schema.get("parameters", {}).get("required", [])

          param_strs = []
          for p_name, p_info in params.items():
            req_mark = "(bắt buộc)" if p_name in required else "(tùy chọn)"
            p_desc = p_info.get("description", "")
            p_type = p_info.get("type", "string")
            param_strs.append(f"     - {p_name} ({p_type}, {req_mark}): {p_desc}")

          lines.append(f"  • Tool: {tname} — {desc}")
          if param_strs:
            lines.append("    Tham số:")
            lines.extend(param_strs)
        else:
          lines.append(f"  • Tool: {tname}")
      lines.append("")
    return "\n".join(lines)

  def create_prompt(self, sample: dict, user_text: str) -> List[Dict[str, str]]:
    """Tạo messages cấu trúc chuẩn chat template cho Qwen LLM."""
    ctx = sample.get("context", {})
    installed_tools_doc = self.build_tool_specs(ctx.get("installed_apps", {}))

    system_content = f"""Bạn là LifeMate - Trợ lý ảo AI thông minh điều khiển thiết bị, quản lý thông tin, tác vụ cá nhân trên thiết bị người dùng.
Nhiệm vụ của bạn: Đọc ngữ cảnh, hiểu yêu cầu của người dùng và xuất ra danh sách các lệnh gọi công cụ (tool calls) cần thực hiện theo thứ tự.

### DANH SÁCH CÔNG CỤ ĐƯỢC CÀI ĐẶT TRÊN THIẾT BỊ:
{installed_tools_doc}

### QUY TRÌNH TƯƠNG TÁC NHIỀU BƯỚC (MULTI-TURN REACT):
Quá trình tương tác là vòng lặp phản hồi nhiều bước (ReAct):
1. Bước tra cứu thông tin (Read):
   - Ở lượt đầu tiên, hãy gọi ĐẦY ĐỦ TẤT CẢ các công cụ tra cứu cần thiết cho mọi vế trong yêu cầu của người dùng:
     * Thời tiết (weather): Nếu có kiểm tra thời tiết, gọi 'weather.get_forecast' với tham số 'date' (YYYY-MM-DD). Tuyệt đối không truyền 'location' nếu hỏi thời tiết tại thành phố của người dùng.
     * Lịch trình (calendar): Nếu yêu cầu dời lịch, hủy lịch hoặc kiểm tra lịch, BẮT BUỘC gọi 'calendar.list_events' để lấy 'event_id' và thời gian 'start', 'end' của sự kiện.
     * Danh bạ (contacts): Nếu có nhắn tin hoặc gọi điện cho ai, BẮT BUỘC gọi 'contacts.search' với tên người đó để lấy 'contact_id'.
2. Bước thực thi hành động (Write):
   - Khi đã nhận kết quả tra cứu:
     * Dời lịch / Đổi giờ sự kiện: BẮT BUỘC dùng 'calendar.update_event' với 'event_id' lấy từ 'calendar.list_events' (tuyệt đối KHÔNG dùng 'calendar.create_event'). Cập nhật cả 'start' và 'end' tương ứng với khoảng thời gian dời (ví dụ: sự kiện 08:00 đến 09:00 dời muộn 1 tiếng thành start 09:00:00+07:00, end 10:00:00+07:00).
     * Gửi tin nhắn: Dùng 'sms.send' với 'contact_id' lấy từ 'contacts.search' và nội dung 'content' ngắn gọn, thông báo đúng giờ mới (ví dụ: "mai chạy bộ dời sang 09:00 nhé"). Không truyền 'phone_number'.
3. Bước kết thúc (Stop):
   - Khi TẤT CẢ các hành động theo yêu cầu của người dùng ĐÃ HOÀN TẤT (hoặc điều kiện không xảy ra), BẮT BUỘC trả về mảng rỗng: [] để kết thúc vòng lặp.
   - TUYỆT ĐỐI KHÔNG lặp lại các công cụ đã có trong danh sách đã thực thi.

### ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Chỉ trả về DUY NHẤT một chuỗi JSON List:
[
  {{"name": "<tên_tool>", "arguments": {{...}}}}
]
Hoặc [] nếu đã hoàn thành toàn bộ tác vụ. Tuyệt đối không giải thích dài dòng hay viết chữ ngoài mảng JSON.
"""

    user_profile = ctx.get("user_profile", {})
    context_str = f"""--- NGỮ CẢNH HIỆN TẠI ---
- Thời gian hiện tại: {ctx.get('current_time')} (Timezone: {ctx.get('timezone', 'Asia/Ho_Chi_Minh')})
- Vị trí hiện tại: {ctx.get('location')}
- Hồ sơ người dùng: Tên={user_profile.get('name')}, Nhà={user_profile.get('home_address')}, Công ty={user_profile.get('work_address')}, Thành phố={user_profile.get('city')}
"""
    if sample.get("history"):
      history_json = json.dumps(sample["history"], ensure_ascii=False)
      context_str += f"- Lịch sử hội thoại: {history_json}\n"

    user_content = f"""{context_str}
Yêu cầu của người dùng: "{user_text}"

Dự đoán danh sách tool calls (JSON List):"""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages
