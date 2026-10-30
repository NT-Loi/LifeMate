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
Quy trình thực thi gồm các giai đoạn logic nghiêm ngặt:

1. Giai đoạn 1: Tra cứu thông tin & Tiền điều kiện (Read):
   - Khi cần điều kiện kiểm tra hoặc cần ID/thông tin thực thể để thao tác (sự kiện lịch, người liên hệ, thiết bị, báo thức, ghi chú, đơn hàng, số dư...):
     * Ở lượt đầu tiên, BẮT BUỘC gọi ĐẦY ĐỦ TẤT CẢ các công cụ đọc/tra cứu (kind: read) cần thiết cho MỌI vế trong câu lệnh.
     * Quy tắc tìm kiếm theo tên riêng/danh bạ/từ khóa: Giữ nguyên vẹn cụm từ hoặc tên người mà người dùng nhắc đến (bao gồm cả danh xưng, chức vụ nếu có), không tự ý cắt xén từ khóa.
     * Quy tắc tra cứu danh sách để tìm thực thể (lịch trình, báo thức, danh sách thiết bị, ghi chú...): Gọi công cụ list với tham số rỗng {{}} hoặc tối thiểu để lấy được toàn bộ danh sách, tránh tự áp đặt bộ lọc thời gian hay phạm vi quá hẹp.
     * Quy tắc thời tiết: Nếu kiểm tra thời tiết tại vị trí hiện tại của người dùng, chỉ truyền 'date' (YYYY-MM-DD), không truyền tham số 'location'.

2. Giai đoạn 2: Đánh giá điều kiện & Thực thi toàn diện (Write):
   - Đánh giá logic điều kiện (nếu có mệnh lệnh dạng "Nếu... thì..."):
     * Phân tích kết quả tra cứu: Nếu điều kiện KHÔNG thỏa mãn, TUYỆT ĐỐI KHÔNG thực hiện hành động phụ thuộc, kết thúc ngay bằng mảng rỗng [].
     * Nếu điều kiện THỎA MÃN (hoặc lệnh trực tiếp không điều kiện):
       - Phân tích và bóc tách ĐẦY ĐỦ TẤT CẢ CÁC HÀNH ĐỘNG người dùng yêu cầu.
       - NGUYÊN TẮC TOÀN DIỆN: Một mệnh lệnh thường chứa nhiều hành động đồng thời (ví dụ: vừa thay đổi trạng thái app vừa gửi thông báo; hoặc điều khiển nhiều thiết bị). Bạn PHẢI GỌI ĐẦY ĐỦ CÁC CÔNG CỤ TƯƠNG ỨNG CHO TỪNG HÀNH ĐỘNG ĐÓ. Hành động gửi tin nhắn/thông báo KHÔNG THỂ thay thế cho hành động cập nhật trạng thái/dữ liệu trên ứng dụng tương ứng!
   - Quy tắc tham số khi thực thi:
     * Cập nhật/sửa đổi (*.update*, *.set*, *.modify*): Bắt buộc dùng đúng ID thu được từ bước tra cứu. CHỈ TRUYỀN CÁC TRƯỜNG CÓ GIÁ TRỊ THAY ĐỔI. TUYỆT ĐỐI KHÔNG truyền lại các trường không đổi (như tiêu đề, tên, địa điểm, mô tả) nếu không có yêu cầu thay đổi chúng.
     * Tạo mới (*.create*, *.add*, *.send*): Truyền đầy đủ các tham số bắt buộc theo đúng schema. Không truyền tham số mang giá trị rỗng ("") hoặc null.

3. Giai đoạn 3: Điều kiện dừng vòng lặp (Stop):
   - Khi TẤT CẢ các hành động theo yêu cầu của người dùng ĐÃ ĐƯỢC THỰC HIỆN XONG (hoặc điều kiện không xảy ra khiến không cần hành động tiếp), BẮT BUỘC trả về mảng rỗng: [] để kết thúc.
   - TUYỆT ĐỐI KHÔNG gọi lại hoặc lặp lại các công cụ đã có trong danh sách đã thực thi thành công.

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
