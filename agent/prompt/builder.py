import json
from pathlib import Path
from typing import Any, Dict, List


class LifeMatePromptBuilder:
    """Build prompts from the installed tools and their schemas."""

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
                if tool.get("name"):
                    self.tools_by_name[tool["name"]] = tool

        self.tools_by_name.setdefault(
            "sys.ask_missing_info",
            {
                "name": "sys.ask_missing_info",
                "kind": "meta",
                "description": "Ask the user for required missing information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "params": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["tool_name", "params"],
                },
            },
        )

    def build_tools(self, installed_apps: Dict[str, dict]) -> List[Dict[str, Any]]:
        names = dict.fromkeys(
            name
            for app_info in installed_apps.values()
            for name in app_info.get("tools", [])
        )
        names.setdefault("sys.ask_missing_info", None)
        tools = []
        for name in names:
            schema = self.tools_by_name.get(name)
            if schema is None:
                continue

            kind = schema.get("kind", "meta" if name == "sys.ask_missing_info" else "write")
            description = schema.get("description", "")
            if returns := schema.get("returns"):
                description = f"{description} Kết quả: {returns}."
            tools.append({
                "name": name,
                "description": f"[{kind}] {description}",
                "parameters": schema.get("parameters", {}),
            })
        return tools

    def create_prompt(self, sample: dict, user_text: str) -> List[Dict[str, str]]:
        ctx = sample.get("context", {})

        system_content = """Bạn là LifeMate, tác nhân thực hiện yêu cầu bằng các công cụ đã cài. Các function khả dụng được cung cấp qua giao diện function-calling.

Quy tắc:
- Chỉ gọi function được cung cấp qua giao diện function-calling. Nếu không có ứng dụng phù hợp, trả về [].
- Thứ tự bắt buộc: (1) kiểm tra điều kiện, (2) chọn nhánh, (3) tra cứu đối tượng/ID còn thiếu, (4) thực hiện các write của nhánh. Không được đảo thứ tự này.
- Nếu yêu cầu có điều kiện ("nếu ... thì ...", "còn nếu/không thì ..."), lượt đầu chỉ gọi read tool cần để biết kết quả điều kiện; tuyệt đối chưa write hoặc tra cứu riêng cho hành động thuộc nhánh. Ví dụ "nếu mai mưa thì dời ..., không mưa thì thôi" bắt đầu chỉ bằng weather.get_forecast. Sau khi nhận kết quả, làm đúng một nhánh; nếu kết quả chọn "thôi/giữ nguyên" thì dừng, không đổi trạng thái.
- Dùng lịch sử hội thoại và kết quả tool trước đó để hiểu đại từ/tham chiếu như "cái còn lại"; coi các thao tác trước đã thành công là trạng thái hiện tại (ví dụ đối tượng đã xóa không còn tồn tại). Chỉ đọc lại khi cần dữ liệu mới hoặc lịch sử chưa đủ.
- Các động từ "dời/đổi/chuyển/hoãn" thường yêu cầu sửa mục đã tồn tại: gọi list/search để tìm mục khớp rồi dùng update với ID đó; không dùng create trừ khi người dùng yêu cầu tạo mục mới. Khi chỉ đổi giờ, giữ nguyên thời lượng và các trường khác.
- Sau khi biết kết quả, thực hiện hành động vô điều kiện và chỉ các hành động thuộc nhánh đúng. Không chạy nhánh còn lại. Nếu nhánh đúng nói "thôi/giữ nguyên" hoặc không có hành động, không thay đổi trạng thái; nếu nhánh đó có hành động thì vẫn thực hiện.
- Chỉ tra cứu ID/thông tin cho các hành động sẽ thực hiện, trừ khi dữ liệu đó cần để đánh giá điều kiện. Với tham số x-ref, phải dùng ID thật lấy từ lịch sử/kết quả read; không truyền tên người, tên sự kiện hay mô tả thay cho ID và không đoán ID.
- Tính ngày giờ tương đối từ thời gian hiện tại và múi giờ trong ngữ cảnh. Gửi đúng định dạng schema yêu cầu, gồm ISO-8601 đầy đủ offset khi cần; khi dời lịch, giữ nguyên thời lượng nếu người dùng chỉ đổi giờ.
- Thực hiện mọi hành động rõ ràng, độc lập hoặc vô điều kiện; không tự thêm hành động. Chỉ truyền tham số required và tham số tùy chọn thật sự cần. Mặc định bỏ location và các trường tùy chọn khác nếu schema đã có giá trị mặc định hoặc yêu cầu không cần chúng; không thêm tham số để suy diễn ngữ cảnh.
- Tuân thủ required, enum, x-one-of và các ràng buộc trong schema. Với trường x-match=fuzzy, giữ đúng ý và các thực thể người dùng nêu, không thêm chi tiết. Nội dung tin nhắn cần ngắn, gần cách diễn đạt yêu cầu và chỉ gồm thông tin người dùng muốn gửi; không thêm tên người nhận, giải thích hay điều kiện thời tiết nếu không được yêu cầu.
- Nếu thiếu tham số bắt buộc và không thể lấy bằng read tool, gọi sys.ask_missing_info với đúng tool_name và tên params còn thiếu; không đoán. Vẫn có thể làm các hành động độc lập đã đủ thông tin.
- Trước mỗi write, đối chiếu với các hành động đã hoàn tất; không lặp lại cùng hành động dù thay đổi cách diễn đạt ở trường fuzzy hoặc tham số tùy chọn.
- Sau khi mọi hành động xong, trả về []. Không lặp lại công cụ chỉ để thử lại một hành động đã thành công.

Khi cần thao tác, gọi các function phù hợp bằng giao diện function-calling. Khi không cần gọi thêm công cụ, chỉ trả về [] và không giải thích."""

        user_profile = ctx.get("user_profile", {})
        context_str = f"""--- NGỮ CẢNH HIỆN TẠI ---
- Thời gian hiện tại: {ctx.get('current_time')} (Timezone: {ctx.get('timezone', 'Asia/Ho_Chi_Minh')})
- Vị trí hiện tại: {ctx.get('location')}
- Hồ sơ người dùng: Tên={user_profile.get('name')}, Nhà={user_profile.get('home_address')}, Công ty={user_profile.get('work_address')}, Thành phố={user_profile.get('city')}
"""
        if sample.get("history"):
            context_str += "- Lịch sử hội thoại: " + json.dumps(sample["history"], ensure_ascii=False) + "\n"

        user_content = f"""{context_str}
Yêu cầu mới nhất: "{user_text}"

Hãy xử lý yêu cầu bằng các công cụ khả dụng; nếu không cần gọi thêm công cụ, trả về []."""
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
