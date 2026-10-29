# LifeMate - Agentic AI That Turns Your Voice Into Action
Track LifeMate — Zalo AI Challenge 2026. Trợ lý giọng nói đa phương thức (ASR + Qwen-Agent ReAct Tool Calling).

---

## 🚀 Quickstart & Hướng dẫn chạy

### 1. Chạy trên Kaggle (Khuyên dùng: 2x Tesla T4 GPUs)
Mở hoặc tải file [kaggle_run.ipynb](kaggle_run.ipynb) lên Kaggle Notebooks:
* **Accelerator:** `GPU T4 x2`
* **Internet:** `Internet On`
* **Dataset:** Nhấn `+ Add Data` và chọn dataset `public_test` của bạn.
* Chạy notebook hoặc nhấn **Save Version -> Save & Run All (Commit)** để chạy nền trong 30-40 phút, kết quả lưu tại `/kaggle/working/submission.jsonl`.

### 2. Chạy trên Google Colab (1x Tesla T4 GPU)
Mở file [colab_run.ipynb](colab_run.ipynb) trên Google Colab:
* **Runtime:** `T4 GPU`
* Mount Google Drive chứa thư mục `public_test`.
* Chạy notebook với 4-bit quantization NF4 tự động kích hoạt để chống tràn VRAM.

### 3. Chạy trên Local Machine
```bash
# Cài đặt môi trường
uv sync

# Chạy suy luận trên toàn bộ public_test
python main.py --mode public_test --num-samples -1 --output submission.jsonl
```