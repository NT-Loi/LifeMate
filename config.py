import os
from pathlib import Path

# Thư mục gốc repo
BASE_DIR = Path(__file__).resolve().parent

# Đường dẫn thư mục dữ liệu (khi chạy trên Colab, chỉ cần sửa đường dẫn ở đây thành đường dẫn trên Drive)
# Ví dụ trên Google Drive:
# PUBLIC_TEST_DATA_DIR = "/content/drive/MyDrive/ZAIC2026/public_test"
TRAIN_DATA_DIR = os.getenv(
    "TRAIN_DATA_DIR",
    str(BASE_DIR / "data/train+public_test/train+public_test/train"),
)
PUBLIC_TEST_DATA_DIR = os.getenv(
    "PUBLIC_TEST_DATA_DIR",
    str(BASE_DIR / "data/train+public_test/train+public_test/public_test"),
)


def get_mode_config(mode: str, data_dir: str = None) -> dict:
  """Tự động sinh cấu hình chuẩn cho một mode (train hoặc public_test)."""
  mode = (mode or "train").lower()
  raw_dir = (
      data_dir
      or (TRAIN_DATA_DIR if mode == "train" else PUBLIC_TEST_DATA_DIR)
  )

  p = Path(raw_dir)
  if not p.is_absolute():
    p = (BASE_DIR / p).resolve()
  else:
    p = p.resolve()

  # Tìm file catalog JSON
  catalog_candidates = [
      p / f"{mode}_tools_catalog.json",
      p / "tools_catalog.json",
  ]
  catalog_path = next((c for c in catalog_candidates if c.exists()), None)
  if catalog_path is None:
    # Thử quét bất kỳ file *catalog*.json nào trong thư mục
    found = list(p.glob("*catalog*.json"))
    catalog_path = found[0] if found else p / f"{mode}_tools_catalog.json"

  # Tìm thư mục zaic_tools
  tools_dir = p / "zaic_tools"
  if not tools_dir.exists() and (p / "tools").exists():
    tools_dir = p / "tools"

  return {
      "data_dir": str(p),
      "tools_dir": str(tools_dir),
      "catalog_path": str(catalog_path),
  }


MODE = {
    "train": get_mode_config("train", TRAIN_DATA_DIR),
    "public_test": get_mode_config("public_test", PUBLIC_TEST_DATA_DIR),
}