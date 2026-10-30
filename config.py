import os
from pathlib import Path

def get_data_root() -> Path:
    # 1. Environment variable if set
    if "LIFEMATE_DATA_DIR" in os.environ:
        p = Path(os.environ["LIFEMATE_DATA_DIR"])
        if p.is_dir():
            return p

    # 2. Local default structure
    local_nested = Path("data/train+public_test/train+public_test")
    if local_nested.is_dir():
        return local_nested
    if Path("data/train").is_dir():
        return Path("data")

    # 3. Kaggle input search
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for cat in kaggle_input.rglob("*tools_catalog.json"):
            # If cat is in train/train_tools_catalog.json, parent.parent is data root
            if cat.parent.name in ("train", "public_test") and (cat.parent / "zaic_tools").is_dir():
                return cat.parent.parent

    return local_nested

DATA_ROOT = get_data_root()
TRAIN_DATA_DIR = DATA_ROOT / "train"
PUBLIC_TEST_DATA_DIR = DATA_ROOT / "public_test"

MODE = {
    "train": {
        "data_dir": TRAIN_DATA_DIR,
        "data_path": TRAIN_DATA_DIR / "train.jsonl",
        "tools_dir": TRAIN_DATA_DIR / "zaic_tools",
        "catalog_path": TRAIN_DATA_DIR / "train_tools_catalog.json"
    },
    "public_test": {
        "data_dir": PUBLIC_TEST_DATA_DIR,
        "data_path": PUBLIC_TEST_DATA_DIR / "public_test.jsonl",
        "tools_dir": PUBLIC_TEST_DATA_DIR / "zaic_tools",
        "catalog_path": PUBLIC_TEST_DATA_DIR / "public_test_tools_catalog.json"
    } 
}