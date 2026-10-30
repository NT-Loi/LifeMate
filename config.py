from pathlib import Path

TRAIN_DATA_DIR = Path("data/train+public_test/train+public_test/train")
PUBLIC_TEST_DATA_DIR = Path("data/train+public_test/train+public_test/public_test")
MODE = {
    "train": {
        "data_dir": TRAIN_DATA_DIR,
        "tools_dir": TRAIN_DATA_DIR / "zaic_tools",
        "catalog_path": TRAIN_DATA_DIR / "train_tools_catalog.json"
    },
    "public_test": {
        "data_dir": PUBLIC_TEST_DATA_DIR,
        "tools_dir": PUBLIC_TEST_DATA_DIR / "zaic_tools",
        "catalog_path": PUBLIC_TEST_DATA_DIR / "public_test_tools_catalog.json"
    } 
}