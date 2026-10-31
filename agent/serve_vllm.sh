#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-outputs/lifemate-qwen35-4b-sft}"
if (($# > 0)); then
  shift
fi
base_dir="$output_dir/base-4bit"

for file in "$base_dir/config.json" "$output_dir/adapter_config.json"; do
  if [[ ! -f "$file" ]]; then
    echo "Required training artifact not found: $file" >&2
    exit 1
  fi
done

python - "$base_dir/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    quantization = json.load(source).get("quantization_config", {})

expected = (
    quantization.get("load_in_4bit")
    and quantization.get("bnb_4bit_quant_type") == "nf4"
    and quantization.get("bnb_4bit_use_double_quant")
)
if not expected:
    raise SystemExit(f"Expected a pre-quantized NF4 4-bit checkpoint, got: {quantization}")
PY

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ("vllm", "vllm-bnb-plugin"):
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError as error:
        raise SystemExit(f"Install {package} in this environment before serving") from error
PY

exec vllm serve "$base_dir" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-dtype bfloat16 \
  --lora-modules "lifemate=$output_dir" \
  "$@"
