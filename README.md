# LifeMate - Agentic AI That Turns Your Voice Into Action


## ASR
Test ASR models:
```bash
uv pip install -r models/asr/requirements-<model-name>.txt
uv run python -m models.asr.<model-name>
```

## LLM
Test LLM models:
```bash
uv pip install -r models/llm/requirements-qwen.txt
uv run python -m models.llm.<model-name>
```

### Google Cloud G2 / L4

The optional `gcp-l4` extra pins a Python 3.12, PyTorch 2.10 + CUDA 12.8 environment and uses prebuilt FlashAttention 2 and causal-conv1d wheels.

```bash
uv sync --extra gcp-l4
uv run --extra gcp-l4 python -m agent.lifemate
```

The model loaders use pinned revisions and local-only loading. Cache the selected model files while building the offline image; the Kaggle notebook defaults to Qwen3.5-2B and uses that model ID for both caching and evaluation.

```python
from huggingface_hub import snapshot_download
from models.llm.qwen import MODEL_REVISIONS as QWEN_REVISIONS
from models.asr.gipformer import DEFAULT_VERSION, MODEL_REVISIONS as GIPFORMER_REVISIONS, REPO_IDS

MODEL_ID = "Qwen/Qwen3.5-2B"
snapshot_download(repo_id=MODEL_ID, revision=QWEN_REVISIONS[MODEL_ID])
snapshot_download(repo_id=REPO_IDS[DEFAULT_VERSION], revision=GIPFORMER_REVISIONS[DEFAULT_VERSION])
```

## LifeMate Agent
Test agent:
```bash
uv run python -m agent.lifemate
```

Select a model directly when creating an agent:
```python
from agent.lifemate import LifeMateAgent

agent = LifeMateAgent(asr=None, model_id="Qwen/Qwen3.5-2B")
```

## Evaluation
Evaluate agent on train data:
```bash
# Evaluate 10 samples (auto-detects single or multi-GPU):
uv run python evaluate.py --num-samples 10

# Multi-GPU evaluation (e.g. T4 x 2 on Kaggle):
uv run python evaluate.py --num-samples 10 --gpus 0,1

# Evaluate a specific sample:
uv run python evaluate.py --id train_0001

# Select a model by its Hugging Face model ID:
uv run python evaluate.py --model-id Qwen/Qwen3.5-2B

# Evaluate entire train set on 2 GPUs:
uv run python evaluate.py --num-samples -1 --gpus 0,1
```

## Kaggle Notebook (T4 x 2 GPU)
You can upload or open [kaggle_evaluation.ipynb](kaggle_evaluation.ipynb) on Kaggle with the `GPU T4 x 2` accelerator. It utilizes both T4 GPUs simultaneously via data-parallel workers, doubling evaluation throughput.
