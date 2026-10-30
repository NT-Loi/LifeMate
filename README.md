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
uv run python -m models.llm.<model-name>
```

## LifeMate Agent
Test agent:
```bash
uv run python -m agent.lifemate
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

# Evaluate entire train set on 2 GPUs:
uv run python evaluate.py --num-samples -1 --gpus 0,1
```

## Kaggle Notebook (T4 x 2 GPU)
You can upload or open [kaggle_evaluation.ipynb](kaggle_evaluation.ipynb) on Kaggle with the `GPU T4 x 2` accelerator. It utilizes both T4 GPUs simultaneously via data-parallel workers, doubling evaluation throughput.