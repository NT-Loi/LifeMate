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