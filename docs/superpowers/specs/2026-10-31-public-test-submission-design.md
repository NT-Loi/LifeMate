# Public Test Submission Runner

## Intent

Run every sample in the configured `public_test` split, write the required `submission.jsonl` at the repository root, and retain timestamped conversations and logs for diagnosis.

## Chosen approach

Add a small `main.py` that reads every row from `MODE["public_test"]["data_path"]`, delegates inference and artifact writing to `evaluate.run_evaluation` with Gipformer ASR, then copies its `predictions.jsonl` to root `submission.jsonl`. Keep run details under `eval_results/public_test_<timestamp>/`.

For public-test samples, skip gold-state replay because labels are absent and the submission contract allows exactly one `ToolEnv` per sample. `LifeMateAgent.predict` owns that environment; the evaluator records its transcript and final state hash. Do not publish a root submission if any sample failed.

Add a dedicated `kaggle_public_test.ipynb` that checks the Kaggle GPUs/data, downloads pinned Gipformer and Qwen3.5-2B assets, and runs `main.py` for the full split. Expose model ID and GPU selection through the script so the notebook can override them.

## Alternatives

- Reimplement the inference loop in `main.py`: rejected because it would duplicate evaluation and artifact handling.
- Add submission execution to `kaggle_evaluation.ipynb`: workable, but mixes train/debug evaluation with the public-test submission flow. A dedicated notebook is the default unless the user prefers otherwise.

## Scope and constraints

- Preserve `evaluate.py` tool-call behavior and use its existing timestamped logs, worker logs, conversations, and summaries.
- Include every public-test row; do not read any `*_environment` files.
- Use Gipformer ASR for public-test audio and keep its transcript in saved conversations.
- Exactly one `ToolEnv` is created per public-test sample; publish `submission.jsonl` only after every sample succeeds.
- Write each submission row as `id`, `tool_calls`, and `state_hash`.
- Do not add model or runtime dependencies beyond the existing Kaggle setup.
