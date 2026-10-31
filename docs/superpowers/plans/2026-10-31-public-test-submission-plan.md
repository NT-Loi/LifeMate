# Public Test Submission Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the complete public-test split through Gipformer ASR and LifeMate, produce a valid root `submission.jsonl`, and save conversations and logs.

**Architecture:** Extend the existing evaluator so public-test mode accepts an ASR and skips gold-state replay, preserving exactly one `ToolEnv` per sample. A small `main.py` calls that runner, refuses to publish if rows are missing or any sample failed, then copies the evaluator's predictions file to the repository root. A dedicated Kaggle notebook installs pinned ASR/LLM requirements and invokes the full-run CLI.

**Tech Stack:** Python, existing `evaluate.py`/`LifeMateAgent`, Gipformer ONNX ASR, Qwen Transformers, Jupyter notebook.

**Spec:** `docs/superpowers/specs/2026-10-31-public-test-submission-design.md`

## Global Constraints

- Include every public-test row; do not read any `*_environment` files.
- Use Gipformer ASR for public-test audio and keep its transcript in saved conversations.
- Exactly one `ToolEnv` is created per public-test sample; publish `submission.jsonl` only after every sample succeeds.
- Write each submission row as `id`, `tool_calls`, and `state_hash`.
- Do not add model or runtime dependencies beyond the existing Kaggle setup.

## Review Focus

- Public-test rows have `audio` but no `audio_text`: pass Gipformer through single- and multi-GPU evaluator paths and store the recognized transcript.
- Public-test rows have no gold `label`: skip gold `ToolEnv` replay while leaving train scoring unchanged.
- Missing dataset or empty split: fail before loading ASR/LLM and leave the current root submission untouched.
- Any ASR/LLM/sample error or incomplete result list: retain timestamped diagnostics but do not publish a new root submission.
- Kaggle exposes zero, one, or two GPUs: construct CLI arguments from detected hardware and run the complete split without a sample limit.

---

### Task 1: Make the shared evaluator support public-test audio

**Files:**
- Modify: `evaluate.py`

**Interfaces:**
- Add `asr: str | None = None` to `run_evaluation` and `_run_evaluation` and pass it to `LifeMateAgent` in auto-shard, single-device, and worker paths.
- Add the same `asr` value to `_worker_process` arguments.

- [ ] Pass the selected ASR through every runner path; keep the default `None` so existing train evaluation remains unchanged.
- [ ] In `_process_single_sample`, skip gold-environment construction/replay for `mode == "public_test"`; record the agent's generated transcript, calls, and `state_hash`. Keep the existing gold replay and scoring for train.
- [ ] Run `python3 -m py_compile evaluate.py` and inspect the resulting diff for all three device paths and the one-ToolEnv public-test branch.

### Task 2: Add the full-split CLI and root submission output

**Files:**
- Create: `main.py`

**Consumes:** Task 1's `run_evaluation(..., asr="Gipformer")` interface.

- [ ] Parse `--model-id`, `--gpus`, `--max-new-tokens` (default 512), and `--output-dir`; do not expose a sample limit because this entrypoint always runs the full public-test split.
- [ ] Load every JSONL row from `MODE["public_test"]["data_path"]`; fail before model initialization when the file is missing or empty.
- [ ] Call `run_evaluation` with `mode="public_test"` and Gipformer ASR. After it returns, require one result per input and no `error` results before copying `predictions.jsonl` to root `submission.jsonl`.
- [ ] Keep timestamped evaluator outputs intact and print the submission/artifact paths. Run `python3 -m py_compile main.py` and `python3 main.py --help`.

### Task 3: Add the Kaggle public-test notebook

**Files:**
- Create: `kaggle_public_test.ipynb`

**Consumes:** Task 2's full-run CLI.

- [ ] Add cells to check accelerator and public-test data, set the CUDA allocator option before importing Torch, and install the existing Qwen and Gipformer requirement files.
- [ ] Download pinned Qwen3.5-2B and Gipformer 1.5 assets before inference; choose up to two visible GPUs and call `main.py` without a sample limit.
- [ ] Copy the generated `submission.jsonl` to `/kaggle/working/submission.jsonl` and show its row count plus links/paths for timestamped conversations and logs.
- [ ] Validate notebook JSON and Python cell syntax; ensure all cells are unexecuted and contain no output artifacts.

## Final Verification

- Run `python3 -m py_compile evaluate.py main.py`.
- Validate the notebook JSON and Python cell syntax, then run `git diff --check`.
- Do not run full inference locally; validate the full split from the Kaggle notebook because the local machine has no GPU.
