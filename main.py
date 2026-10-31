"""Run the complete public-test split and write the competition submission."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

from config import MODE


def load_public_test_samples(data_path: Path) -> list[dict]:
    if not data_path.is_file():
        raise SystemExit(f"Public-test data file not found: {data_path}")

    samples = []
    with data_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"Invalid JSON at {data_path}:{line_number}: {error}") from error
            if not isinstance(sample, dict) or not sample.get("id"):
                raise SystemExit(f"Invalid public-test sample at {data_path}:{line_number}: missing id")
            samples.append(sample)

    if not samples:
        raise SystemExit(f"Public-test split is empty: {data_path}")
    return samples


def validate_predictions(predictions_path: Path, expected_ids: list[str]) -> None:
    if not predictions_path.is_file():
        raise RuntimeError(f"Evaluator did not write predictions: {predictions_path}")

    rows = []
    with predictions_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) != {"id", "tool_calls", "state_hash"}:
                raise RuntimeError(f"Invalid submission fields at {predictions_path}:{line_number}")
            if not isinstance(row["tool_calls"], list):
                raise RuntimeError(f"tool_calls must be a list at {predictions_path}:{line_number}")
            if not isinstance(row["state_hash"], str) or not row["state_hash"]:
                raise RuntimeError(f"Missing state_hash at {predictions_path}:{line_number}")
            rows.append(row)

    actual_ids = [row["id"] for row in rows]
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Incomplete or reordered predictions: expected {len(expected_ids)} rows, "
            f"got {len(actual_ids)}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gipformer and LifeMate on every public-test sample."
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-2B",
        help="Hugging Face model ID (default: Qwen/Qwen3.5-2B).",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="GPU IDs, e.g. '0' or '0,1'; omit to use all available devices.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per agent turn (default: 512).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for timestamped predictions, conversations, and logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_public_test_samples(Path(MODE["public_test"]["data_path"]))
    expected_ids = [sample["id"] for sample in samples]
    if len(set(expected_ids)) != len(expected_ids):
        raise SystemExit("Public-test split contains duplicate sample IDs.")

    from evaluate import run_evaluation

    output_dir = args.output_dir or (
        Path("eval_results")
        / f"public_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    results = run_evaluation(
        samples=samples,
        mode="public_test",
        gpus=args.gpus,
        output_dir=output_dir,
        model_id=args.model_id,
        asr="Gipformer",
        max_new_tokens=args.max_new_tokens,
    )

    errors = [result for result in results if result.get("error")]
    if len(results) != len(samples) or errors:
        failed_ids = [result.get("id") for result in errors]
        raise RuntimeError(
            f"Refusing to publish an incomplete submission: expected {len(samples)} "
            f"results, got {len(results)}, failed IDs={failed_ids}. "
            f"Diagnostics are in {output_dir}."
        )

    predictions_path = Path(output_dir) / "predictions.jsonl"
    validate_predictions(predictions_path, expected_ids)

    submission_path = Path("submission.jsonl")
    temporary_path = submission_path.with_suffix(".jsonl.tmp")
    try:
        shutil.copyfile(predictions_path, temporary_path)
        os.replace(temporary_path, submission_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Submission: {submission_path.resolve()}")
    print(f"Artifacts: {Path(output_dir).resolve()}")
    print(f"Samples: {len(samples)}")
    print(f"Conversations: {(Path(output_dir) / 'conversations.txt').resolve()}")
    print(f"Run log: {(Path(output_dir) / 'run.log').resolve()}")


if __name__ == "__main__":
    main()
