#!/usr/bin/env python
"""Evaluate a selected BC5CDR RE checkpoint on a held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_re import (
    MARKER_TOKENS,
    JsonlRelationDataset,
    RelationBatchCollator,
    binary_metrics,
    evaluate,
    limited_dataset,
)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_best_checkpoint(metrics_glob: str, include_smoke: bool) -> dict[str, Any]:
    candidates = []
    for path in sorted(Path().glob(metrics_glob)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("smoke_test") and not include_smoke:
            continue
        checkpoint_dir = Path(data["checkpoint_dir"])
        if checkpoint_dir.exists():
            candidates.append({**data, "metrics_path": str(path)})

    if not candidates:
        raise FileNotFoundError(
            "No usable RE checkpoint found from metrics files. "
            "Run full training first, or pass --checkpoint explicitly."
        )

    return max(candidates, key=lambda row: float(row.get("best_dev_f1", -1.0)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BC5CDR RE checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--metrics-glob", type=str, default="results/re/*_metrics.json")
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--test", type=Path, default=Path("data/processed/re/test.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/re/best_test_metrics.json"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("predictions/re/best_test_predictions.jsonl"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.test.exists():
        print(f"Missing required file: {args.test}")
        return 2

    selected_run = None
    checkpoint = args.checkpoint
    threshold = args.threshold
    if checkpoint is None:
        selected_run = select_best_checkpoint(args.metrics_glob, args.include_smoke)
        checkpoint = Path(selected_run["checkpoint_dir"])
        if threshold is None:
            threshold = float(selected_run.get("best_threshold", 0.5))

    if threshold is None:
        threshold = 0.5

    if not checkpoint.exists():
        print(f"Missing checkpoint directory: {checkpoint}")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.resize_token_embeddings(len(tokenizer))

    dataset = limited_dataset(JsonlRelationDataset(args.test, tokenizer, args.max_length), args.test_limit)
    collator = RelationBatchCollator(pad_token_id=tokenizer.pad_token_id or 0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    metrics, prediction_rows = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        class_weights=None,
        thresholds=[threshold],
    )

    labels = [row["label"] for row in prediction_rows]
    probabilities = [row["cid_probability"] for row in prediction_rows]
    test_metrics = binary_metrics(labels, probabilities, threshold)
    test_metrics.update(
        {
            "loss": metrics["loss"],
            "cropped_count": metrics["cropped_count"],
            "missing_marker_count": metrics["missing_marker_count"],
        }
    )

    for row in prediction_rows:
        row["prediction"] = int(row["cid_probability"] >= threshold)

    summary = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "test_path": str(args.test),
        "selected_run": selected_run,
        "test": test_metrics,
    }
    save_json(args.output, summary)
    save_jsonl(args.predictions, prediction_rows)

    print(f"checkpoint={checkpoint}")
    print(f"threshold={threshold:.2f}")
    print(f"test_precision={test_metrics['precision']:.4f}")
    print(f"test_recall={test_metrics['recall']:.4f}")
    print(f"test_f1={test_metrics['f1']:.4f}")
    print(f"Metrics: {args.output}")
    print(f"Predictions: {args.predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
