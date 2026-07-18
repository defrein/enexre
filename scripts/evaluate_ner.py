#!/usr/bin/env python
"""Evaluate the selected BC5CDR NER checkpoint on a held-out split."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


IGNORE_INDEX = -100


class JsonlNerDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "pmid": row["pmid"],
            "chunk_index": row["chunk_index"],
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
            "tokens": row.get("tokens"),
            "offset_mapping": row.get("offset_mapping"),
        }


@dataclass
class NerBatchCollator:
    pad_token_id: int

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(row["input_ids"]) for row in rows)

        input_ids = []
        attention_mask = []
        labels = []
        metadata = []

        for row in rows:
            length = len(row["input_ids"])
            pad_length = max_length - length
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad_length)
            attention_mask.append(row["attention_mask"] + [0] * pad_length)
            labels.append(row["labels"] + [IGNORE_INDEX] * pad_length)
            metadata.append(
                {
                    "pmid": row["pmid"],
                    "chunk_index": row["chunk_index"],
                    "tokens": row.get("tokens"),
                    "offset_mapping": row.get("offset_mapping"),
                }
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "metadata": metadata,
        }


def load_label_map(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(index): label for index, label in data["id_to_label"].items()}


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
            "No usable checkpoint found from metrics files. "
            "Run full training first, or pass --checkpoint explicitly."
        )

    return max(candidates, key=lambda row: float(row.get("best_dev_f1", -1.0)))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id_to_label: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    steps = 0
    true_sequences: list[list[str]] = []
    pred_sequences: list[list[str]] = []
    prediction_rows: list[dict[str, Any]] = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = outputs.logits.argmax(dim=-1)

        total_loss += float(outputs.loss.item())
        steps += 1

        for metadata, pred_row, label_row in zip(
            batch["metadata"],
            preds.cpu().tolist(),
            labels.cpu().tolist(),
        ):
            true_labels = []
            pred_labels = []
            tokens = []
            offsets = []

            raw_tokens = metadata.get("tokens") or []
            raw_offsets = metadata.get("offset_mapping") or []

            for token_index, (pred_id, label_id) in enumerate(zip(pred_row, label_row)):
                if label_id == IGNORE_INDEX:
                    continue
                true_label = id_to_label[int(label_id)]
                pred_label = id_to_label[int(pred_id)]
                true_labels.append(true_label)
                pred_labels.append(pred_label)
                if token_index < len(raw_tokens):
                    tokens.append(raw_tokens[token_index])
                if token_index < len(raw_offsets):
                    offsets.append(raw_offsets[token_index])

            true_sequences.append(true_labels)
            pred_sequences.append(pred_labels)
            prediction_rows.append(
                {
                    "pmid": metadata["pmid"],
                    "chunk_index": metadata["chunk_index"],
                    "tokens": tokens,
                    "offset_mapping": offsets,
                    "true_labels": true_labels,
                    "pred_labels": pred_labels,
                }
            )

    metrics = {
        "loss": total_loss / max(steps, 1),
        "precision": precision_score(true_sequences, pred_sequences, zero_division=0),
        "recall": recall_score(true_sequences, pred_sequences, zero_division=0),
        "f1": f1_score(true_sequences, pred_sequences, zero_division=0),
        "classification_report": classification_report(
            true_sequences,
            pred_sequences,
            zero_division=0,
            output_dict=True,
        ),
    }
    return metrics, prediction_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BC5CDR NER checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--metrics-glob", type=str, default="results/ner/*_metrics.json")
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--test", type=Path, default=Path("data/processed/ner/test.jsonl"))
    parser.add_argument("--label-map", type=Path, default=Path("data/processed/ner/label_map.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/ner/best_test_metrics.json"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("predictions/ner/best_test_predictions.jsonl"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for path in [args.test, args.label_map]:
        if not path.exists():
            print(f"Missing required file: {path}")
            return 2

    selected_run = None
    checkpoint = args.checkpoint
    if checkpoint is None:
        selected_run = select_best_checkpoint(args.metrics_glob, args.include_smoke)
        checkpoint = Path(selected_run["checkpoint_dir"])

    if not checkpoint.exists():
        print(f"Missing checkpoint directory: {checkpoint}")
        return 2

    id_to_label = load_label_map(args.label_map)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint).to(device)

    dataset = JsonlNerDataset(args.test)
    collator = NerBatchCollator(pad_token_id=tokenizer.pad_token_id or 0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    metrics, prediction_rows = evaluate(model, dataloader, device, id_to_label)
    summary = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "test_path": str(args.test),
        "selected_run": selected_run,
        "test": metrics,
    }

    save_json(args.output, summary)
    save_jsonl(args.predictions, prediction_rows)

    print(f"checkpoint={checkpoint}")
    print(f"test_precision={metrics['precision']:.4f}")
    print(f"test_recall={metrics['recall']:.4f}")
    print(f"test_f1={metrics['f1']:.4f}")
    print(f"Metrics: {args.output}")
    print(f"Predictions: {args.predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
