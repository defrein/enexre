#!/usr/bin/env python
"""Train PubMedBERT NER model for BC5CDR.

Default mode trains one NER configuration from configs/config_ner.yaml.
Use --smoke-test to run a tiny CPU-friendly check before doing real training.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup


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
            "offset_mapping": row.get("offset_mapping"),
            "tokens": row.get("tokens"),
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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_label_map(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    label_to_id = {label: int(index) for label, index in data["label_to_id"].items()}
    id_to_label = {int(index): label for index, label in data["id_to_label"].items()}
    return label_to_id, id_to_label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def predictions_to_sequences(
    predictions: list[list[int]],
    labels: list[list[int]],
    id_to_label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    true_sequences: list[list[str]] = []
    pred_sequences: list[list[str]] = []

    for pred_row, label_row in zip(predictions, labels):
        true_labels: list[str] = []
        pred_labels: list[str] = []
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id == IGNORE_INDEX:
                continue
            true_labels.append(id_to_label[int(label_id)])
            pred_labels.append(id_to_label[int(pred_id)])
        true_sequences.append(true_labels)
        pred_sequences.append(pred_labels)

    return true_sequences, pred_sequences


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id_to_label: dict[int, str],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    steps = 0
    all_predictions: list[list[int]] = []
    all_labels: list[list[int]] = []
    prediction_rows: list[dict[str, Any]] = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        preds = logits.argmax(dim=-1)

        total_loss += float(outputs.loss.item())
        steps += 1
        all_predictions.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        for metadata, pred_row, label_row in zip(
            batch["metadata"],
            preds.cpu().tolist(),
            labels.cpu().tolist(),
        ):
            true_labels = []
            pred_labels = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == IGNORE_INDEX:
                    continue
                true_labels.append(id_to_label[int(label_id)])
                pred_labels.append(id_to_label[int(pred_id)])
            prediction_rows.append(
                {
                    "pmid": metadata["pmid"],
                    "chunk_index": metadata["chunk_index"],
                    "true_labels": true_labels,
                    "pred_labels": pred_labels,
                }
            )

    true_sequences, pred_sequences = predictions_to_sequences(all_predictions, all_labels, id_to_label)
    metrics = {
        "loss": total_loss / max(steps, 1),
        "precision": precision_score(true_sequences, pred_sequences, zero_division=0),
        "recall": recall_score(true_sequences, pred_sequences, zero_division=0),
        "f1": f1_score(true_sequences, pred_sequences, zero_division=0),
    }
    return metrics, prediction_rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_one_run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    label_to_id, id_to_label = load_label_map(args.label_map)
    seed = int(args.seed if args.seed is not None else config["training"]["random_seeds"][0])

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    checkpoint = args.model_checkpoint or config["checkpoint"]
    learning_rate = args.learning_rate or float(config["training"]["learning_rates"][0])
    batch_size = args.batch_size or int(config["training"]["batch_sizes"][0])
    epochs = args.epochs or int(config["training"]["max_epochs"])
    weight_decay = args.weight_decay if args.weight_decay is not None else float(config["training"]["weight_decay"])
    patience = args.patience if args.patience is not None else int(config["training"]["early_stopping_patience"])

    if args.smoke_test:
        epochs = min(epochs, 1)
        patience = 1

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_dataset = limited_dataset(JsonlNerDataset(args.train), args.train_limit)
    dev_dataset = limited_dataset(JsonlNerDataset(args.dev), args.dev_limit)

    collator = NerBatchCollator(pad_token_id=tokenizer.pad_token_id or 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_training_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * args.warmup_ratio),
        num_training_steps=total_training_steps,
    )

    run_name = args.run_name or f"seed{seed}_lr{learning_rate}_bs{batch_size}"
    if args.smoke_test:
        run_name = f"smoke_{run_name}"

    checkpoint_dir = args.checkpoints_dir / run_name
    results_dir = args.results_dir
    logs_dir = args.logs_dir
    predictions_dir = args.predictions_dir

    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        progress = tqdm(train_loader, desc=f"Training epoch {epoch}/{epochs}")

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item())
            epoch_steps += 1
            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break

        dev_metrics, dev_predictions = evaluate(model, dev_loader, device, id_to_label)
        train_loss = epoch_loss / max(epoch_steps, 1)
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "dev": dev_metrics,
        }
        history.append(epoch_record)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"dev_precision={dev_metrics['precision']:.4f} "
            f"dev_recall={dev_metrics['recall']:.4f} "
            f"dev_f1={dev_metrics['f1']:.4f}"
        )

        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_epoch = epoch
            stale_epochs = 0
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            save_jsonl(predictions_dir / f"{run_name}_dev_predictions.jsonl", dev_predictions)
        else:
            stale_epochs += 1

        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

        if stale_epochs >= patience:
            print(f"Early stopping after {stale_epochs} stale epoch(s).")
            break

    summary = {
        "run_name": run_name,
        "seed": seed,
        "checkpoint": checkpoint,
        "device": str(device),
        "smoke_test": args.smoke_test,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_dev_f1": best_f1,
        "checkpoint_dir": str(checkpoint_dir),
        "history": history,
    }
    save_json(results_dir / f"{run_name}_metrics.json", summary)
    save_json(logs_dir / f"{run_name}_training_log.json", summary)

    if args.smoke_test and args.cleanup_smoke_checkpoint and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        summary["checkpoint_removed_after_smoke_test"] = True
        save_json(results_dir / f"{run_name}_metrics.json", summary)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC5CDR NER token classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/config_ner.yaml"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/ner/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/processed/ner/dev.jsonl"))
    parser.add_argument("--label-map", type=Path, default=Path("data/processed/ner/label_map.json"))
    parser.add_argument("--model-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--cleanup-smoke-checkpoint", action="store_true")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints/ner"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/ner"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/ner"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions/ner"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for path in [args.train, args.dev, args.label_map, args.config]:
        if not path.exists():
            print(f"Missing required file: {path}")
            return 2

    if args.smoke_test:
        args.train_limit = args.train_limit or 8
        args.dev_limit = args.dev_limit or 8
        args.batch_size = args.batch_size or 2
        args.max_train_steps = args.max_train_steps or 2

    summary = train_one_run(args)
    print(f"Best dev F1: {summary['best_dev_f1']:.4f}")
    print(f"Metrics: {args.results_dir / (summary['run_name'] + '_metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
