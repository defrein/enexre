#!/usr/bin/env python
"""Train PubMedBERT relation extraction model for BC5CDR.

Default mode trains one RE configuration from configs/config_re.yaml.
Use --smoke-test to run a tiny CPU-friendly check before full training.
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
import torch.nn.functional as F
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


MARKER_TOKENS = ["[CHEM]", "[/CHEM]", "[DISEASE]", "[/DISEASE]"]


class JsonlRelationDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded, crop_report = encode_marked_text(
            tokenizer=self.tokenizer,
            text=row["marked_text"],
            max_length=self.max_length,
        )
        if not crop_report["all_markers_present"]:
            fallback_text = closest_pair_marked_text(
                text=row["text"],
                chemical_mentions=row["chemical_mentions"],
                disease_mentions=row["disease_mentions"],
            )
            encoded, crop_report = encode_marked_text(
                tokenizer=self.tokenizer,
                text=fallback_text,
                max_length=self.max_length,
            )
            crop_report["used_closest_pair_fallback"] = True
        else:
            crop_report["used_closest_pair_fallback"] = False
        return {
            "pmid": row["pmid"],
            "chemical_id": row["chemical_id"],
            "disease_id": row["disease_id"],
            "label": int(row["label"]),
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "crop_report": crop_report,
        }


@dataclass
class RelationBatchCollator:
    pad_token_id: int

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        attention_mask = []
        labels = []
        metadata = []

        for row in rows:
            pad_length = max_length - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad_length)
            attention_mask.append(row["attention_mask"] + [0] * pad_length)
            labels.append(row["label"])
            metadata.append(
                {
                    "pmid": row["pmid"],
                    "chemical_id": row["chemical_id"],
                    "disease_id": row["disease_id"],
                    "crop_report": row["crop_report"],
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


def marker_ranges(text: str, start_marker: str, end_marker: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    search_start = 0
    while True:
        start = text.find(start_marker, search_start)
        if start == -1:
            break
        end = text.find(end_marker, start + len(start_marker))
        if end == -1:
            break
        ranges.append((start, end + len(end_marker)))
        search_start = end + len(end_marker)
    return ranges


def closest_target_marker_span(text: str) -> tuple[int, int] | None:
    chemical_ranges = marker_ranges(text, "[CHEM]", "[/CHEM]")
    disease_ranges = marker_ranges(text, "[DISEASE]", "[/DISEASE]")
    spans = [
        (min(chem_start, disease_start), max(chem_end, disease_end))
        for chem_start, chem_end in chemical_ranges
        for disease_start, disease_end in disease_ranges
    ]
    if not spans:
        return None
    return min(spans, key=lambda span: span[1] - span[0])


def closest_pair_marked_text(
    text: str,
    chemical_mentions: list[dict[str, Any]],
    disease_mentions: list[dict[str, Any]],
    context_chars: int = 700,
) -> str:
    pairs = [
        (chemical, disease)
        for chemical in chemical_mentions
        for disease in disease_mentions
    ]
    if not pairs:
        return text

    chemical, disease = min(
        pairs,
        key=lambda pair: max(pair[0]["end"], pair[1]["end"]) - min(pair[0]["start"], pair[1]["start"]),
    )
    pair_start = min(chemical["start"], disease["start"])
    pair_end = max(chemical["end"], disease["end"])
    if pair_end - pair_start > context_chars * 2:
        return " ".join(
            [
                marked_local_snippet(text, chemical, "[CHEM]", "[/CHEM]", context_chars),
                marked_local_snippet(text, disease, "[DISEASE]", "[/DISEASE]", context_chars),
            ]
        )

    insertions = [
        (chemical["start"], "[CHEM] "),
        (chemical["end"], " [/CHEM]"),
        (disease["start"], "[DISEASE] "),
        (disease["end"], " [/DISEASE]"),
    ]
    marked = text
    for position, marker in sorted(insertions, key=lambda item: item[0], reverse=True):
        marked = marked[:position] + marker + marked[position:]
    return marked


def marked_local_snippet(
    text: str,
    mention: dict[str, Any],
    start_marker: str,
    end_marker: str,
    context_chars: int,
) -> str:
    snippet_start = max(0, mention["start"] - context_chars // 2)
    snippet_end = min(len(text), mention["end"] + context_chars // 2)
    relative_start = mention["start"] - snippet_start
    relative_end = mention["end"] - snippet_start
    snippet = text[snippet_start:snippet_end]
    return (
        snippet[:relative_start]
        + start_marker
        + " "
        + snippet[relative_start:relative_end]
        + " "
        + end_marker
        + snippet[relative_end:]
    )


def encode_marked_text(tokenizer: Any, text: str, max_length: int) -> tuple[dict[str, list[int]], dict[str, Any]]:
    content_limit = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    full = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = full["input_ids"]
    offsets = full["offset_mapping"]
    cropped = len(input_ids) > content_limit
    crop_start = 0
    crop_end = len(input_ids)

    if cropped:
        span = closest_target_marker_span(text)
        if span is None:
            center = len(input_ids) // 2
            crop_start = max(0, center - content_limit // 2)
        else:
            marker_start_char, marker_end_char = span
            marker_token_indices = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > marker_start_char and start < marker_end_char
            ]
            if marker_token_indices:
                marker_start = marker_token_indices[0]
                marker_end = marker_token_indices[-1] + 1
                marker_width = marker_end - marker_start
                if marker_width >= content_limit:
                    crop_start = marker_start
                else:
                    left_context = (content_limit - marker_width) // 2
                    crop_start = marker_start - left_context
            else:
                center = len(input_ids) // 2
                crop_start = center - content_limit // 2

        crop_start = max(0, min(crop_start, len(input_ids) - content_limit))
        crop_end = crop_start + content_limit
        input_ids = input_ids[crop_start:crop_end]

    input_ids_with_special_tokens = []
    if tokenizer.cls_token_id is not None:
        input_ids_with_special_tokens.append(tokenizer.cls_token_id)
    input_ids_with_special_tokens.extend(input_ids)
    if tokenizer.sep_token_id is not None:
        input_ids_with_special_tokens.append(tokenizer.sep_token_id)
    encoded = {
        "input_ids": input_ids_with_special_tokens,
        "attention_mask": [1] * len(input_ids_with_special_tokens),
    }
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    marker_presence = {marker: marker in tokens for marker in MARKER_TOKENS}
    crop_report = {
        "cropped": cropped,
        "original_token_count": len(full["input_ids"]) + tokenizer.num_special_tokens_to_add(pair=False),
        "final_token_count": len(encoded["input_ids"]),
        "crop_start": crop_start,
        "crop_end": crop_end,
        "all_markers_present": all(marker_presence.values()),
        "marker_presence": marker_presence,
    }
    return encoded, crop_report


def class_weights_from_dataset(dataset: Dataset, device: torch.device) -> torch.Tensor:
    labels = []
    source = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(source))
    for index in indices:
        labels.append(int(source.rows[index]["label"]))

    positive = sum(labels)
    negative = len(labels) - positive
    positive_weight = negative / positive if positive else 1.0
    return torch.tensor([1.0, positive_weight], dtype=torch.float, device=device)


def binary_metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, Any]:
    predictions = [int(probability >= threshold) for probability in probabilities]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    tp = sum(1 for gold, pred in zip(labels, predictions) if gold == 1 and pred == 1)
    fp = sum(1 for gold, pred in zip(labels, predictions) if gold == 0 and pred == 1)
    fn = sum(1 for gold, pred in zip(labels, predictions) if gold == 1 and pred == 0)
    tn = sum(1 for gold, pred in zip(labels, predictions) if gold == 0 and pred == 0)
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor | None,
    thresholds: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    steps = 0
    labels: list[int] = []
    probabilities: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    cropped_count = 0
    missing_marker_count = 0
    closest_pair_fallback_count = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = F.cross_entropy(logits, batch_labels, weight=class_weights)
        probs = torch.softmax(logits, dim=-1)[:, 1]

        total_loss += float(loss.item())
        steps += 1

        batch_probs = probs.cpu().tolist()
        batch_label_values = batch_labels.cpu().tolist()
        labels.extend(batch_label_values)
        probabilities.extend(batch_probs)

        for metadata, probability, label in zip(batch["metadata"], batch_probs, batch_label_values):
            crop_report = metadata["crop_report"]
            cropped_count += int(crop_report["cropped"])
            missing_marker_count += int(not crop_report["all_markers_present"])
            closest_pair_fallback_count += int(crop_report.get("used_closest_pair_fallback", False))
            prediction_rows.append(
                {
                    "pmid": metadata["pmid"],
                    "chemical_id": metadata["chemical_id"],
                    "disease_id": metadata["disease_id"],
                    "label": int(label),
                    "cid_probability": float(probability),
                    "crop_report": crop_report,
                }
            )

    threshold_metrics = [binary_metrics(labels, probabilities, threshold) for threshold in thresholds]
    selected = max(threshold_metrics, key=lambda item: (item["f1"], item["precision"], item["recall"]))
    metrics = {
        "loss": total_loss / max(steps, 1),
        "threshold_metrics": threshold_metrics,
        "selected_threshold": selected["threshold"],
        "precision": selected["precision"],
        "recall": selected["recall"],
        "f1": selected["f1"],
        "true_positive": selected["true_positive"],
        "false_positive": selected["false_positive"],
        "false_negative": selected["false_negative"],
        "true_negative": selected["true_negative"],
        "cropped_count": cropped_count,
        "missing_marker_count": missing_marker_count,
        "closest_pair_fallback_count": closest_pair_fallback_count,
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
    seed = int(args.seed if args.seed is not None else config["training"]["random_seeds"][0])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = args.model_checkpoint or config["checkpoint"]
    learning_rate = args.learning_rate or float(config["training"]["learning_rates"][0])
    batch_size = args.batch_size or int(config["training"]["batch_sizes"][0])
    epochs = args.epochs or int(config["training"]["max_epochs"])
    weight_decay = args.weight_decay if args.weight_decay is not None else float(config["training"]["weight_decay"])
    thresholds = [float(value) for value in (args.thresholds or config["thresholds"])]
    max_length = int(args.max_length or config["training"]["max_sequence_length"])

    if args.smoke_test:
        epochs = min(epochs, 1)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
        id2label={0: "non-CID", 1: "CID"},
        label2id={"non-CID": 0, "CID": 1},
        ignore_mismatched_sizes=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_dataset = limited_dataset(JsonlRelationDataset(args.train, tokenizer, max_length), args.train_limit)
    dev_dataset = limited_dataset(JsonlRelationDataset(args.dev, tokenizer, max_length), args.dev_limit)

    collator = RelationBatchCollator(pad_token_id=tokenizer.pad_token_id or 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    class_weights = None if args.no_class_weights else class_weights_from_dataset(train_dataset, device)
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
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    stale_epochs = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        progress = tqdm(train_loader, desc=f"Training RE epoch {epoch}/{epochs}")

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, labels, weight=class_weights)
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

        dev_metrics, dev_predictions = evaluate(model, dev_loader, device, class_weights, thresholds)
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
            f"dev_threshold={dev_metrics['selected_threshold']:.2f} "
            f"dev_precision={dev_metrics['precision']:.4f} "
            f"dev_recall={dev_metrics['recall']:.4f} "
            f"dev_f1={dev_metrics['f1']:.4f}"
        )

        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_epoch = epoch
            best_threshold = dev_metrics["selected_threshold"]
            stale_epochs = 0
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            save_jsonl(args.predictions_dir / f"{run_name}_dev_predictions.jsonl", dev_predictions)
        else:
            stale_epochs += 1

        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

        if stale_epochs >= args.patience:
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
        "max_length": max_length,
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "best_epoch": best_epoch,
        "best_dev_f1": best_f1,
        "best_threshold": best_threshold,
        "checkpoint_dir": str(checkpoint_dir),
        "history": history,
    }
    save_json(args.results_dir / f"{run_name}_metrics.json", summary)
    save_json(args.logs_dir / f"{run_name}_training_log.json", summary)

    if args.smoke_test and args.cleanup_smoke_checkpoint and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        summary["checkpoint_removed_after_smoke_test"] = True
        save_json(args.results_dir / f"{run_name}_metrics.json", summary)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BC5CDR RE sequence classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/config_re.yaml"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/re/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/processed/re/dev.jsonl"))
    parser.add_argument("--model-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--thresholds", type=float, nargs="*", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--cleanup-smoke-checkpoint", action="store_true")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints/re"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/re"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs/re"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions/re"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in [args.train, args.dev, args.config]:
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
    print(f"Best threshold: {summary['best_threshold']:.2f}")
    print(f"Metrics: {args.results_dir / (summary['run_name'] + '_metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
