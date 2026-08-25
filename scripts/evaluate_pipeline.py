#!/usr/bin/env python
"""Evaluate the end-to-end BC5CDR NER-RE pipeline on the test split.

Tahap 11 uses predicted NER spans to form Chemical-Disease candidates, maps
exact span/type matches to gold MeSH IDs for scoring only, and evaluates the
selected RE checkpoint on those candidates.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from build_ner_dataset import document_text
from build_re_dataset import (
    MARKER_TOKENS,
    cid_pairs,
    count_marker_tokens,
    marked_text_for_pair,
    valid_mesh_ids,
)
from train_re import JsonlRelationDataset, RelationBatchCollator, evaluate
from validate_bc5cdr import Annotation, parse_pubtator


Span = tuple[str, int, int, str]
RelationKey = tuple[str, str, str]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_best_re_checkpoint(metrics_glob: str, include_smoke: bool) -> dict[str, Any]:
    candidates = []
    for path in sorted(Path().glob(metrics_glob)):
        data = read_json(path)
        if "checkpoint_dir" not in data:
            continue
        if data.get("smoke_test") and not include_smoke:
            continue
        checkpoint_dir = Path(data["checkpoint_dir"])
        if checkpoint_dir.exists():
            candidates.append({**data, "metrics_path": str(path)})

    if not candidates:
        raise FileNotFoundError(
            "No usable RE checkpoint found. Pass --re-checkpoint or run RE training first."
        )

    return max(candidates, key=lambda row: float(row.get("best_dev_f1", -1.0)))


def close_active_span(active: dict[str, Any] | None, spans: list[dict[str, Any]]) -> None:
    if active is not None:
        spans.append(active)


def extract_predicted_spans(ner_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    spans_by_pmid: dict[str, dict[tuple[int, int, str], dict[str, Any]]] = defaultdict(dict)

    for row in ner_rows:
        pmid = str(row["pmid"])
        active: dict[str, Any] | None = None
        row_spans: list[dict[str, Any]] = []

        for label, offset in zip(row["pred_labels"], row["offset_mapping"]):
            start, end = int(offset[0]), int(offset[1])
            if start == end:
                continue

            prefix, _, entity_type = label.partition("-")
            if label == "O" or prefix not in {"B", "I"}:
                close_active_span(active, row_spans)
                active = None
                continue

            if prefix == "B" or active is None or active["entity_type"] != entity_type:
                close_active_span(active, row_spans)
                active = {"start": start, "end": end, "entity_type": entity_type}
            else:
                active["end"] = end

        close_active_span(active, row_spans)

        for span in row_spans:
            key = (span["start"], span["end"], span["entity_type"])
            existing = spans_by_pmid[pmid].get(key)
            if existing is None:
                spans_by_pmid[pmid][key] = {"pmid": pmid, **span}

    return {
        pmid: sorted(spans.values(), key=lambda item: (item["start"], item["end"], item["entity_type"]))
        for pmid, spans in spans_by_pmid.items()
    }


def build_gold_indexes(test_path: Path) -> tuple[dict[str, Any], dict[str, dict[tuple[int, int, str], Annotation]], set[RelationKey]]:
    documents = {document.pmid: document for document in parse_pubtator(test_path)}
    annotation_index: dict[str, dict[tuple[int, int, str], Annotation]] = {}
    gold_relations: set[RelationKey] = set()

    for pmid, document in documents.items():
        text = document_text(document)
        annotation_index[pmid] = {}
        for annotation in document.annotations:
            key = (annotation.start, annotation.end, annotation.entity_type)
            if text[annotation.start : annotation.end] == annotation.mention:
                annotation_index[pmid][key] = annotation
        for chemical_id, disease_id in cid_pairs(document):
            gold_relations.add((pmid, chemical_id, disease_id))

    return documents, annotation_index, gold_relations


def matched_mentions_for_document(
    pmid: str,
    predicted_spans: list[dict[str, Any]],
    gold_annotations: dict[tuple[int, int, str], Annotation],
    text: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    mentions_by_type_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    report = {
        "pmid": pmid,
        "predicted_spans": len(predicted_spans),
        "matched_spans": 0,
        "unmatched_spans": 0,
        "matched_chemical_concepts": 0,
        "matched_disease_concepts": 0,
    }

    for span in predicted_spans:
        key = (span["start"], span["end"], span["entity_type"])
        annotation = gold_annotations.get(key)
        if annotation is None:
            report["unmatched_spans"] += 1
            continue

        mesh_ids = valid_mesh_ids(annotation.mesh_id)
        if not mesh_ids:
            report["unmatched_spans"] += 1
            continue

        report["matched_spans"] += 1
        for mesh_id in mesh_ids:
            mention = Annotation(
                pmid=pmid,
                start=span["start"],
                end=span["end"],
                mention=text[span["start"] : span["end"]],
                entity_type=span["entity_type"],
                mesh_id=mesh_id,
            )
            mentions_by_type_id[f"{span['entity_type']}::{mesh_id}"].append(asdict(mention))

    report["matched_chemical_concepts"] = sum(
        1 for key in mentions_by_type_id if key.startswith("Chemical::")
    )
    report["matched_disease_concepts"] = sum(
        1 for key in mentions_by_type_id if key.startswith("Disease::")
    )
    return mentions_by_type_id, report


def build_pipeline_candidates(
    test_path: Path,
    ner_predictions_path: Path,
    output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], set[RelationKey]]:
    documents, gold_annotation_index, gold_relations = build_gold_indexes(test_path)
    predicted_spans_by_pmid = extract_predicted_spans(load_jsonl(ner_predictions_path))
    examples: list[dict[str, Any]] = []
    document_reports = []

    for pmid, document in documents.items():
        text = document_text(document)
        mentions_by_type_id, report = matched_mentions_for_document(
            pmid=pmid,
            predicted_spans=predicted_spans_by_pmid.get(pmid, []),
            gold_annotations=gold_annotation_index[pmid],
            text=text,
        )
        chemical_ids = sorted(
            key.split("::", maxsplit=1)[1]
            for key in mentions_by_type_id
            if key.startswith("Chemical::")
        )
        disease_ids = sorted(
            key.split("::", maxsplit=1)[1]
            for key in mentions_by_type_id
            if key.startswith("Disease::")
        )
        positives = cid_pairs(document)

        for chemical_id in chemical_ids:
            for disease_id in disease_ids:
                chemical_mentions = [
                    Annotation(**mention)
                    for mention in mentions_by_type_id[f"Chemical::{chemical_id}"]
                ]
                disease_mentions = [
                    Annotation(**mention)
                    for mention in mentions_by_type_id[f"Disease::{disease_id}"]
                ]
                marked_text, has_overlap = marked_text_for_pair(
                    text=text,
                    chemical_mentions=chemical_mentions,
                    disease_mentions=disease_mentions,
                )
                examples.append(
                    {
                        "pmid": pmid,
                        "chemical_id": chemical_id,
                        "disease_id": disease_id,
                        "label": int((chemical_id, disease_id) in positives),
                        "text": text,
                        "marked_text": marked_text,
                        "chemical_mentions": [asdict(mention) for mention in chemical_mentions],
                        "disease_mentions": [asdict(mention) for mention in disease_mentions],
                        "marker_counts": count_marker_tokens(marked_text),
                        "marker_overlap": has_overlap,
                        "source": "predicted_ner_exact_span_mapped_to_gold_mesh_for_evaluation",
                    }
                )

        report["candidate_pairs"] = len(chemical_ids) * len(disease_ids)
        document_reports.append(report)

    save_jsonl(output_path, examples)
    summary = {
        "documents": len(documents),
        "candidate_pairs": len(examples),
        "gold_cid_relations": len(gold_relations),
        "predicted_spans": sum(report["predicted_spans"] for report in document_reports),
        "matched_spans": sum(report["matched_spans"] for report in document_reports),
        "unmatched_spans": sum(report["unmatched_spans"] for report in document_reports),
        "documents_without_candidates": sum(
            1 for report in document_reports if report["candidate_pairs"] == 0
        ),
        "marker_overlap_count": sum(1 for example in examples if example["marker_overlap"]),
        "document_reports": document_reports,
    }
    return examples, summary, gold_relations


def relation_metrics(gold_relations: set[RelationKey], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_relations = {
        (str(row["pmid"]), row["chemical_id"], row["disease_id"])
        for row in prediction_rows
        if int(row["prediction"]) == 1
    }
    true_positive = len(predicted_relations & gold_relations)
    false_positive = len(predicted_relations - gold_relations)
    false_negative = len(gold_relations - predicted_relations)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_relations": len(predicted_relations),
        "gold_relations": len(gold_relations),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BC5CDR NER-RE pipeline.")
    parser.add_argument("--test", type=Path, default=Path("data/bc5cdr/test.txt"))
    parser.add_argument(
        "--ner-predictions",
        type=Path,
        default=Path("predictions/ner/best_test_predictions.jsonl"),
    )
    parser.add_argument("--re-checkpoint", type=Path, default=None)
    parser.add_argument("--re-metrics-glob", type=str, default="results/re/*_metrics.json")
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/processed/pipeline/test_candidates.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pipeline/best_test_metrics.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("predictions/pipeline/best_test_predictions.jsonl"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in [args.test, args.ner_predictions]:
        if not path.exists():
            print(f"Missing required file: {path}")
            return 2

    selected_re_run = None
    re_checkpoint = args.re_checkpoint
    threshold = args.threshold
    if re_checkpoint is None:
        selected_re_run = select_best_re_checkpoint(args.re_metrics_glob, args.include_smoke)
        re_checkpoint = Path(selected_re_run["checkpoint_dir"])
        if threshold is None:
            threshold = float(selected_re_run.get("best_threshold", 0.5))

    if threshold is None:
        threshold = 0.5

    if not re_checkpoint.exists():
        print(f"Missing RE checkpoint directory: {re_checkpoint}")
        return 2

    _, candidate_summary, gold_relations = build_pipeline_candidates(
        test_path=args.test,
        ner_predictions_path=args.ner_predictions,
        output_path=args.candidates,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(re_checkpoint, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    model = AutoModelForSequenceClassification.from_pretrained(re_checkpoint).to(device)
    model.resize_token_embeddings(len(tokenizer))

    dataset = JsonlRelationDataset(args.candidates, tokenizer, args.max_length)
    if args.test_limit is not None:
        dataset.rows = dataset.rows[: args.test_limit]
    collator = RelationBatchCollator(pad_token_id=tokenizer.pad_token_id or 0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model_metrics, prediction_rows = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        class_weights=None,
        thresholds=[threshold],
    )
    for row in prediction_rows:
        row["prediction"] = int(row["cid_probability"] >= threshold)

    pipeline_metrics = relation_metrics(gold_relations, prediction_rows)
    pipeline_metrics.update(
        {
            "threshold": threshold,
            "loss": model_metrics["loss"],
            "cropped_count": model_metrics["cropped_count"],
            "missing_marker_count": model_metrics["missing_marker_count"],
            "closest_pair_fallback_count": model_metrics["closest_pair_fallback_count"],
        }
    )

    summary = {
        "status": "pipeline_test",
        "ner_predictions": str(args.ner_predictions),
        "re_checkpoint": str(re_checkpoint),
        "selected_re_run": selected_re_run,
        "candidate_path": str(args.candidates),
        "prediction_path": str(args.predictions),
        "candidate_summary": candidate_summary,
        "pipeline_test": pipeline_metrics,
        "note": (
            "Predicted NER spans are mapped to gold MeSH IDs only when span and type "
            "match exactly; unmatched NER spans are excluded from RE candidates and "
            "missed gold relations are counted as pipeline false negatives."
        ),
    }
    save_json(args.output, summary)
    save_jsonl(args.predictions, prediction_rows)

    print(f"NER predictions: {args.ner_predictions}")
    print(f"RE checkpoint: {re_checkpoint}")
    print(f"threshold={threshold:.2f}")
    print(f"candidate_pairs={candidate_summary['candidate_pairs']}")
    print(f"pipeline_precision={pipeline_metrics['precision']:.4f}")
    print(f"pipeline_recall={pipeline_metrics['recall']:.4f}")
    print(f"pipeline_f1={pipeline_metrics['f1']:.4f}")
    print(f"Metrics: {args.output}")
    print(f"Predictions: {args.predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
