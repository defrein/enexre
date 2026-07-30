#!/usr/bin/env python
"""Build BC5CDR relation extraction candidates and co-occurrence baseline.

This implements Tahap 6-8 from PENELITIAN_STEP.md for the gold-entity RE
setting:
  - generate every unique Chemical-Disease concept pair per document,
  - label pairs from CID annotations,
  - mark all mentions of the target pair in the document text,
  - report candidate counts and the co-occurrence baseline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from validate_bc5cdr import Annotation, Document, parse_pubtator, split_mesh_ids


VALID_ENTITY_TYPES = {"Chemical", "Disease"}
MARKER_TOKENS = ["[CHEM]", "[/CHEM]", "[DISEASE]", "[/DISEASE]"]
INVALID_CONCEPT_IDS = {"", "-", "-1"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def document_text(document: Document) -> str:
    return f"{document.title} {document.abstract}"


def valid_mesh_ids(mesh_id: str) -> list[str]:
    return [mesh for mesh in split_mesh_ids(mesh_id) if mesh not in INVALID_CONCEPT_IDS]


def valid_annotations(document: Document, text: str) -> list[Annotation]:
    annotations: list[Annotation] = []
    for annotation in document.annotations:
        if annotation.entity_type not in VALID_ENTITY_TYPES:
            continue
        if not valid_mesh_ids(annotation.mesh_id):
            continue
        if not (0 <= annotation.start < annotation.end <= len(text)):
            continue
        if text[annotation.start : annotation.end] != annotation.mention:
            continue
        annotations.append(annotation)
    return annotations


def concept_ids_by_type(annotations: list[Annotation]) -> dict[str, set[str]]:
    concept_ids: dict[str, set[str]] = {"Chemical": set(), "Disease": set()}
    for annotation in annotations:
        concept_ids[annotation.entity_type].update(valid_mesh_ids(annotation.mesh_id))
    return concept_ids


def cid_pairs(document: Document) -> set[tuple[str, str]]:
    return {
        (relation.chemical_id, relation.disease_id)
        for relation in document.relations
        if relation.relation_type == "CID"
    }


def target_mentions(
    annotations: list[Annotation],
    entity_type: str,
    mesh_id: str,
) -> list[Annotation]:
    return [
        annotation
        for annotation in annotations
        if annotation.entity_type == entity_type
        and mesh_id in valid_mesh_ids(annotation.mesh_id)
    ]


def marked_text_for_pair(
    text: str,
    chemical_mentions: list[Annotation],
    disease_mentions: list[Annotation],
) -> tuple[str, bool]:
    insertions: list[tuple[int, str]] = []

    for mention in chemical_mentions:
        insertions.append((mention.start, "[CHEM] "))
        insertions.append((mention.end, " [/CHEM]"))

    for mention in disease_mentions:
        insertions.append((mention.start, "[DISEASE] "))
        insertions.append((mention.end, " [/DISEASE]"))

    spans = [(mention.start, mention.end) for mention in chemical_mentions + disease_mentions]
    has_overlap = any(
        left_start < right_end and right_start < left_end
        for index, (left_start, left_end) in enumerate(spans)
        for right_start, right_end in spans[index + 1 :]
    )

    marked = text
    for position, marker in sorted(insertions, key=lambda item: item[0], reverse=True):
        marked = marked[:position] + marker + marked[position:]

    return marked, has_overlap


def count_marker_tokens(marked_text: str) -> dict[str, int]:
    return {marker: marked_text.count(marker) for marker in MARKER_TOKENS}


def build_document_examples(
    document: Document,
    tokenizer: Any | None,
    max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = document_text(document)
    annotations = valid_annotations(document, text)
    concepts = concept_ids_by_type(annotations)
    positives = cid_pairs(document)

    examples: list[dict[str, Any]] = []
    report = {
        "pmid": document.pmid,
        "chemical_concepts": len(concepts["Chemical"]),
        "disease_concepts": len(concepts["Disease"]),
        "candidate_pairs": 0,
        "positive_pairs": 0,
        "negative_pairs": 0,
        "marked_inputs_over_max_length": 0,
        "marker_overlap_count": 0,
        "missing_marker_count": 0,
    }

    for chemical_id in sorted(concepts["Chemical"]):
        for disease_id in sorted(concepts["Disease"]):
            chemical_mentions = target_mentions(annotations, "Chemical", chemical_id)
            disease_mentions = target_mentions(annotations, "Disease", disease_id)
            marked_text, has_overlap = marked_text_for_pair(
                text=text,
                chemical_mentions=chemical_mentions,
                disease_mentions=disease_mentions,
            )
            marker_counts = count_marker_tokens(marked_text)
            label = int((chemical_id, disease_id) in positives)
            token_count = None
            over_max_length = None

            if tokenizer is not None:
                token_count = len(tokenizer(marked_text, add_special_tokens=True)["input_ids"])
                over_max_length = token_count > max_length

            missing_marker = not all(marker_counts[marker] > 0 for marker in MARKER_TOKENS)

            examples.append(
                {
                    "pmid": document.pmid,
                    "chemical_id": chemical_id,
                    "disease_id": disease_id,
                    "label": label,
                    "text": text,
                    "marked_text": marked_text,
                    "chemical_mentions": [
                        asdict(mention) for mention in chemical_mentions
                    ],
                    "disease_mentions": [
                        asdict(mention) for mention in disease_mentions
                    ],
                    "marker_counts": marker_counts,
                    "token_count": token_count,
                    "over_max_length": over_max_length,
                }
            )

            report["candidate_pairs"] += 1
            report["positive_pairs"] += label
            report["negative_pairs"] += 1 - label
            report["marked_inputs_over_max_length"] += int(bool(over_max_length))
            report["marker_overlap_count"] += int(has_overlap)
            report["missing_marker_count"] += int(missing_marker)

    return examples, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def baseline_metrics(positive_pairs: int, negative_pairs: int) -> dict[str, Any]:
    tp = positive_pairs
    fp = negative_pairs
    fn = 0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def process_split(
    split_name: str,
    path: Path,
    tokenizer: Any | None,
    max_length: int,
    output_dir: Path,
) -> dict[str, Any]:
    documents = parse_pubtator(path)
    examples: list[dict[str, Any]] = []
    document_reports: list[dict[str, Any]] = []

    for document in tqdm(documents, desc=f"Building RE {split_name}"):
        document_examples, document_report = build_document_examples(
            document=document,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        examples.extend(document_examples)
        document_reports.append(document_report)

    write_jsonl(output_dir / f"{split_name}.jsonl", examples)

    label_counts = Counter(example["label"] for example in examples)
    over_max_length = sum(1 for example in examples if example["over_max_length"])
    marker_missing = sum(
        1
        for example in examples
        if not all(example["marker_counts"][marker] > 0 for marker in MARKER_TOKENS)
    )

    return {
        "split": split_name,
        "source_path": str(path),
        "documents": len(documents),
        "candidate_pairs": len(examples),
        "cid_pairs": label_counts[1],
        "non_cid_pairs": label_counts[0],
        "marked_inputs_over_max_length": over_max_length,
        "missing_marker_count": marker_missing,
        "marker_overlap_count": sum(
            report["marker_overlap_count"] for report in document_reports
        ),
        "baseline_cooccurrence": baseline_metrics(
            positive_pairs=label_counts[1],
            negative_pairs=label_counts[0],
        ),
        "document_reports": document_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BC5CDR RE candidate dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/config_re.yaml"))
    parser.add_argument("--train", type=Path, default=Path("data/bc5cdr/train.txt"))
    parser.add_argument("--dev", type=Path, default=Path("data/bc5cdr/dev.txt"))
    parser.add_argument("--test", type=Path, default=Path("data/bc5cdr/test.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/re"))
    parser.add_argument("--report", type=Path, default=Path("results/re_preprocessing_report.json"))
    parser.add_argument("--skip-tokenizer", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = config["checkpoint"]
    max_length = int(config["training"]["max_sequence_length"])

    missing = [path for path in [args.train, args.dev, args.test] if not path.exists()]
    if missing:
        print("Missing input files:")
        for path in missing:
            print(f"  - {path}")
        return 2

    tokenizer = None
    if not args.skip_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        tokenizer.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    split_reports = []
    for split_name, path in [
        ("train", args.train),
        ("dev", args.dev),
        ("test", args.test),
    ]:
        split_reports.append(
            process_split(
                split_name=split_name,
                path=path,
                tokenizer=tokenizer,
                max_length=max_length,
                output_dir=args.output_dir,
            )
        )

    summary = {
        "subsets": {
            split["split"]: {
                key: split[key]
                for key in [
                    "documents",
                    "candidate_pairs",
                    "cid_pairs",
                    "non_cid_pairs",
                    "marked_inputs_over_max_length",
                    "missing_marker_count",
                    "marker_overlap_count",
                    "baseline_cooccurrence",
                ]
            }
            for split in split_reports
        },
        "passed": all(
            split["missing_marker_count"] == 0
            and split["marker_overlap_count"] == 0
            for split in split_reports
        ),
    }

    report = {
        "checkpoint": checkpoint,
        "max_sequence_length": max_length,
        "marker_tokens": MARKER_TOKENS,
        "long_text_strategy": "full_marked_document_recorded; training script should crop around target markers if token_count > max_sequence_length",
        "output_dir": str(args.output_dir),
        "summary": summary,
        "splits": split_reports,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"RE dataset written to {args.output_dir}")
    print(f"Report written to {args.report}")
    print(f"Passed: {summary['passed']}")
    for split in split_reports:
        baseline = split["baseline_cooccurrence"]
        print(
            f"{split['split']}: docs={split['documents']}, "
            f"CID={split['cid_pairs']}, non-CID={split['non_cid_pairs']}, "
            f"total={split['candidate_pairs']}, "
            f"over_max_len={split['marked_inputs_over_max_length']}, "
            f"baseline_f1={baseline['f1']:.4f}"
        )

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
