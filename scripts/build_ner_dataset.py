#!/usr/bin/env python
"""Build token-level NER data from BC5CDR PubTator files.

This script implements Tahap 3 from PENELITIAN_STEP.md:
  - combine title and abstract with one space,
  - convert Chemical and Disease spans to BIO labels,
  - tokenize with PubMedBERT,
  - align labels using offset_mapping,
  - write JSONL files for training/evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from validate_bc5cdr import Annotation, Document, parse_pubtator


LABELS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
VALID_ENTITY_TYPES = {"Chemical", "Disease"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def document_text(document: Document) -> str:
    return f"{document.title} {document.abstract}"


def valid_annotations(document: Document, text: str) -> tuple[list[Annotation], list[dict[str, Any]]]:
    annotations: list[Annotation] = []
    invalid: list[dict[str, Any]] = []

    for annotation in document.annotations:
        if annotation.entity_type not in VALID_ENTITY_TYPES:
            invalid.append(
                {
                    "pmid": document.pmid,
                    "mention": annotation.mention,
                    "entity_type": annotation.entity_type,
                    "reason": "unsupported_entity_type",
                }
            )
            continue

        if not (0 <= annotation.start < annotation.end <= len(text)):
            invalid.append(
                {
                    "pmid": document.pmid,
                    "mention": annotation.mention,
                    "start": annotation.start,
                    "end": annotation.end,
                    "reason": "offset_out_of_range",
                }
            )
            continue

        if text[annotation.start : annotation.end] != annotation.mention:
            invalid.append(
                {
                    "pmid": document.pmid,
                    "mention": annotation.mention,
                    "text_at_offset": text[annotation.start : annotation.end],
                    "start": annotation.start,
                    "end": annotation.end,
                    "reason": "offset_text_mismatch",
                }
            )
            continue

        annotations.append(annotation)

    annotations.sort(key=lambda item: (item.start, item.end, item.entity_type))
    return annotations, invalid


def build_char_labels(
    document: Document,
    text: str,
    annotations: list[Annotation],
) -> tuple[list[tuple[str, int] | None], list[dict[str, Any]]]:
    char_labels: list[tuple[str, int] | None] = [None] * len(text)
    conflicts: list[dict[str, Any]] = []

    for annotation in annotations:
        for index in range(annotation.start, annotation.end):
            existing = char_labels[index]
            new_value = (annotation.entity_type, annotation.start)
            if existing is not None and existing != new_value:
                conflicts.append(
                    {
                        "pmid": document.pmid,
                        "position": index,
                        "existing": existing[0],
                        "new": annotation.entity_type,
                        "mention": annotation.mention,
                    }
                )
                continue
            char_labels[index] = new_value

    return char_labels, conflicts


def token_label_for_span(
    char_labels: list[tuple[str, int] | None],
    start: int,
    end: int,
) -> str:
    tagged_positions = [
        (index, value)
        for index, value in enumerate(char_labels[start:end], start=start)
        if value is not None
    ]

    if not tagged_positions:
        return "O"

    first_index, first_value = tagged_positions[0]
    entity_type, entity_start = first_value

    if first_index == entity_start:
        return f"B-{entity_type}"
    return f"I-{entity_type}"


def align_document(
    document: Document,
    tokenizer: Any,
    max_length: int,
    stride: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text = document_text(document)
    annotations, invalid_annotations = valid_annotations(document, text)
    char_labels, conflicts = build_char_labels(document, text, annotations)

    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    examples: list[dict[str, Any]] = []
    label_counter: Counter[str] = Counter()
    entity_token_counter: Counter[str] = Counter()
    chunks = len(encoded["input_ids"])

    for chunk_index in range(chunks):
        input_ids = encoded["input_ids"][chunk_index]
        attention_mask = encoded["attention_mask"][chunk_index]
        offset_mapping = encoded["offset_mapping"][chunk_index]
        sequence_ids = encoded.sequence_ids(chunk_index)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        labels: list[int] = []
        label_names: list[str] = []

        for offset, sequence_id in zip(offset_mapping, sequence_ids):
            start, end = offset
            if sequence_id != 0 or start == end:
                label_name = "IGN"
                label_id = -100
            else:
                label_name = token_label_for_span(char_labels, start, end)
                label_id = LABEL_TO_ID[label_name]
                label_counter[label_name] += 1
                if label_name != "O":
                    entity_token_counter[label_name] += 1

            labels.append(label_id)
            label_names.append(label_name)

        examples.append(
            {
                "pmid": document.pmid,
                "chunk_index": chunk_index,
                "num_chunks": chunks,
                "text": text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "label_names": label_names,
                "tokens": tokens,
                "offset_mapping": [list(offset) for offset in offset_mapping],
            }
        )

    report = {
        "pmid": document.pmid,
        "text_length": len(text),
        "annotation_count": len(document.annotations),
        "valid_annotation_count": len(annotations),
        "invalid_annotations": invalid_annotations,
        "char_label_conflicts": conflicts,
        "chunks": chunks,
        "uses_overflow": chunks > 1,
        "label_counts": dict(label_counter),
        "entity_token_counts": dict(entity_token_counter),
    }
    return examples, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_split(
    split_name: str,
    path: Path,
    tokenizer: Any,
    max_length: int,
    stride: int,
    output_dir: Path,
) -> dict[str, Any]:
    documents = parse_pubtator(path)
    examples: list[dict[str, Any]] = []
    document_reports: list[dict[str, Any]] = []
    label_counter: Counter[str] = Counter()

    for document in tqdm(documents, desc=f"Building {split_name}"):
        document_examples, document_report = align_document(
            document=document,
            tokenizer=tokenizer,
            max_length=max_length,
            stride=stride,
        )
        examples.extend(document_examples)
        document_reports.append(document_report)
        label_counter.update(document_report["label_counts"])

    write_jsonl(output_dir / f"{split_name}.jsonl", examples)

    invalid_annotation_count = sum(
        len(report["invalid_annotations"]) for report in document_reports
    )
    conflict_count = sum(len(report["char_label_conflicts"]) for report in document_reports)
    overflow_document_count = sum(1 for report in document_reports if report["uses_overflow"])

    return {
        "split": split_name,
        "source_path": str(path),
        "documents": len(documents),
        "examples": len(examples),
        "overflow_documents": overflow_document_count,
        "invalid_annotation_count": invalid_annotation_count,
        "char_label_conflict_count": conflict_count,
        "label_counts": dict(label_counter),
        "document_reports": document_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BC5CDR NER BIO dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/config_ner.yaml"))
    parser.add_argument("--train", type=Path, default=Path("data/bc5cdr/train.txt"))
    parser.add_argument("--dev", type=Path, default=Path("data/bc5cdr/dev.txt"))
    parser.add_argument("--test", type=Path, default=Path("data/bc5cdr/test.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ner"))
    parser.add_argument("--report", type=Path, default=Path("results/ner_preprocessing_report.json"))
    parser.add_argument("--stride", type=int, default=128)
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

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset_mapping alignment.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    label_map = {
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(index): label for label, index in LABEL_TO_ID.items()},
        "ignore_index": -100,
    }
    (args.output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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
                stride=args.stride,
                output_dir=args.output_dir,
            )
        )

    report = {
        "checkpoint": checkpoint,
        "max_sequence_length": max_length,
        "stride": args.stride,
        "output_dir": str(args.output_dir),
        "labels": LABELS,
        "splits": split_reports,
        "passed": all(
            split["invalid_annotation_count"] == 0
            and split["char_label_conflict_count"] == 0
            for split in split_reports
        ),
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"NER dataset written to {args.output_dir}")
    print(f"Report written to {args.report}")
    print(f"Passed: {report['passed']}")
    for split in split_reports:
        print(
            f"{split['split']}: docs={split['documents']}, "
            f"examples={split['examples']}, "
            f"overflow_docs={split['overflow_documents']}, "
            f"invalid_ann={split['invalid_annotation_count']}, "
            f"conflicts={split['char_label_conflict_count']}"
        )

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
