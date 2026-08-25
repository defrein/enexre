#!/usr/bin/env python
"""Summarize BC5CDR RE and NER-RE pipeline errors for Tahap 13."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_ner_dataset import document_text
from build_re_dataset import cid_pairs, valid_annotations, valid_mesh_ids
from validate_bc5cdr import parse_pubtator


RelationKey = tuple[str, str, str]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def relation_key(row: dict[str, Any]) -> RelationKey:
    return (str(row["pmid"]), row["chemical_id"], row["disease_id"])


def build_gold_context(test_path: Path) -> tuple[set[RelationKey], dict[str, dict[str, Any]]]:
    gold_relations: set[RelationKey] = set()
    context: dict[str, dict[str, Any]] = {}

    for document in parse_pubtator(test_path):
        text = document_text(document)
        annotations = valid_annotations(document, text)
        mentions_by_mesh: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for annotation in annotations:
            for mesh_id in valid_mesh_ids(annotation.mesh_id):
                mentions_by_mesh[mesh_id].append(
                    {
                        "start": annotation.start,
                        "end": annotation.end,
                        "mention": annotation.mention,
                        "type": annotation.entity_type,
                    }
                )

        for chemical_id, disease_id in cid_pairs(document):
            gold_relations.add((document.pmid, chemical_id, disease_id))

        context[document.pmid] = {
            "text": text,
            "mentions_by_mesh": dict(mentions_by_mesh),
        }

    return gold_relations, context


def concise_evidence(text: str, mentions: list[dict[str, Any]], radius: int = 140) -> str:
    if not mentions:
        return text[: radius * 2].strip()
    start = max(0, min(mention["start"] for mention in mentions) - radius)
    end = min(len(text), max(mention["end"] for mention in mentions) + radius)
    snippet = text[start:end].strip()
    return " ".join(snippet.split())


def example_row(
    key: RelationKey,
    context: dict[str, dict[str, Any]],
    probability: float | None,
    reason: str,
) -> dict[str, Any]:
    pmid, chemical_id, disease_id = key
    doc = context[pmid]
    chemical_mentions = doc["mentions_by_mesh"].get(chemical_id, [])
    disease_mentions = doc["mentions_by_mesh"].get(disease_id, [])
    return {
        "pmid": pmid,
        "chemical_id": chemical_id,
        "disease_id": disease_id,
        "chemical_mentions": [mention["mention"] for mention in chemical_mentions[:5]],
        "disease_mentions": [mention["mention"] for mention in disease_mentions[:5]],
        "cid_probability": probability,
        "reason": reason,
        "evidence": concise_evidence(doc["text"], chemical_mentions + disease_mentions),
    }


def relation_sets(predictions: list[dict[str, Any]]) -> tuple[set[RelationKey], dict[RelationKey, dict[str, Any]]]:
    by_key = {relation_key(row): row for row in predictions}
    positive = {key for key, row in by_key.items() if int(row["prediction"]) == 1}
    return positive, by_key


def summarize_binary_errors(
    name: str,
    predictions: list[dict[str, Any]],
    gold_relations: set[RelationKey],
    context: dict[str, dict[str, Any]],
    sample_size: int,
) -> dict[str, Any]:
    predicted_positive, by_key = relation_sets(predictions)
    candidate_keys = set(by_key)
    false_positives = sorted(predicted_positive - gold_relations)
    false_negatives = sorted(gold_relations - predicted_positive)
    missing_candidate_fns = [key for key in false_negatives if key not in candidate_keys]
    rejected_candidate_fns = [key for key in false_negatives if key in candidate_keys]

    fp_examples = sorted(
        false_positives,
        key=lambda key: by_key[key]["cid_probability"],
        reverse=True,
    )[:sample_size]
    rejected_fn_examples = sorted(
        rejected_candidate_fns,
        key=lambda key: by_key[key]["cid_probability"],
    )[:sample_size]

    return {
        "name": name,
        "counts": {
            "candidate_pairs": len(candidate_keys),
            "false_positive": len(false_positives),
            "false_negative": len(false_negatives),
            "false_negative_missing_candidate": len(missing_candidate_fns),
            "false_negative_rejected_candidate": len(rejected_candidate_fns),
        },
        "false_positive_examples": [
            example_row(
                key=key,
                context=context,
                probability=by_key[key]["cid_probability"],
                reason="Predicted CID but pair is not a gold CID relation.",
            )
            for key in fp_examples
        ],
        "false_negative_rejected_examples": [
            example_row(
                key=key,
                context=context,
                probability=by_key[key]["cid_probability"],
                reason="Gold CID candidate was formed, but RE probability was below threshold.",
            )
            for key in rejected_fn_examples
        ],
        "false_negative_missing_candidate_examples": [
            example_row(
                key=key,
                context=context,
                probability=None,
                reason="Gold CID pair was not evaluated because NER-derived candidate was missing.",
            )
            for key in missing_candidate_fns[:sample_size]
        ],
    }


def pipeline_candidate_loss_breakdown(
    pipeline_candidates: list[dict[str, Any]],
    gold_relations: set[RelationKey],
    context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates_by_pmid: dict[str, set[tuple[str, str]]] = defaultdict(set)
    chemical_ids_by_pmid: dict[str, set[str]] = defaultdict(set)
    disease_ids_by_pmid: dict[str, set[str]] = defaultdict(set)

    for row in pipeline_candidates:
        pmid = str(row["pmid"])
        candidates_by_pmid[pmid].add((row["chemical_id"], row["disease_id"]))
        chemical_ids_by_pmid[pmid].add(row["chemical_id"])
        disease_ids_by_pmid[pmid].add(row["disease_id"])

    reasons = Counter()
    examples = []
    for pmid, chemical_id, disease_id in sorted(gold_relations):
        if (chemical_id, disease_id) in candidates_by_pmid[pmid]:
            continue
        chemical_present = chemical_id in chemical_ids_by_pmid[pmid]
        disease_present = disease_id in disease_ids_by_pmid[pmid]
        if not chemical_present and not disease_present:
            reason = "chemical_and_disease_missing"
        elif not chemical_present:
            reason = "chemical_missing"
        elif not disease_present:
            reason = "disease_missing"
        else:
            reason = "pair_not_formed"
        reasons[reason] += 1
        if len(examples) < 10:
            examples.append(
                example_row(
                    key=(pmid, chemical_id, disease_id),
                    context=context,
                    probability=None,
                    reason=reason,
                )
            )

    return {
        "missing_gold_relation_reasons": dict(reasons),
        "examples": examples,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    pipeline = report["pipeline"]
    re_gold = report["re_gold"]
    lines = [
        "# Error Analysis Summary",
        "",
        "## Aggregate Counts",
        "",
        "| Evaluation | Candidates | FP | FN | FN missing candidate | FN rejected candidate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in [re_gold, pipeline]:
        counts = item["counts"]
        lines.append(
            f"| {item['name']} | {counts['candidate_pairs']} | {counts['false_positive']} | "
            f"{counts['false_negative']} | {counts['false_negative_missing_candidate']} | "
            f"{counts['false_negative_rejected_candidate']} |"
        )

    lines.extend(["", "## Pipeline Missing-Candidate Breakdown", ""])
    for reason, count in report["pipeline_candidate_loss"]["missing_gold_relation_reasons"].items():
        lines.append(f"- {reason}: {count}")

    lines.extend(["", "## Pipeline False Positive Examples", ""])
    for row in pipeline["false_positive_examples"][:5]:
        lines.append(
            f"- PMID {row['pmid']} {row['chemical_id']} -> {row['disease_id']} "
            f"p={row['cid_probability']:.4f}; chem={row['chemical_mentions'][:2]}; "
            f"disease={row['disease_mentions'][:2]}"
        )

    lines.extend(["", "## Pipeline False Negative Examples", ""])
    for row in (
        pipeline["false_negative_rejected_examples"][:3]
        + pipeline["false_negative_missing_candidate_examples"][:3]
    ):
        probability = "missing" if row["cid_probability"] is None else f"{row['cid_probability']:.4f}"
        lines.append(
            f"- PMID {row['pmid']} {row['chemical_id']} -> {row['disease_id']} "
            f"p={probability}; reason={row['reason']}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BC5CDR relation extraction errors.")
    parser.add_argument("--test", type=Path, default=Path("data/bc5cdr/test.txt"))
    parser.add_argument("--re-predictions", type=Path, default=Path("predictions/re/best_test_predictions.jsonl"))
    parser.add_argument(
        "--pipeline-predictions",
        type=Path,
        default=Path("predictions/pipeline/best_test_predictions.jsonl"),
    )
    parser.add_argument(
        "--pipeline-candidates",
        type=Path,
        default=Path("data/processed/pipeline/test_candidates.jsonl"),
    )
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/error_analysis/error_analysis.json"))
    parser.add_argument("--markdown", type=Path, default=Path("results/error_analysis/error_analysis.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = [args.test, args.re_predictions, args.pipeline_predictions, args.pipeline_candidates]
    for path in required:
        if not path.exists():
            print(f"Missing required file: {path}")
            return 2

    gold_relations, context = build_gold_context(args.test)
    re_predictions = load_jsonl(args.re_predictions)
    pipeline_predictions = load_jsonl(args.pipeline_predictions)
    pipeline_candidates = load_jsonl(args.pipeline_candidates)

    report = {
        "gold_relations": len(gold_relations),
        "re_gold": summarize_binary_errors(
            name="RE with gold entities",
            predictions=re_predictions,
            gold_relations=gold_relations,
            context=context,
            sample_size=args.sample_size,
        ),
        "pipeline": summarize_binary_errors(
            name="Pipeline NER-RE",
            predictions=pipeline_predictions,
            gold_relations=gold_relations,
            context=context,
            sample_size=args.sample_size,
        ),
        "pipeline_candidate_loss": pipeline_candidate_loss_breakdown(
            pipeline_candidates=pipeline_candidates,
            gold_relations=gold_relations,
            context=context,
        ),
    }
    save_json(args.output, report)
    write_markdown(args.markdown, report)

    pipeline_counts = report["pipeline"]["counts"]
    loss = report["pipeline_candidate_loss"]["missing_gold_relation_reasons"]
    print(f"Error analysis written to {args.output}")
    print(f"Markdown summary written to {args.markdown}")
    print(
        "pipeline: "
        f"FP={pipeline_counts['false_positive']}, "
        f"FN={pipeline_counts['false_negative']}, "
        f"FN_missing_candidate={pipeline_counts['false_negative_missing_candidate']}, "
        f"FN_rejected_candidate={pipeline_counts['false_negative_rejected_candidate']}"
    )
    print(f"missing_candidate_breakdown={loss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
