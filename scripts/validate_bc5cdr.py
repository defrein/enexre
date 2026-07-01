#!/usr/bin/env python
"""Validate BC5CDR PubTator files for the research protocol.

Expected input files:
  data/bc5cdr/train.txt
  data/bc5cdr/dev.txt
  data/bc5cdr/test.txt

Alternative paths can be provided with command-line arguments.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


VALID_ENTITY_TYPES = {"Chemical", "Disease"}
EXPECTED_DOC_COUNTS = {
    "train": 500,
    "dev": 500,
    "test": 500,
}


@dataclass
class Annotation:
    pmid: str
    start: int
    end: int
    mention: str
    entity_type: str
    mesh_id: str


@dataclass
class Relation:
    pmid: str
    relation_type: str
    chemical_id: str
    disease_id: str


@dataclass
class Document:
    pmid: str
    title: str = ""
    abstract: str = ""
    annotations: list[Annotation] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


def split_mesh_ids(mesh_id: str) -> list[str]:
    return [part for part in mesh_id.split("|") if part and part != "-"]


def parse_pubtator(path: Path) -> list[Document]:
    documents: dict[str, Document] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue

            if "|t|" in line:
                pmid, title = line.split("|t|", maxsplit=1)
                documents.setdefault(pmid, Document(pmid=pmid)).title = title
                continue

            if "|a|" in line:
                pmid, abstract = line.split("|a|", maxsplit=1)
                documents.setdefault(pmid, Document(pmid=pmid)).abstract = abstract
                continue

            parts = line.split("\t")
            pmid = parts[0]
            document = documents.setdefault(pmid, Document(pmid=pmid))

            if len(parts) >= 6:
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid annotation offset") from exc

                document.annotations.append(
                    Annotation(
                        pmid=pmid,
                        start=start,
                        end=end,
                        mention=parts[3],
                        entity_type=parts[4],
                        mesh_id=parts[5],
                    )
                )
                continue

            if len(parts) == 4:
                document.relations.append(
                    Relation(
                        pmid=pmid,
                        relation_type=parts[1],
                        chemical_id=parts[2],
                        disease_id=parts[3],
                    )
                )
                continue

            raise ValueError(f"{path}:{line_number}: unsupported PubTator line: {line}")

    return list(documents.values())


def matching_text_strategy(document: Document, annotation: Annotation) -> str | None:
    candidates = {
        "title_space_abstract": f"{document.title} {document.abstract}",
        "title_newline_abstract": f"{document.title}\n{document.abstract}",
        "title_abstract_no_separator": f"{document.title}{document.abstract}",
    }

    for strategy, text in candidates.items():
        if 0 <= annotation.start <= annotation.end <= len(text):
            if text[annotation.start : annotation.end] == annotation.mention:
                return strategy
    return None


def validate_subset(name: str, path: Path) -> dict:
    documents = parse_pubtator(path)
    invalid_annotations: list[dict] = []
    invalid_relations: list[dict] = []
    duplicate_relations: list[dict] = []
    relation_counter: Counter[tuple[str, str, str]] = Counter()
    text_strategy_counter: Counter[str] = Counter()
    entity_mentions = Counter()
    unique_concepts: dict[str, set[str]] = {
        "Chemical": set(),
        "Disease": set(),
    }

    for document in documents:
        if not document.title and not document.abstract:
            invalid_annotations.append(
                {
                    "pmid": document.pmid,
                    "reason": "document_has_no_title_or_abstract",
                }
            )

        entity_ids_by_type: dict[str, set[str]] = {
            "Chemical": set(),
            "Disease": set(),
        }

        for annotation in document.annotations:
            entity_mentions[annotation.entity_type] += 1

            if annotation.entity_type not in VALID_ENTITY_TYPES:
                invalid_annotations.append(
                    {
                        "pmid": annotation.pmid,
                        "mention": annotation.mention,
                        "entity_type": annotation.entity_type,
                        "reason": "invalid_entity_type",
                    }
                )
                continue

            mesh_ids = split_mesh_ids(annotation.mesh_id)
            if not mesh_ids:
                invalid_annotations.append(
                    {
                        "pmid": annotation.pmid,
                        "mention": annotation.mention,
                        "entity_type": annotation.entity_type,
                        "reason": "missing_mesh_id",
                    }
                )
            else:
                entity_ids_by_type[annotation.entity_type].update(mesh_ids)
                unique_concepts[annotation.entity_type].update(mesh_ids)

            strategy = matching_text_strategy(document, annotation)
            if strategy is None:
                invalid_annotations.append(
                    {
                        "pmid": annotation.pmid,
                        "mention": annotation.mention,
                        "entity_type": annotation.entity_type,
                        "start": annotation.start,
                        "end": annotation.end,
                        "reason": "offset_text_mismatch",
                    }
                )
            else:
                text_strategy_counter[strategy] += 1

        for relation in document.relations:
            relation_key = (relation.pmid, relation.chemical_id, relation.disease_id)
            relation_counter[relation_key] += 1

            if relation.relation_type != "CID":
                invalid_relations.append(
                    {
                        "pmid": relation.pmid,
                        "relation_type": relation.relation_type,
                        "chemical_id": relation.chemical_id,
                        "disease_id": relation.disease_id,
                        "reason": "invalid_relation_type",
                    }
                )

            if relation.chemical_id not in entity_ids_by_type["Chemical"]:
                invalid_relations.append(
                    {
                        "pmid": relation.pmid,
                        "chemical_id": relation.chemical_id,
                        "disease_id": relation.disease_id,
                        "reason": "chemical_id_not_found_in_document",
                    }
                )

            if relation.disease_id not in entity_ids_by_type["Disease"]:
                invalid_relations.append(
                    {
                        "pmid": relation.pmid,
                        "chemical_id": relation.chemical_id,
                        "disease_id": relation.disease_id,
                        "reason": "disease_id_not_found_in_document",
                    }
                )

    for (pmid, chemical_id, disease_id), count in relation_counter.items():
        if count > 1:
            duplicate_relations.append(
                {
                    "pmid": pmid,
                    "chemical_id": chemical_id,
                    "disease_id": disease_id,
                    "count": count,
                }
            )

    expected_documents = EXPECTED_DOC_COUNTS.get(name)
    document_count_ok = expected_documents is None or len(documents) == expected_documents

    return {
        "subset": name,
        "path": str(path),
        "documents": len(documents),
        "expected_documents": expected_documents,
        "document_count_ok": document_count_ok,
        "pmids": [document.pmid for document in documents],
        "chemical_mentions": entity_mentions["Chemical"],
        "disease_mentions": entity_mentions["Disease"],
        "unique_chemical_concepts": len(unique_concepts["Chemical"]),
        "unique_disease_concepts": len(unique_concepts["Disease"]),
        "cid_relations": sum(1 for document in documents for relation in document.relations if relation.relation_type == "CID"),
        "invalid_annotations": invalid_annotations,
        "invalid_relations": invalid_relations,
        "duplicate_relations": duplicate_relations,
        "text_offset_match_strategies": dict(text_strategy_counter),
    }


def find_duplicate_pmids(subset_results: Iterable[dict]) -> list[dict]:
    pmid_to_subsets: dict[str, list[str]] = defaultdict(list)
    for result in subset_results:
        for pmid in result["pmids"]:
            pmid_to_subsets[pmid].append(result["subset"])

    return [
        {"pmid": pmid, "subsets": subsets}
        for pmid, subsets in sorted(pmid_to_subsets.items())
        if len(set(subsets)) > 1
    ]


def build_summary(subset_results: list[dict], duplicate_pmids: list[dict]) -> dict:
    compact_subsets = {}
    for result in subset_results:
        compact_subsets[result["subset"]] = {
            "documents": result["documents"],
            "expected_documents": result["expected_documents"],
            "document_count_ok": result["document_count_ok"],
            "chemical_mentions": result["chemical_mentions"],
            "disease_mentions": result["disease_mentions"],
            "unique_chemical_concepts": result["unique_chemical_concepts"],
            "unique_disease_concepts": result["unique_disease_concepts"],
            "cid_relations": result["cid_relations"],
            "invalid_annotation_count": len(result["invalid_annotations"]),
            "invalid_relation_count": len(result["invalid_relations"]),
            "duplicate_relation_count": len(result["duplicate_relations"]),
            "text_offset_match_strategies": result["text_offset_match_strategies"],
        }

    return {
        "subsets": compact_subsets,
        "duplicate_pmids_across_subsets": duplicate_pmids,
        "passed": all(
            result["document_count_ok"]
            and not result["invalid_annotations"]
            and not result["invalid_relations"]
            and not result["duplicate_relations"]
            for result in subset_results
        )
        and not duplicate_pmids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BC5CDR PubTator files.")
    parser.add_argument("--train", type=Path, default=Path("data/bc5cdr/train.txt"))
    parser.add_argument("--dev", type=Path, default=Path("data/bc5cdr/dev.txt"))
    parser.add_argument("--test", type=Path, default=Path("data/bc5cdr/test.txt"))
    parser.add_argument("--output", type=Path, default=Path("results/dataset_validation.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subset_paths = {
        "train": args.train,
        "dev": args.dev,
        "test": args.test,
    }

    missing_paths = [str(path) for path in subset_paths.values() if not path.exists()]
    if missing_paths:
        print("Missing BC5CDR files:")
        for path in missing_paths:
            print(f"  - {path}")
        print()
        print("Expected default layout:")
        print("  data/bc5cdr/train.txt")
        print("  data/bc5cdr/dev.txt")
        print("  data/bc5cdr/test.txt")
        return 2

    subset_results = [validate_subset(name, path) for name, path in subset_paths.items()]
    duplicate_pmids = find_duplicate_pmids(subset_results)
    summary = build_summary(subset_results, duplicate_pmids)

    report = {
        "summary": summary,
        "details": [
            {
                key: value
                for key, value in result.items()
                if key != "pmids"
            }
            for result in subset_results
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Validation report written to {args.output}")
    print(f"Passed: {summary['passed']}")
    for subset, values in summary["subsets"].items():
        print(
            f"{subset}: docs={values['documents']}, "
            f"Chemical={values['chemical_mentions']}, "
            f"Disease={values['disease_mentions']}, "
            f"CID={values['cid_relations']}, "
            f"invalid_ann={values['invalid_annotation_count']}, "
            f"invalid_rel={values['invalid_relation_count']}, "
            f"duplicate_rel={values['duplicate_relation_count']}"
        )
    print(f"duplicate_pmids_across_subsets={len(duplicate_pmids)}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
