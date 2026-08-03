#!/usr/bin/env python
"""Build Neo4j-ready Chemical-Disease graph artifacts from pipeline predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_mentions(mentions: list[dict[str, Any]], limit: int = 5) -> str:
    values = []
    seen = set()
    for mention in mentions:
        text = mention.get("mention", "").strip()
        if text and text.lower() not in seen:
            values.append(text)
            seen.add(text.lower())
        if len(values) >= limit:
            break
    return "|".join(values)


def snippet_for_mentions(text: str, mentions: list[dict[str, Any]], radius: int = 180) -> str:
    if not mentions:
        return " ".join(text[: radius * 2].split())
    start = max(0, min(int(mention["start"]) for mention in mentions) - radius)
    end = min(len(text), max(int(mention["end"]) for mention in mentions) + radius)
    return " ".join(text[start:end].split())


def index_candidates(candidates: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row["pmid"]), row["chemical_id"], row["disease_id"]): row
        for row in candidates
    }


def build_graph_rows(
    predictions: list[dict[str, Any]],
    candidates_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    chemicals: dict[str, dict[str, Any]] = {}
    diseases: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for row in predictions:
        if int(row["prediction"]) != 1:
            continue

        pmid = str(row["pmid"])
        chemical_id = row["chemical_id"]
        disease_id = row["disease_id"]
        key = (pmid, chemical_id, disease_id)
        candidate = candidates_by_key.get(key, {})
        chemical_mentions = candidate.get("chemical_mentions", [])
        disease_mentions = candidate.get("disease_mentions", [])
        text = candidate.get("text", "")

        chemicals.setdefault(
            chemical_id,
            {
                "mesh_id": chemical_id,
                "label": first_mentions(chemical_mentions) or chemical_id,
                "mention_examples": first_mentions(chemical_mentions),
            },
        )
        diseases.setdefault(
            disease_id,
            {
                "mesh_id": disease_id,
                "label": first_mentions(disease_mentions) or disease_id,
                "mention_examples": first_mentions(disease_mentions),
            },
        )
        edges.append(
            {
                "chemical_id": chemical_id,
                "disease_id": disease_id,
                "pmid": pmid,
                "confidence": f"{float(row['cid_probability']):.8f}",
                "source": "pipeline_ner_re",
                "chemical_mentions": first_mentions(chemical_mentions),
                "disease_mentions": first_mentions(disease_mentions),
                "evidence": snippet_for_mentions(text, chemical_mentions + disease_mentions),
            }
        )

    chemical_rows = sorted(chemicals.values(), key=lambda item: item["mesh_id"])
    disease_rows = sorted(diseases.values(), key=lambda item: item["mesh_id"])
    edge_rows = sorted(edges, key=lambda item: (item["pmid"], item["chemical_id"], item["disease_id"]))
    return chemical_rows, disease_rows, edge_rows


def validate_graph(
    chemical_rows: list[dict[str, Any]],
    disease_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    chemical_ids = [row["mesh_id"] for row in chemical_rows]
    disease_ids = [row["mesh_id"] for row in disease_rows]
    edge_keys = [
        (row["chemical_id"], row["disease_id"], row["pmid"])
        for row in edge_rows
    ]
    duplicate_chemical_ids = sorted(
        mesh_id for mesh_id, count in Counter(chemical_ids).items() if count > 1
    )
    duplicate_disease_ids = sorted(
        mesh_id for mesh_id, count in Counter(disease_ids).items() if count > 1
    )
    duplicate_edges = [
        {"chemical_id": chemical_id, "disease_id": disease_id, "pmid": pmid, "count": count}
        for (chemical_id, disease_id, pmid), count in Counter(edge_keys).items()
        if count > 1
    ]
    missing_pmid_edges = [
        row for row in edge_rows if not row.get("pmid")
    ]
    missing_confidence_edges = [
        row for row in edge_rows if row.get("confidence") in {None, ""}
    ]
    invalid_confidence_edges = []
    for row in edge_rows:
        try:
            confidence = float(row["confidence"])
        except (TypeError, ValueError):
            invalid_confidence_edges.append(row)
            continue
        if confidence < 0.0 or confidence > 1.0:
            invalid_confidence_edges.append(row)

    unknown_endpoint_edges = [
        row
        for row in edge_rows
        if row["chemical_id"] not in set(chemical_ids)
        or row["disease_id"] not in set(disease_ids)
    ]

    summary = {
        "chemical_nodes": len(chemical_rows),
        "disease_nodes": len(disease_rows),
        "cid_relationships": len(edge_rows),
        "duplicate_chemical_node_count": len(duplicate_chemical_ids),
        "duplicate_disease_node_count": len(duplicate_disease_ids),
        "duplicate_relationship_count": len(duplicate_edges),
        "missing_pmid_relationship_count": len(missing_pmid_edges),
        "missing_confidence_relationship_count": len(missing_confidence_edges),
        "invalid_confidence_relationship_count": len(invalid_confidence_edges),
        "unknown_endpoint_relationship_count": len(unknown_endpoint_edges),
    }
    return {
        "summary": summary,
        "passed": all(value == 0 for key, value in summary.items() if key.endswith("_count")),
        "details": {
            "duplicate_chemical_ids": duplicate_chemical_ids,
            "duplicate_disease_ids": duplicate_disease_ids,
            "duplicate_edges": duplicate_edges,
            "missing_pmid_edges": missing_pmid_edges[:20],
            "missing_confidence_edges": missing_confidence_edges[:20],
            "invalid_confidence_edges": invalid_confidence_edges[:20],
            "unknown_endpoint_edges": unknown_endpoint_edges[:20],
        },
    }


def write_import_cypher(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """CREATE CONSTRAINT chemical_mesh_id IF NOT EXISTS
FOR (c:Chemical) REQUIRE c.mesh_id IS UNIQUE;

CREATE CONSTRAINT disease_mesh_id IF NOT EXISTS
FOR (d:Disease) REQUIRE d.mesh_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///chemical_nodes.csv' AS row
MERGE (c:Chemical {mesh_id: row.mesh_id})
SET c.label = row.label,
    c.mention_examples = row.mention_examples;

LOAD CSV WITH HEADERS FROM 'file:///disease_nodes.csv' AS row
MERGE (d:Disease {mesh_id: row.mesh_id})
SET d.label = row.label,
    d.mention_examples = row.mention_examples;

LOAD CSV WITH HEADERS FROM 'file:///cid_edges.csv' AS row
MATCH (c:Chemical {mesh_id: row.chemical_id})
MATCH (d:Disease {mesh_id: row.disease_id})
MERGE (c)-[r:CID {pmid: row.pmid, chemical_id: row.chemical_id, disease_id: row.disease_id}]->(d)
SET r.confidence = toFloat(row.confidence),
    r.source = row.source,
    r.chemical_mentions = row.chemical_mentions,
    r.disease_mentions = row.disease_mentions,
    r.evidence = row.evidence;
""",
        encoding="utf-8",
    )


def write_validation_cypher(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """MATCH (n)
WITH labels(n) AS labels, n.mesh_id AS mesh_id, count(n) AS total
WHERE mesh_id IS NOT NULL AND total > 1
RETURN labels, mesh_id, total;

MATCH ()-[r:CID]->()
WHERE r.pmid IS NULL
RETURN count(r) AS relationships_without_pmid;

MATCH ()-[r:CID]->()
WHERE r.confidence IS NULL
RETURN count(r) AS relationships_without_confidence;

MATCH ()-[r:CID]->()
WITH r.chemical_id AS chemical_id, r.disease_id AS disease_id, r.pmid AS pmid, count(r) AS total
WHERE total > 1
RETURN chemical_id, disease_id, pmid, total;

MATCH (a)-[r:CID]->(b)
WHERE NOT a:Chemical OR NOT b:Disease
RETURN count(r) AS invalid_cid_direction;

MATCH ()-[r:CID]->()
RETURN count(r) AS cid_relationships;
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Neo4j graph CSV artifacts.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("predictions/pipeline/best_test_predictions.jsonl"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/processed/pipeline/test_candidates.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/graph"))
    parser.add_argument("--report", type=Path, default=Path("results/graph/graph_validation.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in [args.predictions, args.candidates]:
        if not path.exists():
            print(f"Missing required file: {path}")
            return 2

    predictions = load_jsonl(args.predictions)
    candidates = load_jsonl(args.candidates)
    chemical_rows, disease_rows, edge_rows = build_graph_rows(
        predictions=predictions,
        candidates_by_key=index_candidates(candidates),
    )

    write_csv(
        args.output_dir / "chemical_nodes.csv",
        ["mesh_id", "label", "mention_examples"],
        chemical_rows,
    )
    write_csv(
        args.output_dir / "disease_nodes.csv",
        ["mesh_id", "label", "mention_examples"],
        disease_rows,
    )
    write_csv(
        args.output_dir / "cid_edges.csv",
        [
            "chemical_id",
            "disease_id",
            "pmid",
            "confidence",
            "source",
            "chemical_mentions",
            "disease_mentions",
            "evidence",
        ],
        edge_rows,
    )
    write_import_cypher(args.output_dir / "neo4j_import.cypher")
    write_validation_cypher(args.output_dir / "neo4j_validation_queries.cypher")

    validation = validate_graph(chemical_rows, disease_rows, edge_rows)
    report = {
        "source_predictions": str(args.predictions),
        "source_candidates": str(args.candidates),
        "output_dir": str(args.output_dir),
        "files": {
            "chemical_nodes": str(args.output_dir / "chemical_nodes.csv"),
            "disease_nodes": str(args.output_dir / "disease_nodes.csv"),
            "cid_edges": str(args.output_dir / "cid_edges.csv"),
            "neo4j_import": str(args.output_dir / "neo4j_import.cypher"),
            "neo4j_validation_queries": str(args.output_dir / "neo4j_validation_queries.cypher"),
        },
        **validation,
    }
    save_json(args.report, report)

    summary = validation["summary"]
    print(f"Graph artifacts written to {args.output_dir}")
    print(f"Validation report written to {args.report}")
    print(f"Passed: {validation['passed']}")
    print(
        f"chemical_nodes={summary['chemical_nodes']}, "
        f"disease_nodes={summary['disease_nodes']}, "
        f"cid_relationships={summary['cid_relationships']}"
    )
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
