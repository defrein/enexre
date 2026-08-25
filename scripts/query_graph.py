#!/usr/bin/env python
"""Query the Neo4j Chemical-Disease graph prototype.

Examples:
  python scripts/query_graph.py --pmid 18801087
  python scripts/query_graph.py --chemical-id D004280
  python scripts/query_graph.py --disease-id D016171 --json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from neo4j import GraphDatabase


DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "enexre12345"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the ENEXRE Neo4j graph.")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", DEFAULT_URI))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", DEFAULT_USER))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", DEFAULT_PASSWORD))
    parser.add_argument("--pmid", default=None, help="Filter CID relationships by PMID.")
    parser.add_argument("--chemical-id", default=None, help="Filter by Chemical MeSH ID.")
    parser.add_argument("--disease-id", default=None, help="Filter by Disease MeSH ID.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text table.")
    return parser.parse_args()


def build_query(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    filters = []
    params: dict[str, Any] = {"limit": args.limit}

    if args.pmid:
        filters.append("r.pmid = $pmid")
        params["pmid"] = args.pmid
    if args.chemical_id:
        filters.append("c.mesh_id = $chemical_id")
        params["chemical_id"] = args.chemical_id
    if args.disease_id:
        filters.append("d.mesh_id = $disease_id")
        params["disease_id"] = args.disease_id

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = f"""
    MATCH (c:Chemical)-[r:CID]->(d:Disease)
    {where_clause}
    RETURN
      c.mesh_id AS chemical_id,
      c.label AS chemical,
      d.mesh_id AS disease_id,
      d.label AS disease,
      r.confidence AS confidence,
      r.pmid AS pmid,
      r.evidence AS evidence
    ORDER BY r.confidence DESC, r.pmid
    LIMIT $limit
    """
    return query, params


def query_graph(args: argparse.Namespace) -> list[dict[str, Any]]:
    query, params = build_query(args)
    with GraphDatabase.driver(args.uri, auth=(args.user, args.password)) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No CID relationships found.")
        return

    headers = ["pmid", "chemical", "disease", "confidence"]
    widths = {
        "pmid": 10,
        "chemical": 28,
        "disease": 32,
        "confidence": 10,
    }
    print(
        "  ".join(header.upper().ljust(widths[header]) for header in headers)
    )
    print(
        "  ".join("-" * widths[header] for header in headers)
    )
    for row in rows:
        values = {
            "pmid": str(row["pmid"]),
            "chemical": str(row["chemical"])[: widths["chemical"]],
            "disease": str(row["disease"])[: widths["disease"]],
            "confidence": f"{float(row['confidence']):.4f}",
        }
        print(
            "  ".join(values[header].ljust(widths[header]) for header in headers)
        )


def main() -> int:
    args = parse_args()
    rows = query_graph(args)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
