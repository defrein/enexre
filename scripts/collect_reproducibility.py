#!/usr/bin/env python
"""Collect reproducibility evidence for the BC5CDR experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any


IMPORTANT_FILES = [
    "data/data_manifest.json",
    "configs/config_ner.yaml",
    "configs/config_re.yaml",
    "requirements.txt",
    "requirements.lock.txt",
    "scripts/validate_bc5cdr.py",
    "scripts/build_ner_dataset.py",
    "scripts/train_ner.py",
    "scripts/evaluate_ner.py",
    "scripts/build_re_dataset.py",
    "scripts/train_re.py",
    "scripts/evaluate_re.py",
    "scripts/evaluate_pipeline.py",
    "scripts/analyze_errors.py",
    "scripts/build_graph.py",
    "scripts/query_graph.py",
    "scripts/prototype_app.py",
    "scripts/run_external_pubmed.py",
    "results/dataset_validation.json",
    "results/ner_preprocessing_report.json",
    "results/re_preprocessing_report.json",
    "results/ner/final_three_seed_test_summary.json",
    "results/re/final_re_summary.json",
    "results/pipeline/best_test_metrics.json",
    "results/error_analysis/error_analysis.json",
    "results/graph/graph_validation.json",
    "results/graph/neo4j_runtime_validation.json",
    "results/external_pubmed/external_pubmed_summary.json",
    "results/external_pubmed/manual_review_summary.md",
]


COMMANDS = [
    ".venv/Scripts/python.exe scripts/validate_bc5cdr.py",
    ".venv/Scripts/python.exe scripts/build_ner_dataset.py",
    ".venv/Scripts/python.exe scripts/train_ner.py --smoke-test --cpu --cleanup-smoke-checkpoint",
    "python scripts/train_ner.py --seed 13 --learning-rate 5e-5 --batch-size 8 --run-name final_seed13_lr5e-5_bs8",
    "python scripts/train_ner.py --seed 42 --learning-rate 5e-5 --batch-size 8 --run-name final_seed42_lr5e-5_bs8",
    "python scripts/train_ner.py --seed 100 --learning-rate 5e-5 --batch-size 8 --run-name final_seed100_lr5e-5_bs8",
    ".venv/Scripts/python.exe scripts/build_re_dataset.py",
    "python scripts/train_re.py --seed 13 --learning-rate 3e-5 --batch-size 8 --run-name re_seed13_lr3e-5_bs8",
    "python scripts/train_re.py --seed 42 --learning-rate 3e-5 --batch-size 8 --run-name re_seed42_lr3e-5_bs8",
    "python scripts/train_re.py --seed 100 --learning-rate 3e-5 --batch-size 8 --run-name re_seed100_lr3e-5_bs8",
    ".venv/Scripts/python.exe scripts/evaluate_re.py --cpu",
    ".venv/Scripts/python.exe scripts/evaluate_pipeline.py --cpu",
    ".venv/Scripts/python.exe scripts/analyze_errors.py",
    ".venv/Scripts/python.exe scripts/build_graph.py",
    ".venv/Scripts/python.exe scripts/query_graph.py --pmid 18801087 --limit 5",
    ".venv/Scripts/python.exe scripts/prototype_app.py --host 127.0.0.1 --port 8000",
    ".venv/Scripts/python.exe scripts/run_external_pubmed.py --articles-input data/external_pubmed/articles.jsonl --count 5 --candidate-granularity surface --cpu",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def git_info() -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"])
    branch = command_output(["git", "branch", "--show-current"])
    status = command_output(["git", "status", "--short"])
    return {
        "commit": commit,
        "branch": branch,
        "status_short": status or "",
    }


def file_records() -> list[dict[str, Any]]:
    rows = []
    for file_name in IMPORTANT_FILES:
        path = Path(file_name)
        rows.append(
            {
                "path": file_name,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256(path) if path.exists() else None,
            }
        )
    return rows


def load_json_if_exists(path: str) -> Any | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def compact_results() -> dict[str, Any]:
    ner = load_json_if_exists("results/ner/final_three_seed_test_summary.json")
    re_summary = load_json_if_exists("results/re/final_re_summary.json")
    pipeline = load_json_if_exists("results/pipeline/best_test_metrics.json")
    graph = load_json_if_exists("results/graph/graph_validation.json")
    external = load_json_if_exists("results/external_pubmed/external_pubmed_summary.json")
    return {
        "ner": ner.get("aggregate") if ner else None,
        "re": re_summary.get("test") if re_summary else None,
        "pipeline": pipeline.get("pipeline_test") if pipeline else None,
        "graph": graph.get("summary") if graph else None,
        "external_pubmed": {
            key: external.get(key)
            for key in [
                "article_count",
                "entity_count",
                "chemical_count",
                "disease_count",
                "candidate_pairs",
                "candidate_granularity",
                "predicted_cid_relations",
                "threshold",
            ]
        }
        if external
        else None,
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    results = report["compact_results"]
    lines = [
        "# Reproducibility Report",
        "",
        "## Environment",
        "",
        f"- Python: {report['environment']['python']}",
        f"- Platform: {report['environment']['platform']}",
        f"- PyTorch: {report['environment']['packages'].get('torch')}",
        f"- Transformers: {report['environment']['packages'].get('transformers')}",
        "- Checkpoint Transformers metadata: "
        f"NER={report['environment']['checkpoint_transformers_versions'].get('ner')}, "
        f"RE={report['environment']['checkpoint_transformers_versions'].get('re')}",
        f"- Neo4j Python driver: {report['environment']['packages'].get('neo4j')}",
        "",
        "## Git",
        "",
        f"- Branch: {report['git']['branch']}",
        f"- Commit: {report['git']['commit']}",
        f"- Status: {'clean' if not report['git']['status_short'] else 'dirty'}",
        "",
        "## Key Results",
        "",
    ]

    if results["ner"]:
        lines.append(f"- NER test F1 mean: {results['ner']['test_f1']['mean']:.4f}")
    if results["re"]:
        lines.append(f"- RE gold test F1: {results['re']['f1']:.4f}")
    if results["pipeline"]:
        lines.append(f"- Pipeline NER-RE test F1: {results['pipeline']['f1']:.4f}")
    if results["graph"]:
        lines.append(
            f"- Graph: {results['graph']['chemical_nodes']} Chemical nodes, "
            f"{results['graph']['disease_nodes']} Disease nodes, "
            f"{results['graph']['cid_relationships']} CID relationships"
        )
    if results["external_pubmed"]:
        lines.append(
            "- External PubMed: "
            f"{results['external_pubmed']['article_count']} abstracts, "
            f"{results['external_pubmed']['candidate_pairs']} candidates, "
            f"{results['external_pubmed']['predicted_cid_relations']} predicted CID relations, "
            f"granularity={results['external_pubmed']['candidate_granularity']}, no gold labels"
        )

    lines.extend(["", "## Reproduction Commands", ""])
    lines.extend(f"- `{command}`" for command in report["commands"])
    lines.extend(["", "## File Checksums", ""])
    lines.append("| File | Exists | SHA256 |")
    lines.append("| --- | ---: | --- |")
    for row in report["files"]:
        lines.append(f"| `{row['path']}` | {row['exists']} | `{row['sha256'] or ''}` |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ner_checkpoint_config = load_json_if_exists(
        "checkpoints/ner/final_seed42_lr5e-5_bs8/config.json"
    ) or {}
    re_checkpoint_config = load_json_if_exists(
        "checkpoints/re/re_seed13_lr3e-5_bs8/config.json"
    ) or {}
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                name: package_version(name)
                for name in [
                    "torch",
                    "transformers",
                    "datasets",
                    "evaluate",
                    "seqeval",
                    "scikit-learn",
                    "pandas",
                    "pyyaml",
                    "neo4j",
                ]
            },
            "checkpoint_transformers_versions": {
                "ner": ner_checkpoint_config.get("transformers_version"),
                "re": re_checkpoint_config.get("transformers_version"),
            },
        },
        "git": git_info(),
        "git_note": (
            "status_short may be dirty when this report is generated before the final "
            "commit because the report and newly created artifacts are part of the "
            "current reproducibility evidence."
        ),
        "commands": COMMANDS,
        "compact_results": compact_results(),
        "files": file_records(),
    }
    save_json(Path("results/reproducibility/reproducibility_report.json"), report)
    write_markdown(Path("results/reproducibility/reproducibility_report.md"), report)
    print("Reproducibility report written to results/reproducibility")
    print(f"tracked_files={sum(1 for row in report['files'] if row['exists'])}/{len(report['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
