#!/usr/bin/env python
"""Generate report figures from saved experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path("results/figures")


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def annotate_bars(ax: plt.Axes, bars, decimals: int = 3) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_ner_seed_metrics() -> None:
    report = load_json("results/ner/final_three_seed_test_summary.json")
    runs = sorted(report["runs"], key=lambda row: row["seed"])
    seeds = [str(row["seed"]) for row in runs]
    metrics = [
        ("Precision", "test_precision"),
        ("Recall", "test_recall"),
        ("F1-Score", "test_f1"),
    ]
    x = np.arange(len(seeds))
    width = 0.24
    colors = ["#31708f", "#5a8f3a", "#b45f3c"]

    _, ax = plt.subplots(figsize=(7.2, 4.2))
    for index, (label, key) in enumerate(metrics):
        values = [row[key] for row in runs]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=colors[index])
        annotate_bars(ax, bars)

    ax.set_title("Hasil NER pada Test Set per Random Seed")
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Skor")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylim(0.84, 0.93)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    save_figure(FIGURE_DIR / "ner_test_metrics_by_seed.png")


def plot_ner_entity_f1() -> None:
    report = load_json("results/ner/final_three_seed_test_summary.json")
    aggregate = report["aggregate"]
    labels = ["Chemical", "Disease"]
    means = [aggregate["chemical_f1"]["mean"], aggregate["disease_f1"]["mean"]]
    stds = [aggregate["chemical_f1"]["std"], aggregate["disease_f1"]["std"]]

    _, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=["#31708f", "#b45f3c"])
    annotate_bars(ax, bars)
    ax.set_title("Perbandingan F1-Score NER Berdasarkan Tipe Entitas")
    ax.set_ylabel("F1-Score rata-rata")
    ax.set_ylim(0.80, 0.96)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_figure(FIGURE_DIR / "ner_entity_f1_comparison.png")


def plot_re_gold_vs_pipeline() -> None:
    re_gold = load_json("results/re/best_test_metrics.json")["test"]
    pipeline = load_json("results/pipeline/best_test_metrics.json")["pipeline_test"]
    metrics = [("Precision", "precision"), ("Recall", "recall"), ("F1-Score", "f1")]
    labels = [label for label, _ in metrics]
    re_values = [re_gold[key] for _, key in metrics]
    pipeline_values = [pipeline[key] for _, key in metrics]
    x = np.arange(len(labels))
    width = 0.34

    _, ax = plt.subplots(figsize=(7.0, 4.2))
    bars_a = ax.bar(x - width / 2, re_values, width, label="RE gold entities", color="#31708f")
    bars_b = ax.bar(x + width / 2, pipeline_values, width, label="Pipeline NER-RE", color="#b45f3c")
    annotate_bars(ax, bars_a)
    annotate_bars(ax, bars_b)
    ax.set_title("Perbandingan RE Gold Entities dan Pipeline End-to-End")
    ax.set_ylabel("Skor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.55, 0.73)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    save_figure(FIGURE_DIR / "re_gold_vs_pipeline_metrics.png")


def plot_pipeline_error_breakdown() -> None:
    report = load_json("results/error_analysis/error_analysis.json")
    pipeline_counts = report["pipeline"]["counts"]
    missing_reasons = report["pipeline_candidate_loss"]["missing_gold_relation_reasons"]
    labels = [
        "False positive",
        "FN ditolak RE",
        "FN kandidat hilang",
        "Disease hilang",
        "Chemical hilang",
        "Keduanya hilang",
    ]
    values = [
        pipeline_counts["false_positive"],
        pipeline_counts["false_negative_rejected_candidate"],
        pipeline_counts["false_negative_missing_candidate"],
        missing_reasons["disease_missing"],
        missing_reasons["chemical_missing"],
        missing_reasons["chemical_and_disease_missing"],
    ]

    _, ax = plt.subplots(figsize=(8.0, 4.6))
    bars = ax.bar(labels, values, color=["#31708f", "#b45f3c", "#5a8f3a", "#8b6f47", "#6d5a8f", "#9a4f64"])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Ringkasan Kesalahan Pipeline NER-RE")
    ax.set_ylabel("Jumlah kasus")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_figure(FIGURE_DIR / "pipeline_error_breakdown.png")


def plot_graph_summary() -> None:
    summary = load_json("results/graph/graph_validation.json")["summary"]
    labels = ["Chemical nodes", "Disease nodes", "CID relationships"]
    values = [summary["chemical_nodes"], summary["disease_nodes"], summary["cid_relationships"]]

    _, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(labels, values, color=["#31708f", "#b45f3c", "#5a8f3a"])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Ringkasan Artefak Knowledge Graph")
    ax.set_ylabel("Jumlah")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_figure(FIGURE_DIR / "knowledge_graph_summary.png")


def plot_external_diagnostic() -> None:
    candidate_rows = load_jsonl("data/external_pubmed/candidate_pairs.jsonl")
    relation_rows = load_jsonl("data/external_pubmed/predicted_relations.jsonl")
    pmids = sorted({row["pmid"] for row in candidate_rows})
    candidates = [sum(1 for row in candidate_rows if row["pmid"] == pmid) for pmid in pmids]
    predicted = [sum(1 for row in relation_rows if row["pmid"] == pmid) for pmid in pmids]

    _, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))
    colors = ["#31708f", "#b45f3c", "#5a8f3a", "#8b6f47", "#6d5a8f"]
    for ax, values, title, ylabel in [
        (axes[0], candidates, "Kandidat RE per PMID", "Kandidat"),
        (axes[1], predicted, "Prediksi CID per PMID", "Relasi CID"),
    ]:
        bars = ax.bar(pmids, values, color=colors)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=28)

    save_figure(FIGURE_DIR / "external_pubmed_candidate_diagnostic.png")


def main() -> int:
    plot_ner_seed_metrics()
    plot_ner_entity_f1()
    plot_re_gold_vs_pipeline()
    plot_pipeline_error_breakdown()
    plot_graph_summary()
    plot_external_diagnostic()
    print(f"Figures written to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
