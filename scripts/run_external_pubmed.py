#!/usr/bin/env python
"""Run the NER-RE pipeline on PubMed abstracts outside BC5CDR.

This implements Tahap 16 as a reproducible prototype. External PubMed
abstracts do not have BC5CDR gold MeSH IDs, so candidate generation uses
surface-form grouping by default to approximate the concept-level RE input
used during BC5CDR training. The output is intended for manual review.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

from build_re_dataset import MARKER_TOKENS, count_marker_tokens
from train_re import JsonlRelationDataset, RelationBatchCollator, evaluate as evaluate_re
from validate_bc5cdr import parse_pubtator


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
LABELS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]


@dataclass(frozen=True)
class PubMedArticle:
    pmid: str
    title: str
    abstract: str


@dataclass(frozen=True)
class EntitySpan:
    entity_id: str
    entity_type: str
    start: int
    end: int
    mention: str


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bc5cdr_pmids(paths: list[Path]) -> set[str]:
    pmids = set()
    for path in paths:
        if path.exists():
            pmids.update(document.pmid for document in parse_pubtator(path))
    return pmids


def eutils_get(endpoint: str, params: dict[str, Any], delay_seconds: float) -> bytes:
    url = f"{EUTILS_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read()
    time.sleep(delay_seconds)
    return payload


def search_pubmed(query: str, retmax: int, delay_seconds: float) -> list[str]:
    payload = eutils_get(
        "esearch.fcgi",
        {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
            "tool": "enexre",
        },
        delay_seconds,
    )
    data = json.loads(payload.decode("utf-8"))
    return [str(pmid) for pmid in data["esearchresult"]["idlist"]]


def text_content(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def fetch_pubmed(pmids: list[str], delay_seconds: float) -> list[PubMedArticle]:
    if not pmids:
        return []
    payload = eutils_get(
        "efetch.fcgi",
        {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "tool": "enexre",
        },
        delay_seconds,
    )
    root = ET.fromstring(payload)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid = text_content(article.find(".//PMID"))
        title = text_content(article.find(".//ArticleTitle"))
        abstract_parts = [
            text_content(part)
            for part in article.findall(".//Abstract/AbstractText")
            if text_content(part)
        ]
        abstract = " ".join(abstract_parts)
        if pmid and title and abstract:
            articles.append(PubMedArticle(pmid=pmid, title=title, abstract=abstract))
    return articles


def select_articles(
    query: str,
    count: int,
    excluded_pmids: set[str],
    delay_seconds: float,
) -> list[PubMedArticle]:
    candidate_pmids = search_pubmed(query=query, retmax=max(count * 6, 30), delay_seconds=delay_seconds)
    candidate_pmids = [pmid for pmid in candidate_pmids if pmid not in excluded_pmids]
    articles = fetch_pubmed(candidate_pmids[: max(count * 3, count)], delay_seconds=delay_seconds)
    return articles[:count]


def article_text(article: PubMedArticle) -> str:
    return f"{article.title} {article.abstract}"


@torch.no_grad()
def predict_ner(
    articles: list[PubMedArticle],
    checkpoint: Path,
    max_length: int,
    stride: int,
    device: torch.device,
) -> dict[str, list[EntitySpan]]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint).to(device)
    model.eval()
    id_to_label = {index: label for index, label in enumerate(LABELS)}
    spans_by_pmid: dict[str, dict[tuple[int, int, str], EntitySpan]] = {}

    for article in articles:
        text = article_text(article)
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        article_spans: dict[tuple[int, int, str], EntitySpan] = {}

        for chunk_index, input_ids in enumerate(encoded["input_ids"]):
            attention_mask = encoded["attention_mask"][chunk_index]
            offsets = encoded["offset_mapping"][chunk_index]
            sequence_ids = encoded.sequence_ids(chunk_index)
            outputs = model(
                input_ids=torch.tensor([input_ids], dtype=torch.long, device=device),
                attention_mask=torch.tensor([attention_mask], dtype=torch.long, device=device),
            )
            pred_ids = outputs.logits.argmax(dim=-1)[0].cpu().tolist()
            active: dict[str, Any] | None = None

            for pred_id, offset, sequence_id in zip(pred_ids, offsets, sequence_ids):
                start, end = int(offset[0]), int(offset[1])
                if sequence_id != 0 or start == end:
                    continue
                label = id_to_label[int(pred_id)]
                prefix, _, entity_type = label.partition("-")
                if label == "O" or prefix not in {"B", "I"}:
                    if active is not None:
                        key = (active["start"], active["end"], active["entity_type"])
                        article_spans[key] = span_from_active(article.pmid, text, active)
                    active = None
                    continue
                if prefix == "B" or active is None or active["entity_type"] != entity_type:
                    if active is not None:
                        key = (active["start"], active["end"], active["entity_type"])
                        article_spans[key] = span_from_active(article.pmid, text, active)
                    active = {"start": start, "end": end, "entity_type": entity_type}
                else:
                    active["end"] = end

            if active is not None:
                key = (active["start"], active["end"], active["entity_type"])
                article_spans[key] = span_from_active(article.pmid, text, active)

        spans_by_pmid[article.pmid] = sorted(article_spans.values(), key=lambda item: (item.start, item.end))

    return spans_by_pmid


def span_from_active(pmid: str, text: str, active: dict[str, Any]) -> EntitySpan:
    start = int(active["start"])
    end = int(active["end"])
    entity_type = active["entity_type"]
    return EntitySpan(
        entity_id=f"{entity_type}:{start}-{end}",
        entity_type=entity_type,
        start=start,
        end=end,
        mention=text[start:end],
    )


def normalize_surface(value: str) -> str:
    return " ".join(value.casefold().split())


def grouped_spans_by_surface(spans: list[EntitySpan]) -> dict[str, list[EntitySpan]]:
    groups: dict[str, list[EntitySpan]] = defaultdict(list)
    for span in spans:
        key = normalize_surface(span.mention)
        if key:
            groups[key].append(span)
    return {
        key: sorted(group, key=lambda item: (item.start, item.end))
        for key, group in sorted(groups.items())
    }


def insert_markers_for_pair(
    text: str,
    chemical_mentions: list[EntitySpan],
    disease_mentions: list[EntitySpan],
) -> str:
    insertions: list[tuple[int, str]] = []
    for chemical in chemical_mentions:
        insertions.append((chemical.start, "[CHEM] "))
        insertions.append((chemical.end, " [/CHEM]"))
    for disease in disease_mentions:
        insertions.append((disease.start, "[DISEASE] "))
        insertions.append((disease.end, " [/DISEASE]"))

    marked = text
    for position, marker in sorted(insertions, key=lambda item: item[0], reverse=True):
        marked = marked[:position] + marker + marked[position:]
    return marked


def local_snippet(text: str, spans: list[EntitySpan], radius: int = 180) -> str:
    if not spans:
        return " ".join(text[: radius * 2].split())
    start = max(0, min(span.start for span in spans) - radius)
    end = min(len(text), max(span.end for span in spans) + radius)
    return " ".join(text[start:end].split())


def build_re_candidates(
    articles: list[PubMedArticle],
    spans_by_pmid: dict[str, list[EntitySpan]],
    max_pairs_per_article: int,
    candidate_granularity: str = "surface",
) -> list[dict[str, Any]]:
    rows = []
    for article in articles:
        text = article_text(article)
        chemicals = [span for span in spans_by_pmid.get(article.pmid, []) if span.entity_type == "Chemical"]
        diseases = [span for span in spans_by_pmid.get(article.pmid, []) if span.entity_type == "Disease"]
        pair_count = 0

        if candidate_granularity == "mention":
            chemical_groups = {
                f"{span.start}-{span.end}-{index}": [span]
                for index, span in enumerate(chemicals)
            }
            disease_groups = {
                f"{span.start}-{span.end}-{index}": [span]
                for index, span in enumerate(diseases)
            }
        elif candidate_granularity == "surface":
            chemical_groups = grouped_spans_by_surface(chemicals)
            disease_groups = grouped_spans_by_surface(diseases)
        else:
            raise ValueError("candidate_granularity must be either 'surface' or 'mention'.")

        for chemical_key, chemical_mentions in chemical_groups.items():
            for disease_key, disease_mentions in disease_groups.items():
                if pair_count >= max_pairs_per_article:
                    break
                pair_count += 1
                chemical = chemical_mentions[0]
                disease = disease_mentions[0]
                marked_text = insert_markers_for_pair(
                    text=text,
                    chemical_mentions=chemical_mentions,
                    disease_mentions=disease_mentions,
                )
                rows.append(
                    {
                        "pmid": article.pmid,
                        "title": article.title,
                        "chemical_id": (
                            chemical.entity_id
                            if candidate_granularity == "mention"
                            else f"ChemicalSurface:{chemical_key}"
                        ),
                        "disease_id": (
                            disease.entity_id
                            if candidate_granularity == "mention"
                            else f"DiseaseSurface:{disease_key}"
                        ),
                        "chemical_mention": chemical.mention,
                        "disease_mention": disease.mention,
                        "label": 0,
                        "text": text,
                        "marked_text": marked_text,
                        "chemical_mentions": [
                            {
                                "pmid": article.pmid,
                                "start": mention.start,
                                "end": mention.end,
                                "mention": mention.mention,
                                "entity_type": "Chemical",
                                "mesh_id": (
                                    mention.entity_id
                                    if candidate_granularity == "mention"
                                    else f"ChemicalSurface:{chemical_key}"
                                ),
                            }
                            for mention in chemical_mentions
                        ],
                        "disease_mentions": [
                            {
                                "pmid": article.pmid,
                                "start": mention.start,
                                "end": mention.end,
                                "mention": mention.mention,
                                "entity_type": "Disease",
                                "mesh_id": (
                                    mention.entity_id
                                    if candidate_granularity == "mention"
                                    else f"DiseaseSurface:{disease_key}"
                                ),
                            }
                            for mention in disease_mentions
                        ],
                        "marker_counts": count_marker_tokens(marked_text),
                        "candidate_granularity": candidate_granularity,
                        "evidence": local_snippet(text, chemical_mentions + disease_mentions),
                    }
                )
            if pair_count >= max_pairs_per_article:
                break
    return rows


def run_re(
    candidate_path: Path,
    checkpoint: Path,
    threshold: float,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.resize_token_embeddings(len(tokenizer))
    dataset = JsonlRelationDataset(candidate_path, tokenizer, max_length)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=RelationBatchCollator(pad_token_id=tokenizer.pad_token_id or 0),
    )
    _, prediction_rows = evaluate_re(
        model=model,
        dataloader=dataloader,
        device=device,
        class_weights=None,
        thresholds=[threshold],
    )
    candidate_rows = read_jsonl(candidate_path)
    enriched = []
    for candidate, prediction in zip(candidate_rows, prediction_rows):
        enriched.append(
            {
                **candidate,
                "cid_probability": prediction["cid_probability"],
                "prediction": int(prediction["cid_probability"] >= threshold),
                "crop_report": prediction["crop_report"],
            }
        )
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external PubMed NER-RE evaluation.")
    parser.add_argument(
        "--query",
        default='("drug-induced"[Title/Abstract] OR "adverse effect"[Title/Abstract]) AND (disease[Title/Abstract] OR toxicity[Title/Abstract]) AND 2020:2026[pdat]',
    )
    parser.add_argument(
        "--pmids",
        default="",
        help=(
            "Optional comma-separated fixed PubMed IDs. When provided, the script "
            "fetches these articles directly instead of running a dynamic PubMed search."
        ),
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--delay-seconds", type=float, default=0.34)
    parser.add_argument("--ner-checkpoint", type=Path, default=Path("checkpoints/ner/final_seed42_lr5e-5_bs8"))
    parser.add_argument("--re-checkpoint", type=Path, default=Path("checkpoints/re/re_seed13_lr3e-5_bs8"))
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pairs-per-article", type=int, default=80)
    parser.add_argument(
        "--candidate-granularity",
        choices=["surface", "mention"],
        default="surface",
        help=(
            "surface groups repeated mentions with the same normalized surface form "
            "to approximate BC5CDR concept-level RE input; mention reproduces the "
            "initial per-mention diagnostic setting."
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/external_pubmed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/external_pubmed"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in [args.ner_checkpoint, args.re_checkpoint]:
        if not path.exists():
            print(f"Missing checkpoint: {path}")
            return 2

    excluded = bc5cdr_pmids(
        [Path("data/bc5cdr/train.txt"), Path("data/bc5cdr/dev.txt"), Path("data/bc5cdr/test.txt")]
    )
    fixed_pmids = [pmid.strip() for pmid in args.pmids.split(",") if pmid.strip()]
    if fixed_pmids:
        selected_pmids = [pmid for pmid in fixed_pmids if pmid not in excluded]
        skipped_pmids = [pmid for pmid in fixed_pmids if pmid in excluded]
        articles = fetch_pubmed(selected_pmids, delay_seconds=args.delay_seconds)
    else:
        skipped_pmids = []
        articles = select_articles(
            query=args.query,
            count=args.count,
            excluded_pmids=excluded,
            delay_seconds=args.delay_seconds,
        )
    if not articles:
        print("No PubMed articles with abstracts were fetched.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    spans_by_pmid = predict_ner(
        articles=articles,
        checkpoint=args.ner_checkpoint,
        max_length=args.max_length,
        stride=args.stride,
        device=device,
    )
    candidates = build_re_candidates(
        articles=articles,
        spans_by_pmid=spans_by_pmid,
        max_pairs_per_article=args.max_pairs_per_article,
        candidate_granularity=args.candidate_granularity,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    article_rows = [
        {"pmid": article.pmid, "title": article.title, "abstract": article.abstract}
        for article in articles
    ]
    entity_rows = [
        {
            "pmid": pmid,
            "entity_id": span.entity_id,
            "entity_type": span.entity_type,
            "start": span.start,
            "end": span.end,
            "mention": span.mention,
        }
        for pmid, spans in spans_by_pmid.items()
        for span in spans
    ]
    candidate_path = args.output_dir / "candidate_pairs.jsonl"
    save_jsonl(args.output_dir / "articles.jsonl", article_rows)
    save_jsonl(args.output_dir / "predicted_entities.jsonl", entity_rows)
    save_jsonl(candidate_path, candidates)

    predictions = []
    if candidates:
        predictions = run_re(
            candidate_path=candidate_path,
            checkpoint=args.re_checkpoint,
            threshold=args.threshold,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )
    save_jsonl(args.output_dir / "scored_candidate_pairs.jsonl", predictions)
    predicted_relations = [row for row in predictions if row["prediction"] == 1]
    save_jsonl(args.output_dir / "predicted_relations.jsonl", predicted_relations)

    summary = {
        "status": "external_pubmed_pipeline",
        "query": args.query,
        "fixed_pmids": fixed_pmids,
        "skipped_bc5cdr_pmids": skipped_pmids,
        "excluded_bc5cdr_pmids": len(excluded),
        "article_count": len(articles),
        "entity_count": len(entity_rows),
        "chemical_count": sum(1 for row in entity_rows if row["entity_type"] == "Chemical"),
        "disease_count": sum(1 for row in entity_rows if row["entity_type"] == "Disease"),
        "candidate_pairs": len(candidates),
        "candidate_granularity": args.candidate_granularity,
        "predicted_cid_relations": len(predicted_relations),
        "threshold": args.threshold,
        "ner_checkpoint": str(args.ner_checkpoint),
        "re_checkpoint": str(args.re_checkpoint),
        "device": str(device),
        "outputs": {
            "articles": str(args.output_dir / "articles.jsonl"),
            "predicted_entities": str(args.output_dir / "predicted_entities.jsonl"),
            "candidate_pairs": str(candidate_path),
            "scored_candidate_pairs": str(args.output_dir / "scored_candidate_pairs.jsonl"),
            "predicted_relations": str(args.output_dir / "predicted_relations.jsonl"),
        },
        "manual_review_note": (
            "External PubMed data has no BC5CDR gold CID labels here; predicted "
            "relations must be manually reviewed for validity."
        ),
    }
    save_json(args.results_dir / "external_pubmed_summary.json", summary)

    print(f"Fetched articles: {len(articles)}")
    print(f"Predicted entities: {len(entity_rows)}")
    print(f"Candidate pairs: {len(candidates)}")
    print(f"Predicted CID relations: {len(predicted_relations)}")
    print(f"Summary: {args.results_dir / 'external_pubmed_summary.json'}")
    print(f"Predicted relations: {args.output_dir / 'predicted_relations.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
