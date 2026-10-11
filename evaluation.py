"""
Enterprise GraphRAG — Evaluation Package.

Consolidated Evaluation Subsystem: BenchmarkRunner, compute_retrieval_metrics, get_sample_evaluation_dataset.
"""

from __future__ import annotations

import sys
from typing import Any


def compute_retrieval_metrics(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int = 10,
) -> dict[str, float]:
    top_k = retrieved_chunk_ids[:k]
    if not top_k or not relevant_chunk_ids:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0}

    hits = [cid in relevant_chunk_ids for cid in top_k]
    num_hits = sum(hits)

    precision = num_hits / len(top_k)
    recall = num_hits / len(relevant_chunk_ids)

    mrr = 0.0
    for idx, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / idx
            break

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mrr": round(mrr, 4),
    }


class BenchmarkRunner:
    """Runs evaluation benchmarks over ground truth datasets."""

    def run_benchmark(self, dataset: list[dict[str, Any]]) -> dict[str, float]:
        precisions, recalls, mrrs = [], [], []
        for item in dataset:
            metrics = compute_retrieval_metrics(
                retrieved_chunk_ids=item.get("retrieved", []),
                relevant_chunk_ids=set(item.get("ground_truth", [])),
            )
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            mrrs.append(metrics["mrr"])

        return {
            "avg_precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "avg_recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "avg_mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
        }


def get_sample_evaluation_dataset() -> list[dict]:
    return [
        {
            "query": "What is GraphRAG?",
            "ground_truth": ["chunk_1", "chunk_2"],
            "retrieved": ["chunk_1", "chunk_5", "chunk_2"],
        }
    ]


for _mod in ["benchmarks", "metrics", "sample_data"]:
    sys.modules[f"evaluation.{_mod}"] = sys.modules[__name__]
