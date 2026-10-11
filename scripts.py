"""
Enterprise GraphRAG — Scripts Package.

Consolidated Scripts Subsystem: run_benchmarks, download_models, setup_neo4j.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from core import app_logger, get_settings
from database import Neo4jClient, ensure_neo4j_schema
from evaluation import BenchmarkRunner, get_sample_evaluation_dataset


def run_benchmarks() -> None:
    """Run evaluation benchmark suite."""
    print("===================================================")
    print(" Enterprise GraphRAG Benchmark Suite")
    print("===================================================")

    dataset = get_sample_evaluation_dataset()
    runner = BenchmarkRunner()

    start = time.time()
    results = runner.run_benchmark(dataset)
    elapsed = time.time() - start

    print(f"Benchmark finished in {elapsed:.2f} seconds.")
    print(f"Precision@10: {results['avg_precision']:.4f}")
    print(f"Recall@10:    {results['avg_recall']:.4f}")
    print(f"MRR:          {results['avg_mrr']:.4f}")


def download_models() -> None:
    """Download required LLM and embedding models for local offline execution."""
    settings = get_settings()
    cache_dir = settings.model.model_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("===================================================")
    print(" Enterprise GraphRAG Model Pre-downloader")
    print("===================================================")

    emb_name = settings.model.embedding_model_name
    print(f"\n[1/2] Downloading embedding model: {emb_name}...")
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(emb_name, cache_folder=str(cache_dir))
        print(f"✓ Embedding model '{emb_name}' downloaded successfully.")
    except Exception as exc:
        print(f"✗ Failed to download embedding model: {exc}")

    gen_name = settings.model.generation_model_name
    hf_token = settings.model.hf_token or os.environ.get("HF_TOKEN")

    print(f"\n[2/2] Downloading LLM generation model: {gen_name}...")
    if not hf_token:
        print(
            "⚠️ WARNING: HF_TOKEN environment variable is missing!\n"
            "Gemma models are gated repositories requiring HuggingFace authentication.\n"
            "Set HF_TOKEN in your .env file or environment before downloading.\n"
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Checking tokenizer for {gen_name}...")
        try:
            AutoTokenizer.from_pretrained(
                gen_name, cache_dir=str(cache_dir), token=hf_token, local_files_only=True
            )
            print(f"✓ Tokenizer already available locally.")
        except Exception:
            AutoTokenizer.from_pretrained(gen_name, cache_dir=str(cache_dir), token=hf_token)

        print(f"Checking weights for {gen_name}...")
        try:
            AutoModelForCausalLM.from_pretrained(
                gen_name,
                cache_dir=str(cache_dir),
                token=hf_token,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            print(f"✓ Generation model '{gen_name}' already available locally.")
        except Exception:
            AutoModelForCausalLM.from_pretrained(
                gen_name,
                cache_dir=str(cache_dir),
                token=hf_token,
                low_cpu_mem_usage=True,
            )
            print(f"✓ Generation model '{gen_name}' downloaded successfully.")
    except Exception as exc:
        print(f"✗ Failed to download/verify generation model: {exc}")

    print("\n===================================================")
    print(" Model download process complete!")
    print(" System is ready for OFFLINE execution.")
    print("===================================================")


def setup_neo4j() -> None:
    """Connect to local Neo4j Community instance and initialize constraints/indexes."""
    settings = get_settings()
    print("===================================================")
    print(" Enterprise GraphRAG Neo4j Setup")
    print("===================================================")
    print(f"Connecting to Neo4j at {settings.neo4j.uri}...")

    client = Neo4jClient(settings.neo4j)
    try:
        client.verify_connectivity()
        print("[OK] Successfully connected to Neo4j.")
        print("Applying schema constraints and indexes...")
        ensure_neo4j_schema(client)
        print("[OK] Neo4j schema setup complete.")
    except Exception as exc:
        print(f"[FAILED] Neo4j setup failed: {exc}")
        print("Ensure Neo4j Community Edition service is installed and running locally.")


for _mod in ["benchmark", "download_models", "setup_neo4j"]:
    sys.modules[f"scripts.{_mod}"] = sys.modules[__name__]
