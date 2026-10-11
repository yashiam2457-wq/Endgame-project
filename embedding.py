"""
Enterprise GraphRAG — Embedding Package.

Consolidated Embedding Subsystem: EmbeddingEncoder, BatchEmbedder, MultiVectorEmbedder.
"""

from __future__ import annotations

import hashlib
import re
import sys
from typing import Any, Optional

import numpy as np

try:
    import torch
except BaseException:
    torch = None

from core import EmbeddingModelNotLoadedError, ModelConfig, embedding_logger
from models import Community, Entity, Relationship
from utils import detect_device, timer

# ── SECTION 1: Encoder & Fallback ─────────────────────────────────────────────


def _deterministic_hash_embedding(text: str, dimension: int = 384) -> np.ndarray:
    """Generate a deterministic dense embedding vector for text using token/ngram hashing."""
    vec = np.zeros(dimension, dtype=np.float32)
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        tokens = [text.lower()] if text.strip() else ["empty"]

    ngrams = list(tokens)
    for i in range(len(tokens) - 1):
        ngrams.append(f"{tokens[i]}_{tokens[i+1]}")

    for item in ngrams:
        h_bytes = hashlib.md5(item.encode("utf-8")).digest()
        idx = int.from_bytes(h_bytes[:4], "little") % dimension
        sign = 1.0 if (h_bytes[4] % 2 == 0) else -1.0
        weight = 1.0 + (int.from_bytes(h_bytes[5:7], "little") % 100) / 100.0
        vec[idx] += sign * weight

    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    else:
        vec[0] = 1.0

    return vec


class EmbeddingEncoder:
    """Wraps SentenceTransformers with lazy loading, GPU support, and deterministic fallback."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model = None
        self._device = detect_device(config.embedding_device)
        self._dimension: int = 384
        self._is_fallback: bool = False

    def load(self) -> None:
        if self._model is not None or self._is_fallback:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self._config.embedding_model_name,
                device=str(self._device),
                cache_folder=str(self._config.model_cache_dir),
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
            embedding_logger.info(f"Embedding model loaded successfully. Dimension: {self._dimension}")
            return
        except BaseException as exc:
            embedding_logger.warning(
                f"Failed to load embedding model on {self._device}: {exc}. Trying CPU..."
            )

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self._config.embedding_model_name,
                device="cpu",
                cache_folder=str(self._config.model_cache_dir),
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
            embedding_logger.info(f"CPU embedding model loaded successfully. Dimension: {self._dimension}")
        except BaseException as exc2:
            embedding_logger.warning(
                f"PyTorch/SentenceTransformers unavailable ({exc2}). Enabling zero-dependency Hash Embedding fallback mode."
            )
            self._model = None
            self._is_fallback = True
            self._dimension = 384

    def _ensure_loaded(self) -> None:
        if self._model is None and not self._is_fallback:
            self.load()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        self._ensure_loaded()
        dim = self._dimension or 384

        if not texts:
            return np.empty((0, dim), dtype=np.float32)

        if self._is_fallback or self._model is None:
            matrix = np.zeros((len(texts), dim), dtype=np.float32)
            for i, text in enumerate(texts):
                matrix[i] = _deterministic_hash_embedding(text, dimension=dim)
            return matrix

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                precision="float32",
            )
            return embeddings.astype(np.float32)
        except Exception as exc:
            embedding_logger.error(f"Encoding failed: {exc}. Using fallback hash vectors.")
            matrix = np.zeros((len(texts), dim), dtype=np.float32)
            for i, text in enumerate(texts):
                matrix[i] = _deterministic_hash_embedding(text, dimension=dim)
            return matrix

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        result = self.encode([text], normalize=normalize)
        return result[0]

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dimension

    @property
    def model_name(self) -> str:
        if self._is_fallback:
            return f"{self._config.embedding_model_name} (Hash Fallback)"
        return self._config.embedding_model_name


# ── SECTION 2: Batch Embedder ────────────────────────────────────────────────


class BatchEmbedder:
    """Handles batch embedding with SQLite cache support."""

    def __init__(self, encoder: EmbeddingEncoder, cache: Optional[Any] = None) -> None:
        self._encoder = encoder
        self._cache = cache

    def embed_batch(
        self,
        texts: list[str],
        cache_prefix: str = "chunk",
        batch_size: int = 64,
    ) -> list[list[float]]:
        if not texts:
            return []

        cache_keys = [self._make_cache_key(t) for t in texts]
        results: list[Optional[list[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        if self._cache is not None:
            for i, key in enumerate(cache_keys):
                cached = self._cache.get("embedding", f"{cache_prefix}:{key}")
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(texts[i])
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        if uncached_texts:
            with timer(f"batch_embed:{len(uncached_texts)}") as t:
                embeddings = self._encoder.encode(
                    uncached_texts, batch_size=batch_size
                )
            embedding_logger.debug(
                f"Embedded {len(uncached_texts)} texts in {t['elapsed_ms']:.0f} ms"
            )

            for local_idx, (global_idx, text) in enumerate(
                zip(uncached_indices, uncached_texts)
            ):
                embedding = embeddings[local_idx].tolist()
                results[global_idx] = embedding

                if self._cache is not None:
                    key = cache_keys[global_idx]
                    self._cache.set("embedding", f"{cache_prefix}:{key}", embedding)

        return [r for r in results if r is not None]

    @staticmethod
    def _make_cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# ── SECTION 3: Multi-Vector Embedder ─────────────────────────────────────────


class MultiVectorEmbedder:
    """Generates embedding vectors for Chunks, Entities, Relationships, and Communities."""

    def __init__(self, encoder: EmbeddingEncoder) -> None:
        self._encoder = encoder

    def embed_chunks(self, chunks: list[Any]) -> list[list[float]]:
        texts = [chunk.text for chunk in chunks]
        if not texts:
            return []
        arr = self._encoder.encode(texts)
        return arr.tolist()

    def embed_entities(self, entities: list[Entity]) -> list[list[float]]:
        texts = [
            f"Entity: {e.name}. Type: {e.entity_type.value}. Description: {e.description or ''}"
            for e in entities
        ]
        if not texts:
            return []
        arr = self._encoder.encode(texts)
        return arr.tolist()

    def embed_relationships(self, relationships: list[Relationship]) -> list[list[float]]:
        texts = [
            f"Relationship: {r.source_entity_name} {r.relationship_type.value} {r.target_entity_name}. "
            f"Evidence: {r.evidence_sentence or r.description or ''}"
            for r in relationships
        ]
        if not texts:
            return []
        arr = self._encoder.encode(texts)
        return arr.tolist()

    def embed_communities(self, communities: list[Community]) -> list[list[float]]:
        texts = [
            f"Community: {c.title}. Summary: {c.summary}. Topics: {', '.join(c.topics)}"
            for c in communities
        ]
        if not texts:
            return []
        arr = self._encoder.encode(texts)
        return arr.tolist()


# ── SECTION 4: Backward Compatibility Submodule Aliases ─────────────────────

for _mod in ["batch", "encoder", "multi_vector"]:
    sys.modules[f"embedding.{_mod}"] = sys.modules[__name__]
