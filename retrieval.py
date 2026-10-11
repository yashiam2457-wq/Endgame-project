"""
Enterprise GraphRAG — Retrieval Package.

Consolidated Retrieval Subsystem: FAISSStore, GraphRetriever, HybridRanker, ContextBuilder, IntentDetector, QueryDecomposer, EntityRecognizer, RetrievalPipeline.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import faiss
import numpy as np

from core import (
    AppConfig,
    FAISSConfig,
    FAISSIndexError,
    FAISSIndexNotFoundError,
    retrieval_logger,
)
from database import CacheStore, Neo4jClient
from embedding import EmbeddingEncoder
from generation import AnswerGenerator, GemmaModel
from graph import GraphTraversal
from models import Citation, QueryIntent, QueryRequest, QueryResponse, SearchResult
from prompts import build_decomposition_prompt, build_intent_prompt
from utils import count_tokens_approx, parse_json_safe, timer, truncate_text

# ── SECTION 1: FAISS Vector Store ────────────────────────────────────────────


class FAISSStore:
    """FAISS vector index with persistence and MMR support."""

    def __init__(self, config: FAISSConfig, model_name: str = "") -> None:
        self._config = config
        self._model_name = model_name
        self._index: Optional[faiss.Index] = None
        self._chunk_ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._dimension = config.dimension
        self._built = False

    def _create_empty_index(self) -> faiss.Index:
        dim = self._dimension
        index_type = self._config.index_type

        if index_type == "Flat":
            base = faiss.IndexFlatIP(dim)
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(dim)
            base = faiss.IndexIVFFlat(
                quantizer, dim, self._config.nlist, faiss.METRIC_INNER_PRODUCT
            )
        elif index_type == "IVFPQ":
            quantizer = faiss.IndexFlatIP(dim)
            m = max(4, dim // 4)
            base = faiss.IndexIVFPQ(quantizer, dim, self._config.nlist, m, 8)
        else:
            raise FAISSIndexError(f"Unknown index type: {index_type}")

        return faiss.IndexIDMap(base)

    def load_if_exists(self) -> bool:
        index_file = self._config.index_path / "index.faiss"
        meta_file = self._config.index_path / "metadata.pkl"

        if not index_file.exists() or not meta_file.exists():
            retrieval_logger.info("No existing FAISS index found.")
            return False

        try:
            self._index = faiss.read_index(str(index_file))
            with meta_file.open("rb") as fh:
                meta = pickle.load(fh)
            self._chunk_ids = meta["chunk_ids"]
            self._metadata = meta["metadata"]
            self._dimension = meta.get("dimension", self._dimension)
            self._built = True
            retrieval_logger.info(
                f"FAISS index loaded: {self._index.ntotal} vectors "
                f"(dim={self._dimension})"
            )
            return True
        except Exception as exc:
            raise FAISSIndexError(
                f"Failed to load FAISS index: {exc}", cause=exc
            ) from exc

    def save(self) -> None:
        if self._index is None:
            raise FAISSIndexError("No index to save.")
        self._config.index_path.mkdir(parents=True, exist_ok=True)
        index_file = self._config.index_path / "index.faiss"
        meta_file = self._config.index_path / "metadata.pkl"

        faiss.write_index(self._index, str(index_file))
        with meta_file.open("wb") as fh:
            pickle.dump(
                {
                    "chunk_ids": self._chunk_ids,
                    "metadata": self._metadata,
                    "dimension": self._dimension,
                    "model_name": self._model_name,
                },
                fh,
            )
        retrieval_logger.info(
            f"FAISS index saved: {self._index.ntotal} vectors → {index_file}"
        )

    def add_chunks(
        self,
        chunks: list[Any],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise FAISSIndexError(
                f"Chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        if not chunks:
            return

        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        if self._index is None:
            self._dimension = vectors.shape[1]
            self._index = self._create_empty_index()

        inner = faiss.downcast_index(self._index.index)
        if hasattr(inner, "is_trained") and not inner.is_trained:
            if len(vectors) >= self._config.nlist:
                retrieval_logger.info(
                    f"Training FAISS IVF index on {len(vectors)} vectors..."
                )
                inner.train(vectors)
            else:
                retrieval_logger.info(
                    f"Corpus size ({len(vectors)} vectors) is smaller than IVF nlist ({self._config.nlist}). "
                    "Using Flat index for exact precision."
                )
                flat_base = faiss.IndexFlatIP(self._dimension)
                new_index = faiss.IndexIDMap(flat_base)
                if self._index.ntotal > 0:
                    try:
                        old_vecs = self._index.reconstruct_n(0, self._index.ntotal)
                        old_ids = np.arange(len(self._chunk_ids) - len(chunks), dtype=np.int64)
                        new_index.add_with_ids(old_vecs, old_ids)
                    except Exception:
                        pass
                self._index = new_index

        start_id = len(self._chunk_ids)
        ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)
        self._index.add_with_ids(vectors, ids)

        for chunk in chunks:
            self._chunk_ids.append(chunk.id)
            doc_fname = getattr(chunk, "filename", None) or getattr(chunk, "document_filename", None) or ""
            self._metadata.append(
                {
                    "namespace": "chunk",
                    "document_id": chunk.document_id,
                    "filename": doc_fname,
                    "document_filename": doc_fname,
                    "text": chunk.text[:500],
                    "page_number": chunk.page_number,
                    "section_header": chunk.section_header,
                    "sequence_index": chunk.sequence_index,
                    "chunk_type": chunk.chunk_type.value,
                }
            )

        self._built = True
        self.save()
        retrieval_logger.info(
            f"Added {len(chunks)} vectors. Total: {self._index.ntotal}"
        )

    def add_vectors(
        self,
        item_ids: list[str],
        embeddings: list[list[float]],
        namespace: str,
        metadata_list: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        if len(item_ids) != len(embeddings):
            raise FAISSIndexError("ID and embedding count mismatch.")
        if not item_ids:
            return

        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        if self._index is None:
            self._dimension = vectors.shape[1]
            self._index = self._create_empty_index()

        start_id = len(self._chunk_ids)
        ids = np.arange(start_id, start_id + len(item_ids), dtype=np.int64)

        inner = faiss.downcast_index(self._index.index)
        if hasattr(inner, "is_trained") and not inner.is_trained:
            if len(vectors) >= self._config.nlist:
                inner.train(vectors)
            else:
                flat_base = faiss.IndexFlatIP(self._dimension)
                new_index = faiss.IndexIDMap(flat_base)
                if self._index.ntotal > 0:
                    try:
                        old_vecs = self._index.reconstruct_n(0, self._index.ntotal)
                        old_ids = np.arange(len(self._chunk_ids), dtype=np.int64)
                        new_index.add_with_ids(old_vecs, old_ids)
                    except Exception:
                        pass
                self._index = new_index

        self._index.add_with_ids(vectors, ids)

        for i, item_id in enumerate(item_ids):
            self._chunk_ids.append(item_id)
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            meta["namespace"] = namespace
            meta["item_id"] = item_id
            self._metadata.append(meta)

        self._built = True
        self.save()
        retrieval_logger.info(f"Added {len(item_ids)} [{namespace}] vectors into FAISS. Total: {self._index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        document_ids: list[str] | str | None = None,
    ) -> list[dict[str, Any]]:
        if self._index is None or not self._built or self.total_vectors == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        inner = faiss.downcast_index(self._index.index)
        if hasattr(inner, "nprobe"):
            inner.nprobe = min(self._config.nprobe, self._config.nlist)

        allowed_docs: set[str] | None = None
        if document_ids:
            if isinstance(document_ids, (list, set, tuple)):
                allowed_docs = set(str(d).strip() for d in document_ids if str(d).strip())
            elif isinstance(document_ids, str) and document_ids.strip():
                allowed_docs = {document_ids.strip()}

        fetch_k = min(max(top_k * 10, 100), self._index.ntotal) if allowed_docs else min(top_k, self._index.ntotal)

        with timer("faiss_search", log=False):
            scores, indices = self._index.search(query, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunk_ids):
                continue
            chunk_id = self._chunk_ids[int(idx)]
            meta = self._metadata[int(idx)]

            if allowed_docs:
                doc_id = str(meta.get("document_id", "")).strip()
                if doc_id not in allowed_docs:
                    continue

            results.append(
                {
                    "chunk_id": chunk_id,
                    "faiss_score": float(score),
                    **meta,
                }
            )
            if len(results) >= top_k:
                break

        return results

    def search_mmr(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        fetch_k: int = 50,
        lambda_mult: float = 0.7,
        document_ids: list[str] | str | None = None,
    ) -> list[dict[str, Any]]:
        candidates = self.search(query_embedding, top_k=fetch_k, document_ids=document_ids)
        if len(candidates) <= top_k:
            return candidates

        selected: list[dict[str, Any]] = []
        remaining = list(candidates)

        best = max(remaining, key=lambda x: x["faiss_score"])
        selected.append(best)
        remaining.remove(best)

        while len(selected) < top_k and remaining:
            best_candidate = None
            best_mmr_score = float("-inf")

            for cand in remaining:
                relevance = cand["faiss_score"]
                max_sim = max(
                    1.0 - abs(cand["faiss_score"] - s["faiss_score"])
                    for s in selected
                )
                mmr = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                if mmr > best_mmr_score:
                    best_mmr_score = mmr
                    best_candidate = cand

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)

        return selected

    def delete_by_document(self, document_id: str) -> int:
        if self._index is None:
            return 0

        ids_to_remove = [
            i
            for i, meta in enumerate(self._metadata)
            if meta.get("document_id") == document_id
        ]
        if not ids_to_remove:
            return 0

        remove_ids = np.array(ids_to_remove, dtype=np.int64)
        self._index.remove_ids(remove_ids)

        for idx in sorted(ids_to_remove, reverse=True):
            self._chunk_ids.pop(idx)
            self._metadata.pop(idx)

        self.save()
        retrieval_logger.info(
            f"Removed {len(ids_to_remove)} vectors for document {document_id}"
        )
        return len(ids_to_remove)

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def is_built(self) -> bool:
        return self._built and self._index is not None


# ── SECTION 2: Graph Retriever ───────────────────────────────────────────────


class GraphRetriever:
    """Retrieves subgraphs and entity neighborhood context from Neo4j/SQLite."""

    def __init__(self, neo4j: Neo4jClient, sqlite: Any = None) -> None:
        if sqlite is None:
            try:
                from core import get_sqlite
                sqlite = get_sqlite()
            except Exception:
                sqlite = None
        self._traversal = GraphTraversal(neo4j, sqlite=sqlite)

    def retrieve_graph_context(
        self,
        entity_names: list[str],
        hop_depth: int = 2,
        limit: int = 20,
        document_ids: list[str] | str | None = None,
    ) -> list[dict[str, Any]]:
        if not entity_names:
            return []

        allowed_docs: set[str] | None = None
        if document_ids:
            if isinstance(document_ids, (list, set, tuple)):
                allowed_docs = set(str(d).strip() for d in document_ids if str(d).strip())
            elif isinstance(document_ids, str) and document_ids.strip():
                allowed_docs = {document_ids.strip()}

        context: list[dict[str, Any]] = []
        for name in entity_names[:5]:
            neighbors = self._traversal.get_neighborhood(
                name, hops=hop_depth, limit=limit
            )
            if allowed_docs:
                for n in neighbors:
                    doc_id = str(n.get("document_id", "")).strip()
                    if not doc_id or doc_id in allowed_docs:
                        context.append(n)
            else:
                context.extend(neighbors)

        return context


# ── SECTION 3: Hybrid Ranker ─────────────────────────────────────────────────


class HybridRanker:
    """Ranks and fuses search results from vector (FAISS) and graph retrieval."""

    def __init__(self, faiss_weight: float = 0.5, graph_weight: float = 0.5) -> None:
        total = faiss_weight + graph_weight
        if total > 0:
            self._faiss_weight = faiss_weight / total
            self._graph_weight = graph_weight / total
        else:
            self._faiss_weight = 0.5
            self._graph_weight = 0.5

    def rank(
        self,
        vector_results: list[dict[str, Any]],
        graph_context: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[SearchResult]:
        graph_entity_set = {
            item.get("name", "").lower(): item for item in graph_context if "name" in item
        }

        ranked_results: list[SearchResult] = []

        for item in vector_results:
            chunk_id = item["chunk_id"]
            doc_id = item.get("document_id", "")
            text = item.get("text", "")
            faiss_score = float(item.get("faiss_score", 0.0))

            text_lower = text.lower()
            matching_entities = [
                name for name in graph_entity_set if name in text_lower
            ]
            graph_score = min(1.0, len(matching_entities) * 0.2)

            fused_score = (self._faiss_weight * faiss_score) + (
                self._graph_weight * graph_score
            )

            ranked_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    document_filename=item.get("filename", item.get("document_filename", "Document")),
                    text=text,
                    score=fused_score,
                    faiss_score=faiss_score,
                    graph_score=graph_score,
                    page_number=item.get("page_number"),
                    section_header=item.get("section_header"),
                    entities=matching_entities,
                    metadata=item,
                )
            )

        ranked_results.sort(key=lambda x: x.score, reverse=True)
        return ranked_results[:top_k]


# ── SECTION 4: Context Builder ───────────────────────────────────────────────


class ContextBuilder:
    """Formats and compresses top retrieval results for the generation model."""

    def __init__(self, max_tokens: int = 3072, dedup_threshold: float = 0.95) -> None:
        self._max_tokens = max_tokens
        self._dedup_threshold = dedup_threshold

    def build_context(self, search_results: list[SearchResult]) -> list[SearchResult]:
        deduped: list[SearchResult] = []
        seen_snippets: set[str] = set()
        accumulated_tokens = 0

        for res in search_results:
            short_text = res.text[:100].strip().lower()
            if short_text in seen_snippets:
                continue
            seen_snippets.add(short_text)

            tokens = res.metadata.get("token_count") or count_tokens_approx(res.text)
            if accumulated_tokens + tokens > self._max_tokens:
                remaining_budget = max(0, self._max_tokens - accumulated_tokens)
                if remaining_budget > 50:
                    res.text = truncate_text(res.text, remaining_budget)
                    deduped.append(res)
                break

            deduped.append(res)
            accumulated_tokens += tokens

        return deduped


# ── SECTION 5: Intent Detector & Query Decomposer ───────────────────────────


class IntentDetector:
    """Classifies user queries into intent categories."""

    def __init__(self, llm: GemmaModel) -> None:
        self._llm = llm

    def detect_intent(self, query: str) -> QueryIntent:
        q_lower = query.lower().strip()

        if any(w in q_lower for w in ["compare", "difference between", "versus", "vs"]):
            return QueryIntent.COMPARATIVE
        if any(w in q_lower for w in ["why", "how does", "explain", "analyze"]):
            return QueryIntent.ANALYTICAL
        if any(w in q_lower for w in ["what is", "who is", "when did", "where is"]):
            return QueryIntent.FACTUAL

        if self._llm and self._llm.is_loaded:
            try:
                prompt = build_intent_prompt(query)
                raw = self._llm.generate(prompt, max_new_tokens=10, do_sample=False).strip().lower()
                for intent in QueryIntent:
                    if intent.value in raw:
                        return intent
            except Exception:
                pass

        return QueryIntent.EXPLORATORY


class QueryDecomposer:
    """Decomposes multi-part queries into independent sub-queries."""

    def __init__(self, llm: GemmaModel) -> None:
        self._llm = llm

    def decompose(self, query: str) -> list[str]:
        if len(query.split()) < 10 and " and " not in query.lower() and " or " not in query.lower():
            return [query]

        if self._llm and self._llm.is_loaded:
            try:
                prompt = build_decomposition_prompt(query)
                raw = self._llm.generate(prompt, max_new_tokens=128, do_sample=False)
                parsed = parse_json_safe(raw)
                if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                    return parsed
            except Exception:
                pass

        return [query]


class EntityRecognizer:
    """Extracts candidate entity names from user queries using Regex/NLP heuristics."""

    def recognize_entities(self, query: str) -> list[str]:
        title_cased = re.findall(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b", query)
        stop_words = {"What", "How", "Why", "Where", "Who", "When", "Which", "Can", "Could", "Tell", "Explain", "Compare", "List"}

        filtered = [e for e in title_cased if e not in stop_words and len(e) > 2]
        return list(dict.fromkeys(filtered))


# ── SECTION 6: Retrieval Pipeline Orchestrator ───────────────────────────────


class RetrievalPipeline:
    """End-to-end online query processing pipeline."""

    def __init__(
        self,
        settings: AppConfig,
        faiss_store: FAISSStore,
        neo4j_client: Neo4jClient,
        encoder: EmbeddingEncoder,
        llm: GemmaModel,
        cache: CacheStore,
    ) -> None:
        self._settings = settings
        self._faiss = faiss_store
        self._neo4j = neo4j_client
        self._encoder = encoder
        self._llm = llm
        self._cache = cache

        self._intent_detector = IntentDetector(llm)
        self._query_decomposer = QueryDecomposer(llm)
        self._entity_recognizer = EntityRecognizer()
        self._graph_retriever = GraphRetriever(neo4j_client)
        self._hybrid_ranker = HybridRanker(
            faiss_weight=settings.retrieval.faiss_weight,
            graph_weight=settings.retrieval.graph_weight,
        )
        self._context_builder = ContextBuilder(
            max_tokens=settings.retrieval.max_context_tokens,
            dedup_threshold=settings.retrieval.dedup_threshold,
        )
        self._generator = AnswerGenerator(llm=llm, cache=cache)

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        query_id = str(uuid4())
        start_total = time.perf_counter()
        query_text = request.query.strip()

        retrieval_logger.info(f"[{query_id}] Query received: '{query_text}' (mode={request.mode.value})")

        with timer(f"query:{query_id}") as t_retrieval:
            target_doc_ids: list[str] | None = None
            if request.filters:
                raw_doc_id = request.filters.get("document_id") or request.filters.get("document_ids")
                if isinstance(raw_doc_id, list):
                    target_doc_ids = [str(d).strip() for d in raw_doc_id if str(d).strip()]
                elif isinstance(raw_doc_id, str) and raw_doc_id.strip():
                    target_doc_ids = [raw_doc_id.strip()]

            intent = self._intent_detector.detect_intent(query_text)
            sub_queries = self._query_decomposer.decompose(query_text)
            query_entities = self._entity_recognizer.recognize_entities(query_text)

            raw_vector_results: list[dict[str, Any]] = []
            seen_chunks: set[str] = set()

            for sq in sub_queries:
                q_emb = self._encoder.encode_single(sq)
                candidates = self._faiss.search_mmr(
                    q_emb,
                    top_k=request.top_k,
                    lambda_mult=self._settings.retrieval.mmr_lambda,
                    document_ids=target_doc_ids,
                )
                for cand in candidates:
                    cid = cand["chunk_id"]
                    if cid not in seen_chunks:
                        if target_doc_ids and str(cand.get("document_id", "")).strip() not in target_doc_ids:
                            continue
                        seen_chunks.add(cid)
                        raw_vector_results.append(cand)

            try:
                from core import get_sqlite
                sqlite_mgr = get_sqlite()
                docs_list, _ = sqlite_mgr.list_documents(page_size=500)
                doc_map = {d["id"]: d["filename"] for d in docs_list if "id" in d and "filename" in d}
                for cand in raw_vector_results:
                    did = cand.get("document_id")
                    if did and did in doc_map:
                        cand["filename"] = doc_map[did]
                        cand["document_filename"] = doc_map[did]

                # Fallback: if vector results are empty or query requests a summary/overview, load document chunks directly from SQLite
                q_lower = query_text.lower()
                is_summary_query = any(w in q_lower for w in ["summarize", "summary", "overview", "about", "outline", "brief", "main points"])
                if not raw_vector_results or is_summary_query:
                    available_docs = docs_list
                    if target_doc_ids:
                        available_docs = [d for d in docs_list if d.get("id") in target_doc_ids]
                    for doc_rec in available_docs[:3]:
                        doc_chunks = sqlite_mgr.get_chunks_for_document(doc_rec["id"])
                        for c in doc_chunks[:5]:
                            cid = c["id"]
                            if cid not in seen_chunks:
                                seen_chunks.add(cid)
                                raw_vector_results.append({
                                    "chunk_id": cid,
                                    "document_id": doc_rec["id"],
                                    "filename": doc_rec.get("filename", "Document"),
                                    "document_filename": doc_rec.get("filename", "Document"),
                                    "text": c["text"],
                                    "faiss_score": 0.75,
                                    "page_number": c.get("page_number"),
                                    "section_header": c.get("section_header"),
                                })
            except Exception as exc:
                retrieval_logger.debug(f"SQLite document fallback notice: {exc}")

            graph_context: list[dict[str, Any]] = []
            if request.include_graph:
                graph_context = self._graph_retriever.retrieve_graph_context(
                    entity_names=query_entities,
                    hop_depth=request.graph_hops,
                    limit=20,
                    document_ids=target_doc_ids,
                )

            ranked_results = self._hybrid_ranker.rank(
                vector_results=raw_vector_results,
                graph_context=graph_context,
                top_k=request.top_k,
            )

            fused_context = self._context_builder.build_context(ranked_results)

        retrieval_ms = t_retrieval["elapsed_ms"]

        response = self._generator.generate(
            query=query_text,
            search_results=fused_context,
            intent=intent,
            entities=query_entities,
            query_id=query_id,
            session_id=request.session_id,
        )

        total_ms = (time.perf_counter() - start_total) * 1000
        response.sub_queries = sub_queries
        response.graph_nodes_visited = len(graph_context)
        response.retrieval_time_ms = retrieval_ms
        response.total_time_ms = total_ms

        retrieval_logger.info(
            f"[{query_id}] Query pipeline complete: total={total_ms:.0f}ms (retrieval={retrieval_ms:.0f}ms)"
        )
        return response


# ── SECTION 7: Backward Compatibility Submodule Aliases ─────────────────────

for _mod in [
    "context_builder",
    "entity_recognizer",
    "faiss_store",
    "graph_retriever",
    "hybrid_ranker",
    "intent_detector",
    "pipeline",
    "query_decomposer",
]:
    sys.modules[f"retrieval.{_mod}"] = sys.modules[__name__]
