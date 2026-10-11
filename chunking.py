"""
Enterprise GraphRAG — Chunking Package.

Consolidated Chunkers: BaseChunker, ChunkerFactory, FixedChunker, SlidingWindowChunker, RecursiveChunker, SentenceChunker, SemanticChunker, HierarchicalChunker.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from core import ChunkingConfig
from models import Chunk, ChunkType, Document
from utils import count_tokens_approx, extract_sentences

# ── SECTION 1: Base Chunker ──────────────────────────────────────────────────


class BaseChunker(ABC):
    """Abstract chunker interface."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        ...

    def _make_chunk(
        self,
        document: Document,
        text: str,
        sequence_index: int,
        start_char: int = 0,
        end_char: int = 0,
        page_number: int | None = None,
        section_header: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        return Chunk(
            document_id=document.id,
            sequence_index=sequence_index,
            text=text,
            start_char=start_char,
            end_char=end_char,
            page_number=page_number,
            section_header=section_header,
            token_count=count_tokens_approx(text),
            metadata=metadata or {},
        )


# ── SECTION 2: Fixed Chunker ─────────────────────────────────────────────────


class FixedChunker(BaseChunker):
    """Splits text into fixed-size character windows."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._size = config.chunk_size * 4
        self._min_length = config.min_length

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.cleaned_text or document.raw_text
        chunks: list[Chunk] = []
        idx = 0
        seq = 0
        while idx < len(text):
            end = min(idx + self._size, len(text))
            segment = text[idx:end].strip()
            if len(segment) >= self._min_length:
                chunks.append(
                    self._make_chunk(
                        document, segment, seq,
                        start_char=idx, end_char=end
                    )
                )
                seq += 1
            idx = end
        return chunks


# ── SECTION 3: Sliding Window Chunker ────────────────────────────────────────


class SlidingWindowChunker(BaseChunker):
    """Chunks text with a sliding window and overlapping stride."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._size = config.chunk_size * 4
        self._overlap = config.chunk_overlap * 4
        self._min_length = config.min_length

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.cleaned_text or document.raw_text
        stride = max(1, self._size - self._overlap)
        chunks: list[Chunk] = []
        idx = 0
        seq = 0
        while idx < len(text):
            end = min(idx + self._size, len(text))
            segment = text[idx:end].strip()
            if len(segment) >= self._min_length:
                chunks.append(
                    self._make_chunk(
                        document, segment, seq,
                        start_char=idx, end_char=end,
                        metadata={"overlap": self._overlap}
                    )
                )
                seq += 1
            if end >= len(text):
                break
            idx += stride
        return chunks


# ── SECTION 4: Recursive Chunker ─────────────────────────────────────────────

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


class RecursiveChunker(BaseChunker):
    """Recursively splits text by progressively smaller separators."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._max_chars = config.chunk_size * 4
        self._overlap_chars = config.chunk_overlap * 4
        self._min_length = config.min_length

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.cleaned_text or document.raw_text
        raw_chunks = self._split_recursive(text, _SEPARATORS)
        chunks: list[Chunk] = []
        pos = 0
        for seq, segment in enumerate(raw_chunks):
            segment = segment.strip()
            if len(segment) < self._min_length:
                continue
            start = text.find(segment, pos)
            if start == -1:
                start = pos
            end = start + len(segment)
            chunks.append(
                self._make_chunk(
                    document, segment, seq,
                    start_char=start, end_char=end
                )
            )
            pos = max(0, end - self._overlap_chars)
        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not separators or len(text) <= self._max_chars:
            return [text] if text.strip() else []

        sep = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(sep) if sep else list(text)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= self._max_chars:
                current = candidate
            else:
                if current:
                    if len(current) > self._max_chars:
                        result.extend(self._split_recursive(current, remaining_seps))
                    else:
                        result.append(current)
                current = part

        if current:
            if len(current) > self._max_chars:
                result.extend(self._split_recursive(current, remaining_seps))
            else:
                result.append(current)

        return result


# ── SECTION 5: Sentence Chunker ──────────────────────────────────────────────


class SentenceChunker(BaseChunker):
    """Groups sentences into chunks that fit within a token budget."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._max_tokens = config.chunk_size
        self._overlap_sentences = max(1, config.chunk_overlap // 50)
        self._min_length = config.min_length

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.cleaned_text or document.raw_text
        sentences = extract_sentences(text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0
        seq = 0
        pos = 0

        for sentence in sentences:
            stokens = count_tokens_approx(sentence)
            if current_tokens + stokens > self._max_tokens and current:
                segment = " ".join(current)
                if len(segment) >= self._min_length:
                    start = text.find(current[0], pos)
                    end = start + len(segment)
                    chunks.append(
                        self._make_chunk(document, segment, seq, start, end)
                    )
                    seq += 1
                    pos = max(0, end - 100)
                current = current[-self._overlap_sentences :]
                current_tokens = sum(count_tokens_approx(s) for s in current)

            current.append(sentence)
            current_tokens += stokens

        if current:
            segment = " ".join(current)
            if len(segment) >= self._min_length:
                start = text.find(current[0], pos)
                end = start + len(segment) if start >= 0 else len(text)
                chunks.append(self._make_chunk(document, segment, seq, start, end))

        return chunks


# ── SECTION 6: Semantic Chunker ──────────────────────────────────────────────

_SIMILARITY_THRESHOLD = 0.75
_WINDOW_SIZE = 3


class SemanticChunker(BaseChunker):
    """Groups semantically coherent sentences into chunks."""

    def __init__(self, config: ChunkingConfig, encoder: object | None = None) -> None:
        self._config = config
        self._encoder = encoder
        self._max_tokens = config.chunk_size
        self._min_length = config.min_length

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.cleaned_text or document.raw_text
        sentences = extract_sentences(text)

        if not sentences:
            return []

        if self._encoder is None or len(sentences) < 3:
            return SentenceChunker(self._config).chunk(document)

        boundaries = self._detect_boundaries(sentences)
        return self._build_chunks(document, sentences, boundaries, text)

    def _detect_boundaries(self, sentences: list[str]) -> list[int]:
        embeddings = self._encoder.encode(sentences)  # type: ignore[attr-defined]
        boundaries = [0]
        for i in range(_WINDOW_SIZE, len(sentences) - _WINDOW_SIZE):
            left = np.mean(embeddings[max(0, i - _WINDOW_SIZE) : i], axis=0)
            right = np.mean(embeddings[i : i + _WINDOW_SIZE], axis=0)
            sim = float(
                np.dot(left, right)
                / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9)
            )
            if sim < _SIMILARITY_THRESHOLD:
                boundaries.append(i)
        boundaries.append(len(sentences))
        return boundaries

    def _build_chunks(
        self,
        document: Document,
        sentences: list[str],
        boundaries: list[int],
        original_text: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        pos = 0
        for seq, (start_idx, end_idx) in enumerate(
            zip(boundaries, boundaries[1:])
        ):
            segment = " ".join(sentences[start_idx:end_idx]).strip()
            if len(segment) < self._min_length:
                continue
            char_start = original_text.find(sentences[start_idx], pos)
            if char_start < 0:
                char_start = pos
            char_end = char_start + len(segment)
            chunks.append(
                self._make_chunk(
                    document, segment, seq,
                    start_char=char_start, end_char=char_end,
                    metadata={"semantic": True}
                )
            )
            pos = char_end
        return chunks


# ── SECTION 7: Hierarchical Chunker ──────────────────────────────────────────


class HierarchicalChunker(BaseChunker):
    """Two-level chunking: large parent chunks + smaller child chunks."""

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config
        parent_config = config.model_copy(
            update={"chunk_size": config.chunk_size * 2}
        )
        child_config = config.model_copy()
        self._parent_chunker = RecursiveChunker(parent_config)
        self._child_chunker = RecursiveChunker(child_config)

    def chunk(self, document: Document) -> list[Chunk]:
        parents = self._parent_chunker.chunk(document)
        all_chunks: list[Chunk] = []
        child_seq = 0

        for parent in parents:
            parent.chunk_type = ChunkType.PARENT
            all_chunks.append(parent)

            parent_doc = document.model_copy(
                update={
                    "id": document.id,
                    "cleaned_text": parent.text,
                    "raw_text": parent.text,
                }
            )
            children = self._child_chunker.chunk(parent_doc)

            for child in children:
                child.chunk_type = ChunkType.CHILD
                child.parent_chunk_id = parent.id
                child.sequence_index = child_seq
                child.start_char = parent.start_char + child.start_char
                child.end_char = parent.start_char + child.end_char
                all_chunks.append(child)
                child_seq += 1

        return all_chunks


# ── SECTION 8: Chunker Factory ───────────────────────────────────────────────


class ChunkerFactory:
    """Creates the appropriate chunker based on configuration."""

    @staticmethod
    def create(
        config: ChunkingConfig, encoder: Optional[object] = None
    ) -> BaseChunker:
        strategy = config.strategy

        if strategy == "fixed":
            return FixedChunker(config)

        if strategy == "sliding":
            return SlidingWindowChunker(config)

        if strategy == "recursive":
            return RecursiveChunker(config)

        if strategy == "sentence":
            return SentenceChunker(config)

        if strategy == "semantic":
            return SemanticChunker(config, encoder=encoder)

        if strategy == "hierarchical":
            return HierarchicalChunker(config)

        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Valid options: fixed, sliding, recursive, sentence, semantic, hierarchical"
        )


# ── SECTION 9: Backward Compatibility Submodule Aliases ─────────────────────

for _mod in ["base", "factory", "fixed", "hierarchical", "recursive", "semantic", "sentence", "sliding"]:
    sys.modules[f"chunking.{_mod}"] = sys.modules[__name__]
