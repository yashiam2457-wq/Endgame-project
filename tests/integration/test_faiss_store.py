"""Integration test for FAISS vector store."""
from __future__ import annotations

import numpy as np
from core.config import FaissConfig
from retrieval.faiss_store import FAISSStore
from models.document import Chunk, ChunkType


def test_faiss_add_and_search(tmp_path):
    config = FaissConfig(index_path=tmp_path / "faiss_idx", dimension=4)
    store = FAISSStore(config)

    chunk = Chunk(
        document_id="doc_1",
        text="Sample text",
        chunk_type=ChunkType.STANDALONE,
    )
    embeddings = [[0.1, 0.2, 0.3, 0.4]]

    store.add_chunks([chunk], embeddings)
    assert store.total_vectors == 1

    query_vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    results = store.search(query_vector, top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == chunk.id
