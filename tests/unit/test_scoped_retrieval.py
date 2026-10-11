"""
Unit tests for Document-Scoped Retrieval & Knowledge Graph Generation.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from core import FaissConfig, SQLiteConfig
from database import SQLiteManager
from graph import NetworkXGraphOps
from models import Chunk, ChunkType
from retrieval import FAISSStore


@pytest.fixture
def faiss_store(tmp_path):
    config = FaissConfig(index_path=tmp_path / "faiss_test", dimension=4, index_type="Flat")
    store = FAISSStore(config)
    
    chunks = [
        Chunk(id="c1", document_id="doc_A", sequence_index=0, text="Doc A chunk 1", chunk_type=ChunkType.STANDALONE),
        Chunk(id="c2", document_id="doc_A", sequence_index=1, text="Doc A chunk 2", chunk_type=ChunkType.STANDALONE),
        Chunk(id="c3", document_id="doc_B", sequence_index=0, text="Doc B chunk 1", chunk_type=ChunkType.STANDALONE),
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    store.add_chunks(chunks, embeddings)
    return store


def test_faiss_store_document_filtering(faiss_store):
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Search doc_A only
    res_A = faiss_store.search(query_vec, top_k=10, document_ids=["doc_A"])
    assert len(res_A) == 2
    assert all(r["document_id"] == "doc_A" for r in res_A)

    # Search doc_B only
    res_B = faiss_store.search(query_vec, top_k=10, document_ids=["doc_B"])
    assert len(res_B) == 1
    assert res_B[0]["document_id"] == "doc_B"

    # Search unselected doc
    res_none = faiss_store.search(query_vec, top_k=10, document_ids=["doc_C"])
    assert len(res_none) == 0


def test_graph_generation_with_nodes_and_edges(tmp_path):
    db_file = tmp_path / "test_graph_gen.db"
    sqlite = SQLiteManager(SQLiteConfig(path=db_file))
    sqlite.initialize()

    nodes = [
        {"id": "n1", "name": "Apple", "canonical_name": "apple", "entity_type": "COMPANY", "document_id": "doc1"},
        {"id": "n2", "name": "Tim Cook", "canonical_name": "tim cook", "entity_type": "PERSON", "document_id": "doc1"},
        {"id": "n3", "name": "Google", "canonical_name": "google", "entity_type": "COMPANY", "document_id": "doc2"},
    ]
    edges = [
        {
            "id": "e1",
            "source_canonical": "tim cook",
            "target_canonical": "apple",
            "source_name": "Tim Cook",
            "target_name": "Apple",
            "rel_type": "CEO_OF",
            "document_id": "doc1",
        }
    ]

    sqlite.upsert_graph_nodes(nodes)
    sqlite.upsert_graph_edges(edges)

    # Scoped doc1 graph nodes & edges
    doc1_nodes = sqlite.get_graph_nodes_for_documents(["doc1"])
    doc1_edges = sqlite.get_graph_edges_for_documents(["doc1"])

    nx_ops = NetworkXGraphOps()
    nx_ops.build_from_dicts(doc1_nodes, doc1_edges)
    vis = nx_ops.to_cytoscape_format()

    assert len(vis.nodes) == 2
    assert len(vis.edges) == 1
    assert vis.edges[0]["data"]["label"] == "CEO_OF"
