"""
Unit tests for Local Embedded Persistent Graph Store Fallback when Neo4j is offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from core import SQLiteConfig
from database import SQLiteManager
from graph import GraphBuilder, GraphTraversal
from models import Community, Document, Entity, ExtractionResult, Relationship, SupportedFileType


@pytest.fixture
def sqlite_mgr(tmp_path):
    """Temporary SQLite database fixture."""
    db_file = tmp_path / "test_graph.db"
    mgr = SQLiteManager(SQLiteConfig(path=db_file))
    mgr.initialize()
    return mgr


def test_sqlite_graph_tables_crud(sqlite_mgr):
    """Verify SQLite CRUD operations for nodes, edges, and communities."""
    nodes = [
        {"id": "n1", "name": "Python", "canonical_name": "python", "entity_type": "LANGUAGE", "description": "Programming language"},
        {"id": "n2", "name": "FastAPI", "canonical_name": "fastapi", "entity_type": "FRAMEWORK", "description": "Web framework"},
    ]
    edges = [
        {
            "id": "e1",
            "source_canonical": "fastapi",
            "target_canonical": "python",
            "source_name": "FastAPI",
            "target_name": "Python",
            "rel_type": "WRITTEN_IN",
            "confidence": 0.99,
            "evidence_sentence": "FastAPI is written in Python.",
        }
    ]
    communities = [
        {"id": "c1", "community_id": 1, "title": "Python Ecosystem", "summary": "Web frameworks", "entity_ids": ["n1", "n2"]}
    ]

    n_count = sqlite_mgr.upsert_graph_nodes(nodes)
    e_count = sqlite_mgr.upsert_graph_edges(edges)
    c_count = sqlite_mgr.upsert_graph_communities(communities)

    assert n_count == 2
    assert e_count == 1
    assert c_count == 1

    fetched_nodes = sqlite_mgr.get_all_graph_nodes()
    fetched_edges = sqlite_mgr.get_all_graph_edges()

    assert len(fetched_nodes) == 2
    assert fetched_nodes[0]["name"] in ["Python", "FastAPI"]
    assert len(fetched_edges) == 1
    assert fetched_edges[0]["rel_type"] == "WRITTEN_IN"


def test_graph_builder_fallback_when_neo4j_disconnected(sqlite_mgr):
    """Verify GraphBuilder persists to SQLite fallback when Neo4j is offline."""
    mock_neo4j = MagicMock()
    mock_neo4j.is_connected = False

    builder = GraphBuilder(mock_neo4j, sqlite=sqlite_mgr)

    doc = Document(id="d1", filename="test.pdf", file_type=SupportedFileType.PDF, sha256="hash123")
    e1 = Entity.from_extracted("PostgreSQL", "DATABASE", "c1", "d1")
    e2 = Entity.from_extracted("SQLAlchemy", "LIBRARY", "c1", "d1")
    r1 = Relationship.from_extracted("SQLAlchemy", "PostgreSQL", "CONNECTS_TO", "c1", "d1")

    res = ExtractionResult(chunk_id="c1", document_id="d1", entities=[e1, e2], relationships=[r1])

    n_count, r_count = builder.build_from_extractions([res], doc)

    assert n_count == 2
    assert r_count == 1

    saved_nodes = sqlite_mgr.get_all_graph_nodes()
    assert len(saved_nodes) == 2


def test_graph_traversal_fallback(sqlite_mgr):
    """Verify GraphTraversal queries SQLite fallback when Neo4j is offline."""
    mock_neo4j = MagicMock()
    mock_neo4j.is_connected = False

    nodes = [
        {"id": "n1", "name": "PyTorch", "canonical_name": "pytorch", "entity_type": "LIBRARY"},
        {"id": "n2", "name": "TensorFlow", "canonical_name": "tensorflow", "entity_type": "LIBRARY"},
    ]
    sqlite_mgr.upsert_graph_nodes(nodes)

    traversal = GraphTraversal(mock_neo4j, sqlite=sqlite_mgr)

    results = traversal.get_neighborhood("PyTorch")
    assert len(results) == 1
    assert results[0]["name"] == "PyTorch"

    search_res = traversal.search_entities("Tensor")
    assert len(search_res) == 1
    assert search_res[0]["name"] == "TensorFlow"
