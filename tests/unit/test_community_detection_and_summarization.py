"""
Unit tests for Community Detection, Multi-Level Summarization, Multi-Vector Embeddings, and Expanded Label Taxonomy.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from database import _NODE_LABELS
from embedding import MultiVectorEmbedder
from generation import MultiLevelSummarizer
from graph import CommunityDetector
from models import Chunk, Community, Entity, EntityType, Relationship, RelationshipType


REQUIRED_23_LABELS = [
    "Document",
    "Chunk",
    "Entity",
    "Community",
    "Topic",
    "Keyword",
    "Section",
    "Page",
    "Author",
    "Organization",
    "Technology",
    "Concept",
    "Algorithm",
    "Dataset",
    "Model",
    "Library",
    "Framework",
    "Language",
    "Repository",
    "File",
    "Function",
    "Method",
    "API",
]


def test_all_23_node_labels_registered():
    """Verify all 23 requested node labels exist in _NODE_LABELS database schema."""
    for label in REQUIRED_23_LABELS:
        assert label in _NODE_LABELS, f"Label {label} missing from _NODE_LABELS schema"


def test_entity_type_enum_covers_labels():
    """Verify EntityType enum includes corresponding uppercase values."""
    for label in REQUIRED_23_LABELS:
        assert hasattr(EntityType, label.upper()), f"EntityType missing {label.upper()}"


def test_community_detection():
    """Verify CommunityDetector partitions graph entities into communities."""
    detector = CommunityDetector()

    e1 = Entity.from_extracted(name="FastAPI", entity_type="FRAMEWORK", chunk_id="c1", document_id="d1")
    e2 = Entity.from_extracted(name="Pydantic", entity_type="LIBRARY", chunk_id="c1", document_id="d1")
    e3 = Entity.from_extracted(name="Python", entity_type="LANGUAGE", chunk_id="c1", document_id="d1")

    r1 = Relationship.from_extracted(
        source_name="FastAPI",
        target_name="Pydantic",
        rel_type="USES",
        chunk_id="c1",
        document_id="d1",
        source_id=e1.id,
        target_id=e2.id,
    )
    r2 = Relationship.from_extracted(
        source_name="FastAPI",
        target_name="Python",
        rel_type="WRITTEN_IN",
        chunk_id="c1",
        document_id="d1",
        source_id=e1.id,
        target_id=e3.id,
    )

    communities = detector.detect_communities([e1, e2, e3], [r1, r2])

    assert len(communities) >= 1
    comm = communities[0]
    assert isinstance(comm, Community)
    assert len(comm.entity_ids) == 3
    assert "FastAPI" in comm.entity_names


def test_multi_level_summarization():
    """Verify MultiLevelSummarizer produces summaries across all 5 tiers."""
    mock_llm = MagicMock()
    summarizer = MultiLevelSummarizer(mock_llm)

    e1 = Entity.from_extracted(name="Gemma", entity_type="MODEL", chunk_id="c1", document_id="d1")
    r1 = Relationship.from_extracted(
        source_name="Gemma",
        target_name="Transformer",
        rel_type="IMPLEMENTS",
        chunk_id="c1",
        document_id="d1",
        evidence_sentence="Gemma implements the Transformer architecture.",
    )
    comm = Community(
        community_id=1,
        title="AI Models",
        level=0,
        summary="Cluster of AI models and architectures.",
        entity_names=["Gemma"],
        topics=["MODEL", "CONCEPT"],
        keywords=["gemma", "model"],
    )

    pkg = summarizer.generate_all_summaries("d1", [e1], [r1], [comm])

    assert len(pkg.entity_summaries) == 1
    assert pkg.entity_summaries[0].entity_name == "Gemma"

    assert len(pkg.relationship_summaries) == 1
    assert pkg.relationship_summaries[0].rel_type == "IMPLEMENTS"

    assert len(pkg.community_summaries) == 1
    assert pkg.community_summaries[0].title == "AI Models"

    assert len(pkg.keyword_summaries) >= 1
    assert len(pkg.topic_summaries) >= 1


def test_multi_vector_embedder():
    """Verify MultiVectorEmbedder generates embeddings for Chunks, Entities, Relationships, and Communities."""
    mock_encoder = MagicMock()
    mock_encoder.encode.side_effect = lambda texts: np.random.randn(len(texts), 384).astype(np.float32)

    mv_embedder = MultiVectorEmbedder(mock_encoder)

    chunk = Chunk(id="c1", document_id="d1", sequence_index=0, text="Test chunk content")
    entity = Entity.from_extracted(name="Neo4j", entity_type="DATABASE", chunk_id="c1", document_id="d1")
    rel = Relationship.from_extracted(
        source_name="System", target_name="Neo4j", rel_type="USES", chunk_id="c1", document_id="d1"
    )
    comm = Community(community_id=1, title="Databases", summary="Database community", topics=["DATABASE"])

    c_vecs = mv_embedder.embed_chunks([chunk])
    e_vecs = mv_embedder.embed_entities([entity])
    r_vecs = mv_embedder.embed_relationships([rel])
    comm_vecs = mv_embedder.embed_communities([comm])

    assert len(c_vecs) == 1 and len(c_vecs[0]) == 384
    assert len(e_vecs) == 1 and len(e_vecs[0]) == 384
    assert len(r_vecs) == 1 and len(r_vecs[0]) == 384
    assert len(comm_vecs) == 1 and len(comm_vecs[0]) == 384
