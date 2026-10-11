"""Unit tests for GraphRAG relationship types and metadata fields."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from database import _RELATIONSHIP_TYPES
from generation import EntityExtractor
from graph import GraphBuilder
from models import Chunk, Document, GraphEdge, Relationship, RelationshipType, SupportedFileType


EXPECTED_25_RELATIONSHIPS = [
    "USES",
    "IMPLEMENTS",
    "DEPENDS_ON",
    "IMPORTS",
    "CALLS",
    "CREATED_BY",
    "LOCATED_IN",
    "PART_OF",
    "BELONGS_TO",
    "RELATED_TO",
    "MENTIONS",
    "CAUSES",
    "LEADS_TO",
    "REFERENCES",
    "WORKS_WITH",
    "SUPPORTS",
    "COMPATIBLE_WITH",
    "TRAINED_ON",
    "GENERATES",
    "CONSUMES",
    "CONNECTS_TO",
    "HAS_PROPERTY",
    "HAS_VERSION",
    "OWNS",
    "PRODUCES",
]


def test_all_25_relationship_types_exist():
    """Verify all 25 requested relationship types are defined in RelationshipType enum."""
    for rel_name in EXPECTED_25_RELATIONSHIPS:
        enum_item = RelationshipType(rel_name)
        assert enum_item.value == rel_name


def test_relationship_schema_contains_all_25_types():
    """Verify database schema list includes all 25 relationship types."""
    for rel_name in EXPECTED_25_RELATIONSHIPS:
        assert rel_name in _RELATIONSHIP_TYPES


def test_relationship_model_required_fields():
    """Verify Relationship model stores confidence, evidence_sentence, document_id, chunk_id, and page_number."""
    rel = Relationship.from_extracted(
        source_name="Service A",
        target_name="Database B",
        rel_type="CONNECTS_TO",
        chunk_id="chunk-123",
        document_id="doc-456",
        description="Service A connects to Database B over TLS",
        confidence=0.98,
        evidence_sentence="Service A connects to Database B on port 5432 using TLS.",
        page_number=4,
    )

    assert rel.relationship_type == RelationshipType.CONNECTS_TO
    assert rel.confidence == 0.98
    assert rel.evidence_sentence == "Service A connects to Database B on port 5432 using TLS."
    assert rel.document_id == "doc-456"
    assert rel.chunk_id == "chunk-123"
    assert rel.page_number == 4


def test_graphedge_model_required_fields():
    """Verify GraphEdge model stores confidence, evidence_sentence, document_id, chunk_id, and page_number."""
    edge = GraphEdge(
        source_id="s1",
        target_id="t1",
        source_name="Model X",
        target_name="Dataset Y",
        relationship_type="TRAINED_ON",
        confidence=0.92,
        evidence_sentence="Model X was trained on Dataset Y for 10 epochs.",
        document_id="doc-789",
        chunk_id="chunk-321",
        page_number=12,
    )

    assert edge.relationship_type == "TRAINED_ON"
    assert edge.confidence == 0.92
    assert edge.evidence_sentence == "Model X was trained on Dataset Y for 10 epochs."
    assert edge.document_id == "doc-789"
    assert edge.chunk_id == "chunk-321"
    assert edge.page_number == 12


def test_parse_extraction_with_confidence_and_evidence():
    """Test EntityExtractor parses confidence and evidence_sentence from LLM JSON output."""
    mock_llm = MagicMock()
    extractor = EntityExtractor(mock_llm)

    chunk = Chunk(
        id="chunk-abc",
        document_id="doc-xyz",
        sequence_index=0,
        text="FastAPI uses Pydantic for data validation.",
        page_number=3,
    )

    raw_json = json.dumps({
        "entities": [
            {"name": "FastAPI", "type": "TECHNOLOGY"},
            {"name": "Pydantic", "type": "TECHNOLOGY"}
        ],
        "relationships": [
            {
                "source": "FastAPI",
                "target": "Pydantic",
                "type": "USES",
                "description": "FastAPI uses Pydantic for validation",
                "confidence": 0.96,
                "evidence_sentence": "FastAPI uses Pydantic for data validation."
            }
        ]
    })

    entities, relationships = extractor._parse_extraction(raw_json, chunk, "doc-xyz")

    assert len(entities) == 2
    assert len(relationships) == 1

    rel = relationships[0]
    assert rel.source_entity_name == "FastAPI"
    assert rel.target_entity_name == "Pydantic"
    assert rel.relationship_type == RelationshipType.USES
    assert rel.confidence == 0.96
    assert rel.evidence_sentence == "FastAPI uses Pydantic for data validation."
    assert rel.document_id == "doc-xyz"
    assert rel.chunk_id == "chunk-abc"
    assert rel.page_number == 3
