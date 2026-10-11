"""
Unit tests for Intelligent Relationship Labeling and UNKNOWN Elimination.
"""

from __future__ import annotations

import pytest
from models import EntityType, Relationship, RelationshipType


def test_space_and_hyphen_normalization():
    """Verify string normalization converts spaces and hyphens into standard UPPERCASE_WITH_UNDERSCORES."""
    assert RelationshipType.from_str("depends on") == RelationshipType.DEPENDS_ON
    assert RelationshipType.from_str("created-by") == RelationshipType.CREATED_BY
    assert RelationshipType.from_str("part of") == RelationshipType.PART_OF
    assert RelationshipType.from_str("located in") == RelationshipType.LOCATED_IN
    assert RelationshipType.from_str("works with") == RelationshipType.WORKS_WITH
    assert RelationshipType.from_str("compatible with") == RelationshipType.COMPATIBLE_WITH
    assert RelationshipType.from_str("trained on") == RelationshipType.TRAINED_ON
    assert RelationshipType.from_str("connects to") == RelationshipType.CONNECTS_TO
    assert RelationshipType.from_str("has property") == RelationshipType.HAS_PROPERTY


def test_synonym_and_alias_dictionary():
    """Verify common LLM relationship variations are mapped to standard relationship types."""
    assert RelationshipType.from_str("built_with") == RelationshipType.USES
    assert RelationshipType.from_str("written_in") == RelationshipType.USES
    assert RelationshipType.from_str("extends") == RelationshipType.IMPLEMENTS
    assert RelationshipType.from_str("is_a") == RelationshipType.IMPLEMENTS
    assert RelationshipType.from_str("requires") == RelationshipType.DEPENDS_ON
    assert RelationshipType.from_str("authored_by") == RelationshipType.CREATED_BY
    assert RelationshipType.from_str("contained_in") == RelationshipType.PART_OF
    assert RelationshipType.from_str("cites") == RelationshipType.MENTIONS
    assert RelationshipType.from_str("results_in") == RelationshipType.CAUSES
    assert RelationshipType.from_str("integrates_with") == RelationshipType.WORKS_WITH
    assert RelationshipType.from_str("enables") == RelationshipType.SUPPORTS


def test_entity_type_heuristic_fallbacks():
    """Verify heuristic fallbacks infer relationship type from source and target EntityTypes."""
    # Target = LOCATION -> LOCATED_IN
    res1 = RelationshipType.from_str("custom_link", target_type=EntityType.LOCATION)
    assert res1 == RelationshipType.LOCATED_IN

    # Target = LIBRARY -> DEPENDS_ON
    res2 = RelationshipType.from_str("unknown_rel", target_type=EntityType.LIBRARY)
    assert res2 == RelationshipType.DEPENDS_ON

    # Target = PERSON -> CREATED_BY
    res3 = RelationshipType.from_str("arbitrary_edge", target_type=EntityType.PERSON)
    assert res3 == RelationshipType.CREATED_BY

    # Target = DATASET -> TRAINED_ON
    res4 = RelationshipType.from_str("data_link", target_type=EntityType.DATASET)
    assert res4 == RelationshipType.TRAINED_ON

    # Default fallback -> RELATED_TO (never UNKNOWN)
    res5 = RelationshipType.from_str("xyz_unrecognized")
    assert res5 == RelationshipType.RELATED_TO


def test_relationship_from_extracted_with_types():
    """Verify Relationship.from_extracted resolves raw rel strings using entity types."""
    rel = Relationship.from_extracted(
        source_name="MyClass",
        target_name="BaseClass",
        rel_type="inherits",
        chunk_id="c1",
        document_id="d1",
        source_type=EntityType.CONCEPT,
        target_type=EntityType.CONCEPT,
    )
    assert rel.relationship_type == RelationshipType.IMPLEMENTS
