"""Unit test for text utilities."""
from __future__ import annotations

from utils.text_utils import clean_whitespace, canonicalize_entity_name


def test_clean_whitespace():
    assert clean_whitespace("  Hello   World  \n\n\n ") == "Hello World"


def test_canonicalize_entity():
    assert canonicalize_entity_name("  Google, Inc.! ") == "google inc"
