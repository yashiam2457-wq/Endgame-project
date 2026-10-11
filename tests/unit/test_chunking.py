"""Unit test for chunking strategies."""
from __future__ import annotations

from core import ChunkingConfig
from chunking import ChunkerFactory
from models import Document, SupportedFileType


def test_recursive_chunker():
    config = ChunkingConfig(strategy="recursive", chunk_size=100, chunk_overlap=10)
    chunker = ChunkerFactory.create(config)

    doc = Document(
        sha256="fake_sha256",
        filename="test.txt",
        file_type=SupportedFileType.TXT,
        cleaned_text="This is a test document. " * 20,
    )

    chunks = chunker.chunk(doc)
    assert len(chunks) > 0
    assert chunks[0].document_id == doc.id
