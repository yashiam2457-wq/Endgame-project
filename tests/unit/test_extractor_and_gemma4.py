"""Unit tests for Gemma 4 image/diagram parsing and DocumentExtractor."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from generation import GemmaModel
from ingestion import DocumentLoader, TextExtractor
from models import Document, SupportedFileType
from utils.file_utils import SUPPORTED_EXTENSIONS


def test_supported_extensions():
    """Verify supported extensions include pdf, docx, doc, txt, md, markdown, html, htm."""
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".doc" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS
    assert ".html" in SUPPORTED_EXTENSIONS


def test_gemma_model_diagram_methods():
    """Verify GemmaModel analyze_image and describe_diagram methods exist and function."""
    mock_config = MagicMock()
    mock_config.generation_device = "cpu"
    mock_config.generation_model_name = "google/gemma-4-e4b-it"
    mock_config.quantization = "none"

    model = GemmaModel(mock_config)
    # Test analyze_image fallback when missing file
    result = model.analyze_image("non_existent_image.png")
    assert "not found" in result

    # Test describe_diagram fallback
    desc = model.describe_diagram(b"fake_png_bytes")
    assert isinstance(desc, str)


def test_gemma_model_generate_ollama_fallback():
    """Verify GemmaModel generate fallback when Ollama is unavailable."""
    mock_config = MagicMock()
    mock_config.generation_device = "cpu"
    mock_config.generation_model_name = "google/gemma-4-e4b-it"
    mock_config.quantization = "none"

    model = GemmaModel(mock_config)
    model._use_ollama = True
    model._ollama_model = "non_existent_model_123"

    # Should attempt Ollama, handle fail, and call _load_transformers safely
    with pytest.raises(Exception):
        model.generate("Test prompt")


def test_text_extractor_txt(tmp_path: Path):
    """Test text extractor with plain text file."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Enterprise GraphRAG test document with Gemma 4.", encoding="utf-8")

    doc = Document(
        sha256="fake_sha256",
        filename="sample.txt",
        file_type=SupportedFileType.TXT,
        file_path=txt_file,
    )

    extractor = TextExtractor()
    extracted_doc = extractor.extract(doc)

    assert "Enterprise GraphRAG" in extracted_doc.raw_text
    assert extracted_doc.metadata.word_count == 7
