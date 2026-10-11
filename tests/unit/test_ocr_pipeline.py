"""
Unit test suite for the Enterprise Offline OCR Subsystem.
Tests PDF classification, OpenCV preprocessing, Layout detection, Table extraction,
Document reconstruction, and GraphRAG ingestion payload integration.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from ocr_engine import (
    DocumentReconstructor,
    DocumentType,
    ImagePreprocessor,
    LayoutElementType,
    OCRBoundingBox,
    OCRDocument,
    OCRLayoutBlock,
    OCRPage,
    OCRStats,
    OCRTable,
    PDFClassifier,
    QualityValidator,
    TableExtractor,
)
from ocr_pipeline import OCRPipelineConfig, OCRProcessingPipeline


def test_ocr_bounding_box_geometry():
    bbox = OCRBoundingBox(x0=10.0, y0=20.0, x1=110.0, y1=220.0)
    assert bbox.width == 100.0
    assert bbox.height == 200.0
    assert bbox.area == 20000.0
    assert bbox.to_dict() == {"x0": 10.0, "y0": 20.0, "x1": 110.0, "y1": 220.0}


def test_ocr_table_markdown_generation():
    tbl = OCRTable(
        table_id="tbl_1",
        page_number=1,
        bbox=OCRBoundingBox(0.0, 0.0, 100.0, 100.0),
        headers=["Entity", "Type", "Confidence"],
        rows=[
            ["GraphRAG", "System", "0.99"],
            ["Neo4j", "Database", "0.95"],
        ],
    )
    md = tbl.to_markdown()
    assert "| Entity | Type | Confidence |" in md
    assert "| GraphRAG | System | 0.99 |" in md
    assert "| Neo4j | Database | 0.95 |" in md


def test_document_reconstructor_and_hyphen_merge():
    block1 = OCRLayoutBlock(
        block_id="b1",
        element_type=LayoutElementType.HEADING,
        text="System Architecture",
        confidence=0.99,
        bbox=OCRBoundingBox(0, 0, 100, 20),
        page_number=1,
    )
    block2 = OCRLayoutBlock(
        block_id="b2",
        element_type=LayoutElementType.PARAGRAPH,
        text="The graph algorithm performs multi-\nlevel entity extraction.",
        confidence=0.95,
        bbox=OCRBoundingBox(0, 30, 100, 50),
        page_number=1,
    )
    page1 = OCRPage(
        page_number=1,
        width=595.0,
        height=842.0,
        dpi=300,
        rotation=0,
        document_type=DocumentType.DIGITAL,
        layout_blocks=[block1, block2],
    )

    full_md, full_txt = DocumentReconstructor.reconstruct([page1])
    assert "## System Architecture" in full_md
    assert "multilevel entity extraction" in full_md.lower()


def test_quality_validator_telemetry():
    p1 = OCRPage(
        page_number=1, width=100, height=100, dpi=300, rotation=0,
        document_type=DocumentType.DIGITAL, confidence=0.98
    )
    p2 = OCRPage(
        page_number=2, width=100, height=100, dpi=300, rotation=0,
        document_type=DocumentType.SCANNED, confidence=0.55
    )

    stats = QualityValidator.validate_document([p1, p2])
    assert stats.total_pages == 2
    assert stats.digital_pages == 1
    assert stats.scanned_pages == 1
    assert 2 in stats.low_confidence_pages
    assert stats.average_confidence > 0.70


def test_graphrag_payload_structure():
    block1 = OCRLayoutBlock(
        block_id="b1",
        element_type=LayoutElementType.TITLE,
        text="Enterprise GraphRAG Manual",
        confidence=0.99,
        bbox=OCRBoundingBox(0, 0, 200, 30),
        page_number=1,
    )
    page1 = OCRPage(
        page_number=1,
        width=595.0,
        height=842.0,
        dpi=300,
        rotation=0,
        document_type=DocumentType.DIGITAL,
        layout_blocks=[block1],
        reconstructed_text="Enterprise GraphRAG Manual",
    )
    doc = OCRDocument(
        file_path="sample.pdf",
        filename="sample.pdf",
        doc_type=DocumentType.DIGITAL,
        total_pages=1,
        pages=[page1],
        reconstructed_markdown="# Enterprise GraphRAG Manual",
        reconstructed_text="Enterprise GraphRAG Manual",
        stats=OCRStats(total_pages=1, average_confidence=0.99),
    )

    payload = doc.to_graphrag_payload()
    assert "document_text" in payload
    assert "page_text" in payload
    assert "section_text" in payload
    assert "tables" in payload
    assert "images" in payload
    assert "metadata" in payload
    assert "bounding_boxes" in payload
    assert payload["metadata"]["filename"] == "sample.pdf"
    assert payload["page_numbers"] == [1]


def test_ocr_pipeline_config():
    config = OCRPipelineConfig(dpi=400, deskew=True, binarize=True)
    assert config.dpi == 400
    assert config.deskew is True
    pipeline = OCRProcessingPipeline(config)
    assert pipeline.config.dpi == 400
