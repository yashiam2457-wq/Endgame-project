"""
Enterprise GraphRAG — OCR Pipeline Orchestrator.

Pipeline Manager: Batch Execution, Multithreaded Processing, Disk Caching, Checkpointing,
and GraphRAG Ingestion Interface Adapter.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core import app_logger, performance_logger
from ocr_engine import (
    DocumentReconstructor,
    DocumentType,
    ImageMetadataExtractor,
    ImagePreprocessor,
    OCREngineManager,
    OCRDocument,
    OCRPage,
    PDFClassifier,
    PDFToImageConverter,
    QualityValidator,
)


class OCRPipelineConfig:
    """Configuration settings for the OCR processing pipeline."""

    def __init__(
        self,
        dpi: int = 300,
        deskew: bool = True,
        binarize: bool = True,
        remove_noise: bool = True,
        enhance_contrast: bool = True,
        max_workers: int = 4,
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
        confidence_threshold: float = 0.70,
    ) -> None:
        self.dpi = dpi
        self.deskew = deskew
        self.binarize = binarize
        self.remove_noise = remove_noise
        self.enhance_contrast = enhance_contrast
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        self.confidence_threshold = confidence_threshold
        self.cache_dir = cache_dir or (Path(__file__).parent / "cache" / "ocr")
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class OCRProcessingPipeline:
    """Complete 100% offline OCR preprocessing pipeline for GraphRAG ingestion."""

    def __init__(self, config: Optional[OCRPipelineConfig] = None) -> None:
        self.config = config or OCRPipelineConfig()
        self.engine_manager = OCREngineManager()

    def process_pdf(
        self, pdf_path: Union[str, Path], force_ocr: bool = False
    ) -> OCRDocument:
        """Processes a PDF file and returns a structured OCRDocument."""
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        start_time = time.perf_counter()
        app_logger.info(f"Starting OCR Pipeline for: {path.name}")

        # Check cache
        cache_key = self._get_cache_key(path, force_ocr)
        cached_doc = self._load_from_cache(cache_key) if self.config.enable_cache else None
        if cached_doc:
            app_logger.info(f"Loaded OCR pipeline result from cache for: {path.name}")
            return cached_doc

        # Step 1: PDF Classification
        doc_type, page_analysis = PDFClassifier.analyze_pdf(path)
        total_pages = len(page_analysis)
        app_logger.info(
            f"PDF Classification for {path.name}: Type={doc_type.value.upper()}, Total Pages={total_pages}"
        )

        pages: List[OCRPage] = []
        all_tables = []
        all_images = []

        # Step 2-7: Process pages concurrently
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, max(1, total_pages))) as executor:
            future_to_page = {
                executor.submit(self._process_single_page, path, p_num, p_type, force_ocr): p_num
                for p_num, (p_type, _, _) in page_analysis.items()
            }

            for future in as_completed(future_to_page):
                p_num = future_to_page[future]
                try:
                    page_obj, page_tables, page_imgs = future.result()
                    pages.append(page_obj)
                    all_tables.extend(page_tables)
                    all_images.extend(page_imgs)
                except Exception as exc:
                    app_logger.error(f"Error processing page {p_num} of {path.name}: {exc}")
                    # Construct fallback empty page
                    empty_p = OCRPage(
                        page_number=p_num,
                        width=595.0,
                        height=842.0,
                        dpi=self.config.dpi,
                        rotation=0,
                        document_type=DocumentType.SCANNED,
                        confidence=0.0,
                    )
                    pages.append(empty_p)

        pages.sort(key=lambda p: p.page_number)

        # Step 8 & 9: Document Reconstruction & Normalization
        reconstructed_md, reconstructed_txt = DocumentReconstructor.reconstruct(pages)

        # Step 10: Quality Validation
        elapsed = time.perf_counter() - start_time
        stats = QualityValidator.validate_document(pages)
        stats.processing_time_seconds = elapsed

        ocr_doc = OCRDocument(
            file_path=str(path),
            filename=path.name,
            doc_type=doc_type,
            total_pages=total_pages,
            pages=pages,
            reconstructed_markdown=reconstructed_md,
            reconstructed_text=reconstructed_txt,
            extracted_tables=all_tables,
            extracted_images=all_images,
            stats=stats,
            processing_log=[
                f"Processed {total_pages} pages in {elapsed:.2f}s",
                f"Document Classification: {doc_type.value}",
                f"Average Confidence: {stats.average_confidence:.2%}",
            ],
        )

        # Save to cache
        if self.config.enable_cache:
            self._save_to_cache(cache_key, ocr_doc)

        performance_logger.info(
            f"OCR Pipeline complete for {path.name}: {total_pages} pages in {elapsed:.2f}s | Avg Conf={stats.average_confidence:.2%}"
        )
        return ocr_doc

    def _process_single_page(
        self, path: Path, page_num: int, page_type: DocumentType, force_ocr: bool
    ) -> Tuple[OCRPage, List[Any], List[Any]]:
        start_t = time.perf_counter()

        # Extract embedded images
        page_imgs = ImageMetadataExtractor.extract_images_from_pdf(path, page_num)

        # Digital text layer processing
        if page_type == DocumentType.DIGITAL and not force_ocr:
            raw_text, conf, blocks, tables = self.engine_manager.extract_digital_page(path, page_num)
            page_obj = OCRPage(
                page_number=page_num,
                width=595.0,
                height=842.0,
                dpi=self.config.dpi,
                rotation=0,
                document_type=DocumentType.DIGITAL,
                raw_text=raw_text,
                confidence=conf,
                layout_blocks=blocks,
                tables=tables,
                images=page_imgs,
            )
            return page_obj, tables, page_imgs

        # Scanned page processing (PDF -> Image -> Preprocess -> OCR)
        img_bytes, w, h, rot = PDFToImageConverter.render_page(path, page_num, dpi=self.config.dpi)

        # OpenCV Preprocessing
        processed_bytes = ImagePreprocessor.preprocess_image(
            img_bytes,
            deskew=self.config.deskew,
            binarize=self.config.binarize,
            remove_noise=self.config.remove_noise,
            enhance_contrast=self.config.enhance_contrast,
        )

        # OCR Execution
        raw_text, conf, blocks, tables = self.engine_manager.extract_page_ocr(
            processed_bytes, page_num, float(w), float(h)
        )

        page_obj = OCRPage(
            page_number=page_num,
            width=float(w),
            height=float(h),
            dpi=self.config.dpi,
            rotation=rot,
            document_type=DocumentType.SCANNED,
            raw_text=raw_text,
            confidence=conf,
            layout_blocks=blocks,
            tables=tables,
            images=page_imgs,
            image_bytes=processed_bytes,
        )

        return page_obj, tables, page_imgs

    def _get_cache_key(self, path: Path, force_ocr: bool) -> str:
        stat = path.stat()
        key_str = f"{path.name}_{stat.st_size}_{stat.st_mtime}_{self.config.dpi}_{force_ocr}"
        import hashlib
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]

    def _load_from_cache(self, cache_key: str) -> Optional[OCRDocument]:
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        try:
            with cache_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                text = data.get("reconstructed_markdown", "") or data.get("reconstructed_text", "")
                if not text or len(text.strip().split()) < 5:
                    return None
                # Quick rebuild of cached output
                return OCRDocument(
                    file_path=data.get("file_path", ""),
                    filename=data.get("filename", ""),
                    doc_type=DocumentType(data.get("doc_type", "scanned")),
                    total_pages=data.get("total_pages", 0),
                    reconstructed_markdown=data.get("reconstructed_markdown", ""),
                    reconstructed_text=data.get("reconstructed_text", ""),
                )
        except Exception:
            return None

    def _save_to_cache(self, cache_key: str, doc: OCRDocument) -> None:
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        try:
            payload = {
                "file_path": doc.file_path,
                "filename": doc.filename,
                "doc_type": doc.doc_type.value,
                "total_pages": doc.total_pages,
                "reconstructed_markdown": doc.reconstructed_markdown,
                "reconstructed_text": doc.reconstructed_text,
            }
            with cache_file.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            app_logger.debug(f"Failed writing OCR cache: {exc}")


def extract_pdf_with_ocr_pipeline(pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
    """Helper function to run OCR pipeline and return (text, metadata_dict)."""
    pipeline = OCRProcessingPipeline()
    ocr_doc = pipeline.process_pdf(pdf_path)
    payload = ocr_doc.to_graphrag_payload()
    return ocr_doc.reconstructed_markdown or ocr_doc.reconstructed_text, payload
