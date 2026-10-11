"""
Enterprise GraphRAG — Generation Package.

Consolidated Generation Subsystem: GemmaModel, AnswerGenerator, EntityExtractor, MultiLevelSummarizer.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

try:
    import torch
except BaseException:
    torch = None

from core import (
    ModelConfig,
    ModelNotLoadedError,
    generation_logger,
)
from utils import clear_gpu_cache, detect_device
from models import (
    Citation,
    Chunk,
    Community,
    CommunitySummary,
    Entity,
    EntitySummary,
    ExtractionResult,
    KeywordSummary,
    QueryIntent,
    QueryResponse,
    Relationship,
    RelationshipSummary,
    SearchResult,
    SummaryPackage,
    TopicSummary,
)
from prompts import build_answer_prompt, build_entity_extraction_prompt
from utils import canonicalize_entity_name, parse_json_safe, repair_json, validate_entity_json

# ── SECTION 1: Gemma Model ───────────────────────────────────────────────────


class GemmaModel:
    """Manages the Gemma 4 1B Instruct model lifecycle."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._device = detect_device(config.generation_device)
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._use_ollama = False
        self._ollama_model = ""

    def load(self) -> None:
        if self._loaded:
            return

        try:
            import httpx

            resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if resp.status_code == 200:
                models_list = resp.json().get("models", [])
                # Build a mapping from lowercase name -> original name to preserve exact API names
                name_map: dict[str, str] = {
                    m.get("name", "").lower(): m.get("name", "")
                    for m in models_list
                }
                generation_logger.info(f"Ollama local models found: {list(name_map.values())}")

                # Priority: gemma4 > gemma3 > any gemma > first available model
                ollama_match: str | None = None
                for priority_prefix in ("gemma4", "gemma3", "gemma"):
                    for lower_name, orig_name in name_map.items():
                        if lower_name.startswith(priority_prefix):
                            ollama_match = orig_name
                            break
                    if ollama_match:
                        break

                # If no gemma found, fall back to first available model
                if not ollama_match and models_list:
                    ollama_match = models_list[0].get("name", "")

                if ollama_match:
                    self._use_ollama = True
                    self._ollama_model = ollama_match
                    self._loaded = True
                    generation_logger.info(
                        f"Connected to Ollama — using model: '{ollama_match}'"
                    )
                    return
        except Exception as exc:
            generation_logger.debug(f"Ollama local check bypassed: {exc}")

        self._use_ollama = False
        self._load_transformers()

    def _load_transformers(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        generation_logger.info(
            f"Loading Gemma model via Transformers: {self._config.generation_model_name} "
            f"on {self._device} with {self._config.quantization} precision"
        )

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            hf_token = self._config.hf_token or None
            model_path_or_id = self._config.generation_model_name
            is_local = Path(model_path_or_id).exists()

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_path_or_id,
                    cache_dir=str(self._config.model_cache_dir),
                    token=hf_token,
                    local_files_only=True if not is_local else False,
                )
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_path_or_id,
                    cache_dir=str(self._config.model_cache_dir),
                    token=hf_token,
                )

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            quant_config = self._build_quantization_config()
            load_kwargs = self._build_load_kwargs(quant_config)

            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path_or_id,
                    cache_dir=str(self._config.model_cache_dir),
                    token=hf_token,
                    local_files_only=True if not is_local else False,
                    **load_kwargs,
                )
            except Exception:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path_or_id,
                    cache_dir=str(self._config.model_cache_dir),
                    token=hf_token,
                    **load_kwargs,
                )

            if quant_config is None and str(self._device) == "cuda":
                self._model = self._model.to(self._device)

            self._model.eval()
            self._loaded = True

            generation_logger.info("Gemma model ready via Transformers.")

        except Exception as exc:
            generation_logger.error(f"Failed to load Gemma model via Transformers: {exc}")
            raise ModelNotLoadedError(
                f"Cannot load {self._config.generation_model_name}: {exc}.",
                cause=exc,
            ) from exc

    def _build_quantization_config(self) -> Optional[object]:
        if self._config.quantization == "int8" and str(self._device) == "cuda":
            from transformers import BitsAndBytesConfig

            generation_logger.info("Using INT8 quantization via bitsandbytes.")
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _build_load_kwargs(self, quant_config: Optional[object]) -> dict:
        kwargs: dict = {"low_cpu_mem_usage": True}
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
            kwargs["device_map"] = "auto"
        elif self._config.quantization == "fp16" and str(self._device) == "cuda":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32
        return kwargs

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> str:
        self._ensure_loaded()

        max_tokens = max_new_tokens or self._config.max_new_tokens
        temp = temperature or self._config.temperature
        tp = top_p or self._config.top_p

        if getattr(self, "_use_ollama", False):
            try:
                import httpx

                payload = {
                    "model": self._ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": temp,
                        "top_p": tp,
                        "num_predict": max_tokens,
                    },
                }
                resp = httpx.post("http://localhost:11434/api/generate", json=payload, timeout=120.0)
                if resp.status_code == 200:
                    return resp.json().get("response", "").strip()
                else:
                    generation_logger.warning(f"Ollama returned status {resp.status_code}: {resp.text}")
            except Exception as exc:
                generation_logger.warning(f"Ollama generation failed: {exc}. Falling back to Transformers.")

        if self._tokenizer is None or self._model is None:
            try:
                self._load_transformers()
            except Exception as exc:
                generation_logger.error(f"Transformers fallback load failed: {exc}")
                raise ModelNotLoadedError(f"Model generation unavailable: {exc}") from exc

        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=False,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp if do_sample else None,
                    top_p=tp if do_sample else None,
                    do_sample=do_sample,
                    repetition_penalty=self._config.repetition_penalty,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        except torch.cuda.OutOfMemoryError:
            clear_gpu_cache()
            generation_logger.warning("GPU OOM — retrying on CPU.")
            return self._generate_cpu_fallback(prompt, max_tokens)
        except Exception as exc:
            raise ModelNotLoadedError(
                f"Generation failed: {exc}", cause=exc
            ) from exc

    def _generate_cpu_fallback(self, prompt: str, max_new_tokens: int) -> str:
        self._model = self._model.to("cpu")
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        return self._tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    def analyze_image(
        self,
        image_input: bytes | str | Path,
        prompt: str = "Analyze and describe the contents of this image in detail.",
    ) -> str:
        self._ensure_loaded()

        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                return f"[Image file not found: {img_path}]"
            raw_bytes = img_path.read_bytes()
        else:
            raw_bytes = image_input

        b64_image = base64.b64encode(raw_bytes).decode("utf-8")

        if getattr(self, "_use_ollama", False):
            try:
                import httpx

                payload = {
                    "model": self._ollama_model,
                    "prompt": prompt,
                    "images": [b64_image],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 512,
                    },
                }
                resp = httpx.post("http://localhost:11434/api/generate", json=payload, timeout=120.0)
                if resp.status_code == 200:
                    result_text = resp.json().get("response", "").strip()
                    if result_text:
                        return result_text
            except Exception as exc:
                generation_logger.debug(f"Ollama image analysis bypassed: {exc}")

        try:
            from PIL import Image

            img = Image.open(io.BytesIO(raw_bytes))
            ocr_text = ""
            try:
                import pytesseract
                ocr_text = pytesseract.image_to_string(img).strip()
            except Exception:
                pass

            if ocr_text:
                return f"[Extracted Diagram Text]\n{ocr_text}"
        except Exception as exc:
            generation_logger.debug(f"Pillow/OCR processing exception: {exc}")

        return "[Embedded Diagram / Figure Block]"

    def describe_diagram(self, image_input: bytes | str | Path) -> str:
        diagram_prompt = (
            "Thoroughly analyze and describe this diagram, figure, chart, or scanned page in detail. "
            "Extract all visible text, headers, labels, numbers, flowchart nodes, entity relationships, "
            "arrows, process steps, tables, and structural connections into a clean, structured Markdown text format."
        )
        return self.analyze_image(image_input, prompt=diagram_prompt)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._config.generation_model_name


# ── SECTION 2: Answer Generator ──────────────────────────────────────────────

_CONFIDENCE_PATTERN = re.compile(
    r"##\s*Confidence\s*\n(.+)", re.IGNORECASE
)
_CONFIDENCE_LEVELS = {"high": 0.9, "medium": 0.6, "low": 0.3}


class AnswerGenerator:
    """Generates grounded answers with citations using Gemma 4 1B."""

    def __init__(
        self,
        llm: GemmaModel,
        cache: Optional[Any] = None,
        fallback_message: str = (
            "I don't have sufficient information in the provided documents "
            "to answer this question."
        ),
    ) -> None:
        self._llm = llm
        self._cache = cache
        self._fallback = fallback_message

    def generate(
        self,
        query: str,
        search_results: list[SearchResult],
        intent: QueryIntent = QueryIntent.UNKNOWN,
        entities: Optional[list[str]] = None,
        query_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> QueryResponse:
        qid = query_id or str(uuid4())

        if not search_results:
            return self._empty_response(query, qid, intent, session_id)

        cache_key = self._make_cache_key(query, search_results)
        if self._cache is not None:
            cached = self._cache.get("llm", f"answer:{cache_key}")
            if cached is not None:
                generation_logger.debug(f"Cache hit for query: {query[:50]}")
                return QueryResponse.model_validate(cached)

        context_chunks = [
            {
                "text": r.text,
                "document_filename": r.document_filename,
                "document_id": r.document_id,
                "chunk_id": r.chunk_id,
                "sequence_index": r.metadata.get("sequence_index", 0),
                "score": r.score,
            }
            for r in search_results
        ]

        prompt = build_answer_prompt(
            query=query,
            context_chunks=context_chunks,
            entities=entities or [],
        )

        start_gen = time.perf_counter()
        try:
            raw_answer = self._llm.generate(prompt, do_sample=True)
            gen_ms = (time.perf_counter() - start_gen) * 1000

            answer_text = self._extract_answer_section(raw_answer)
            reasoning = self._extract_section(raw_answer, "Reasoning")
            confidence = self._extract_confidence(raw_answer)
        except Exception as exc:
            gen_ms = (time.perf_counter() - start_gen) * 1000
            generation_logger.warning(
                f"LLM generation model fallback triggered ({exc}). Synthesizing from document context."
            )
            context_snippets = [
                f"• [{r.document_filename}]: {r.text[:300]}..."
                for r in search_results[:3]
            ]
            answer_text = (
                "Based on the retrieved document context:\n\n"
                + "\n\n".join(context_snippets)
            )
            reasoning = f"Direct context synthesis from top {len(context_snippets)} relevant document chunks."
            confidence = 0.75

        citations = self._build_citations(search_results)

        response = QueryResponse(
            query_id=qid,
            query=query,
            intent=intent,
            answer=answer_text or self._fallback,
            reasoning=reasoning,
            confidence_score=confidence,
            citations=citations,
            search_results=search_results,
            entities_detected=entities or [],
            generation_time_ms=gen_ms,
            model_name=self._llm.model_name,
            session_id=session_id,
        )

        if self._cache is not None:
            self._cache.set(
                "llm",
                f"answer:{cache_key}",
                response.model_dump(mode="json"),
            )

        generation_logger.info(
            f"Generated answer in {gen_ms:.0f} ms | "
            f"confidence={confidence:.2f} | citations={len(citations)}"
        )
        return response

    def _extract_answer_section(self, text: str) -> str:
        section = self._extract_section(text, "Answer")
        if section:
            return section
        return text.strip()

    def _extract_section(self, text: str, heading: str) -> Optional[str]:
        pattern = re.compile(
            rf"##\s*{re.escape(heading)}\s*\n(.*?)(?=##|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(text)
        return match.group(1).strip() if match else None

    def _extract_confidence(self, text: str) -> float:
        match = _CONFIDENCE_PATTERN.search(text)
        if not match:
            return 0.5
        level_text = match.group(1).lower()
        for level, score in _CONFIDENCE_LEVELS.items():
            if level in level_text:
                return score
        return 0.5

    def _build_citations(self, results: list[SearchResult]) -> list[Citation]:
        citations: list[Citation] = []
        for i, result in enumerate(results, 1):
            citations.append(
                Citation(
                    citation_id=f"[{result.document_filename}-Chunk{i}]",
                    document_id=result.document_id,
                    document_filename=result.document_filename,
                    chunk_id=result.chunk_id,
                    chunk_text_snippet=result.text[:200] + "..."
                    if len(result.text) > 200
                    else result.text,
                    page_number=result.page_number,
                    relevance_score=result.score,
                    entity_ids=result.entities,
                )
            )
        return citations

    def _empty_response(
        self,
        query: str,
        query_id: str,
        intent: QueryIntent,
        session_id: Optional[str],
    ) -> QueryResponse:
        guidance = (
            "No document content is currently available in the system repository.\n\n"
            "To answer queries or generate document summaries, please upload a document "
            "(PDF, TXT, DOCX, MD) using the Document Ingestion panel on the left, then click 'Sync & Index All Pending'."
        )
        return QueryResponse(
            query_id=query_id,
            query=query,
            intent=intent,
            answer=guidance,
            confidence_score=0.0,
            citations=[],
            search_results=[],
            session_id=session_id,
            model_name=self._llm.model_name if self._llm.is_loaded else "not_loaded",
        )

    @staticmethod
    def _make_cache_key(query: str, results: list[SearchResult]) -> str:
        doc_ids = ":".join(sorted(set(r.document_id for r in results if r.document_id)))
        top_ids = ":".join(r.chunk_id for r in results[:3])
        return hashlib.sha256(f"{doc_ids}:{query}:{top_ids}".encode()).hexdigest()[:32]


# ── SECTION 3: Entity Extractor ───────────────────────────────────────────────


class EntityExtractor:
    """Extracts entities and relationships from document chunks using Gemma."""

    def __init__(
        self,
        llm: GemmaModel,
        cache: Optional[Any] = None,
        max_retries: int = 3,
    ) -> None:
        self._llm = llm
        self._cache = cache
        self._max_retries = max_retries

    def extract(self, chunk: Chunk, document_id: str) -> ExtractionResult:
        if self._cache is not None:
            cached = self._cache.get("llm", f"entity:{chunk.id}")
            if cached is not None:
                return self._deserialize_result(cached, chunk, document_id)

        start_time = time.perf_counter()
        prompt = build_entity_extraction_prompt(chunk.text)
        raw_json = self._generate_with_retry(prompt, chunk.id)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        entities, relationships = self._parse_extraction(raw_json, chunk, document_id)

        result = ExtractionResult(
            chunk_id=chunk.id,
            document_id=document_id,
            entities=entities,
            relationships=relationships,
            extraction_time_ms=elapsed_ms,
            model_name=self._llm.model_name,
        )

        if self._cache is not None:
            self._cache.set("llm", f"entity:{chunk.id}", self._serialize_result(result))

        generation_logger.debug(
            f"Extracted {len(entities)} entities, {len(relationships)} relationships "
            f"from chunk {chunk.id[:8]} in {elapsed_ms:.0f} ms"
        )
        return result

    def _generate_with_retry(self, prompt: str, chunk_id: str) -> str:
        for attempt in range(1, self._max_retries + 1):
            try:
                # Use JSON-mode Ollama call for entity extraction to force valid JSON output
                if getattr(self._llm, "_use_ollama", False):
                    import httpx
                    payload = {
                        "model": self._llm._ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0.05,
                            "top_p": 0.9,
                            "num_predict": 4096,
                        },
                    }
                    resp = httpx.post("http://localhost:11434/api/generate", json=payload, timeout=120.0)
                    if resp.status_code == 200:
                        output = resp.json().get("response", "").strip()
                    else:
                        output = self._llm.generate(prompt, do_sample=False)
                else:
                    output = self._llm.generate(prompt, do_sample=False)

                data = parse_json_safe(output)
                if data is not None and validate_entity_json(data):
                    return output
                repaired = repair_json(output)
                if repaired:
                    data = parse_json_safe(repaired)
                    if data and validate_entity_json(data):
                        return repaired
                generation_logger.warning(
                    f"Attempt {attempt}/{self._max_retries}: "
                    f"Invalid JSON from chunk {chunk_id[:8]}: {output[:120]!r}"
                )
                time.sleep(0.5)
            except Exception as exc:
                generation_logger.warning(
                    f"Generation error on attempt {attempt}: {exc}"
                )
                time.sleep(1.0)

        generation_logger.error(
            f"Entity extraction failed after {self._max_retries} retries "
            f"for chunk {chunk_id[:8]}. Returning empty result."
        )
        return '{"entities": [], "relationships": []}'

    def _parse_extraction(
        self, raw: str, chunk: Chunk, document_id: str
    ) -> tuple[list[Entity], list[Relationship]]:
        data = parse_json_safe(raw)
        if data is None or not validate_entity_json(data):
            return [], []

        entities: list[Entity] = []
        entity_name_to_id: dict[str, str] = {}
        entity_type_map: dict[str, Any] = {}

        for item in data.get("entities", []):
            name = item.get("name", "").strip()
            if not name:
                continue
            entity = Entity.from_extracted(
                name=name,
                entity_type=item.get("type", "UNKNOWN"),
                chunk_id=chunk.id,
                document_id=document_id,
                description=item.get("description"),
            )
            entities.append(entity)
            canon = canonicalize_entity_name(name)
            entity_name_to_id[canon] = entity.id
            entity_type_map[canon] = entity.entity_type

        relationships: list[Relationship] = []
        for item in data.get("relationships", []):
            src = item.get("source", "").strip()
            tgt = item.get("target", "").strip()
            if not src or not tgt:
                continue

            raw_conf = item.get("confidence", 1.0)
            try:
                conf = float(raw_conf)
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                conf = 1.0

            evidence = item.get("evidence_sentence") or item.get("evidence")
            if isinstance(evidence, str):
                evidence = evidence.strip()
            else:
                evidence = None

            src_canon = canonicalize_entity_name(src)
            tgt_canon = canonicalize_entity_name(tgt)

            rel = Relationship.from_extracted(
                source_name=src,
                target_name=tgt,
                rel_type=item.get("type", "RELATED_TO"),
                chunk_id=chunk.id,
                document_id=document_id,
                source_id=entity_name_to_id.get(src_canon),
                target_id=entity_name_to_id.get(tgt_canon),
                description=item.get("description"),
                confidence=conf,
                evidence_sentence=evidence,
                page_number=getattr(chunk, "page_number", None),
                source_type=entity_type_map.get(src_canon),
                target_type=entity_type_map.get(tgt_canon),
            )
            relationships.append(rel)

        return entities, relationships

    def _serialize_result(self, result: ExtractionResult) -> dict:
        return result.model_dump(mode="json")

    def _deserialize_result(
        self, data: dict, chunk: Chunk, document_id: str
    ) -> ExtractionResult:
        return ExtractionResult.model_validate(data)


# ── SECTION 4: Multi-Level Summarizer ────────────────────────────────────────


class MultiLevelSummarizer:
    """Orchestrates generation of multi-level hierarchical summaries."""

    def __init__(self, llm: GemmaModel) -> None:
        self._llm = llm

    def generate_all_summaries(
        self,
        document_id: str,
        entities: list[Entity],
        relationships: list[Relationship],
        communities: list[Community],
        chunk_texts: Optional[list[str]] = None,
    ) -> SummaryPackage:
        entity_sums = self.summarize_entities(entities)
        rel_sums = self.summarize_relationships(relationships)
        comm_sums = self.summarize_communities(communities, entities)
        kw_sums = self.summarize_keywords(entities, communities)
        topic_sums = self.summarize_topics(communities, chunk_texts)

        return SummaryPackage(
            document_id=document_id,
            entity_summaries=entity_sums,
            relationship_summaries=rel_sums,
            community_summaries=comm_sums,
            keyword_summaries=kw_sums,
            topic_summaries=topic_sums,
        )

    def summarize_entities(self, entities: list[Entity]) -> list[EntitySummary]:
        summaries: list[EntitySummary] = []
        for e in entities[:20]:
            desc = e.description or f"Named entity of type {e.entity_type.value}"
            summary_text = f"{e.name} ({e.entity_type.value}): {desc}"
            summaries.append(
                EntitySummary(
                    entity_id=e.id,
                    entity_name=e.name,
                    entity_type=e.entity_type.value,
                    summary=summary_text,
                    keywords=[e.name.lower(), e.entity_type.value.lower()],
                )
            )
        return summaries

    def summarize_relationships(self, relationships: list[Relationship]) -> list[RelationshipSummary]:
        summaries: list[RelationshipSummary] = []
        for r in relationships[:30]:
            ev = f" Supported by sentence: '{r.evidence_sentence}'." if r.evidence_sentence else ""
            summary_text = (
                f"Relationship {r.source_entity_name} --[{r.relationship_type.value}]--> "
                f"{r.target_entity_name}.{ev}"
            )
            summaries.append(
                RelationshipSummary(
                    relationship_id=r.id,
                    source_name=r.source_entity_name,
                    target_name=r.target_entity_name,
                    rel_type=r.relationship_type.value,
                    summary=summary_text,
                    confidence=r.confidence,
                )
            )
        return summaries

    def summarize_communities(
        self, communities: list[Community], entities: list[Entity]
    ) -> list[CommunitySummary]:
        summaries: list[CommunitySummary] = []
        for comm in communities:
            findings = [f"Contains entities: {', '.join(comm.entity_names[:5])}"]
            if comm.topics:
                findings.append(f"Primary topics: {', '.join(comm.topics)}")

            summary_text = (
                f"Community '{comm.title}' (Level {comm.level}): {comm.summary} "
                f"Key entities include {', '.join(comm.entity_names[:4])}."
            )

            summaries.append(
                CommunitySummary(
                    community_id=comm.id,
                    title=comm.title,
                    level=comm.level,
                    summary=summary_text,
                    findings=findings,
                    rating=0.9,
                )
            )
        return summaries

    def summarize_keywords(
        self, entities: list[Entity], communities: list[Community]
    ) -> list[KeywordSummary]:
        kw_map: dict[str, list[str]] = {}
        for e in entities:
            kw = e.name.lower()
            kw_map.setdefault(kw, []).append(e.name)

        for c in communities:
            for kw in c.keywords:
                kw_map.setdefault(kw.lower(), []).append(c.title)

        summaries: list[KeywordSummary] = []
        for kw, assoc in list(kw_map.items())[:25]:
            summaries.append(
                KeywordSummary(
                    keyword=kw,
                    score=1.0,
                    summary=f"Key term '{kw}' referenced across {len(assoc)} entities/communities.",
                    associated_entities=list(set(assoc))[:5],
                )
            )
        return summaries

    def summarize_topics(
        self, communities: list[Community], chunk_texts: Optional[list[str]] = None
    ) -> list[TopicSummary]:
        topic_map: dict[str, list[str]] = {}
        for c in communities:
            for t in c.topics:
                topic_map.setdefault(t, []).append(c.id)

        if not topic_map:
            topic_map["General Overview"] = [c.id for c in communities]

        summaries: list[TopicSummary] = []
        for topic, comm_ids in topic_map.items():
            summaries.append(
                TopicSummary(
                    topic_name=topic,
                    summary=f"Thematic area covering {topic} across {len(comm_ids)} graph communities.",
                    keywords=[topic.lower()],
                    community_ids=comm_ids,
                )
            )
        return summaries


# ── SECTION 5: Backward Compatibility Submodule Aliases ─────────────────────

for _mod in ["entity_extractor", "generator", "model", "summarizer"]:
    sys.modules[f"generation.{_mod}"] = sys.modules[__name__]
