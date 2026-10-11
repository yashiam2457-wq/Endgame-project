"""
Enterprise GraphRAG — Domain Models Package.

Consolidated Pydantic Models & Schemas for API, Documents, Entities, Graph, Queries, Communities, Summaries.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

# ── SECTION 1: Generic API Wrappers & Request/Response Schemas ───────────────


class APIResponse(BaseModel):
    """Generic API success response wrapper."""

    success: bool = True
    message: str = "OK"
    data: Any = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Generic API error response."""

    success: bool = False
    error: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseModel):
    """Paginated list response."""

    items: list[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


class DocumentSummary(BaseModel):
    """Lightweight document info for list views."""

    id: str
    filename: str
    file_type: str
    status: str
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    chunk_count: int = 0
    entity_count: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentDetail(DocumentSummary):
    """Full document info including metadata."""

    sha256: str
    title: Optional[str] = None
    author: Optional[str] = None
    version: int = 1
    file_size_bytes: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document."""

    document_id: str
    deleted_chunks: int
    deleted_entities: int
    deleted_relationships: int
    success: bool = True


class ReindexRequest(BaseModel):
    """Request to reindex one or all documents."""

    document_id: Optional[str] = None


class UploadResponse(BaseModel):
    """Response after successful file upload."""

    document_id: str
    filename: str
    file_size_bytes: int
    sha256: str
    status: str = "pending"
    message: str = "Upload successful. Use /index to trigger indexing."


class IndexRequest(BaseModel):
    """Request to index a specific document or all pending documents."""

    document_id: Optional[str] = None
    force: bool = False


class IndexResponse(BaseModel):
    """Status response for an indexing job."""

    job_id: str
    status: str
    document_id: Optional[str] = None
    message: str = ""


class IndexProgress(BaseModel):
    """Polling response showing index job progress."""

    job_id: str
    status: str
    progress_pct: float = 0.0
    current_step: str = ""
    total_chunks: int = 0
    processed_chunks: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    elapsed_ms: float = 0.0
    error: Optional[str] = None


class GraphQueryRequest(BaseModel):
    """Request to query the knowledge graph."""

    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    relationship_type: Optional[str] = None
    hop_depth: int = Field(default=2, ge=1, le=5)
    limit: int = Field(default=100, ge=1, le=1000)
    document_ids: Optional[list[str]] = None


class ShortestPathRequest(BaseModel):
    """Request for shortest path between two entities."""

    source_entity: str
    target_entity: str
    max_hops: int = Field(default=5, ge=1, le=10)


class ConfigPatchRequest(BaseModel):
    """Partial config update request."""

    section: str
    updates: dict[str, Any]


class SystemStats(BaseModel):
    """Comprehensive system statistics."""

    document_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    query_count: int = 0
    avg_query_latency_ms: float = 0.0
    faiss_index_size: int = 0
    graph_node_count: int = 0
    graph_edge_count: int = 0
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    cache_hit_rate: float = 0.0
    uptime_seconds: float = 0.0


class HealthCheck(BaseModel):
    """Service health status."""

    status: str
    version: str
    neo4j: str
    faiss: str
    embedding_model: str
    llm_model: str
    sqlite: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Prometheus-compatible metrics response."""

    metrics: dict[str, Any] = Field(default_factory=dict)


# ── SECTION 2: Document Models ───────────────────────────────────────────────


class DocumentStatus(str, Enum):
    """Processing state of a document in the system."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class SupportedFileType(str, Enum):
    """File types supported by the ingestion pipeline."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "md"
    HTML = "html"


class DocumentMetadata(BaseModel):
    """Rich metadata extracted from a document."""

    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    file_type: Optional[str] = None
    source_path: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Represents a document loaded into the system."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    sha256: str
    filename: str
    file_type: SupportedFileType
    file_path: Optional[Path] = None
    raw_text: str = ""
    cleaned_text: str = ""
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1

    model_config = {"arbitrary_types_allowed": True}


class ChunkType(str, Enum):
    """Classification of a text chunk in the hierarchy."""

    PARENT = "parent"
    CHILD = "child"
    STANDALONE = "standalone"


class Chunk(BaseModel):
    """A text segment derived from a document during chunking."""

    model_config = {"extra": "allow"}

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    document_filename: Optional[str] = None
    sequence_index: int = 0
    text: str
    chunk_type: ChunkType = ChunkType.STANDALONE
    parent_chunk_id: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    embedding: Optional[list[float]] = None
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── SECTION 3: Entity & Relationship Models ──────────────────────────────────


class EntityType(str, Enum):
    """Supported entity types and node labels."""

    DOCUMENT = "DOCUMENT"
    CHUNK = "CHUNK"
    ENTITY = "ENTITY"
    COMMUNITY = "COMMUNITY"
    TOPIC = "TOPIC"
    KEYWORD = "KEYWORD"
    SECTION = "SECTION"
    PAGE = "PAGE"
    AUTHOR = "AUTHOR"
    ORGANIZATION = "ORGANIZATION"
    TECHNOLOGY = "TECHNOLOGY"
    CONCEPT = "CONCEPT"
    ALGORITHM = "ALGORITHM"
    DATASET = "DATASET"
    MODEL = "MODEL"
    LIBRARY = "LIBRARY"
    FRAMEWORK = "FRAMEWORK"
    LANGUAGE = "LANGUAGE"
    REPOSITORY = "REPOSITORY"
    FILE = "FILE"
    FUNCTION = "FUNCTION"
    METHOD = "METHOD"
    API = "API"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    DATE = "DATE"
    NUMBER = "NUMBER"
    POLICY = "POLICY"
    LAW = "LAW"
    COMPANY = "COMPANY"
    DEPENDENCY = "DEPENDENCY"
    UNKNOWN = "UNKNOWN"


class Entity(BaseModel):
    """A named entity extracted from a document chunk."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    canonical_name: str
    entity_type: EntityType
    description: Optional[str] = None
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_extracted(
        cls,
        name: str,
        entity_type: str,
        chunk_id: str,
        document_id: str,
        description: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Entity:
        canonical = name.strip().lower()
        try:
            etype = EntityType(entity_type.upper())
        except ValueError:
            etype = EntityType.UNKNOWN
        return cls(
            name=name.strip(),
            canonical_name=canonical,
            entity_type=etype,
            description=description,
            source_chunk_ids=[chunk_id],
            source_document_ids=[document_id],
            confidence=confidence,
        )


class RelationshipType(str, Enum):
    """Common relationship types in the knowledge graph."""

    USES = "USES"
    IMPLEMENTS = "IMPLEMENTS"
    DEPENDS_ON = "DEPENDS_ON"
    IMPORTS = "IMPORTS"
    CALLS = "CALLS"
    CREATED_BY = "CREATED_BY"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    BELONGS_TO = "BELONGS_TO"
    RELATED_TO = "RELATED_TO"
    MENTIONS = "MENTIONS"
    CAUSES = "CAUSES"
    LEADS_TO = "LEADS_TO"
    REFERENCES = "REFERENCES"
    WORKS_WITH = "WORKS_WITH"
    SUPPORTS = "SUPPORTS"
    COMPATIBLE_WITH = "COMPATIBLE_WITH"
    TRAINED_ON = "TRAINED_ON"
    GENERATES = "GENERATES"
    CONSUMES = "CONSUMES"
    CONNECTS_TO = "CONNECTS_TO"
    HAS_PROPERTY = "HAS_PROPERTY"
    HAS_VERSION = "HAS_VERSION"
    OWNS = "OWNS"
    PRODUCES = "PRODUCES"
    WORKS_FOR = "WORKS_FOR"
    REGULATES = "REGULATES"
    CAUSED_BY = "CAUSED_BY"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"

    @classmethod
    def from_str(
        cls,
        rel_type: str,
        source_type: Optional[EntityType] = None,
        target_type: Optional[EntityType] = None,
    ) -> RelationshipType:
        if not rel_type or not rel_type.strip():
            return cls._heuristic_fallback(source_type, target_type)

        raw = rel_type.strip().upper()
        try:
            return cls(raw)
        except ValueError:
            pass

        norm = re.sub(r"[\s\-]+", "_", raw).strip("_")
        try:
            return cls(norm)
        except ValueError:
            pass

        alias_map: dict[str, RelationshipType] = {
            "USE": cls.USES, "USING": cls.USES, "UTILIZES": cls.USES, "UTILIZE": cls.USES,
            "EMPLOY": cls.USES, "BUILT_WITH": cls.USES, "WRITTEN_IN": cls.USES, "POWERED_BY": cls.USES,
            "IMPLEMENT": cls.IMPLEMENTS, "IMPLEMENTED_BY": cls.IMPLEMENTS, "EXTENDS": cls.IMPLEMENTS,
            "INHERITS": cls.IMPLEMENTS, "IS_A": cls.IMPLEMENTS, "SUBCLASS_OF": cls.IMPLEMENTS,
            "DEPENDS": cls.DEPENDS_ON, "REQUIRES": cls.DEPENDS_ON, "REQUIRE": cls.DEPENDS_ON,
            "PREREQUISITE": cls.DEPENDS_ON, "NEEDS": cls.DEPENDS_ON, "NEED": cls.DEPENDS_ON,
            "IMPORT": cls.IMPORTS, "IMPORTING": cls.IMPORTS, "INCLUDES": cls.IMPORTS, "INCLUDE": cls.IMPORTS,
            "CONTAINS": cls.IMPORTS, "CONTAIN": cls.IMPORTS,
            "CALL": cls.CALLS, "INVOKES": cls.CALLS, "INVOKE": cls.CALLS, "TRIGGERS": cls.CALLS, "EXECUTES": cls.CALLS,
            "AUTHOR": cls.CREATED_BY, "AUTHORED_BY": cls.CREATED_BY, "DEVELOPED_BY": cls.CREATED_BY,
            "MADE_BY": cls.CREATED_BY, "WRITTEN_BY": cls.CREATED_BY, "WORKS_FOR": cls.CREATED_BY,
            "BASED_IN": cls.LOCATED_IN, "SITUATED_IN": cls.LOCATED_IN, "FOUND_IN": cls.LOCATED_IN,
            "COMPONENT_OF": cls.PART_OF, "CONTAINED_IN": cls.PART_OF, "MEMBER_OF": cls.PART_OF, "SUBSET_OF": cls.PART_OF,
            "OWNED_BY": cls.BELONGS_TO,
            "ASSOCIATED_WITH": cls.RELATED_TO, "CONNECTED_WITH": cls.RELATED_TO, "LINKED_TO": cls.RELATED_TO,
            "MENTION": cls.MENTIONS, "CITES": cls.MENTIONS, "CITE": cls.MENTIONS, "DISCUSSES": cls.MENTIONS,
            "CAUSE": cls.CAUSES, "CAUSED_BY": cls.CAUSES, "RESULTS_IN": cls.CAUSES,
            "REFERENCE": cls.REFERENCES, "REFERENCING": cls.REFERENCES, "POINTS_TO": cls.REFERENCES,
            "COLLABORATES_WITH": cls.WORKS_WITH, "INTEGRATES_WITH": cls.WORKS_WITH,
            "SUPPORT": cls.SUPPORTS, "SUPPORTING": cls.SUPPORTS, "ENABLES": cls.SUPPORTS,
            "COMPATIBLE": cls.COMPATIBLE_WITH,
            "FINE_TUNED_ON": cls.TRAINED_ON, "FIT_ON": cls.TRAINED_ON,
            "GENERATE": cls.GENERATES, "EMITS": cls.GENERATES, "YIELDS": cls.GENERATES,
            "CONSUME": cls.CONSUMES, "TAKES": cls.CONSUMES, "READS": cls.CONSUMES,
            "CONNECTS": cls.CONNECTS_TO, "LINKS": cls.CONNECTS_TO,
            "HAS_ATTRIBUTE": cls.HAS_PROPERTY, "PROPERTY_OF": cls.HAS_PROPERTY,
            "VERSION_OF": cls.HAS_VERSION, "VERSION": cls.HAS_VERSION,
            "OWN": cls.OWNS, "POSSESSES": cls.OWNS,
            "PRODUCE": cls.PRODUCES, "CREATES": cls.PRODUCES, "MAKES": cls.PRODUCES, "BUILDS": cls.PRODUCES,
        }

        if norm in alias_map:
            return alias_map[norm]

        return cls._heuristic_fallback(source_type, target_type)

    @classmethod
    def _heuristic_fallback(
        cls,
        source_type: Optional[EntityType],
        target_type: Optional[EntityType],
    ) -> RelationshipType:
        if target_type == EntityType.LOCATION:
            return cls.LOCATED_IN
        if target_type in (EntityType.LIBRARY, EntityType.FRAMEWORK, EntityType.TECHNOLOGY, EntityType.DEPENDENCY):
            return cls.DEPENDS_ON
        if target_type in (EntityType.PERSON, EntityType.AUTHOR):
            return cls.CREATED_BY
        if target_type == EntityType.DATASET:
            return cls.TRAINED_ON
        if target_type in (EntityType.DOCUMENT, EntityType.FILE, EntityType.REPOSITORY):
            return cls.REFERENCES
        return cls.RELATED_TO


class Relationship(BaseModel):
    """A directional relationship between two entities."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_id: str
    target_entity_id: str
    source_entity_name: str
    target_entity_name: str
    relationship_type: RelationshipType
    description: Optional[str] = None
    confidence: float = 1.0
    evidence_sentence: Optional[str] = None
    document_id: str = ""
    chunk_id: str = ""
    page_number: Optional[int] = None
    weight: float = 1.0
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_extracted(
        cls,
        source_name: str,
        target_name: str,
        rel_type: str,
        chunk_id: str,
        document_id: str,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        description: Optional[str] = None,
        confidence: float = 1.0,
        evidence_sentence: Optional[str] = None,
        page_number: Optional[int] = None,
        weight: float = 1.0,
        source_type: Optional[EntityType] = None,
        target_type: Optional[EntityType] = None,
    ) -> Relationship:
        rtype = RelationshipType.from_str(rel_type, source_type=source_type, target_type=target_type)
        return cls(
            source_entity_id=source_id or str(uuid4()),
            target_entity_id=target_id or str(uuid4()),
            source_entity_name=source_name.strip(),
            target_entity_name=target_name.strip(),
            relationship_type=rtype,
            description=description,
            confidence=confidence,
            evidence_sentence=evidence_sentence,
            document_id=document_id,
            chunk_id=chunk_id,
            page_number=page_number,
            weight=weight,
            source_chunk_ids=[chunk_id] if chunk_id else [],
            source_document_ids=[document_id] if document_id else [],
        )


class ExtractionResult(BaseModel):
    """Container for entities and relationships extracted from one chunk."""

    chunk_id: str
    document_id: str
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    extraction_time_ms: float = 0.0
    model_name: str = ""


# ── SECTION 4: Graph Models ──────────────────────────────────────────────────


class GraphNode(BaseModel):
    """A node in the knowledge graph (entity)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    neo4j_id: Optional[int] = None
    name: str
    canonical_name: str
    entity_type: str
    description: Optional[str] = None
    properties: dict[str, Any] = Field(default_factory=dict)
    degree: int = 0
    document_count: int = 0


class GraphEdge(BaseModel):
    """An edge in the knowledge graph (relationship)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    source_name: str
    target_name: str
    relationship_type: str
    confidence: float = 1.0
    evidence_sentence: Optional[str] = None
    document_id: str = ""
    chunk_id: str = ""
    page_number: Optional[int] = None
    weight: float = 1.0
    description: Optional[str] = None
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphPath(BaseModel):
    """A path through the knowledge graph."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    length: int = 0
    total_weight: float = 0.0


class GraphStats(BaseModel):
    """Statistics about the knowledge graph."""

    total_nodes: int = 0
    total_edges: int = 0
    entity_type_counts: dict[str, int] = Field(default_factory=dict)
    relationship_type_counts: dict[str, int] = Field(default_factory=dict)
    avg_degree: float = 0.0
    max_degree: int = 0
    connected_components: int = 0
    density: float = 0.0
    top_entities: list[dict[str, Any]] = Field(default_factory=list)
    top_relationships: list[dict[str, Any]] = Field(default_factory=list)


class GraphVisualizationData(BaseModel):
    """Data package for frontend graph visualization."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    stats: GraphStats = Field(default_factory=GraphStats)
    query_entity: Optional[str] = None


# ── SECTION 5: Query & Retrieval Models ──────────────────────────────────────


class QueryIntent(str, Enum):
    """Detected intent of a user query."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"
    NAVIGATIONAL = "navigational"
    UNKNOWN = "unknown"


class SearchMode(str, Enum):
    """Retrieval mode selection."""

    SEMANTIC = "semantic"
    GRAPH = "graph"
    HYBRID = "hybrid"


class Citation(BaseModel):
    """A single citation reference linking answer text to a source."""

    citation_id: str
    document_id: str
    document_filename: str
    chunk_id: str
    chunk_text_snippet: str
    page_number: Optional[int] = None
    relevance_score: float = 0.0
    entity_ids: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """A single retrieval result."""

    chunk_id: str
    document_id: str
    document_filename: str
    text: str
    score: float
    faiss_score: Optional[float] = None
    graph_score: Optional[float] = None
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    entities: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    graph_neighbors: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Incoming query from user."""

    query: str = Field(..., min_length=1, max_length=4096)
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=10, ge=1, le=100)
    graph_hops: int = Field(default=2, ge=0, le=5)
    include_graph: bool = True
    include_citations: bool = True
    stream: bool = False
    session_id: Optional[str] = None
    filters: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Complete response to a user query."""

    query_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    intent: QueryIntent = QueryIntent.UNKNOWN
    answer: str
    reasoning: Optional[str] = None
    confidence_score: float = 0.0
    citations: list[Citation] = Field(default_factory=list)
    search_results: list[SearchResult] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    entities_detected: list[str] = Field(default_factory=list)
    graph_nodes_visited: int = 0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    model_name: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None


class ConversationMessage(BaseModel):
    """One message in a conversation history."""

    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query_id: Optional[str] = None


# ── SECTION 6: Community & Summary Models ───────────────────────────────────


class Community(BaseModel):
    """A detected entity community in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    community_id: int
    title: str
    level: int = 0
    summary: str = ""
    entity_ids: list[str] = Field(default_factory=list)
    entity_names: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CommunityHierarchy(BaseModel):
    """Hierarchical organization of communities."""

    level: int
    communities: list[Community] = Field(default_factory=list)
    parent_community_ids: dict[str, str] = Field(default_factory=dict)


class EntitySummary(BaseModel):
    entity_id: str
    entity_name: str
    entity_type: str
    summary: str
    keywords: list[str] = Field(default_factory=list)


class RelationshipSummary(BaseModel):
    relationship_id: str
    source_name: str
    target_name: str
    rel_type: str
    summary: str
    confidence: float = 1.0


class CommunitySummary(BaseModel):
    community_id: str
    title: str
    level: int = 0
    summary: str
    findings: list[str] = Field(default_factory=list)
    rating: float = 1.0


class KeywordSummary(BaseModel):
    keyword: str
    score: float = 1.0
    summary: str = ""
    associated_entities: list[str] = Field(default_factory=list)


class TopicSummary(BaseModel):
    topic_name: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    community_ids: list[str] = Field(default_factory=list)


class SummaryPackage(BaseModel):
    document_id: str = ""
    entity_summaries: list[EntitySummary] = Field(default_factory=list)
    relationship_summaries: list[RelationshipSummary] = Field(default_factory=list)
    community_summaries: list[CommunitySummary] = Field(default_factory=list)
    keyword_summaries: list[KeywordSummary] = Field(default_factory=list)
    topic_summaries: list[TopicSummary] = Field(default_factory=list)
    created_at: str = ""


# ── SECTION 7: Backward Compatibility Submodule Aliases ─────────────────────

for _mod in ["api", "community", "document", "entity", "graph", "query", "summary"]:
    sys.modules[f"models.{_mod}"] = sys.modules[__name__]
