"""
Enterprise GraphRAG — Core Package.

Consolidated Core Subsystem: Config, Logging, Exceptions, Dependencies, Events.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Literal, Optional

import yaml
from fastapi import Depends, FastAPI
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:
    class SettingsConfigDict(dict):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

    class BaseSettings(BaseModel):
        model_config = {}


# ── SECTION 1: Config ────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent


def _load_yaml_config() -> dict[str, Any]:
    """Load config.yaml from the project root."""
    config_path = _PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class ModelConfig(BaseSettings):
    """LLM and embedding model settings."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    generation_model_name: str = Field(
        default="gemma3:1b",
        alias="GENERATION_MODEL_NAME",
        description="Model name or HuggingFace ID for generation.",
    )
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
        description="SentenceTransformer model ID.",
    )
    model_cache_dir: Path = Field(
        default=Path("./checkpoints"),
        alias="MODEL_CACHE_DIR",
    )
    generation_device: str = Field(default="auto", alias="GENERATION_DEVICE")
    embedding_device: str = Field(default="auto", alias="EMBEDDING_DEVICE")
    quantization: Literal["fp16", "int8", "none"] = Field(
        default="fp16", alias="QUANTIZATION"
    )
    max_new_tokens: int = Field(default=1024, alias="MAX_NEW_TOKENS")
    temperature: float = Field(default=0.1, alias="TEMPERATURE")
    top_p: float = Field(default=0.9, alias="TOP_P")
    repetition_penalty: float = Field(default=1.1, alias="REPETITION_PENALTY")
    batch_inference_size: int = Field(default=4, alias="BATCH_INFERENCE_SIZE")
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")

    @field_validator("model_cache_dir", mode="before")
    @classmethod
    def resolve_cache_dir(cls, v: Any) -> Path:
        """Resolve relative paths to absolute."""
        return _PROJECT_ROOT / Path(v)


class Neo4jConfig(BaseSettings):
    """Neo4j connection settings."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_", extra="ignore")

    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="neo4j")
    password: str = Field(default="password")
    database: str = Field(default="neo4j")
    max_connection_pool_size: int = Field(default=50)
    connection_timeout: int = Field(default=30)


class FaissConfig(BaseSettings):
    """FAISS vector store settings."""

    model_config = SettingsConfigDict(env_prefix="FAISS_", extra="ignore")

    index_path: Path = Field(default=Path("./data/faiss_index"))
    index_type: Literal["Flat", "IVFFlat", "IVFPQ"] = Field(default="Flat")
    nlist: int = Field(default=100)
    nprobe: int = Field(default=10)
    top_k: int = Field(default=10, alias="FAISS_TOP_K")
    use_gpu: bool = Field(default=False)
    dimension: int = Field(default=384)

    @field_validator("index_path", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        return _PROJECT_ROOT / Path(v)


FAISSConfig = FaissConfig



class SQLiteConfig(BaseSettings):
    """SQLite storage settings."""

    model_config = SettingsConfigDict(env_prefix="SQLITE_", extra="ignore")

    path: Path = Field(default=Path("./data/graphrag.db"))
    cache_path: Path = Field(default=Path("./cache/cache.db"))

    @field_validator("path", "cache_path", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        return _PROJECT_ROOT / Path(v)


class ChunkingConfig(BaseSettings):
    """Text chunking settings."""

    model_config = SettingsConfigDict(env_prefix="CHUNK_", extra="ignore")

    strategy: Literal[
        "fixed", "sliding", "recursive", "sentence", "semantic", "hierarchical"
    ] = Field(default="recursive")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    min_length: int = Field(default=50, alias="CHUNK_MIN_LENGTH")
    max_length: int = Field(default=2048, alias="CHUNK_MAX_LENGTH")


class RetrievalConfig(BaseSettings):
    """Hybrid retrieval settings."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    top_k: int = Field(default=10, alias="RETRIEVAL_TOP_K")
    graph_hop_depth: int = Field(default=2, alias="GRAPH_HOP_DEPTH")
    faiss_weight: float = Field(default=0.5, alias="HYBRID_FAISS_WEIGHT")
    graph_weight: float = Field(default=0.5, alias="HYBRID_GRAPH_WEIGHT")
    rerank_top_k: int = Field(default=5, alias="RERANK_TOP_K")
    mmr_lambda: float = Field(default=0.7, alias="MMR_LAMBDA")
    max_context_tokens: int = Field(default=3072, alias="MAX_CONTEXT_TOKENS")
    enable_compression: bool = Field(default=True)
    dedup_threshold: float = Field(default=0.95)

    @model_validator(mode="after")
    def validate_weights(self) -> "RetrievalConfig":
        """Ensure FAISS + graph weights are normalized."""
        total = self.faiss_weight + self.graph_weight
        if total > 0:
            self.faiss_weight /= total
            self.graph_weight /= total
        return self


class LoggingConfig(BaseSettings):
    """Logging settings."""

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    level: str = Field(default="INFO")
    log_dir: Path = Field(default=Path("./logs"))
    rotation: str = Field(default="100 MB")
    retention: str = Field(default="30 days")

    @field_validator("log_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: Any) -> Path:
        return _PROJECT_ROOT / Path(v)


class AppConfig(BaseSettings):
    """Root application configuration."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="Enterprise GraphRAG", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    app_env: str = Field(default="production", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    app_secret_key: str = Field(
        default="change-me-to-a-random-64-char-secret-key",
        alias="APP_SECRET_KEY",
    )

    enable_auth: bool = Field(default=False, alias="ENABLE_AUTH")
    api_key: str = Field(default="", alias="API_KEY")

    model: ModelConfig = Field(default_factory=ModelConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    faiss: FaissConfig = Field(default_factory=FaissConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    data_dir: Path = Field(default=_PROJECT_ROOT / "data")
    upload_dir: Path = Field(default=_PROJECT_ROOT / "data" / "uploads")
    static_dir: Path = Field(default=_PROJECT_ROOT / "frontend" / "static")
    templates_dir: Path = Field(default=_PROJECT_ROOT / "frontend" / "templates")

    @model_validator(mode="after")
    def ensure_directories(self) -> "AppConfig":
        """Create all required directories on startup."""
        dirs = [
            self.data_dir,
            self.upload_dir,
            self.faiss.index_path.parent,
            self.sqlite.path.parent,
            self.sqlite.cache_path.parent,
            self.logging.log_dir,
            self.model.model_cache_dir,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def from_yaml(cls) -> "AppConfig":
        """Load config merging YAML defaults with env-var overrides."""
        yaml_data = _load_yaml_config()
        overrides: dict[str, Any] = {}
        if "app" in yaml_data:
            app_section = yaml_data["app"]
            overrides["APP_HOST"] = app_section.get("host", "0.0.0.0")
            overrides["APP_PORT"] = app_section.get("port", 8000)
            overrides["APP_DEBUG"] = app_section.get("debug", False)
        for key, val in overrides.items():
            if key not in os.environ:
                os.environ[key] = str(val)
        return cls()


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """Return singleton application settings (cached)."""
    return AppConfig.from_yaml()


# ── SECTION 2: Logging ───────────────────────────────────────────────────────


def configure_logging(config: LoggingConfig) -> None:
    """Set up Loguru logging with multiple sinks."""
    log_dir = config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level=config.level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    _sink_configs = [
        ("app", "app.log", "INFO"),
        ("retrieval", "retrieval.log", "DEBUG"),
        ("graph", "graph.log", "DEBUG"),
        ("embedding", "embedding.log", "DEBUG"),
        ("generation", "generation.log", "DEBUG"),
        ("performance", "performance.log", "INFO"),
        ("api", "api.log", "INFO"),
    ]

    for name, filename, level in _sink_configs:
        logger.add(
            log_dir / filename,
            level=level,
            rotation=config.rotation,
            retention=config.retention,
            encoding="utf-8",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            filter=lambda record, n=name: record["extra"].get("sink") == n,
            backtrace=True,
            diagnose=False,
            enqueue=True,
        )

    logger.add(
        log_dir / "errors.log",
        level="ERROR",
        rotation=config.rotation,
        retention=config.retention,
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}\n{exception}"
        ),
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )


def get_logger(sink: str) -> "logger":
    """Return a logger bound to a specific subsystem sink."""
    return logger.bind(sink=sink)


app_logger = get_logger("app")
retrieval_logger = get_logger("retrieval")
graph_logger = get_logger("graph")
embedding_logger = get_logger("embedding")
generation_logger = get_logger("generation")
performance_logger = get_logger("performance")
api_logger = get_logger("api")


# ── SECTION 3: Exceptions ────────────────────────────────────────────────────


class GraphRAGError(Exception):
    """Base exception for all Enterprise GraphRAG errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to API-friendly dict."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(GraphRAGError):
    status_code = 500
    error_code = "CONFIGURATION_ERROR"


class DocumentError(GraphRAGError):
    status_code = 422
    error_code = "DOCUMENT_ERROR"


class DocumentNotFoundError(DocumentError):
    status_code = 404
    error_code = "DOCUMENT_NOT_FOUND"


class DocumentParsingError(DocumentError):
    status_code = 422
    error_code = "DOCUMENT_PARSE_ERROR"


class DuplicateDocumentError(DocumentError):
    status_code = 409
    error_code = "DUPLICATE_DOCUMENT"


class UnsupportedFileTypeError(DocumentError):
    status_code = 415
    error_code = "UNSUPPORTED_FILE_TYPE"


class EmbeddingError(GraphRAGError):
    status_code = 500
    error_code = "EMBEDDING_ERROR"


class EmbeddingModelNotLoadedError(EmbeddingError):
    status_code = 503
    error_code = "EMBEDDING_MODEL_NOT_LOADED"


class GenerationError(GraphRAGError):
    status_code = 500
    error_code = "GENERATION_ERROR"


class ModelNotLoadedError(GenerationError):
    status_code = 503
    error_code = "MODEL_NOT_LOADED"


class EntityExtractionError(GenerationError):
    status_code = 422
    error_code = "ENTITY_EXTRACTION_ERROR"


class JSONParseError(GenerationError):
    status_code = 422
    error_code = "JSON_PARSE_ERROR"


class GraphError(GraphRAGError):
    status_code = 500
    error_code = "GRAPH_ERROR"


class Neo4jConnectionError(GraphError):
    status_code = 503
    error_code = "NEO4J_CONNECTION_ERROR"


class CypherExecutionError(GraphError):
    status_code = 500
    error_code = "CYPHER_EXECUTION_ERROR"


class EntityNotFoundError(GraphError):
    status_code = 404
    error_code = "ENTITY_NOT_FOUND"


class RetrievalError(GraphRAGError):
    status_code = 500
    error_code = "RETRIEVAL_ERROR"


class FAISSIndexError(RetrievalError):
    status_code = 500
    error_code = "FAISS_INDEX_ERROR"


class FAISSIndexNotFoundError(RetrievalError):
    status_code = 503
    error_code = "FAISS_INDEX_NOT_FOUND"


class AuthenticationError(GraphRAGError):
    status_code = 401
    error_code = "AUTHENTICATION_ERROR"


class AuthorizationError(GraphRAGError):
    status_code = 403
    error_code = "AUTHORIZATION_ERROR"


class ValidationError(GraphRAGError):
    status_code = 422
    error_code = "VALIDATION_ERROR"


class RateLimitError(GraphRAGError):
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"


# ── SECTION 4: Dependencies ──────────────────────────────────────────────────


def get_app_settings() -> AppConfig:
    return get_settings()


SettingsDep = Annotated[AppConfig, Depends(get_app_settings)]


@lru_cache(maxsize=1)
def _get_sqlite_manager() -> "SQLiteManager":  # noqa: F821
    from database import SQLiteManager

    settings = get_settings()
    return SQLiteManager(settings.sqlite)


def get_sqlite() -> "SQLiteManager":  # noqa: F821
    return _get_sqlite_manager()


SQLiteDep = Annotated["SQLiteManager", Depends(get_sqlite)]  # noqa: F821


@lru_cache(maxsize=1)
def _get_neo4j_client() -> "Neo4jClient":  # noqa: F821
    from database import Neo4jClient

    settings = get_settings()
    return Neo4jClient(settings.neo4j)


def get_neo4j() -> "Neo4jClient":  # noqa: F821
    return _get_neo4j_client()


Neo4jDep = Annotated["Neo4jClient", Depends(get_neo4j)]  # noqa: F821


@lru_cache(maxsize=1)
def _get_cache_store() -> "CacheStore":  # noqa: F821
    from database import CacheStore

    settings = get_settings()
    return CacheStore(settings.sqlite.cache_path)


def get_cache() -> "CacheStore":  # noqa: F821
    return _get_cache_store()


CacheDep = Annotated["CacheStore", Depends(get_cache)]  # noqa: F821


@lru_cache(maxsize=1)
def _get_embedding_encoder() -> "EmbeddingEncoder":  # noqa: F821
    from embedding import EmbeddingEncoder

    settings = get_settings()
    return EmbeddingEncoder(settings.model)


def get_encoder() -> "EmbeddingEncoder":  # noqa: F821
    return _get_embedding_encoder()


EncoderDep = Annotated["EmbeddingEncoder", Depends(get_encoder)]  # noqa: F821


@lru_cache(maxsize=1)
def _get_llm_model() -> "GemmaModel":  # noqa: F821
    from generation import GemmaModel

    settings = get_settings()
    return GemmaModel(settings.model)


def get_llm() -> "GemmaModel":  # noqa: F821
    return _get_llm_model()


LLMDep = Annotated["GemmaModel", Depends(get_llm)]  # noqa: F821


@lru_cache(maxsize=1)
def _get_faiss_store() -> "FAISSStore":  # noqa: F821
    from retrieval import FAISSStore

    settings = get_settings()
    return FAISSStore(settings.faiss, settings.model.embedding_model_name)


def get_faiss() -> "FAISSStore":  # noqa: F821
    return _get_faiss_store()


FaissDep = Annotated["FAISSStore", Depends(get_faiss)]  # noqa: F821


@lru_cache(maxsize=1)
def _get_retrieval_pipeline() -> "RetrievalPipeline":  # noqa: F821
    from retrieval import RetrievalPipeline

    settings = get_settings()
    return RetrievalPipeline(
        settings=settings,
        faiss_store=_get_faiss_store(),
        neo4j_client=_get_neo4j_client(),
        encoder=_get_embedding_encoder(),
        llm=_get_llm_model(),
        cache=_get_cache_store(),
    )


def get_retrieval_pipeline() -> "RetrievalPipeline":  # noqa: F821
    return _get_retrieval_pipeline()


RetrievalDep = Annotated["RetrievalPipeline", Depends(get_retrieval_pipeline)]  # noqa: F821


@lru_cache(maxsize=1)
def _get_ingestion_pipeline() -> "IngestionPipeline":  # noqa: F821
    from ingestion import IngestionPipeline

    settings = get_settings()
    return IngestionPipeline(
        settings=settings,
        sqlite=_get_sqlite_manager(),
        faiss_store=_get_faiss_store(),
        neo4j_client=_get_neo4j_client(),
        encoder=_get_embedding_encoder(),
        llm=_get_llm_model(),
        cache=_get_cache_store(),
    )


def get_ingestion_pipeline() -> "IngestionPipeline":  # noqa: F821
    return _get_ingestion_pipeline()


IngestionDep = Annotated["IngestionPipeline", Depends(get_ingestion_pipeline)]  # noqa: F821


# ── SECTION 5: Events ────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager."""
    settings = get_settings()
    configure_logging(settings.logging)
    app_logger.info("=== Enterprise GraphRAG starting ===")
    app_logger.info(f"Environment: {settings.app_env}")
    app_logger.info(f"Debug mode: {settings.app_debug}")

    try:
        await _startup(settings)
        app_logger.info("=== GraphRAG ready — all systems online ===")
        yield
    finally:
        await _shutdown()
        app_logger.info("=== GraphRAG shutdown complete ===")


async def _startup(settings: AppConfig) -> None:
    """Initialize all subsystems on startup."""
    from database import SQLiteManager, Neo4jClient, CacheStore, ensure_neo4j_schema

    app_logger.info("Initializing SQLite metadata store...")
    sqlite = _get_sqlite_manager()
    sqlite.initialize()
    app_logger.info("SQLite ready.")

    app_logger.info(f"Connecting to Neo4j at {settings.neo4j.uri} ...")
    neo4j_connected = False
    neo4j_last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            neo4j = _get_neo4j_client()
            neo4j.verify_connectivity()
            ensure_neo4j_schema(neo4j)
            app_logger.info("● Neo4j Online — connected and schema verified.")
            neo4j_connected = True
            break
        except Exception as exc:
            neo4j_last_exc = exc
            app_logger.warning(
                f"Neo4j connection attempt {attempt}/3 failed: {exc}. "
                + ("Retrying in 3s..." if attempt < 3 else "")
            )
            if attempt < 3:
                import asyncio as _asyncio
                await _asyncio.sleep(3)

    if not neo4j_connected:
        _neo4j_uri = settings.neo4j.uri
        app_logger.error(
            f"\n{'='*60}\n"
            f"  ✖  Neo4j is REQUIRED for GraphRAG but is not reachable.\n"
            f"  URI: {_neo4j_uri}\n"
            f"  Credentials: {settings.neo4j.username} / {settings.neo4j.password}\n\n"
            f"  To start Neo4j locally, open a NEW terminal and run:\n"
            f"    powershell -ExecutionPolicy Bypass -File start_neo4j.ps1\n\n"
            f"  Then restart: python run.py\n"
            f"{'='*60}"
        )
        raise RuntimeError(
            f"Neo4j is REQUIRED for GraphRAG. Cannot connect to {_neo4j_uri}. "
            f"Run start_neo4j.ps1 in a separate terminal first, then restart run.py."
        )

    app_logger.info("Initializing cache store...")
    cache = _get_cache_store()
    cache.initialize()
    app_logger.info("Cache store ready.")

    app_logger.info("Pre-warming embedding model...")
    try:
        encoder = _get_embedding_encoder()
        encoder.load()
        app_logger.info(f"Embedding model loaded: {settings.model.embedding_model_name}")
    except Exception as exc:
        app_logger.warning(f"Embedding model pre-warm failed: {exc}")

    app_logger.info("Pre-warming Gemma 4 LLM (connecting to Ollama)...")
    try:
        llm = _get_llm_model()
        llm.load()
        app_logger.info(f"Gemma model ready: '{llm._ollama_model}' via Ollama.")
    except Exception as exc:
        app_logger.warning(f"LLM pre-warm failed (will retry on first use): {exc}")

    app_logger.info("Loading FAISS index...")
    try:
        faiss = _get_faiss_store()
        faiss.load_if_exists()
        if faiss.total_vectors == 0:
            chunks_count = sqlite.count_chunks()
            if chunks_count > 0:
                app_logger.info(f"Auto-syncing FAISS vector index from {chunks_count} SQLite document chunks...")
                docs_list, _ = sqlite.list_documents(page_size=500)
                all_chunks = []
                for d in docs_list:
                    doc_chunks = sqlite.get_chunks_for_document(d["id"])
                    all_chunks.extend(doc_chunks)
                if all_chunks:
                    encoder = _get_embedding_encoder()
                    texts = [c.get("text", "") for c in all_chunks]
                    embeddings_arr = encoder.encode(texts)
                    embeddings = embeddings_arr.tolist() if hasattr(embeddings_arr, "tolist") else embeddings_arr

                    from models import Chunk, ChunkType
                    chunk_objs = [
                        Chunk(
                            id=c["id"],
                            document_id=c["document_id"],
                            sequence_index=c.get("sequence_index", 0),
                            text=c.get("text", ""),
                            chunk_type=ChunkType(c.get("chunk_type", "PARAGRAPH")),
                            section_header=c.get("section_header"),
                            page_number=c.get("page_number"),
                        )
                        for c in all_chunks
                    ]
                    faiss.add_chunks(chunk_objs, embeddings)
                    app_logger.info(f"FAISS index auto-synced: {faiss.total_vectors} vectors ready.")
        app_logger.info("FAISS index ready.")
    except Exception as exc:
        app_logger.warning(f"FAISS index initialization note: {exc}")


async def _shutdown() -> None:
    """Gracefully close all subsystem connections."""
    try:
        neo4j = _get_neo4j_client()
        neo4j.close()
        app_logger.info("Neo4j connection closed.")
    except Exception as exc:
        app_logger.warning(f"Error closing Neo4j: {exc}")

    try:
        sqlite = _get_sqlite_manager()
        sqlite.close()
        app_logger.info("SQLite connection closed.")
    except Exception as exc:
        app_logger.warning(f"Error closing SQLite: {exc}")


# ── SECTION 6: Backward Compatibility Submodule Aliases ─────────────────────

for _mod in ["config", "logging", "exceptions", "dependencies", "events"]:
    sys.modules[f"core.{_mod}"] = sys.modules[__name__]
