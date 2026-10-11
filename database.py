"""
Enterprise GraphRAG — Database Package.

Consolidated Database Subsystem: CacheStore, Neo4jClient, Neo4j Schema, SQLiteManager.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from neo4j import GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable, TransientError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core import (
    CypherExecutionError,
    Neo4jConfig,
    Neo4jConnectionError,
    SQLiteConfig,
    app_logger,
    graph_logger,
)

# ── SECTION 1: Cache Store ───────────────────────────────────────────────────

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    value_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL,
    hit_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_cache_namespace ON cache_entries(namespace);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);
"""

_TTL_DEFAULTS: dict[str, int] = {
    "embedding": 86400,
    "llm": 3600,
    "graph": 1800,
    "query": 900,
    "metadata": 86400,
}


class CacheStore:
    """SQLite-backed cache with namespace isolation and TTL expiry."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._hits = 0
        self._misses = 0

    def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_CACHE_SCHEMA)
        app_logger.info(f"Cache store initialized at {self._db_path}")

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _make_key(namespace: str, raw_key: str) -> str:
        digest = hashlib.sha256(f"{namespace}:{raw_key}".encode()).hexdigest()
        return digest

    def get(self, namespace: str, raw_key: str) -> Optional[Any]:
        key = self._make_key(namespace, raw_key)
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value_json, expires_at FROM cache_entries WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                self._misses += 1
                return None
            if row["expires_at"] is not None and row["expires_at"] < now:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                self._misses += 1
                return None
            conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE key = ?",
                (key,),
            )
        self._hits += 1
        return json.loads(row["value_json"])

    def set(
        self,
        namespace: str,
        raw_key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        key = self._make_key(namespace, raw_key)
        effective_ttl = ttl or _TTL_DEFAULTS.get(namespace, 3600)
        now = time.time()
        expires_at = now + effective_ttl
        value_json = json.dumps(value, default=str)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries (key, namespace, value_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at,
                    hit_count = 0
                """,
                (key, namespace, value_json, now, expires_at),
            )

    def delete(self, namespace: str, raw_key: str) -> None:
        key = self._make_key(namespace, raw_key)
        with self._connect() as conn:
            conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))

    def clear_namespace(self, namespace: str) -> int:
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE namespace = ?",
                (namespace,),
            ).fetchone()[0]
            conn.execute(
                "DELETE FROM cache_entries WHERE namespace = ?", (namespace,)
            )
        return count

    def evict_expired(self) -> int:
        now = time.time()
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE expires_at < ?", (now,)
            ).fetchone()[0]
            conn.execute(
                "DELETE FROM cache_entries WHERE expires_at < ?", (now,)
            )
        return count

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM cache_entries"
            ).fetchone()[0]
            by_ns = conn.execute(
                "SELECT namespace, COUNT(*) as cnt FROM cache_entries GROUP BY namespace"
            ).fetchall()
        return {
            "total_entries": total,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "by_namespace": {r["namespace"]: r["cnt"] for r in by_ns},
        }


# ── SECTION 2: Neo4j Client ─────────────────────────────────────────────────


class Neo4jClient:
    """Thread-safe Neo4j driver wrapper."""

    def __init__(self, config: Neo4jConfig) -> None:
        self._config = config
        self._driver = None
        self._connected = False

    def _ensure_driver(self) -> None:
        if self._driver is not None:
            return
        try:
            self._driver = GraphDatabase.driver(
                self._config.uri,
                auth=(self._config.username, self._config.password),
                max_connection_pool_size=self._config.max_connection_pool_size,
                connection_timeout=self._config.connection_timeout,
                encrypted=False,
            )
        except Exception as exc:
            raise Neo4jConnectionError(
                f"Cannot create Neo4j driver: {exc}", cause=exc
            ) from exc

    def verify_connectivity(self) -> None:
        self._ensure_driver()
        try:
            self._driver.verify_connectivity()
            self._connected = True
            graph_logger.info(f"Neo4j connected: {self._config.uri}")
        except ServiceUnavailable as exc:
            raise Neo4jConnectionError(
                f"Neo4j not available at {self._config.uri}. "
                "Ensure Neo4j Community Edition is running.",
                cause=exc,
            ) from exc

    @property
    def is_connected(self) -> bool:
        return self._connected

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        self._ensure_driver()
        if not self._connected:
            try:
                self.verify_connectivity()
            except Neo4jConnectionError:
                raise
        with self._driver.session(database=self._config.database) as sess:
            yield sess

    @retry(
        retry=retry_if_exception_type(TransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def run_query(
        self,
        cypher: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        params = parameters or {}
        try:
            with self.session() as sess:
                result = sess.run(cypher, **params)
                return [dict(record) for record in result]
        except TransientError:
            raise
        except Exception as exc:
            raise CypherExecutionError(
                f"Cypher execution failed: {exc}",
                details={"cypher": cypher[:200], "params": str(params)[:200]},
                cause=exc,
            ) from exc

    def run_write_query(
        self,
        cypher: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        params = parameters or {}
        try:
            with self.session() as sess:
                result = sess.execute_write(
                    lambda tx: list(tx.run(cypher, **params))
                )
                return [dict(r) for r in result]
        except Exception as exc:
            raise CypherExecutionError(
                f"Write query failed: {exc}",
                details={"cypher": cypher[:200]},
                cause=exc,
            ) from exc

    def run_batch_write(
        self,
        cypher: str,
        batch: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> int:
        total = 0
        for i in range(0, len(batch), batch_size):
            sub_batch = batch[i : i + batch_size]
            self.run_write_query(cypher, {"batch": sub_batch})
            total += len(sub_batch)
        return total

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            self._connected = False
            graph_logger.info("Neo4j driver closed.")


# ── SECTION 3: Neo4j Schema ──────────────────────────────────────────────────

_CONSTRAINT_QUERIES = [
    """
    CREATE CONSTRAINT entity_unique IF NOT EXISTS
    FOR (e:Entity) REQUIRE (e.canonical_name, e.entity_type) IS UNIQUE
    """,
    """
    CREATE CONSTRAINT document_id_unique IF NOT EXISTS
    FOR (d:Document) REQUIRE d.document_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
    FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT community_id_unique IF NOT EXISTS
    FOR (c:Community) REQUIRE c.community_id IS UNIQUE
    """,
]

_INDEX_QUERIES = [
    """
    CREATE TEXT INDEX entity_name_text IF NOT EXISTS
    FOR (e:Entity) ON (e.name)
    """,
    """
    CREATE INDEX entity_type_idx IF NOT EXISTS
    FOR (e:Entity) ON (e.entity_type)
    """,
    """
    CREATE INDEX chunk_document_idx IF NOT EXISTS
    FOR (c:Chunk) ON (c.document_id)
    """,
]

_NODE_LABELS = [
    "Document", "Chunk", "Entity", "Community", "Topic", "Keyword", "Section",
    "Page", "Author", "Organization", "Technology", "Concept", "Algorithm",
    "Dataset", "Model", "Library", "Framework", "Language", "Repository",
    "File", "Function", "Method", "API",
]
_RELATIONSHIP_TYPES = [
    "USES", "IMPLEMENTS", "DEPENDS_ON", "IMPORTS", "CALLS", "CREATED_BY",
    "LOCATED_IN", "PART_OF", "BELONGS_TO", "RELATED_TO", "MENTIONS", "CAUSES",
    "LEADS_TO", "REFERENCES", "WORKS_WITH", "SUPPORTS", "COMPATIBLE_WITH",
    "TRAINED_ON", "GENERATES", "CONSUMES", "CONNECTS_TO", "HAS_PROPERTY",
    "HAS_VERSION", "OWNS", "PRODUCES", "WORKS_FOR", "REGULATES", "CAUSED_BY",
    "ASSOCIATED_WITH", "HAS_CHUNK", "MENTIONS_ENTITY",
]


def ensure_neo4j_schema(neo4j_client: Neo4jClient) -> None:
    graph_logger.info("Ensuring Neo4j schema (constraints + indexes)...")
    for query in _CONSTRAINT_QUERIES:
        try:
            neo4j_client.run_write_query(query.strip())
            graph_logger.debug(f"Constraint applied: {query.strip()[:60]}...")
        except Exception as exc:
            graph_logger.warning(f"Constraint query skipped (may exist): {exc}")

    for query in _INDEX_QUERIES:
        try:
            neo4j_client.run_write_query(query.strip())
            graph_logger.debug(f"Index applied: {query.strip()[:60]}...")
        except Exception as exc:
            graph_logger.warning(f"Index query skipped (may exist): {exc}")

    graph_logger.info("Neo4j schema verification complete.")


# ── SECTION 4: SQLite Manager ────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    sha256 TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    title TEXT,
    author TEXT,
    page_count INTEGER,
    word_count INTEGER,
    file_size_bytes INTEGER,
    version INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    sequence_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'standalone',
    parent_chunk_id TEXT,
    start_char INTEGER NOT NULL DEFAULT 0,
    end_char INTEGER NOT NULL DEFAULT 0,
    page_number INTEGER,
    section_header TEXT,
    token_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS query_log (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    intent TEXT,
    answer TEXT,
    confidence_score REAL,
    retrieval_time_ms REAL,
    generation_time_ms REAL,
    total_time_ms REAL,
    session_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    description TEXT,
    document_id TEXT,
    chunk_id TEXT,
    properties_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id TEXT PRIMARY KEY,
    source_canonical TEXT NOT NULL,
    target_canonical TEXT NOT NULL,
    source_name TEXT NOT NULL,
    target_name TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    evidence_sentence TEXT,
    document_id TEXT,
    chunk_id TEXT,
    page_number INTEGER,
    weight REAL DEFAULT 1.0,
    description TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_communities (
    id TEXT PRIMARY KEY,
    community_id INTEGER UNIQUE NOT NULL,
    title TEXT NOT NULL,
    level INTEGER DEFAULT 0,
    summary TEXT,
    entity_ids_json TEXT DEFAULT '[]',
    keywords_json TEXT DEFAULT '[]',
    topics_json TEXT DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_sequence ON chunks(document_id, sequence_index);
CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_query_log_created ON query_log(created_at);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_canonical ON graph_nodes(canonical_name);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_canonical);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_canonical);
"""


class SQLiteManager:
    """Manages the SQLite metadata store."""

    def __init__(self, config: SQLiteConfig) -> None:
        self._db_path = config.path
        self._connection: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
        app_logger.info(f"SQLite initialized at {self._db_path}")

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert_document(self, doc: dict[str, Any]) -> None:
        sql = """
            INSERT INTO documents (
                id, sha256, filename, file_type, file_path, status,
                title, author, page_count, word_count, file_size_bytes,
                version, metadata_json, created_at, updated_at
            ) VALUES (
                :id, :sha256, :filename, :file_type, :file_path, :status,
                :title, :author, :page_count, :word_count, :file_size_bytes,
                :version, :metadata_json, :created_at, :updated_at
            )
            ON CONFLICT(sha256) DO UPDATE SET
                filename=excluded.filename,
                file_path=excluded.file_path,
                status=excluded.status,
                title=COALESCE(excluded.title, documents.title),
                author=COALESCE(excluded.author, documents.author),
                page_count=COALESCE(excluded.page_count, documents.page_count),
                word_count=COALESCE(excluded.word_count, documents.word_count),
                file_size_bytes=COALESCE(excluded.file_size_bytes, documents.file_size_bytes),
                version=excluded.version,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
        """
        with self._connect() as conn:
            conn.execute(sql, doc)

    def get_document(self, document_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT d.*, COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                WHERE d.id = ?
                GROUP BY d.id
                """, (document_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_document_by_sha256(self, sha256: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT d.*, COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                WHERE d.sha256 = ?
                GROUP BY d.id
                """, (sha256,)
            ).fetchone()
        return dict(row) if row else None

    def list_documents(
        self,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        offset = (page - 1) * page_size
        where_count = "WHERE status = ?" if status else ""
        where_join = "WHERE d.status = ?" if status else ""
        params_count = (status,) if status else ()
        params_list = (status, page_size, offset) if status else (page_size, offset)

        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM documents {where_count}", params_count
            ).fetchone()[0]
            sql = f"""
                SELECT d.*, COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                {where_join}
                GROUP BY d.id
                ORDER BY d.created_at DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(sql, params_list).fetchall()
        return [dict(r) for r in rows], total

    def update_document_status(self, document_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
                (status, datetime.utcnow().isoformat(), document_id),
            )

    def delete_document(self, document_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        sql = """
            INSERT OR REPLACE INTO chunks (
                id, document_id, sequence_index, text, chunk_type,
                parent_chunk_id, start_char, end_char, page_number,
                section_header, token_count, metadata_json, created_at
            ) VALUES (
                :id, :document_id, :sequence_index, :text, :chunk_type,
                :parent_chunk_id, :start_char, :end_char, :page_number,
                :section_header, :token_count, :metadata_json, :created_at
            )
        """
        with self._connect() as conn:
            conn.executemany(sql, chunks)

    def get_chunks_for_document(self, document_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY sequence_index",
                (document_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete_chunks_for_document(self, document_id: str) -> int:
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)
            ).fetchone()[0]
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        return count

    def count_chunks(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def log_query(self, query_record: dict[str, Any]) -> None:
        sql = """
            INSERT INTO query_log (
                id, query, intent, answer, confidence_score,
                retrieval_time_ms, generation_time_ms, total_time_ms,
                session_id, created_at
            ) VALUES (
                :id, :query, :intent, :answer, :confidence_score,
                :retrieval_time_ms, :generation_time_ms, :total_time_ms,
                :session_id, :created_at
            )
        """
        with self._connect() as conn:
            conn.execute(sql, query_record)

    def get_query_stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_queries,
                    AVG(total_time_ms) as avg_latency_ms,
                    MIN(total_time_ms) as min_latency_ms,
                    MAX(total_time_ms) as max_latency_ms
                FROM query_log
                """
            ).fetchone()
        return dict(row) if row else {}

    def upsert_graph_nodes(self, nodes: list[dict[str, Any]]) -> int:
        sql = """
            INSERT OR REPLACE INTO graph_nodes (
                id, name, canonical_name, entity_type, description,
                document_id, chunk_id, properties_json, created_at
            ) VALUES (
                :id, :name, :canonical_name, :entity_type, :description,
                :document_id, :chunk_id, :properties_json, :created_at
            )
        """
        now = datetime.utcnow().isoformat()
        records = [
            {
                "id": n.get("id"),
                "name": n.get("name", ""),
                "canonical_name": n.get("canonical_name", ""),
                "entity_type": n.get("entity_type", "UNKNOWN"),
                "description": n.get("description", ""),
                "document_id": n.get("document_id", ""),
                "chunk_id": n.get("chunk_id", ""),
                "properties_json": json.dumps(n.get("properties", {})),
                "created_at": now,
            }
            for n in nodes
        ]
        with self._connect() as conn:
            conn.executemany(sql, records)
        return len(records)

    def upsert_graph_edges(self, edges: list[dict[str, Any]]) -> int:
        sql = """
            INSERT OR REPLACE INTO graph_edges (
                id, source_canonical, target_canonical, source_name, target_name,
                rel_type, confidence, evidence_sentence, document_id, chunk_id,
                page_number, weight, description, created_at
            ) VALUES (
                :id, :source_canonical, :target_canonical, :source_name, :target_name,
                :rel_type, :confidence, :evidence_sentence, :document_id, :chunk_id,
                :page_number, :weight, :description, :created_at
            )
        """
        now = datetime.utcnow().isoformat()
        records = [
            {
                "id": e.get("id"),
                "source_canonical": e.get("source_canonical", ""),
                "target_canonical": e.get("target_canonical", ""),
                "source_name": e.get("source_name", e.get("source_canonical", "")),
                "target_name": e.get("target_name", e.get("target_canonical", "")),
                "rel_type": e.get("rel_type", "RELATED_TO"),
                "confidence": e.get("confidence", 1.0),
                "evidence_sentence": e.get("evidence_sentence", ""),
                "document_id": e.get("document_id", ""),
                "chunk_id": e.get("chunk_id", ""),
                "page_number": e.get("page_number"),
                "weight": e.get("weight", 1.0),
                "description": e.get("description", ""),
                "created_at": now,
            }
            for e in edges
        ]
        with self._connect() as conn:
            conn.executemany(sql, records)
        return len(records)

    def upsert_graph_communities(self, communities: list[dict[str, Any]]) -> int:
        sql = """
            INSERT OR REPLACE INTO graph_communities (
                id, community_id, title, level, summary,
                entity_ids_json, keywords_json, topics_json, created_at
            ) VALUES (
                :id, :community_id, :title, :level, :summary,
                :entity_ids_json, :keywords_json, :topics_json, :created_at
            )
        """
        now = datetime.utcnow().isoformat()
        records = [
            {
                "id": c.get("id"),
                "community_id": c.get("community_id"),
                "title": c.get("title", ""),
                "level": c.get("level", 0),
                "summary": c.get("summary", ""),
                "entity_ids_json": json.dumps(c.get("entity_ids", [])),
                "keywords_json": json.dumps(c.get("keywords", [])),
                "topics_json": json.dumps(c.get("topics", [])),
                "created_at": now,
            }
            for c in communities
        ]
        with self._connect() as conn:
            conn.executemany(sql, records)
        return len(records)

    def get_all_graph_nodes(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM graph_nodes LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_all_graph_edges(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM graph_edges LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_graph_edges_for_nodes(self, node_identifiers: list[str], limit: int = 200) -> list[dict[str, Any]]:
        if not node_identifiers:
            return []
        clean_ids = list(set(str(i).strip() for i in node_identifiers if i))
        if not clean_ids:
            return []

        placeholders = ",".join("?" for _ in clean_ids)
        sql = f"""
            SELECT * FROM graph_edges
            WHERE source_canonical IN ({placeholders})
               OR target_canonical IN ({placeholders})
               OR source_name IN ({placeholders})
               OR target_name IN ({placeholders})
               OR id IN ({placeholders})
            LIMIT ?
        """
        params = clean_ids * 5 + [limit]
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_graph_nodes_for_documents(self, document_ids: list[str] | None, limit: int = 100) -> list[dict[str, Any]]:
        if document_ids is None:
            return self.get_all_graph_nodes(limit=limit)
        clean_ids = list(set(str(i).strip() for i in document_ids if i))
        if not clean_ids:
            return []
        placeholders = ",".join("?" for _ in clean_ids)
        sql = f"""
            SELECT * FROM graph_nodes
            WHERE document_id IN ({placeholders})
               OR canonical_name IN (SELECT source_canonical FROM graph_edges WHERE document_id IN ({placeholders}))
               OR canonical_name IN (SELECT target_canonical FROM graph_edges WHERE document_id IN ({placeholders}))
               OR name IN (SELECT source_name FROM graph_edges WHERE document_id IN ({placeholders}))
               OR name IN (SELECT target_name FROM graph_edges WHERE document_id IN ({placeholders}))
            LIMIT ?
        """
        params = clean_ids * 5 + [limit]
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_graph_edges_for_documents(self, document_ids: list[str] | None, limit: int = 200) -> list[dict[str, Any]]:
        if document_ids is None:
            return self.get_all_graph_edges(limit=limit)
        clean_ids = list(set(str(i).strip() for i in document_ids if i))
        if not clean_ids:
            return []
        placeholders = ",".join("?" for _ in clean_ids)
        sql = f"SELECT * FROM graph_edges WHERE document_id IN ({placeholders}) LIMIT ?"
        params = clean_ids + [limit]
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def delete_graph_data_for_document(self, document_id: str) -> tuple[int, int]:
        with self._connect() as conn:
            edges_cursor = conn.execute("DELETE FROM graph_edges WHERE document_id = ?", (document_id,))
            deleted_edges = edges_cursor.rowcount

            nodes_cursor = conn.execute("DELETE FROM graph_nodes WHERE document_id = ?", (document_id,))
            deleted_nodes = nodes_cursor.rowcount

            conn.execute("""
                DELETE FROM graph_nodes
                WHERE document_id NOT IN (SELECT id FROM documents)
            """)
            conn.execute("""
                DELETE FROM graph_edges
                WHERE document_id NOT IN (SELECT id FROM documents)
            """)

        return deleted_nodes, deleted_edges

    def purge_all_graph_data_if_empty(self) -> None:
        with self._connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            if doc_count == 0:
                conn.execute("DELETE FROM graph_nodes")
                conn.execute("DELETE FROM graph_edges")

    def close(self) -> None:
        pass


# ── SECTION 5: Backward Compatibility Submodule Aliases ─────────────────────

for _mod in ["cache_store", "neo4j_client", "schema", "sqlite"]:
    sys.modules[f"database.{_mod}"] = sys.modules[__name__]
