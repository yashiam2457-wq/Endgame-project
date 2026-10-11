"""
Enterprise GraphRAG — Graph Operations Package.

Consolidated Graph Subsystem: GraphBuilder, CommunityDetector, Cypher generators, NetworkXGraphOps, GraphTraversal, GraphVisualizer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from core import graph_logger
from database import Neo4jClient
from models import Community, Document, ExtractionResult, GraphEdge, GraphNode, GraphPath, GraphStats, GraphVisualizationData, Entity, Relationship
from utils import canonicalize_entity_name

# ── SECTION 1: Cypher Query Helpers ──────────────────────────────────────────


def build_neighborhood_query(
    entity_name: str, hop_depth: int = 2, limit: int = 50
) -> tuple[str, dict[str, Any]]:
    cypher = """
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower($name)
       OR toLower(e.canonical_name) CONTAINS toLower($name)
    WITH e LIMIT 5
    MATCH (e)-[r]-(neighbor:Entity)
    RETURN e.name AS source, type(r) AS relationship, neighbor.name AS target, neighbor.entity_type AS target_type
    LIMIT $limit
    """
    return cypher.strip(), {"name": entity_name, "limit": limit}


def build_shortest_path_query(
    source_name: str, target_name: str, max_hops: int = 5
) -> tuple[str, dict[str, Any]]:
    cypher = f"""
    MATCH (a:Entity), (b:Entity)
    WHERE toLower(a.name) CONTAINS toLower($source)
      AND toLower(b.name) CONTAINS toLower($target)
    WITH a, b LIMIT 1
    MATCH path = shortestPath((a)-[*..{max_hops}]-(b))
    RETURN [n IN nodes(path) | n.name] AS nodes, [r IN relationships(path) | type(r)] AS relationships
    """
    return cypher.strip(), {"source": source_name, "target": target_name}


# ── SECTION 2: Graph Builder ─────────────────────────────────────────────────

_UPSERT_ENTITY_CYPHER = """
UNWIND $batch AS item
MERGE (e:Entity {canonical_name: item.canonical_name, entity_type: item.entity_type})
ON CREATE SET
    e.id = item.id,
    e.name = item.name,
    e.description = item.description,
    e.source_chunks = [item.chunk_id],
    e.source_documents = [item.document_id],
    e.created_at = datetime()
ON MATCH SET
    e.source_chunks = CASE
        WHEN item.chunk_id IN e.source_chunks THEN e.source_chunks
        ELSE e.source_chunks + [item.chunk_id]
    END,
    e.source_documents = CASE
        WHEN item.document_id IN e.source_documents THEN e.source_documents
        ELSE e.source_documents + [item.document_id]
    END,
    e.updated_at = datetime()
RETURN e.canonical_name AS canonical_name, e.id AS id
"""

_UPSERT_RELATIONSHIP_CYPHER = """
UNWIND $batch AS item
MATCH (src:Entity {canonical_name: item.source_canonical})
MATCH (tgt:Entity {canonical_name: item.target_canonical})
MERGE (src)-[r:RELATED_TO {rel_type: item.rel_type}]->(tgt)
ON CREATE SET
    r.id = item.id,
    r.description = item.description,
    r.confidence = item.confidence,
    r.evidence_sentence = item.evidence_sentence,
    r.document_id = item.document_id,
    r.chunk_id = item.chunk_id,
    r.page_number = item.page_number,
    r.weight = item.weight,
    r.source_chunks = [item.chunk_id],
    r.created_at = datetime()
ON MATCH SET
    r.weight = r.weight + item.weight,
    r.confidence = CASE WHEN item.confidence > r.confidence THEN item.confidence ELSE r.confidence END,
    r.evidence_sentence = COALESCE(r.evidence_sentence, item.evidence_sentence),
    r.source_chunks = CASE
        WHEN item.chunk_id IN r.source_chunks THEN r.source_chunks
        ELSE r.source_chunks + [item.chunk_id]
    END
"""

_UPSERT_DOCUMENT_CYPHER = """
MERGE (d:Document {document_id: $doc_id})
ON CREATE SET
    d.filename = $filename,
    d.file_type = $file_type,
    d.sha256 = $sha256,
    d.created_at = datetime()
ON MATCH SET
    d.updated_at = datetime()
"""

_LINK_ENTITY_DOCUMENT_CYPHER = """
UNWIND $batch AS item
MATCH (e:Entity {canonical_name: item.canonical_name, entity_type: item.entity_type})
MATCH (d:Document {document_id: item.document_id})
MERGE (d)-[:MENTIONS_ENTITY]->(e)
"""

_UPSERT_COMMUNITY_CYPHER = """
UNWIND $batch AS item
MERGE (c:Community {community_id: item.community_id})
ON CREATE SET
    c.id = item.id,
    c.title = item.title,
    c.level = item.level,
    c.summary = item.summary,
    c.keywords = item.keywords,
    c.topics = item.topics,
    c.created_at = datetime()
ON MATCH SET
    c.title = item.title,
    c.summary = item.summary,
    c.keywords = item.keywords,
    c.topics = item.topics,
    c.updated_at = datetime()
WITH c, item
UNWIND item.entity_ids AS eid
MATCH (e:Entity {id: eid})
MERGE (e)-[:IN_COMMUNITY]->(c)
"""


class GraphBuilder:
    """Constructs the knowledge graph from entity/relationship extractions."""

    def __init__(self, neo4j: Neo4jClient, sqlite: Any = None) -> None:
        self._neo4j = neo4j
        if sqlite is None:
            try:
                from core import get_sqlite
                self._sqlite = get_sqlite()
            except Exception:
                self._sqlite = None
        else:
            self._sqlite = sqlite

    def build_from_extractions(
        self,
        extraction_results: list[ExtractionResult],
        document: Document,
    ) -> tuple[int, int]:
        entity_batch: list[dict[str, Any]] = []
        relationship_batch: list[dict[str, Any]] = []
        seen_entities: set[str] = set()

        for result in extraction_results:
            for entity in result.entities:
                dedup_key = f"{entity.canonical_name}:{entity.entity_type.value}"
                if dedup_key in seen_entities:
                    continue
                seen_entities.add(dedup_key)
                entity_batch.append(
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "canonical_name": entity.canonical_name,
                        "entity_type": entity.entity_type.value,
                        "description": entity.description or "",
                        "chunk_id": result.chunk_id,
                        "document_id": document.id,
                    }
                )

            for rel in result.relationships:
                relationship_batch.append(
                    {
                        "id": rel.id,
                        "source_canonical": canonicalize_entity_name(rel.source_entity_name),
                        "target_canonical": canonicalize_entity_name(rel.target_entity_name),
                        "source_name": rel.source_entity_name,
                        "target_name": rel.target_entity_name,
                        "rel_type": rel.relationship_type.value,
                        "description": rel.description or "",
                        "confidence": rel.confidence,
                        "evidence_sentence": rel.evidence_sentence or "",
                        "document_id": rel.document_id or document.id,
                        "chunk_id": rel.chunk_id or result.chunk_id,
                        "page_number": rel.page_number,
                        "weight": rel.weight,
                    }
                )

        n_count = 0
        r_count = 0
        if self._sqlite is not None:
            n_count = self._sqlite.upsert_graph_nodes(entity_batch)
            r_count = self._sqlite.upsert_graph_edges(relationship_batch)
            graph_logger.info(f"Persisted graph: {n_count} nodes, {r_count} edges in SQLite.")

        if not self._neo4j.is_connected:
            graph_logger.info("Neo4j not connected — using SQLite persistent graph store.")
            return n_count or len(entity_batch), r_count or len(relationship_batch)

        try:
            self._neo4j.run_write_query(
                _UPSERT_DOCUMENT_CYPHER,
                {
                    "doc_id": document.id,
                    "filename": document.filename,
                    "file_type": document.file_type.value,
                    "sha256": document.sha256,
                },
            )
        except Exception as exc:
            graph_logger.warning(f"Document upsert failed: {exc}")

        entity_count = 0
        if entity_batch:
            try:
                entity_count = self._neo4j.run_batch_write(
                    _UPSERT_ENTITY_CYPHER, entity_batch
                )
            except Exception as exc:
                graph_logger.error(f"Entity batch insert failed: {exc}")

        rel_count = 0
        if relationship_batch and entity_count > 0:
            try:
                rel_count = self._neo4j.run_batch_write(
                    _UPSERT_RELATIONSHIP_CYPHER, relationship_batch
                )
            except Exception as exc:
                graph_logger.error(f"Relationship batch insert failed: {exc}")

        if entity_batch:
            link_batch = [
                {
                    "canonical_name": e["canonical_name"],
                    "entity_type": e["entity_type"],
                    "document_id": document.id,
                }
                for e in entity_batch
            ]
            try:
                self._neo4j.run_batch_write(
                    _LINK_ENTITY_DOCUMENT_CYPHER, link_batch
                )
            except Exception as exc:
                graph_logger.warning(f"Entity-document linking failed: {exc}")

        graph_logger.info(
            f"Graph built: {entity_count} entities, {rel_count} relationships "
            f"for document {document.filename}"
        )
        return entity_count, rel_count

    def build_communities(self, communities: list[Any]) -> int:
        if not communities:
            return 0

        batch = [
            {
                "id": c.id,
                "community_id": c.community_id,
                "title": c.title,
                "level": c.level,
                "summary": c.summary,
                "keywords": c.keywords,
                "topics": c.topics,
                "entity_ids": c.entity_ids,
            }
            for c in communities
        ]

        c_count = 0
        if self._sqlite is not None:
            c_count = self._sqlite.upsert_graph_communities(batch)
            graph_logger.info(f"Persisted {c_count} communities in SQLite.")

        if not self._neo4j.is_connected:
            return c_count or len(communities)

        try:
            count = self._neo4j.run_batch_write(_UPSERT_COMMUNITY_CYPHER, batch)
            graph_logger.info(f"Successfully persisted {len(communities)} communities in Neo4j.")
            return count
        except Exception as exc:
            graph_logger.error(f"Community batch insert failed: {exc}")
            return 0


# ── SECTION 3: Community Detector ───────────────────────────────────────────


class CommunityDetector:
    """Performs community detection on knowledge graph structures."""

    def detect_communities(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        min_community_size: int = 1,
    ) -> list[Community]:
        if not entities:
            return []

        g = nx.Graph()
        entity_map = {e.id: e for e in entities}
        name_to_id = {canonicalize_entity_name(e.name): e.id for e in entities}

        for e in entities:
            g.add_node(e.id, name=e.name, entity_type=e.entity_type.value)

        for rel in relationships:
            src_id = rel.source_entity_id if rel.source_entity_id in entity_map else name_to_id.get(canonicalize_entity_name(rel.source_entity_name))
            tgt_id = rel.target_entity_id if rel.target_entity_id in entity_map else name_to_id.get(canonicalize_entity_name(rel.target_entity_name))

            if src_id and tgt_id and src_id in g and tgt_id in g:
                g.add_edge(src_id, tgt_id, rel_type=rel.relationship_type.value, weight=rel.weight)

        partition_sets = []
        if g.number_of_edges() > 0:
            try:
                partition_sets = list(nx.community.louvain_communities(g, weight="weight"))
            except Exception:
                try:
                    partition_sets = list(nx.community.greedy_modularity_communities(g, weight="weight"))
                except Exception as exc:
                    graph_logger.warning(f"Community detection fallback triggered: {exc}")
                    partition_sets = [set(g.nodes())]
        else:
            partition_sets = [set(comp) for comp in nx.connected_components(g)]

        communities: list[Community] = []
        for idx, node_set in enumerate(partition_sets):
            if len(node_set) < min_community_size:
                continue

            member_entities = [entity_map[nid] for nid in node_set if nid in entity_map]
            entity_names = [e.name for e in member_entities]
            entity_ids = [e.id for e in member_entities]

            rel_ids = [
                r.id for r in relationships
                if r.source_entity_id in node_set or r.target_entity_id in node_set
            ]

            primary_types = list(set(e.entity_type.value for e in member_entities))
            title = f"Community {idx + 1}: {', '.join(entity_names[:3])}" if entity_names else f"Community {idx + 1}"
            keywords = list(set(primary_types + [name.lower() for name in entity_names[:5]]))

            community = Community(
                community_id=idx + 1,
                title=title,
                level=0,
                summary=f"Cluster of {len(member_entities)} entities focusing on {', '.join(entity_names[:4])}.",
                entity_ids=entity_ids,
                entity_names=entity_names,
                relationship_ids=rel_ids,
                keywords=keywords,
                topics=primary_types[:3],
            )
            communities.append(community)

        graph_logger.info(
            f"Community detection complete: {len(communities)} communities "
            f"detected from {len(entities)} entities."
        )
        return communities


# ── SECTION 4: NetworkX Graph Operations ────────────────────────────────────


class NetworkXGraphOps:
    """In-memory graph operations using NetworkX."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    def add_node(
        self,
        node_id: str,
        name: str,
        entity_type: str,
        description: str = "",
        **kwargs: Any,
    ) -> None:
        self._graph.add_node(
            node_id,
            name=name,
            entity_type=entity_type,
            description=description,
            **kwargs,
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self._graph.add_edge(
            source_id,
            target_id,
            rel_type=rel_type,
            weight=weight,
            **kwargs,
        )

    def build_from_dicts(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        self._graph.clear()
        node_id_map: dict[str, str] = {}

        for node in nodes:
            name = str(node.get("name") or node.get("canonical_name") or node.get("id") or "Entity").strip()
            canonical = str(node.get("canonical_name") or name).lower().strip()
            nid = str(node.get("id") or "").strip()

            primary_key = canonical if canonical else name
            if not primary_key:
                continue

            node_id_map[primary_key] = primary_key
            node_id_map[primary_key.lower()] = primary_key
            node_id_map[name] = primary_key
            node_id_map[name.lower()] = primary_key
            if nid:
                node_id_map[nid] = primary_key

            self.add_node(
                node_id=primary_key,
                name=name,
                entity_type=str(node.get("entity_type") or "CONCEPT").upper(),
                description=str(node.get("description") or ""),
            )

        for edge in edges:
            raw_src = str(edge.get("source_canonical") or edge.get("source_name") or edge.get("source_id") or "").strip()
            raw_tgt = str(edge.get("target_canonical") or edge.get("target_name") or edge.get("target_id") or "").strip()

            src = node_id_map.get(raw_src) or node_id_map.get(raw_src.lower()) or (raw_src.lower() if raw_src else "")
            tgt = node_id_map.get(raw_tgt) or node_id_map.get(raw_tgt.lower()) or (raw_tgt.lower() if raw_tgt else "")

            if src and tgt and src != tgt:
                if not self._graph.has_node(src):
                    src_name = str(edge.get("source_name") or raw_src)
                    self.add_node(node_id=src, name=src_name, entity_type="CONCEPT")
                if not self._graph.has_node(tgt):
                    tgt_name = str(edge.get("target_name") or raw_tgt)
                    self.add_node(node_id=tgt, name=tgt_name, entity_type="CONCEPT")

                self.add_edge(
                    source_id=src,
                    target_id=tgt,
                    rel_type=str(edge.get("rel_type") or edge.get("relationship_type") or "RELATED_TO").upper(),
                    weight=float(edge.get("weight", 1.0)),
                    confidence=float(edge.get("confidence", 1.0)),
                    evidence_sentence=edge.get("evidence_sentence"),
                    document_id=str(edge.get("document_id", "")),
                    chunk_id=str(edge.get("chunk_id", "")),
                    page_number=edge.get("page_number"),
                )

    def compute_stats(self) -> GraphStats:
        num_nodes = self._graph.number_of_nodes()
        num_edges = self._graph.number_of_edges()

        if num_nodes == 0:
            return GraphStats()

        degrees = dict(self._graph.degree())
        avg_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0.0
        max_degree = max(degrees.values()) if degrees else 0

        type_counts: dict[str, int] = {}
        for _, attrs in self._graph.nodes(data=True):
            etype = attrs.get("entity_type", "UNKNOWN")
            type_counts[etype] = type_counts.get(etype, 0) + 1

        rel_counts: dict[str, int] = {}
        for _, _, attrs in self._graph.edges(data=True):
            rtype = attrs.get("rel_type", "RELATED_TO")
            rel_counts[rtype] = rel_counts.get(rtype, 0) + 1

        undirected = self._graph.to_undirected()
        num_components = nx.number_connected_components(undirected)
        density = nx.density(self._graph)

        top_nodes_sorted = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        top_entities = [
            {
                "name": self._graph.nodes[n].get("name", n),
                "degree": deg,
                "type": self._graph.nodes[n].get("entity_type", "UNKNOWN"),
            }
            for n, deg in top_nodes_sorted
        ]

        return GraphStats(
            total_nodes=num_nodes,
            total_edges=num_edges,
            entity_type_counts=type_counts,
            relationship_type_counts=rel_counts,
            avg_degree=avg_degree,
            max_degree=max_degree,
            connected_components=num_components,
            density=density,
            top_entities=top_entities,
        )

    def to_cytoscape_format(self) -> GraphVisualizationData:
        cy_nodes = []
        for nid, attrs in self._graph.nodes(data=True):
            cy_nodes.append(
                {
                    "data": {
                        "id": str(nid),
                        "label": attrs.get("name", str(nid)),
                        "type": attrs.get("entity_type", "UNKNOWN"),
                        "description": attrs.get("description", ""),
                    }
                }
            )

        cy_edges = []
        for i, (src, tgt, attrs) in enumerate(self._graph.edges(data=True)):
            cy_edges.append(
                {
                    "data": {
                        "id": f"e_{i}_{src}_{tgt}",
                        "source": str(src),
                        "target": str(tgt),
                        "label": attrs.get("rel_type", "RELATED_TO"),
                        "weight": attrs.get("weight", 1.0),
                        "confidence": attrs.get("confidence", 1.0),
                        "evidence_sentence": attrs.get("evidence_sentence"),
                        "document_id": attrs.get("document_id", ""),
                        "chunk_id": attrs.get("chunk_id", ""),
                        "page_number": attrs.get("page_number"),
                    }
                }
            )

        return GraphVisualizationData(
            nodes=cy_nodes,
            edges=cy_edges,
            stats=self.compute_stats(),
        )

    def generate_pyvis_html(self, output_path: Path) -> str:
        try:
            from pyvis.network import Network

            net = Network(
                height="600px",
                width="100%",
                directed=True,
                notebook=False,
                bgcolor="#1e1e2e",
                font_color="#cdd6f4",
            )
            net.from_nx(self._graph)
            net.toggle_physics(True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            net.write_html(str(output_path))
            graph_logger.info(f"PyVis graph saved to {output_path}")
            return str(output_path)
        except Exception as exc:
            graph_logger.error(f"PyVis generation failed: {exc}")
            return ""


# ── SECTION 5: Graph Traversal ───────────────────────────────────────────────

_NEIGHBORHOOD_SIMPLE_CYPHER = """
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($name)
   OR toLower(e.canonical_name) CONTAINS toLower($name)
WITH e LIMIT 3
OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
WITH e, collect({
    node: neighbor,
    rel_type: type(r),
    rel_props: properties(r)
}) AS neighbors
RETURN e, neighbors
LIMIT $limit
"""

_MULTIHOP_CYPHER = """
MATCH path = (start:Entity)-[*1..$hops]-(end:Entity)
WHERE toLower(start.name) CONTAINS toLower($start_entity)
WITH path, end, length(path) AS path_length
ORDER BY path_length
RETURN
    [n IN nodes(path) | {
        name: n.name,
        entity_type: n.entity_type,
        canonical_name: n.canonical_name
    }] AS nodes,
    [r IN relationships(path) | {
        type: type(r),
        weight: r.weight
    }] AS rels,
    path_length
LIMIT $limit
"""

_SHORTEST_PATH_CYPHER = """
MATCH (a:Entity), (b:Entity)
WHERE toLower(a.name) CONTAINS toLower($source)
  AND toLower(b.name) CONTAINS toLower($target)
WITH a, b LIMIT 1
MATCH path = shortestPath((a)-[*..10]-(b))
RETURN
    [n IN nodes(path) | {
        name: n.name,
        entity_type: n.entity_type
    }] AS nodes,
    [r IN relationships(path) | type(r)] AS rel_types,
    length(path) AS length
"""

_ENTITY_SEARCH_CYPHER = """
MATCH (e:Entity)
WHERE toLower(e.name) CONTAINS toLower($query)
   OR toLower(e.entity_type) = toLower($query)
RETURN e
ORDER BY size(e.source_documents) DESC
LIMIT $limit
"""

_GRAPH_STATS_CYPHER = """
MATCH (e:Entity)
WITH count(e) AS node_count
MATCH ()-[r]->()
WITH node_count, count(r) AS edge_count
RETURN node_count, edge_count
"""


class GraphTraversal:
    """Executes graph traversal queries against Neo4j."""

    def __init__(self, neo4j: Neo4jClient, sqlite: Any = None) -> None:
        self._neo4j = neo4j
        if sqlite is None:
            try:
                from core import get_sqlite
                self._sqlite = get_sqlite()
            except Exception:
                self._sqlite = None
        else:
            self._sqlite = sqlite

    def get_neighborhood(
        self,
        entity_name: str,
        hops: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if not self._neo4j.is_connected:
            if self._sqlite is not None:
                nodes = self._sqlite.get_all_graph_nodes(limit=limit)
                if entity_name:
                    return [n for n in nodes if entity_name.lower() in n.get("name", "").lower()]
                return nodes
            return []

        try:
            results = self._neo4j.run_query(
                _NEIGHBORHOOD_SIMPLE_CYPHER,
                {"name": entity_name, "limit": limit},
            )
            return self._format_neighborhood(results)
        except Exception as exc:
            graph_logger.warning(f"Neighborhood query failed: {exc}")
            return []

    def multihop_traversal(
        self,
        start_entity: str,
        hops: int = 2,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if not self._neo4j.is_connected:
            return []

        try:
            return self._neo4j.run_query(
                _MULTIHOP_CYPHER,
                {"start_entity": start_entity, "hops": hops, "limit": limit},
            )
        except Exception as exc:
            graph_logger.warning(f"Multi-hop traversal failed: {exc}")
            return []

    def shortest_path(
        self,
        source_entity: str,
        target_entity: str,
    ) -> Optional[dict[str, Any]]:
        if not self._neo4j.is_connected:
            return None

        try:
            results = self._neo4j.run_query(
                _SHORTEST_PATH_CYPHER,
                {"source": source_entity, "target": target_entity},
            )
            return results[0] if results else None
        except Exception as exc:
            graph_logger.warning(f"Shortest path query failed: {exc}")
            return None

    def search_entities(
        self,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if not self._neo4j.is_connected:
            if self._sqlite is not None:
                nodes = self._sqlite.get_all_graph_nodes(limit=limit)
                if query:
                    q = query.lower()
                    return [n for n in nodes if q in n.get("name", "").lower() or q in n.get("entity_type", "").lower()]
                return nodes
            return []

        try:
            return self._neo4j.run_query(
                _ENTITY_SEARCH_CYPHER,
                {"query": query, "limit": limit},
            )
        except Exception as exc:
            graph_logger.warning(f"Entity search failed: {exc}")
            return []

    def get_entity_context(
        self, entity_names: list[str], hops: int = 1
    ) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        seen_entities: set[str] = set()

        for name in entity_names[:5]:
            neighborhood = self.get_neighborhood(name, hops=hops, limit=20)
            for item in neighborhood:
                entity_key = item.get("name", "")
                if entity_key and entity_key not in seen_entities:
                    seen_entities.add(entity_key)
                    context.append(item)

        return context

    def get_stats(self) -> dict[str, int]:
        if not self._neo4j.is_connected:
            return {"node_count": 0, "edge_count": 0}

        try:
            results = self._neo4j.run_query(_GRAPH_STATS_CYPHER)
            if results:
                return {
                    "node_count": results[0].get("node_count", 0),
                    "edge_count": results[0].get("edge_count", 0),
                }
        except Exception as exc:
            graph_logger.warning(f"Stats query failed: {exc}")

        return {"node_count": 0, "edge_count": 0}

    def _format_neighborhood(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for row in results:
            entity_node = row.get("e", {})
            neighbors = row.get("neighbors", [])

            if entity_node:
                formatted.append(self._node_to_dict(entity_node))

            for nbr in neighbors:
                if isinstance(nbr, dict) and nbr.get("node"):
                    node_dict = self._node_to_dict(nbr["node"])
                    node_dict["relationship_type"] = nbr.get("rel_type", "RELATED_TO")
                    formatted.append(node_dict)

        return formatted

    @staticmethod
    def _node_to_dict(node: Any) -> dict[str, Any]:
        if isinstance(node, dict):
            return node
        try:
            return dict(node)
        except Exception:
            return {}


# ── SECTION 6: Graph Visualizer ───────────────────────────────────────────────


class GraphVisualizer:
    """Helper to convert graph data into HTML interactive visualizer files."""

    def __init__(self) -> None:
        self._nx_ops = NetworkXGraphOps()

    def render_graph_html(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        output_file: Path,
    ) -> str:
        self._nx_ops.build_from_dicts(nodes, edges)
        return self._nx_ops.generate_pyvis_html(output_file)


# ── SECTION 7: Backward Compatibility Submodule Aliases ─────────────────────

for _mod in ["builder", "community_detector", "cypher", "networkx_ops", "traversal", "visualizer"]:
    sys.modules[f"graph.{_mod}"] = sys.modules[__name__]
