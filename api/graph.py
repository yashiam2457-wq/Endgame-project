"""Graph API router."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from core import get_neo4j, get_sqlite
from graph import GraphTraversal, NetworkXGraphOps
from models import APIResponse, GraphQueryRequest, ShortestPathRequest

router = APIRouter(prefix="", tags=["Graph"], dependencies=[Depends(verify_api_key)])

_EMPTY_GRAPH = {"nodes": [], "edges": []}


@router.post("/graph", response_model=APIResponse)
async def query_knowledge_graph(
    request: GraphQueryRequest,
    neo4j=Depends(get_neo4j),
) -> APIResponse:
    try:
        sqlite = get_sqlite()
        docs, _ = sqlite.list_documents(page_size=1)
        if not docs and not neo4j.is_connected:
            sqlite.purge_all_graph_data_if_empty()
            return APIResponse(success=True, message="Knowledge graph is empty (no indexed documents)", data=_EMPTY_GRAPH)

        traversal = GraphTraversal(neo4j, sqlite)


        # Clean and normalize document_ids filter
        doc_ids = [str(d).strip() for d in request.document_ids if str(d).strip()] if request.document_ids else None
        if doc_ids == []:
            doc_ids = None

        if doc_ids:
            nodes = sqlite.get_graph_nodes_for_documents(doc_ids, limit=request.limit)
            edges = sqlite.get_graph_edges_for_documents(doc_ids, limit=request.limit * 2)
            if request.entity_name:
                q = request.entity_name.lower()
                nodes = [n for n in nodes if q in n.get("name", "").lower() or q in n.get("canonical_name", "").lower()]
                node_ids = [str(n.get("canonical_name") or n.get("name") or n.get("id")) for n in nodes]
                edges = sqlite.get_graph_edges_for_nodes(node_ids, limit=request.limit * 2)

        elif request.entity_name:
            nodes = traversal.get_neighborhood(
                request.entity_name, hops=request.hop_depth, limit=request.limit
            )
            node_ids = []
            for n in nodes:
                for k in ("canonical_name", "name", "id"):
                    v = n.get(k)
                    if v:
                        node_ids.append(str(v))
            edges = sqlite.get_graph_edges_for_nodes(node_ids, limit=request.limit * 2)
        else:
            nodes = sqlite.get_all_graph_nodes(limit=request.limit)
            edges = sqlite.get_all_graph_edges(limit=request.limit * 2)

        if not nodes:
            # No entities extracted yet — return empty graph with informative message
            # (Do NOT show fake Section 1/2/3 nodes — those are chunking artifacts, not knowledge graph data)
            return APIResponse(
                success=True,
                message="Knowledge graph entities not yet extracted. Re-upload your document or wait for entity extraction to complete.",
                data=_EMPTY_GRAPH,
            )

        nx_ops = NetworkXGraphOps()
        nx_ops.build_from_dicts(nodes, edges)
        vis_data = nx_ops.to_cytoscape_format()

        msg = "Neo4j connected" if (neo4j and neo4j.is_connected) else "SQLite persistent store"
        return APIResponse(success=True, message=msg, data=vis_data)


    except Exception as exc:
        return APIResponse(
            success=True,
            message=f"Graph query returned no data: {exc}",
            data=_EMPTY_GRAPH,
        )


@router.post("/graph/shortest-path", response_model=APIResponse)
async def get_shortest_path(
    request: ShortestPathRequest,
    neo4j=Depends(get_neo4j),
) -> APIResponse:
    """Find shortest path between two graph entities."""
    if not neo4j.is_connected:
        return APIResponse(
            success=True,
            message="Knowledge graph not available (Neo4j offline).",
            data={"path": []},
        )
    try:
        traversal = GraphTraversal(neo4j)
        path = traversal.shortest_path(request.source_entity, request.target_entity)
        return APIResponse(success=True, message="Shortest path query complete", data={"path": path})
    except Exception as exc:
        return APIResponse(success=True, message=f"No path found: {exc}", data={"path": []})
