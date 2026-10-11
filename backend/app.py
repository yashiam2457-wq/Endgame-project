"""
Enterprise GraphRAG — FastAPI Application Factory.

Configures application middleware, exception handlers, static file mounting,
and routes all frontend HTML pages and API endpoints.
"""

from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.router import api_router
from core import GraphRAGError, get_settings, lifespan


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-Ready Offline Enterprise GraphRAG System using Gemma 4 1B, Neo4j & FAISS",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static assets
    static_dir = settings.static_dir
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include API endpoints
    app.include_router(api_router)

    # ── Exception Handlers ───────────────────────────────────────────────────

    @app.exception_handler(GraphRAGError)
    async def graphrag_exception_handler(request: Request, exc: GraphRAGError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    # ── Frontend SPA Route ────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def serve_index() -> HTMLResponse:
        """Serve SPA shell template."""
        template_file = settings.templates_dir / "index.html"
        if template_file.exists():
            return HTMLResponse(content=template_file.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Enterprise GraphRAG UI</h1><p>Frontend template initializing...</p>")

    return app
