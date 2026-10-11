"""API endpoint integration tests."""
from __future__ import annotations

from fastapi.testclient import TestClient
from backend.app import create_app

app = create_app()
client = TestClient(app)


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] in ["healthy", "degraded"]


def test_stats_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "ram_used_mb" in data["data"]
