"""Pytest fixtures and environment configuration."""
from __future__ import annotations

import pytest
from pathlib import Path

from core.config import AppConfig


@pytest.fixture
def mock_config(tmp_path: Path) -> AppConfig:
    """Provide a temporary AppConfig instance for testing."""
    return AppConfig(
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "data" / "uploads",
        static_dir=tmp_path / "frontend" / "static",
        templates_dir=tmp_path / "frontend" / "templates",
    )
