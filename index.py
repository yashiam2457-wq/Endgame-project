"""
Enterprise GraphRAG — CLI Indexing Entrypoint.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core.config import get_settings
from core.dependencies import _get_ingestion_pipeline


async def run_cli_indexing() -> None:
    """Run CLI batch indexing over the default upload directory."""
    settings = get_settings()
    pipeline = _get_ingestion_pipeline()

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.upload_dir
    print(f"Starting CLI ingestion for: {target}")

    results = await pipeline.ingest_directory(target)
    print(f"Indexed {len(results)} files.")


def main() -> None:
    asyncio.run(run_cli_indexing())


if __name__ == "__main__":
    main()
