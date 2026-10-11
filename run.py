import os
import subprocess
import sys
from pathlib import Path

# Auto-switch to virtualenv python if invoked via system python
venv_python = Path(__file__).parent / "venv" / "Scripts" / "python.exe"
if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
    cmd = [str(venv_python)] + sys.argv
    sys.exit(subprocess.call(cmd))

import uvicorn
from core.config import get_settings


def main() -> None:
    """Launch the Uvicorn web server hosting the FastAPI application."""
    settings = get_settings()
    uvicorn.run(
        "backend.app:create_app",
        factory=True,
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.logging.level.lower(),
    )



if __name__ == "__main__":
    main()
