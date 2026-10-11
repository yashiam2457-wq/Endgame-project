"""
Enterprise GraphRAG — Utilities Package.

Consolidated Utilities: File Operations, GPU Detection, JSON Parsing/Repair, Text Processing, Performance Timing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, Optional

import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

from core import app_logger, generation_logger, performance_logger

# ── SECTION 1: File Utilities ────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".html", ".htm"}
)


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_supported_files(root: Path, recursive: bool = True) -> Iterator[Path]:
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield root
        return

    glob_pattern = "**/*" if recursive else "*"
    for path in root.glob(glob_pattern):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def safe_copy(src: Path, dst_dir: Path, rename: Optional[str] = None) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_name = rename or src.name
    dst = dst_dir / dst_name
    shutil.copy2(str(src), str(dst))
    return dst


def human_readable_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} TB"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── SECTION 2: GPU Utilities ──────────────────────────────────────────────────


def is_cuda_available() -> bool:
    if not TORCH_AVAILABLE or torch is None:
        return False
    try:
        return torch.cuda.is_available()
    except Exception as exc:
        app_logger.warning(f"CUDA check encountered DLL error: {exc}. Falling back to CPU.")
        return False


def detect_device(preference: str = "auto") -> torch.device:
    if not TORCH_AVAILABLE or torch is None:
        return "cpu"

    if preference == "cpu":
        app_logger.info("Device forced to CPU by configuration.")
        return torch.device("cpu")

    if preference == "cuda" or preference == "auto":
        if is_cuda_available():
            try:
                device = torch.device("cuda")
                props = torch.cuda.get_device_properties(0)
                app_logger.info(
                    f"CUDA device detected: {props.name} "
                    f"({props.total_memory / 1024**3:.1f} GB VRAM)"
                )
                return device
            except Exception as exc:
                app_logger.warning(f"Failed to initialize CUDA device: {exc}. Falling back to CPU.")

    if preference == "cuda":
        app_logger.warning("CUDA requested but not available — falling back to CPU.")

    app_logger.info("Using CPU device.")
    return torch.device("cpu")


def get_gpu_memory_info() -> Optional[dict[str, float]]:
    if not is_cuda_available():
        return None
    try:
        used = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return {
            "used_mb": round(used, 1),
            "reserved_mb": round(reserved, 1),
            "total_mb": round(total, 1),
            "free_mb": round(total - reserved, 1),
        }
    except Exception:
        return None


def get_ram_info() -> dict[str, float]:
    mem = psutil.virtual_memory()
    return {
        "used_mb": round(mem.used / 1024**2, 1),
        "total_mb": round(mem.total / 1024**2, 1),
        "available_mb": round(mem.available / 1024**2, 1),
        "percent": mem.percent,
    }


def clear_gpu_cache() -> None:
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        app_logger.debug("GPU cache cleared.")


def set_optimal_threads(n_threads: Optional[int] = None) -> None:
    if TORCH_AVAILABLE and torch is not None:
        if n_threads is None:
            n_threads = psutil.cpu_count(logical=False) or 4
        torch.set_num_threads(n_threads)
        app_logger.debug(f"PyTorch CPU threads set to {n_threads}.")


def get_system_info() -> dict[str, object]:
    info: dict[str, object] = {
        "pytorch_version": torch.__version__ if TORCH_AVAILABLE and torch else "not_installed",
        "cuda_available": is_cuda_available(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram": get_ram_info(),
    }
    if is_cuda_available() and torch is not None:
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = get_gpu_memory_info()
    return info


# ── SECTION 3: JSON Utilities ────────────────────────────────────────────────


def extract_json_block(text: str) -> Optional[str]:
    md_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    md_match = md_pattern.search(text)
    if md_match:
        return md_match.group(1).strip()

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
            if not in_string:
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[start_idx : i + 1]
    return None


def clean_control_characters(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


def _close_truncated_json(text: str) -> str:
    """Attempts to balance quotes and brackets on truncated JSON output."""
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in ("{", "["):
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

    res = text
    if in_string:
        res += '"'

    res = re.sub(r",\s*$", "", res)

    while stack:
        top = stack.pop()
        if top == "{":
            res += "}"
        elif top == "[":
            res += "]"

    return res


def repair_json(text: str) -> Optional[str]:
    cleaned = clean_control_characters(text)
    repaired = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        json.loads(repaired, strict=False)
        return repaired
    except json.JSONDecodeError:
        pass

    repaired = re.sub(r"(?<![\\])'", '"', repaired)
    try:
        json.loads(repaired, strict=False)
        return repaired
    except json.JSONDecodeError:
        pass

    # Try closing truncated JSON strings and brackets
    closed = _close_truncated_json(cleaned)
    closed_repaired = re.sub(r",\s*([}\]])", r"\1", closed)
    try:
        json.loads(closed_repaired, strict=False)
        return closed_repaired
    except json.JSONDecodeError:
        return None



def parse_json_safe(text: str) -> Optional[Any]:
    if not text or not text.strip():
        return None

    try:
        return json.loads(text.strip(), strict=False)
    except json.JSONDecodeError:
        pass

    extracted = extract_json_block(text)
    target = extracted if extracted is not None else text.strip()

    try:
        return json.loads(target, strict=False)
    except json.JSONDecodeError:
        pass

    repaired = repair_json(target)
    if repaired is not None:
        try:
            return json.loads(repaired, strict=False)
        except json.JSONDecodeError:
            pass

    generation_logger.warning("JSON parsing failed after extraction and auto-repair.")
    return None


def validate_entity_json(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if "entities" not in data or "relationships" not in data:
        return False
    if not isinstance(data["entities"], list):
        return False
    if not isinstance(data["relationships"], list):
        return False
    return True


# ── SECTION 4: Text Utilities ────────────────────────────────────────────────


def normalize_unicode(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    cleaned = "".join(
        ch for ch in normalized if not unicodedata.category(ch).startswith("C")
    )
    return cleaned


def clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_noise(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[-=_]{5,}", "", text)
    lines = [line for line in text.splitlines() if len(line.strip()) != 1]
    return "\n".join(lines)


def canonicalize_entity_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def count_tokens_approx(text: str) -> int:
    words = len(text.split())
    return int(words * 1.3)


def truncate_text(text: str, max_tokens: int, chars_per_token: int = 4) -> str:
    max_chars = max_tokens * chars_per_token
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def extract_sentences(text: str) -> list[str]:
    sentence_pattern = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=[A-Z])"
    )
    sentences = sentence_pattern.split(text)
    return [s.strip() for s in sentences if s.strip()]


def clean_document_text(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_noise(text)
    text = clean_whitespace(text)
    return text


# ── SECTION 5: Performance Timer ────────────────────────────────────────────


@contextmanager
def timer(label: str, log: bool = True) -> Generator[dict[str, float], None, None]:
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        result["elapsed_ms"] = elapsed_ms
        if log:
            performance_logger.info(f"{label}: {elapsed_ms:.2f} ms")


def timed(label: Optional[str] = None) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = label or fn.__qualname__
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                performance_logger.info(f"{name}: {elapsed_ms:.2f} ms")

        return wrapper

    return decorator


def async_timed(label: Optional[str] = None) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            name = label or fn.__qualname__
            start = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                performance_logger.info(f"{name}: {elapsed_ms:.2f} ms")

        return wrapper

    return decorator


# ── SECTION 6: Backward Compatibility Submodule Aliases ─────────────────────

for _mod in ["file_utils", "gpu", "json_utils", "text_utils", "timer"]:
    sys.modules[f"utils.{_mod}"] = sys.modules[__name__]
