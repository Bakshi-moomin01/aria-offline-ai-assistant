"""
utils.py — Utility helpers for the Offline AI Assistant.

Provides system-info collection, byte formatting, a timer context manager,
and directory-safety helpers used across the project.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

import psutil

from models import GpuInfo, SystemInfo

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# System information
# ──────────────────────────────────────────────

def get_system_info() -> SystemInfo:
    """Collect a snapshot of the current host hardware and OS details.

    Returns:
        A populated ``SystemInfo`` model.
    """
    uname = platform.uname()
    mem = psutil.virtual_memory()

    gpus: list[GpuInfo] = []
    try:
        import GPUtil  # type: ignore[import-untyped]
        for gpu in GPUtil.getGPUs():
            gpus.append(
                GpuInfo(
                    id=gpu.id,
                    name=gpu.name,
                    total_memory_mb=gpu.memoryTotal,
                    used_memory_mb=gpu.memoryUsed,
                    free_memory_mb=gpu.memoryFree,
                    load_percent=gpu.load * 100,
                    temperature_c=gpu.temperature,
                )
            )
    except Exception as exc:
        logger.warning("Could not query GPU info: %s", exc)

    cpu_name = platform.processor() or uname.machine
    try:
        import subprocess
        result = subprocess.run(
            ["wmic", "cpu", "get", "Name"],
            capture_output=True, text=True, timeout=5,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if len(lines) >= 2:
            cpu_name = lines[1]
    except Exception:
        pass  # Fall back to platform.processor()

    return SystemInfo(
        os=uname.system,
        os_version=uname.version,
        cpu=cpu_name,
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        ram_total_mb=mem.total / (1024 ** 2),
        ram_available_mb=mem.available / (1024 ** 2),
        ram_used_percent=mem.percent,
        gpus=gpus,
        python_version=sys.version,
        timestamp=datetime.utcnow(),
    )


def get_ram_usage_mb() -> float:
    """Return current process RSS in megabytes."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)


def get_vram_usage_mb() -> float | None:
    """Return total GPU memory used in megabytes (first GPU), or ``None``."""
    try:
        import GPUtil  # type: ignore[import-untyped]
        gpus = GPUtil.getGPUs()
        if gpus:
            return gpus[0].memoryUsed
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────

def format_bytes(num_bytes: int | float) -> str:
    """Human-readable byte string (e.g. ``'1.23 GB'``).

    Args:
        num_bytes: Raw byte count.

    Returns:
        Formatted string with appropriate unit.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


# ──────────────────────────────────────────────
# Timer context manager
# ──────────────────────────────────────────────

class TimerResult:
    """Mutable container populated by the ``timer`` context manager."""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.end: float = 0.0

    @property
    def elapsed_s(self) -> float:
        return self.end - self.start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000


@contextmanager
def timer() -> Generator[TimerResult, None, None]:
    """Context manager that records wall-clock elapsed time.

    Usage::

        with timer() as t:
            do_something()
        print(f"Took {t.elapsed_ms:.1f} ms")
    """
    result = TimerResult()
    result.start = time.perf_counter()
    try:
        yield result
    finally:
        result.end = time.perf_counter()


# ──────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────

def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Target directory path.

    Returns:
        The resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def unique_filename(directory: Path, prefix: str, suffix: str) -> Path:
    """Generate a timestamped filename that is (practically) unique.

    Args:
        directory: Folder for the file.
        prefix:    Name prefix (e.g. ``'benchmark'``).
        suffix:    File extension including the dot (e.g. ``'.json'``).

    Returns:
        Full ``Path`` to the new file.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return directory / f"{prefix}_{ts}{suffix}"
