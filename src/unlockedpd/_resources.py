"""Resource policy helpers for unlockedpd optimized paths.

This module keeps Python ThreadPool fan-out separate from Numba thread control
and provides lightweight memory-budget guards that can trigger pandas fallback
through the normal patch wrapper.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ._config import config

_AUTO_MEMORY_BANDWIDTH_CAP = 8
_PAIRWISE_MEMORY_BANDWIDTH_CAP = 4
_last_selected_path = threading.local()


class ResourceBudgetExceeded(RuntimeError):
    """Raised when an optimized path should fall back to pandas for resources."""


@dataclass(frozen=True)
class MemoryEstimate:
    """Estimated operation memory pressure in bytes."""

    baseline_bytes: int
    optimized_bytes: int
    ratio: float


def cpu_count() -> int:
    """Return logical CPU count with a conservative fallback."""
    return os.cpu_count() or 8


def get_last_selected_path() -> str | None:
    """Return the last optimized-path label recorded in this thread."""
    return getattr(_last_selected_path, "value", None)


def set_last_selected_path(path: str) -> None:
    """Record an optimized-path label for profilers and diagnostics."""
    _last_selected_path.value = path


def _operation_cap(operation: str | None) -> int:
    if operation and "pairwise" in operation:
        return _PAIRWISE_MEMORY_BANDWIDTH_CAP
    return _AUTO_MEMORY_BANDWIDTH_CAP


def resolve_threadpool_workers(
    work_units: int,
    *,
    operation: str | None = None,
    cap: int | None = None,
    min_workers: int = 1,
) -> int:
    """Resolve Python ThreadPool worker count for a specific operation.

    Precedence:
    1. explicit ``config.threadpool_workers`` / runtime assignment;
    2. ``UNLOCKEDPD_THREADPOOL_WORKERS`` parsed by config at initialization;
    3. adaptive auto cap bounded by CPU count, work units, and a memory-bandwidth cap.
    """
    units = max(1, int(work_units or 1))
    configured = int(config.threadpool_workers or 0)
    cpu_cap = max(1, cpu_count())
    if configured > 0:
        resolved_cap = configured
    else:
        resolved_cap = _operation_cap(operation)
    if cap is not None and int(cap) > 0:
        resolved_cap = min(resolved_cap, int(cap))
    workers = min(units, cpu_cap, max(1, resolved_cap))
    return max(int(min_workers), workers)


def threadpool_chunks(work_units: int, *, operation: str | None = None, cap: int | None = None) -> tuple[int, list[tuple[int, int]]]:
    """Return ``(workers, ranges)`` for chunking work units across a ThreadPool."""
    workers = resolve_threadpool_workers(work_units, operation=operation, cap=cap)
    chunk_size = max(1, math.ceil(max(1, int(work_units)) / workers))
    chunks = [
        (start, min(start + chunk_size, int(work_units)))
        for start in range(0, int(work_units), chunk_size)
    ]
    return min(workers, len(chunks)), chunks


def array_nbytes(shape: Sequence[int], dtype=np.float64) -> int:
    """Estimate NumPy array bytes for ``shape`` and ``dtype``."""
    itemsize = np.dtype(dtype).itemsize
    total = itemsize
    for dim in shape:
        total *= max(0, int(dim))
    return int(total)


def dataframe_like_bytes(n_rows: int, n_cols: int, dtype=np.float64) -> int:
    """Estimate the data-buffer size of a numeric DataFrame-like value."""
    return array_nbytes((n_rows, n_cols), dtype)


def pairwise_rolling_memory_estimate(n_rows: int, n_cols: int, dtype=np.float64) -> MemoryEstimate:
    """Estimate pairwise rolling corr/cov memory including duplicate buffers.

    Pandas-compatible output is intrinsically O(rows * cols^2). The optimized
    implementation historically also materialized a flat upper-triangle buffer;
    the estimate intentionally includes both to guard avoidable pressure.
    """
    n_pairs = n_cols * (n_cols + 1) // 2
    itemsize = np.dtype(dtype).itemsize
    input_bytes = n_rows * n_cols * itemsize
    pandas_output_bytes = n_rows * n_cols * n_cols * itemsize
    flat_bytes = n_rows * n_pairs * itemsize
    optimized_bytes = input_bytes + pandas_output_bytes + flat_bytes
    baseline_bytes = max(1, input_bytes + pandas_output_bytes)
    ratio = optimized_bytes / baseline_bytes
    return MemoryEstimate(baseline_bytes=baseline_bytes, optimized_bytes=optimized_bytes, ratio=ratio)


def simple_result_memory_estimate(n_rows: int, n_cols: int, *, intermediates: int = 1, dtype=np.float64) -> MemoryEstimate:
    """Estimate memory for operations whose output shape matches input shape."""
    data_bytes = dataframe_like_bytes(n_rows, n_cols, dtype)
    baseline_bytes = max(1, data_bytes * 2)  # input + pandas output
    optimized_bytes = max(1, data_bytes * (2 + max(0, intermediates)))
    return MemoryEstimate(
        baseline_bytes=baseline_bytes,
        optimized_bytes=optimized_bytes,
        ratio=optimized_bytes / baseline_bytes,
    )


def assert_memory_budget(estimate: MemoryEstimate, *, operation: str) -> None:
    """Raise ``ResourceBudgetExceeded`` when estimated memory ratio is over budget."""
    max_overhead = float(config.max_memory_overhead)
    if estimate.ratio > max_overhead:
        set_last_selected_path("fallback")
        raise ResourceBudgetExceeded(
            f"{operation} estimated RSS overhead {estimate.ratio:.2f}x exceeds "
            f"configured max_memory_overhead={max_overhead:.2f}x"
        )


def use_threadpool_path(work_units: int, *, operation: str | None = None) -> tuple[int, list[tuple[int, int]]]:
    """Record and return ThreadPool plan for an optimized parallel path."""
    workers, chunks = threadpool_chunks(work_units, operation=operation)
    set_last_selected_path("threadpool" if workers > 1 else "serial_numba")
    return workers, chunks


def record_dispatch_path(path: str):
    """Decorator-style helper for statement-like path recording."""
    set_last_selected_path(path)
    return None
