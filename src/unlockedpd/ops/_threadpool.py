"""Thin chunk helpers for bounded ThreadPool fan-out in operation modules."""
from __future__ import annotations

import math
from typing import List, Tuple

from .._resources import resolve_threadpool_workers as _resolve_policy_workers


DEFAULT_THREADPOOL_WORKER_CAP = 32


def resolve_threadpool_workers(
    work_units: int,
    *,
    operation_cap: int = DEFAULT_THREADPOOL_WORKER_CAP,
) -> int:
    """Resolve ThreadPool workers through the shared resource policy."""

    return _resolve_policy_workers(
        work_units=max(1, int(work_units)),
        operation="threadpool",
        operation_cap=operation_cap,
    )


def make_threadpool_chunks(
    work_units: int,
    *,
    operation_cap: int = DEFAULT_THREADPOOL_WORKER_CAP,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Return ``(worker_count, chunks)`` for contiguous work-unit ranges."""

    units = max(0, int(work_units))
    workers = resolve_threadpool_workers(units or 1, operation_cap=operation_cap)
    if units == 0:
        return workers, []

    chunk_size = max(1, math.ceil(units / workers))
    chunks = [
        (start, min(start + chunk_size, units))
        for start in range(0, units, chunk_size)
    ]
    return max(1, min(workers, len(chunks))), chunks
