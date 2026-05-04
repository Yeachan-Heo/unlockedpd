"""Helpers for bounded ThreadPool fan-out in operation modules."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Tuple


DEFAULT_THREADPOOL_WORKER_CAP = 32


def _fallback_threadpool_workers(work_units: int, operation_cap: int) -> int:
    """Resolve workers before the shared resource helper is integrated."""
    units = max(1, int(work_units))
    cpu_count = os.cpu_count() or 8
    cap = max(1, int(operation_cap))
    return max(1, min(units, cpu_count, cap))


def _resource_threadpool_workers(work_units: int, operation_cap: int) -> int:
    """Delegate worker-count policy to the shared resource helper when present."""
    try:
        from .._resources import resolve_threadpool_workers as resolve_workers
    except ImportError:
        return _fallback_threadpool_workers(work_units, operation_cap)

    return resolve_workers(work_units=work_units, operation_cap=operation_cap)


def resolve_threadpool_workers(
    work_units: int,
    *,
    operation_cap: int = DEFAULT_THREADPOOL_WORKER_CAP,
) -> int:
    """Resolve useful ThreadPool workers for a single operation call.

    Resolution intentionally caps workers by available work units so wide/large
    operations keep parallel paths while narrow shapes do not fan out idle
    Python workers.  Config/resource-budget policy lives in
    ``unlockedpd._resources``; this ops helper only owns chunk construction and
    ThreadPool execution.
    """
    return _resource_threadpool_workers(work_units, operation_cap)


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

    chunk_size = max(1, (units + workers - 1) // workers)
    chunks = [
        (i * chunk_size, min((i + 1) * chunk_size, units))
        for i in range(workers)
        if i * chunk_size < units
    ]
    return max(1, len(chunks)), chunks


def run_threadpool_chunks(
    work_units: int,
    process_chunk: Callable[[Tuple[int, int]], None],
    *,
    operation_cap: int = DEFAULT_THREADPOOL_WORKER_CAP,
) -> None:
    """Run ``process_chunk`` across bounded contiguous work chunks."""
    workers, chunks = make_threadpool_chunks(work_units, operation_cap=operation_cap)
    if not chunks:
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_chunk, chunks))
