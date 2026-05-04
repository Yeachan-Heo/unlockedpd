"""Helpers for bounded ThreadPool fan-out in operation modules."""

import os
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from typing import List, Tuple


DEFAULT_THREADPOOL_WORKER_CAP = 32
AUTO_VALUES = {"", "0", "auto", "none", "default"}


def _coerce_positive_int(value) -> int:
    """Return a positive integer value, or 0 when unset/auto/invalid."""
    if value is None:
        return 0

    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in AUTO_VALUES:
            return 0
        try:
            parsed = int(stripped)
        except ValueError:
            return 0
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0

    return parsed if parsed > 0 else 0


def _configured_threadpool_workers() -> int:
    """Read the runtime/env ThreadPool cap without making config mandatory."""
    configured = 0
    try:
        from .._config import config

        configured = _coerce_positive_int(getattr(config, "threadpool_workers", 0))
    except Exception:
        configured = 0

    if configured:
        return configured

    return _coerce_positive_int(os.environ.get("UNLOCKEDPD_THREADPOOL_WORKERS"))


def resolve_threadpool_workers(
    work_units: int,
    *,
    operation_cap: int = DEFAULT_THREADPOOL_WORKER_CAP,
) -> int:
    """Resolve useful ThreadPool workers for a single operation call.

    Resolution intentionally caps workers by available work units so wide/large
    operations keep parallel paths while narrow shapes do not fan out idle
    Python workers.  When worker-2's config/resource lane adds
    ``config.threadpool_workers`` this helper will honor it; until then it
    supports ``UNLOCKEDPD_THREADPOOL_WORKERS`` directly.
    """
    units = max(1, int(work_units))
    cpu_count = os.cpu_count() or 8
    configured = _configured_threadpool_workers()

    candidates = [units, cpu_count, max(1, int(operation_cap))]
    if configured:
        candidates.append(configured)

    return max(1, min(candidates))


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
