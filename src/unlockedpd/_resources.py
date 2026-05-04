"""Shared resource policy helpers for unlockedpd optimized paths.

The config layer owns user/runtime knobs; this module turns those knobs into
bounded resource decisions that operation modules can share without duplicating
ThreadPool or memory-budget policy.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


DEFAULT_MEMORY_BANDWIDTH_WORKER_CAP = 8
PAIRWISE_MEMORY_BANDWIDTH_WORKER_CAP = 4

ShapeLike = Union[int, Sequence[int]]
_last_selected_path = threading.local()


class ResourceBudgetExceeded(RuntimeError):
    """Raised when an optimized path would exceed configured resource budgets."""


@dataclass(frozen=True)
class MemoryEstimate:
    """Memory estimate for a candidate optimized path."""

    baseline_bytes: int
    optimized_bytes: int
    ratio: float


@dataclass(frozen=True)
class ResourceBudgetDecision:
    """Result from checking observed resource ratios against config budgets."""

    pass_budget: bool
    memory_overhead: Optional[float]
    cpu_overhead: Optional[float]
    max_memory_overhead: float
    max_cpu_overhead: float


def _positive_int_or_none(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def logical_cpu_count() -> int:
    """Return logical CPU count with a conservative fallback."""

    return max(1, os.cpu_count() or 1)


def cpu_count() -> int:
    """Backward-compatible alias used by tests and older call sites."""

    return logical_cpu_count()


def get_last_selected_path() -> Optional[str]:
    """Return the last optimized-path label recorded in this thread."""

    return getattr(_last_selected_path, "value", None)


def set_last_selected_path(path: str) -> None:
    """Record an optimized-path label for profilers and diagnostics."""

    _last_selected_path.value = path


def _operation_cap(operation: Optional[str]) -> int:
    operation_text = (operation or "").lower()
    if "pairwise" in operation_text:
        return PAIRWISE_MEMORY_BANDWIDTH_WORKER_CAP
    return DEFAULT_MEMORY_BANDWIDTH_WORKER_CAP


def resolve_threadpool_workers(
    work_units: Optional[int] = None,
    *,
    operation: Optional[str] = None,
    configured_workers: Optional[int] = None,
    operation_cap: Optional[int] = None,
    memory_bandwidth_cap: Optional[int] = None,
    cap: Optional[int] = None,
    min_workers: int = 1,
) -> int:
    """Resolve a bounded ThreadPool worker count.

    ``config.threadpool_workers`` is a cap for Python ThreadPool paths only;
    it intentionally does not affect ``config.num_threads`` / Numba threads.
    Adaptive mode (0/auto) considers logical CPUs, work units, and
    operation-specific memory-bandwidth caps.
    """

    from ._config import config

    configured_cap = (
        config.threadpool_workers
        if configured_workers is None
        else max(0, int(configured_workers))
    )
    op_cap = _operation_cap(operation) if operation_cap is None else operation_cap
    mem_cap = (
        DEFAULT_MEMORY_BANDWIDTH_WORKER_CAP
        if memory_bandwidth_cap is None
        else memory_bandwidth_cap
    )

    candidates = [logical_cpu_count()]
    for candidate in (
        configured_cap,
        _positive_int_or_none(work_units),
        _positive_int_or_none(op_cap),
        _positive_int_or_none(mem_cap),
        _positive_int_or_none(cap),
    ):
        if candidate is not None and candidate > 0:
            candidates.append(candidate)

    selected = max(1, min(candidates))
    return max(1, int(min_workers), selected)


def threadpool_chunks(
    work_units: int,
    *,
    operation: Optional[str] = None,
    cap: Optional[int] = None,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Return ``(workers, ranges)`` for chunking work units across a ThreadPool."""

    units = max(0, int(work_units))
    workers = resolve_threadpool_workers(units or 1, operation=operation, cap=cap)
    if units == 0:
        return workers, []

    chunk_size = max(1, math.ceil(units / workers))
    chunks = [
        (start, min(start + chunk_size, units)) for start in range(0, units, chunk_size)
    ]
    return max(1, min(workers, len(chunks))), chunks


def estimate_array_nbytes(
    shape_or_array: Union[ShapeLike, np.ndarray],
    *,
    dtype: Union[str, np.dtype, type] = np.float64,
    itemsize: Optional[int] = None,
) -> int:
    """Estimate bytes for an ndarray or shape."""

    nbytes = getattr(shape_or_array, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)

    if isinstance(shape_or_array, int):
        elements = shape_or_array
    else:
        elements = math.prod(int(dim) for dim in shape_or_array)

    bytes_per_element = (
        int(itemsize) if itemsize is not None else np.dtype(dtype).itemsize
    )
    return max(0, int(elements) * bytes_per_element)


def estimate_operation_nbytes(
    shape_or_array: Union[ShapeLike, np.ndarray],
    *,
    dtype: Union[str, np.dtype, type] = np.float64,
    inputs: int = 1,
    outputs: int = 1,
    intermediates: int = 0,
) -> int:
    """Estimate bytes retained by a simple array operation."""

    base = estimate_array_nbytes(shape_or_array, dtype=dtype)
    copies = max(0, int(inputs)) + max(0, int(outputs)) + max(0, int(intermediates))
    return base * copies


def overhead_ratio(optimized: float, baseline: float) -> float:
    """Return an optimized/baseline ratio with safe zero handling."""

    if baseline <= 0:
        return math.inf if optimized > 0 else 1.0
    return optimized / baseline


def check_resource_budget(
    *,
    memory_overhead: Optional[float] = None,
    cpu_overhead: Optional[float] = None,
    max_memory_overhead: Optional[float] = None,
    max_cpu_overhead: Optional[float] = None,
) -> ResourceBudgetDecision:
    """Check resource ratios against explicit or global config budgets."""

    from ._config import config

    memory_budget = (
        config.max_memory_overhead
        if max_memory_overhead is None
        else float(max_memory_overhead)
    )
    cpu_budget = (
        config.max_cpu_overhead if max_cpu_overhead is None else float(max_cpu_overhead)
    )

    pass_budget = True
    if memory_overhead is not None and memory_overhead > memory_budget:
        pass_budget = False
    if cpu_overhead is not None and cpu_overhead > cpu_budget:
        pass_budget = False

    return ResourceBudgetDecision(
        pass_budget=pass_budget,
        memory_overhead=memory_overhead,
        cpu_overhead=cpu_overhead,
        max_memory_overhead=memory_budget,
        max_cpu_overhead=cpu_budget,
    )


def simple_result_memory_estimate(
    n_rows: int,
    n_cols: int,
    *,
    dtype: Union[str, np.dtype, type] = np.float64,
    intermediates: int = 1,
) -> MemoryEstimate:
    """Estimate RSS overhead for simple DataFrame-in/DataFrame-out paths."""

    input_bytes = estimate_array_nbytes((n_rows, n_cols), dtype=dtype)
    output_bytes = estimate_array_nbytes((n_rows, n_cols), dtype=dtype)
    baseline_bytes = input_bytes + output_bytes
    optimized_bytes = input_bytes + output_bytes * (1 + max(0, int(intermediates)))
    return MemoryEstimate(
        baseline_bytes=baseline_bytes,
        optimized_bytes=optimized_bytes,
        ratio=overhead_ratio(optimized_bytes, baseline_bytes),
    )


def pairwise_rolling_memory_estimate(
    n_rows: int,
    n_cols: int,
    *,
    dtype: Union[str, np.dtype, type] = np.float64,
    input_copy_bytes: int = 0,
) -> MemoryEstimate:
    """Estimate RSS overhead for pairwise rolling corr/cov."""

    input_bytes = estimate_array_nbytes((n_rows, n_cols), dtype=dtype)
    output_bytes = estimate_array_nbytes((n_rows, n_cols, n_cols), dtype=dtype)
    n_pairs = int(n_cols) * (int(n_cols) + 1) // 2
    pair_index_bytes = n_pairs * 2 * np.dtype(np.int64).itemsize
    baseline_bytes = input_bytes + output_bytes
    optimized_bytes = baseline_bytes + max(0, int(input_copy_bytes)) + pair_index_bytes
    return MemoryEstimate(
        baseline_bytes=baseline_bytes,
        optimized_bytes=optimized_bytes,
        ratio=overhead_ratio(optimized_bytes, baseline_bytes),
    )


def assert_memory_budget(estimate: MemoryEstimate, *, operation: str) -> None:
    """Raise when an estimated optimized path is over the RSS budget."""

    from ._config import config

    max_overhead = float(config.max_memory_overhead)
    if estimate.ratio > max_overhead:
        set_last_selected_path("fallback")
        raise ResourceBudgetExceeded(
            f"{operation} estimated RSS overhead {estimate.ratio:.2f}x exceeds "
            f"configured max_memory_overhead={max_overhead:.2f}x"
        )


def use_threadpool_path(
    work_units: int,
    *,
    operation: Optional[str] = None,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Record and return ThreadPool plan for an optimized parallel path."""

    workers, chunks = threadpool_chunks(work_units, operation=operation)
    set_last_selected_path("threadpool" if workers > 1 else "serial_numba")
    return workers, chunks


def record_dispatch_path(path: str) -> None:
    """Record a dispatch-path label for profiler diagnostics."""

    set_last_selected_path(path)


__all__ = [
    "DEFAULT_MEMORY_BANDWIDTH_WORKER_CAP",
    "PAIRWISE_MEMORY_BANDWIDTH_WORKER_CAP",
    "MemoryEstimate",
    "ResourceBudgetDecision",
    "ResourceBudgetExceeded",
    "assert_memory_budget",
    "check_resource_budget",
    "cpu_count",
    "estimate_array_nbytes",
    "estimate_operation_nbytes",
    "get_last_selected_path",
    "logical_cpu_count",
    "overhead_ratio",
    "pairwise_rolling_memory_estimate",
    "record_dispatch_path",
    "resolve_threadpool_workers",
    "set_last_selected_path",
    "simple_result_memory_estimate",
    "threadpool_chunks",
    "use_threadpool_path",
]
