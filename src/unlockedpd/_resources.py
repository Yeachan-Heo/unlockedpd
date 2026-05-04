"""Shared resource policy helpers for unlockedpd optimized paths.

The config layer owns user/runtime knobs; this module turns those knobs into
bounded resource decisions that operation modules can share without importing
ThreadPool constants from each other.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np


DEFAULT_MEMORY_BANDWIDTH_WORKER_CAP = 32


ShapeLike = Union[int, Sequence[int]]


def _positive_int_or_none(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def logical_cpu_count() -> int:
    """Return a safe logical CPU count for resource decisions."""
    return max(1, os.cpu_count() or 1)


def resolve_threadpool_workers(
    *,
    work_units: Optional[int] = None,
    configured_workers: Optional[int] = None,
    operation_cap: Optional[int] = None,
    memory_bandwidth_cap: int = DEFAULT_MEMORY_BANDWIDTH_WORKER_CAP,
) -> int:
    """Resolve a bounded ThreadPool worker count.

    Precedence is cap-based rather than exact: a runtime/env
    ``config.threadpool_workers`` value limits Python ThreadPool fan-out, while
    Numba ``config.num_threads`` remains independent.

    The selected worker count considers:
    - configured cap (runtime assignment beats env because it updates config);
    - logical CPU count;
    - operation work units such as columns or pairs;
    - a memory-bandwidth or operation-specific cap.
    """
    from ._config import config

    configured_cap = (
        config.threadpool_workers
        if configured_workers is None
        else max(0, int(configured_workers))
    )

    caps = [logical_cpu_count()]
    for cap in (
        configured_cap,
        _positive_int_or_none(work_units),
        _positive_int_or_none(operation_cap),
        _positive_int_or_none(memory_bandwidth_cap),
    ):
        if cap is not None and cap > 0:
            caps.append(cap)

    return max(1, min(caps))


def estimate_array_nbytes(
    shape_or_array: Union[ShapeLike, np.ndarray],
    *,
    dtype: Union[str, np.dtype, type] = np.float64,
    itemsize: Optional[int] = None,
) -> int:
    """Estimate bytes for an ndarray or shape.

    Existing arrays use their actual ``nbytes``. Shape estimates default to
    float64 to match the numeric coercion used by most optimized paths.
    """
    nbytes = getattr(shape_or_array, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)

    if isinstance(shape_or_array, int):
        elements = shape_or_array
    else:
        elements = math.prod(int(dim) for dim in shape_or_array)

    bytes_per_element = int(itemsize) if itemsize is not None else np.dtype(dtype).itemsize
    return max(0, int(elements) * bytes_per_element)


def estimate_operation_nbytes(
    shape_or_array: Union[ShapeLike, np.ndarray],
    *,
    dtype: Union[str, np.dtype, type] = np.float64,
    inputs: int = 1,
    outputs: int = 1,
    extra_bytes: int = 0,
) -> int:
    """Estimate total bytes touched by an operation's inputs/outputs."""
    base = estimate_array_nbytes(shape_or_array, dtype=dtype)
    return base * (int(inputs) + int(outputs)) + max(0, int(extra_bytes))


def overhead_ratio(optimized: float, baseline: float) -> float:
    """Return an optimized/baseline ratio with safe zero handling."""
    if baseline <= 0:
        return math.inf if optimized > 0 else 1.0
    return optimized / baseline


@dataclass(frozen=True)
class ResourceBudgetDecision:
    """Result from checking observed resource ratios against config budgets."""

    pass_budget: bool
    memory_overhead: Optional[float]
    cpu_overhead: Optional[float]
    max_memory_overhead: float
    max_cpu_overhead: float


def check_resource_budget(
    *,
    memory_overhead: Optional[float] = None,
    cpu_overhead: Optional[float] = None,
    max_memory_overhead: Optional[float] = None,
    max_cpu_overhead: Optional[float] = None,
) -> ResourceBudgetDecision:
    """Check resource ratios against explicit or global config budgets."""
    from ._config import config

    memory_budget = config.max_memory_overhead if max_memory_overhead is None else float(max_memory_overhead)
    cpu_budget = config.max_cpu_overhead if max_cpu_overhead is None else float(max_cpu_overhead)

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


__all__ = [
    "DEFAULT_MEMORY_BANDWIDTH_WORKER_CAP",
    "ResourceBudgetDecision",
    "check_resource_budget",
    "estimate_array_nbytes",
    "estimate_operation_nbytes",
    "logical_cpu_count",
    "overhead_ratio",
    "resolve_threadpool_workers",
]
