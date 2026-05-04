"""Thread-safe configuration for unlockedpd.

This module provides the configuration system that controls unlockedpd behavior.
All configuration access is thread-safe using a lock.

Environment Variables:
    UNLOCKEDPD_ENABLED: Set to 'false' to disable all patches (default: 'true')
    UNLOCKEDPD_NUM_THREADS: Number of threads for Numba parallel operations (default: 0 = auto)
    UNLOCKEDPD_THREADPOOL_WORKERS: Python ThreadPool cap (default: 0/auto)
    UNLOCKEDPD_MAX_MEMORY_OVERHEAD: Max optimized RSS overhead ratio (default: 6.0)
    UNLOCKEDPD_MAX_CPU_OVERHEAD: Max optimized process CPU overhead ratio (default: 6.0)
    UNLOCKEDPD_WARMUP: Warmup policy: lazy/none/eager/full (default: lazy)
    UNLOCKEDPD_WARN_ON_FALLBACK: Set to 'true' to warn when falling back to pandas (default: 'false')
    UNLOCKEDPD_PARALLEL_THRESHOLD: Minimum array size for parallel execution (default: 10000)
    UNLOCKEDPD_MAX_MEMORY_OVERHEAD: Maximum optimized-vs-pandas RSS ratio budget (default: 6.0)
    UNLOCKEDPD_MAX_CPU_OVERHEAD: Maximum optimized-vs-pandas CPU seconds ratio budget (default: 6.0)
"""
import math
import os
import threading
from dataclasses import dataclass, field

import numba


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"", "auto", "none"}:
        return default
    try:
        return max(0, int(value))
    except ValueError:
        return default


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _parse_warmup(value: str | None) -> str:
    warmup = (value or "lazy").strip().lower()
    aliases = {
        "0": "none",
        "false": "none",
        "off": "none",
        "disabled": "none",
        "disable": "none",
        "1": "eager",
        "true": "eager",
        "on": "eager",
        "full": "eager",
        "auto": "lazy",
    }
    warmup = aliases.get(warmup, warmup)
    return warmup if warmup in {"none", "lazy", "eager"} else "lazy"


@dataclass
class UnlockedConfig:
    """Thread-safe configuration for unlockedpd.

    Attributes:
        enabled: If False, all patches bypass to original pandas methods.
        num_threads: Number of threads for Numba (0 = auto/default).
        threadpool_workers: Python ThreadPool cap (0 = adaptive auto).
        max_memory_overhead: Max optimized RSS overhead ratio before fallback.
        max_cpu_overhead: Max process CPU-seconds overhead ratio for profiling gates.
        warmup: Import warmup policy: ``lazy``/``none``/``eager``.
        warn_on_fallback: If True, emit warnings when falling back to pandas.
        cache_compiled: If True, cache Numba compiled functions.
        parallel_threshold: Minimum array size before parallel execution is used.
    """
    _enabled: bool = field(default=True, repr=False)
    _num_threads: int = field(default=0, repr=False)
    _threadpool_workers: int = field(default=0, repr=False)
    _max_memory_overhead: float = field(default=6.0, repr=False)
    _max_cpu_overhead: float = field(default=6.0, repr=False)
    _warmup: str = field(default="lazy", repr=False)
    _warn_on_fallback: bool = field(default=False, repr=False)
    _cache_compiled: bool = field(default=True, repr=False)
    _parallel_threshold: int = field(default=10_000, repr=False)
    _max_memory_overhead: float = field(default=6.0, repr=False)
    _max_cpu_overhead: float = field(default=6.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self):
        """Load configuration from environment variables."""
        self._enabled = _parse_bool(os.environ.get('UNLOCKEDPD_ENABLED'), True)
        self._num_threads = _parse_int(os.environ.get('UNLOCKEDPD_NUM_THREADS'), 0)
        self._threadpool_workers = _parse_int(os.environ.get('UNLOCKEDPD_THREADPOOL_WORKERS'), 0)
        self._max_memory_overhead = _parse_float(os.environ.get('UNLOCKEDPD_MAX_MEMORY_OVERHEAD'), 6.0)
        self._max_cpu_overhead = _parse_float(os.environ.get('UNLOCKEDPD_MAX_CPU_OVERHEAD'), 6.0)
        self._warmup = _parse_warmup(os.environ.get('UNLOCKEDPD_WARMUP'))
        self._warn_on_fallback = _parse_bool(os.environ.get('UNLOCKEDPD_WARN_ON_FALLBACK'), False)
        self._cache_compiled = _parse_bool(os.environ.get('UNLOCKEDPD_CACHE_COMPILED'), True)
        self._parallel_threshold = _parse_int(os.environ.get('UNLOCKEDPD_PARALLEL_THRESHOLD'), 10_000)

        # Apply Numba thread config on initialization. This remains Numba-only.
        if self._num_threads > 0:
            numba.set_num_threads(self._num_threads)

    @property
    def enabled(self) -> bool:
        """Whether unlockedpd optimizations are enabled."""
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = bool(value)

    @property
    def num_threads(self) -> int:
        """Number of threads for Numba parallel operations (0 = auto)."""
        with self._lock:
            return self._num_threads

    @num_threads.setter
    def num_threads(self, value: int) -> None:
        with self._lock:
            self._num_threads = max(0, int(value))
            if self._num_threads > 0:
                numba.set_num_threads(self._num_threads)

    @property
    def threadpool_workers(self) -> int:
        """Python ThreadPool worker cap (0 = adaptive auto)."""
        with self._lock:
            return self._threadpool_workers

    @threadpool_workers.setter
    def threadpool_workers(self, value: int | str | None) -> None:
        with self._lock:
            self._threadpool_workers = _parse_int(str(value) if value is not None else None, 0)

    @property
    def max_memory_overhead(self) -> float:
        """Maximum RSS overhead ratio allowed before estimated fallback."""
        with self._lock:
            return self._max_memory_overhead

    @max_memory_overhead.setter
    def max_memory_overhead(self, value: float) -> None:
        with self._lock:
            parsed = float(value)
            if parsed <= 0:
                raise ValueError("max_memory_overhead must be positive")
            self._max_memory_overhead = parsed

    @property
    def max_cpu_overhead(self) -> float:
        """Maximum process CPU-seconds overhead ratio used by profilers/gates."""
        with self._lock:
            return self._max_cpu_overhead

    @max_cpu_overhead.setter
    def max_cpu_overhead(self, value: float) -> None:
        with self._lock:
            parsed = float(value)
            if parsed <= 0:
                raise ValueError("max_cpu_overhead must be positive")
            self._max_cpu_overhead = parsed

    @property
    def warmup(self) -> str:
        """Warmup policy: 'lazy'/'none' avoid import-time full warmup; 'eager' runs it."""
        with self._lock:
            return self._warmup

    @warmup.setter
    def warmup(self, value: str) -> None:
        with self._lock:
            self._warmup = _parse_warmup(value)

    @property
    def warn_on_fallback(self) -> bool:
        """Whether to warn when falling back to original pandas."""
        with self._lock:
            return self._warn_on_fallback

    @warn_on_fallback.setter
    def warn_on_fallback(self, value: bool) -> None:
        with self._lock:
            self._warn_on_fallback = bool(value)

    @property
    def cache_compiled(self) -> bool:
        """Whether to cache Numba compiled functions."""
        with self._lock:
            return self._cache_compiled

    @cache_compiled.setter
    def cache_compiled(self, value: bool) -> None:
        with self._lock:
            self._cache_compiled = bool(value)

    @property
    def parallel_threshold(self) -> int:
        """Minimum array size before parallel execution is used."""
        with self._lock:
            return self._parallel_threshold

    @parallel_threshold.setter
    def parallel_threshold(self, value: int) -> None:
        with self._lock:
            self._parallel_threshold = max(0, int(value))

    @property
    def max_memory_overhead(self) -> float:
        """Maximum optimized-vs-pandas RSS ratio budget."""
        with self._lock:
            return self._max_memory_overhead

    @max_memory_overhead.setter
    def max_memory_overhead(self, value: float) -> None:
        with self._lock:
            self._max_memory_overhead = _coerce_positive_float(value, "max_memory_overhead")

    @property
    def max_cpu_overhead(self) -> float:
        """Maximum optimized-vs-pandas CPU seconds ratio budget."""
        with self._lock:
            return self._max_cpu_overhead

    @max_cpu_overhead.setter
    def max_cpu_overhead(self, value: float) -> None:
        with self._lock:
            self._max_cpu_overhead = _coerce_positive_float(value, "max_cpu_overhead")

    def apply_thread_config(self) -> None:
        """Apply the current Numba thread configuration."""
        with self._lock:
            if self._num_threads > 0:
                numba.set_num_threads(self._num_threads)

    def as_dict(self) -> dict:
        """Return a thread-safe snapshot for profiling/reporting."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "num_threads": self._num_threads,
                "threadpool_workers": self._threadpool_workers,
                "warn_on_fallback": self._warn_on_fallback,
                "cache_compiled": self._cache_compiled,
                "parallel_threshold": self._parallel_threshold,
                "max_memory_overhead": self._max_memory_overhead,
                "max_cpu_overhead": self._max_cpu_overhead,
            }

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"UnlockedConfig(enabled={self._enabled}, "
                f"num_threads={self._num_threads}, "
                f"threadpool_workers={self._threadpool_workers or 'auto'}, "
                f"max_memory_overhead={self._max_memory_overhead}, "
                f"max_cpu_overhead={self._max_cpu_overhead}, "
                f"warmup='{self._warmup}', "
                f"warn_on_fallback={self._warn_on_fallback}, "
                f"cache_compiled={self._cache_compiled}, "
                f"parallel_threshold={self._parallel_threshold}, "
                f"max_memory_overhead={self._max_memory_overhead}, "
                f"max_cpu_overhead={self._max_cpu_overhead})"
            )


# Global configuration instance
config = UnlockedConfig()
