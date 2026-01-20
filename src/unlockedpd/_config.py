"""Thread-safe configuration for unlockedpd.

This module provides the configuration system that controls unlockedpd behavior.
All configuration access is thread-safe using a lock.

Environment Variables:
    UNLOCKEDPD_ENABLED: Set to 'false' to disable all patches (default: 'true')
    UNLOCKEDPD_NUM_THREADS: Number of threads for Numba parallel operations (default: 0 = auto)
    UNLOCKEDPD_WARN_ON_FALLBACK: Set to 'true' to warn when falling back to pandas (default: 'false')
    UNLOCKEDPD_PARALLEL_THRESHOLD: Minimum array size for parallel execution (default: 10000)
"""
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

import numba


@dataclass
class UnlockedConfig:
    """Thread-safe configuration for unlockedpd.

    Uses a lock for all mutable attribute access to ensure
    thread-safety when config is modified from multiple threads.

    Attributes:
        enabled: If False, all patches bypass to original pandas methods
        num_threads: Number of threads for Numba (0 = auto/default)
        warn_on_fallback: If True, emit warnings when falling back to pandas
        cache_compiled: If True, cache Numba compiled functions
        parallel_threshold: Minimum array size before parallel execution is used
    """
    _enabled: bool = field(default=True, repr=False)
    _num_threads: int = field(default=0, repr=False)
    _warn_on_fallback: bool = field(default=False, repr=False)
    _cache_compiled: bool = field(default=True, repr=False)
    _parallel_threshold: int = field(default=10_000, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self):
        """Load configuration from environment variables."""
        self._enabled = os.environ.get('UNLOCKEDPD_ENABLED', 'true').lower() == 'true'

        threads_str = os.environ.get('UNLOCKEDPD_NUM_THREADS', '0')
        self._num_threads = int(threads_str) if threads_str.isdigit() else 0

        self._warn_on_fallback = os.environ.get('UNLOCKEDPD_WARN_ON_FALLBACK', 'false').lower() == 'true'

        threshold_str = os.environ.get('UNLOCKEDPD_PARALLEL_THRESHOLD', '10000')
        self._parallel_threshold = int(threshold_str) if threshold_str.isdigit() else 10_000

        # Apply thread config on initialization
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
            self._num_threads = int(value)
            if value > 0:
                numba.set_num_threads(value)

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
            self._parallel_threshold = int(value)

    def apply_thread_config(self) -> None:
        """Apply the current thread configuration to Numba."""
        with self._lock:
            if self._num_threads > 0:
                numba.set_num_threads(self._num_threads)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"UnlockedConfig(enabled={self._enabled}, "
                f"num_threads={self._num_threads}, "
                f"warn_on_fallback={self._warn_on_fallback}, "
                f"cache_compiled={self._cache_compiled}, "
                f"parallel_threshold={self._parallel_threshold})"
            )


# Global configuration instance
config = UnlockedConfig()
