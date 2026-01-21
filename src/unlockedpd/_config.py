"""Thread-safe configuration for unlockedpd.

This module provides the configuration system that controls unlockedpd behavior.
All configuration access is thread-safe using a lock.

Environment Variables:
    UNLOCKEDPD_ENABLED: Set to 'false' to disable all patches (default: 'true')
    UNLOCKEDPD_NUM_THREADS: Number of threads for Numba parallel operations (default: 0 = auto)
    UNLOCKEDPD_WARN_ON_FALLBACK: Set to 'true' to warn when falling back to pandas (default: 'false')
    UNLOCKEDPD_PARALLEL_THRESHOLD: Minimum array size for parallel execution (default: 10000)
    UNLOCKEDPD_IO_ENABLED: Set to 'false' to disable IO optimizations (default: 'true')
    UNLOCKEDPD_IO_WORKERS: Number of workers for parallel IO (default: 0 = auto)
    UNLOCKEDPD_CSV_THRESHOLD_MB: Minimum CSV file size in MB for parallel reading (default: 50)
    UNLOCKEDPD_PARQUET_THRESHOLD_MB: Minimum Parquet file size in MB for parallel reading (default: 20)
    UNLOCKEDPD_EXCEL_THRESHOLD_MB: Minimum Excel file size in MB for parallel reading (default: 10)
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
    _io_enabled: bool = field(default=True, repr=False)
    _io_workers: int = field(default=0, repr=False)  # 0 = auto
    _csv_threshold_mb: int = field(default=50, repr=False)
    _parquet_threshold_mb: int = field(default=20, repr=False)
    _excel_threshold_mb: int = field(default=10, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self):
        """Load configuration from environment variables."""
        self._enabled = os.environ.get('UNLOCKEDPD_ENABLED', 'true').lower() == 'true'

        threads_str = os.environ.get('UNLOCKEDPD_NUM_THREADS', '0')
        self._num_threads = int(threads_str) if threads_str.isdigit() else 0

        self._warn_on_fallback = os.environ.get('UNLOCKEDPD_WARN_ON_FALLBACK', 'false').lower() == 'true'

        threshold_str = os.environ.get('UNLOCKEDPD_PARALLEL_THRESHOLD', '10000')
        self._parallel_threshold = int(threshold_str) if threshold_str.isdigit() else 10_000

        # IO configuration
        self._io_enabled = os.environ.get('UNLOCKEDPD_IO_ENABLED', 'true').lower() == 'true'

        io_workers_str = os.environ.get('UNLOCKEDPD_IO_WORKERS', '0')
        self._io_workers = int(io_workers_str) if io_workers_str.isdigit() else 0

        csv_thresh = os.environ.get('UNLOCKEDPD_CSV_THRESHOLD_MB', '50')
        self._csv_threshold_mb = int(csv_thresh) if csv_thresh.isdigit() else 50

        parquet_thresh = os.environ.get('UNLOCKEDPD_PARQUET_THRESHOLD_MB', '20')
        self._parquet_threshold_mb = int(parquet_thresh) if parquet_thresh.isdigit() else 20

        excel_thresh = os.environ.get('UNLOCKEDPD_EXCEL_THRESHOLD_MB', '10')
        self._excel_threshold_mb = int(excel_thresh) if excel_thresh.isdigit() else 10

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

    @property
    def io_enabled(self) -> bool:
        """Whether IO optimizations are enabled."""
        with self._lock:
            return self._io_enabled and self._enabled

    @io_enabled.setter
    def io_enabled(self, value: bool) -> None:
        with self._lock:
            self._io_enabled = bool(value)

    @property
    def io_workers(self) -> int:
        """Number of workers for parallel IO (0 = auto)."""
        with self._lock:
            if self._io_workers > 0:
                return self._io_workers
            return min(os.cpu_count() or 4, 16)

    @io_workers.setter
    def io_workers(self, value: int) -> None:
        with self._lock:
            self._io_workers = max(0, int(value))

    @property
    def csv_threshold_mb(self) -> int:
        """Minimum CSV file size in MB for parallel reading."""
        with self._lock:
            return self._csv_threshold_mb

    @csv_threshold_mb.setter
    def csv_threshold_mb(self, value: int) -> None:
        with self._lock:
            self._csv_threshold_mb = max(1, int(value))

    @property
    def parquet_threshold_mb(self) -> int:
        """Minimum Parquet file size in MB for parallel reading."""
        with self._lock:
            return self._parquet_threshold_mb

    @parquet_threshold_mb.setter
    def parquet_threshold_mb(self, value: int) -> None:
        with self._lock:
            self._parquet_threshold_mb = max(1, int(value))

    @property
    def excel_threshold_mb(self) -> int:
        """Minimum Excel file size in MB for parallel reading."""
        with self._lock:
            return self._excel_threshold_mb

    @excel_threshold_mb.setter
    def excel_threshold_mb(self, value: int) -> None:
        with self._lock:
            self._excel_threshold_mb = max(1, int(value))

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
                f"parallel_threshold={self._parallel_threshold}, "
                f"io_enabled={self._io_enabled}, "
                f"io_workers={self._io_workers})"
            )


# Global configuration instance
config = UnlockedConfig()
