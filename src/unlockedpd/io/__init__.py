"""Parallel IO operations for unlockedpd.

This module provides parallelized file reading operations using
ThreadPoolExecutor for concurrent IO. Unlike the compute operations
that use Numba JIT, IO operations leverage Python's GIL release during
C library IO calls for true parallelism.

Key insight: pandas/pyarrow/openpyxl release the GIL during file reads,
so ThreadPoolExecutor achieves true parallelism without Numba.

Patched functions (transparent optimization):
    - pd.read_csv: Parallel row-chunk reading for files > 50MB
    - pd.read_parquet: Parallel row-group reading via pyarrow
    - pd.read_excel: Parallel sheet reading for multi-sheet files

Utility functions (explicit API):
    - read_files_parallel: Read any file list concurrently
    - read_csv_folder: Read all CSVs in a directory
    - read_parquet_folder: Read all Parquets in a directory
    - read_excel_folder: Read all Excel files in a directory

Configuration:
    import unlockedpd
    unlockedpd.config.io_enabled = True  # Enable/disable IO patches
    unlockedpd.config.io_workers = 8     # Set worker count (0 = auto)
    unlockedpd.config.csv_threshold_mb = 50  # Min file size for parallel CSV
"""

from .csv import optimized_read_csv, apply_csv_patches
from .parquet import optimized_read_parquet, apply_parquet_patches
from .excel import optimized_read_excel, apply_excel_patches
from .multi import (
    read_files_parallel,
    read_csv_folder,
    read_parquet_folder,
    read_excel_folder,
)

__all__ = [
    # Optimized readers (usually accessed via patched pd.read_*)
    "optimized_read_csv",
    "optimized_read_parquet",
    "optimized_read_excel",
    # Patch application functions
    "apply_csv_patches",
    "apply_parquet_patches",
    "apply_excel_patches",
    # Multi-file utilities (explicit API)
    "read_files_parallel",
    "read_csv_folder",
    "read_parquet_folder",
    "read_excel_folder",
]


def apply_io_patches() -> None:
    """Apply all IO patches to pandas.

    This patches pd.read_csv, pd.read_parquet, and pd.read_excel
    with parallel reading implementations that automatically fall
    back to pandas for unsupported cases.

    Called automatically on import if config.io_enabled is True.
    """
    apply_csv_patches()
    apply_parquet_patches()
    apply_excel_patches()
