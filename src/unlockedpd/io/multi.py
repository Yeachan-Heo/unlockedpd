"""Multi-file parallel reading utilities.

This module provides helpers for reading multiple files concurrently,
a common pattern in data pipelines dealing with partitioned datasets.
"""

import os
from pathlib import Path
from typing import List, Union, Optional, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob as glob_module

import pandas as pd

from ._base import PathLike, execute_parallel
from ._thresholds import IO_WORKERS


def read_files_parallel(
    files: List[PathLike],
    reader: Callable[[PathLike], pd.DataFrame],
    max_workers: Optional[int] = None,
    ignore_errors: bool = False,
    add_source_column: bool = False,
    source_column_name: str = '_source_file',
) -> pd.DataFrame:
    """Read multiple files in parallel and concatenate results.

    This is the core multi-file reader that works with any pandas
    reader function.

    Args:
        files: List of file paths to read
        reader: Function that reads a single file and returns DataFrame
        max_workers: Number of parallel workers (default: min(cpu_count, 16))
        ignore_errors: If True, skip files that fail to read
        add_source_column: If True, add column with source filename
        source_column_name: Name for the source column

    Returns:
        Concatenated DataFrame from all files

    Example:
        >>> files = glob.glob('data/*.csv')
        >>> df = read_files_parallel(files, pd.read_csv)

        >>> # With custom reader
        >>> reader = lambda f: pd.read_csv(f, usecols=['a', 'b'])
        >>> df = read_files_parallel(files, reader)
    """
    if not files:
        return pd.DataFrame()

    from .._config import config
    workers = max_workers if max_workers is not None else config.io_workers

    results: List[pd.DataFrame] = []
    errors: List[tuple] = []

    def read_file(filepath: PathLike) -> Optional[pd.DataFrame]:
        try:
            df = reader(filepath)
            if add_source_column:
                df[source_column_name] = Path(filepath).name
            return df
        except Exception as e:
            if ignore_errors:
                return None
            raise

    # Execute parallel reads
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {
            executor.submit(read_file, f): f
            for f in files
        }

        for future in as_completed(future_to_file):
            filepath = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                if ignore_errors:
                    errors.append((filepath, str(e)))
                else:
                    raise RuntimeError(f"Failed to read {filepath}: {e}") from e

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def read_csv_folder(
    folder: PathLike,
    pattern: str = "*.csv",
    max_workers: Optional[int] = None,
    ignore_errors: bool = False,
    recursive: bool = False,
    **read_csv_kwargs
) -> pd.DataFrame:
    """Read all CSV files in a folder in parallel.

    Convenience function for the common pattern of reading a folder
    of partitioned CSV files.

    Args:
        folder: Path to folder containing CSV files
        pattern: Glob pattern for file matching (default: "*.csv")
        max_workers: Number of parallel workers
        ignore_errors: Skip files that fail to read
        recursive: If True, search recursively with **/ pattern
        **read_csv_kwargs: Arguments passed to pd.read_csv

    Returns:
        Concatenated DataFrame from all matching files

    Example:
        >>> df = read_csv_folder('data/partitions/', usecols=['id', 'value'])
    """
    folder_path = Path(folder)

    if recursive:
        files = sorted(folder_path.rglob(pattern))
    else:
        files = sorted(folder_path.glob(pattern))

    if not files:
        return pd.DataFrame()

    # Use original read_csv to avoid double-patching overhead
    from .csv import _original_read_csv
    reader = lambda f: _original_read_csv(f, **read_csv_kwargs)

    return read_files_parallel(
        files,
        reader,
        max_workers=max_workers,
        ignore_errors=ignore_errors
    )


def read_parquet_folder(
    folder: PathLike,
    pattern: str = "*.parquet",
    max_workers: Optional[int] = None,
    ignore_errors: bool = False,
    recursive: bool = False,
    **read_parquet_kwargs
) -> pd.DataFrame:
    """Read all Parquet files in a folder in parallel.

    Args:
        folder: Path to folder containing Parquet files
        pattern: Glob pattern for file matching (default: "*.parquet")
        max_workers: Number of parallel workers
        ignore_errors: Skip files that fail to read
        recursive: If True, search recursively with **/ pattern
        **read_parquet_kwargs: Arguments passed to pd.read_parquet

    Returns:
        Concatenated DataFrame from all matching files

    Example:
        >>> df = read_parquet_folder('data/partitions/', columns=['id', 'value'])
    """
    folder_path = Path(folder)

    if recursive:
        files = sorted(folder_path.rglob(pattern))
    else:
        files = sorted(folder_path.glob(pattern))

    if not files:
        return pd.DataFrame()

    # Use original read_parquet
    from .parquet import _original_read_parquet
    reader = lambda f: _original_read_parquet(f, **read_parquet_kwargs)

    return read_files_parallel(
        files,
        reader,
        max_workers=max_workers,
        ignore_errors=ignore_errors
    )


def read_excel_folder(
    folder: PathLike,
    pattern: str = "*.xlsx",
    max_workers: Optional[int] = None,
    ignore_errors: bool = False,
    recursive: bool = False,
    **read_excel_kwargs
) -> pd.DataFrame:
    """Read all Excel files in a folder in parallel.

    Args:
        folder: Path to folder containing Excel files
        pattern: Glob pattern for file matching (default: "*.xlsx")
        max_workers: Number of parallel workers
        ignore_errors: Skip files that fail to read
        recursive: If True, search recursively with **/ pattern
        **read_excel_kwargs: Arguments passed to pd.read_excel

    Returns:
        Concatenated DataFrame from all matching files
    """
    folder_path = Path(folder)

    if recursive:
        files = sorted(folder_path.rglob(pattern))
    else:
        files = sorted(folder_path.glob(pattern))

    if not files:
        return pd.DataFrame()

    # Use original read_excel
    from .excel import _original_read_excel
    reader = lambda f: _original_read_excel(f, **read_excel_kwargs)

    return read_files_parallel(
        files,
        reader,
        max_workers=max_workers,
        ignore_errors=ignore_errors
    )
