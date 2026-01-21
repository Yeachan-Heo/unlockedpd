"""Parallel Parquet reading using row-group parallelism.

Parquet files are organized into row groups, which are natural parallelization
units. PyArrow can read specific row groups, enabling concurrent reading.
"""

import os
from pathlib import Path
from typing import Optional, List, Any
import warnings

import pandas as pd

from ._base import PathLike, is_local_file, get_file_size, execute_parallel
from ._thresholds import IO_WORKERS, PARQUET_PARALLEL_THRESHOLD_BYTES


# Store original for fallback
_original_read_parquet = pd.read_parquet


def optimized_read_parquet(
    path: PathLike,
    columns: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """Read Parquet with parallel row-group reading.

    For Parquet files with multiple row groups, reads groups in parallel
    using ThreadPoolExecutor + PyArrow.

    Args:
        path: Path to Parquet file
        columns: Columns to read (None = all)
        **kwargs: Additional arguments for read_parquet

    Returns:
        DataFrame

    Raises:
        ValueError: If parallel reading is not beneficial (triggers fallback)
    """
    from .._config import config

    # Check if parallel reading is applicable
    if not is_local_file(path):
        raise ValueError("Not a local file path")

    # Check file size against threshold
    file_size = get_file_size(path)
    threshold = config.parquet_threshold_mb * 1024 * 1024

    if file_size < threshold:
        raise ValueError(f"File size {file_size} below threshold {threshold}")

    # Try to import pyarrow
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ValueError("pyarrow required for parallel parquet reading")

    # Get row group metadata
    try:
        parquet_file = pq.ParquetFile(path)
        num_row_groups = parquet_file.metadata.num_row_groups
    except Exception as e:
        raise ValueError(f"Could not read parquet metadata: {e}")

    if num_row_groups <= 1:
        raise ValueError("Single row group, no benefit from parallel reading")

    # Get number of workers (cap at row groups)
    num_workers = min(config.io_workers, num_row_groups)

    # Create tasks for each row group
    def read_row_group(rg_idx: int) -> pd.DataFrame:
        pf = pq.ParquetFile(path)
        table = pf.read_row_group(rg_idx, columns=columns)
        return table.to_pandas()

    tasks = [
        lambda idx=i: read_row_group(idx)
        for i in range(num_row_groups)
    ]

    # Execute in parallel
    chunks = execute_parallel(tasks, num_workers, ordered=True)

    # Concatenate results
    return pd.concat(chunks, ignore_index=True)


def apply_parquet_patches() -> None:
    """Apply Parquet reading patch to pandas.

    Patches pd.read_parquet to use parallel row-group reading for files
    with multiple row groups, with automatic fallback for other cases.
    """
    from .._config import config

    def patched_read_parquet(path, **kwargs):
        if not config.io_enabled:
            return _original_read_parquet(path, **kwargs)

        # Extract columns if provided
        columns = kwargs.pop('columns', None)

        try:
            return optimized_read_parquet(path, columns=columns, **kwargs)
        except (ValueError, ImportError) as e:
            if config.warn_on_fallback:
                warnings.warn(
                    f"unlockedpd: Falling back to pandas read_parquet: {e}",
                    RuntimeWarning,
                    stacklevel=2
                )
            # Restore columns to kwargs for original call
            if columns is not None:
                kwargs['columns'] = columns
            return _original_read_parquet(path, **kwargs)

    # Patch at module level
    pd.read_parquet = patched_read_parquet
