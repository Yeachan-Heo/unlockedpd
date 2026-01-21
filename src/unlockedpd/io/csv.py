"""Parallel CSV reading using row-chunk based parallelism.

This module provides optimized CSV reading by splitting large files into
chunks and reading them concurrently using ThreadPoolExecutor.

Key insight: pandas C parser releases GIL during IO, so ThreadPoolExecutor
achieves true parallelism without needing Numba.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Any
import warnings

import pandas as pd

from ._base import PathLike, is_local_file, is_compressed, get_file_size, execute_parallel
from ._thresholds import (
    IO_WORKERS,
    CSV_PARALLEL_THRESHOLD_BYTES,
    CSV_MIN_ROWS_PER_CHUNK,
    calculate_num_chunks,
)


# Store original for fallback
_original_read_csv = pd.read_csv


def _find_chunk_boundaries(
    filepath: PathLike,
    num_chunks: int
) -> List[Tuple[int, int]]:
    """Find byte boundaries for chunks at newline positions.

    Ensures each chunk starts and ends at a complete row.

    Args:
        filepath: Path to CSV file
        num_chunks: Number of chunks to create

    Returns:
        List of (start_byte, end_byte) tuples
    """
    file_size = get_file_size(filepath)
    if file_size == 0:
        return []

    chunk_size = file_size // num_chunks
    boundaries: List[Tuple[int, int]] = []

    with open(filepath, 'rb') as f:
        # Skip header line
        f.readline()
        header_end = f.tell()

        current_start = header_end

        for i in range(num_chunks - 1):
            # Seek to approximate boundary
            target = header_end + (i + 1) * chunk_size
            if target >= file_size:
                break

            f.seek(target)
            # Read to end of current line (find newline)
            f.readline()
            chunk_end = f.tell()

            if chunk_end > current_start:
                boundaries.append((current_start, chunk_end))
                current_start = chunk_end

        # Last chunk goes to end of file
        if current_start < file_size:
            boundaries.append((current_start, file_size))

    return boundaries


def _read_csv_chunk(
    filepath: PathLike,
    start_byte: int,
    end_byte: int,
    columns: List[str],
    dtype: Optional[dict] = None,
    **kwargs
) -> pd.DataFrame:
    """Read a specific byte range from a CSV file.

    Args:
        filepath: Path to CSV file
        start_byte: Starting byte position
        end_byte: Ending byte position
        columns: Column names (from header)
        dtype: Optional dtype dict
        **kwargs: Additional arguments for read_csv

    Returns:
        DataFrame for this chunk
    """
    # Read the byte range
    with open(filepath, 'rb') as f:
        f.seek(start_byte)
        chunk_bytes = f.read(end_byte - start_byte)

    # Parse with pandas
    from io import BytesIO
    chunk_kwargs = kwargs.copy()
    chunk_kwargs['names'] = columns
    chunk_kwargs['header'] = None
    if dtype is not None:
        chunk_kwargs['dtype'] = dtype

    return pd.read_csv(BytesIO(chunk_bytes), **chunk_kwargs)


def optimized_read_csv(
    filepath_or_buffer: PathLike,
    **kwargs
) -> pd.DataFrame:
    """Read CSV with parallel chunk reading for large files.

    For files larger than the threshold (50MB by default), splits the file
    into chunks and reads them in parallel using ThreadPoolExecutor.

    Args:
        filepath_or_buffer: Path to CSV file
        **kwargs: All pandas read_csv arguments

    Returns:
        DataFrame

    Raises:
        ValueError: If parallel reading is not beneficial (triggers fallback)
    """
    from .._config import config

    # Check if parallel reading is applicable
    if not is_local_file(filepath_or_buffer):
        raise ValueError("Not a local file path")

    if is_compressed(filepath_or_buffer):
        raise ValueError("Compressed files not supported for parallel reading")

    # Check for unsupported parameters that return iterators
    if kwargs.get('iterator', False):
        raise ValueError("iterator=True not supported")
    if kwargs.get('chunksize') is not None:
        raise ValueError("chunksize not supported")

    # Check file size against threshold
    file_size = get_file_size(filepath_or_buffer)
    threshold = config.csv_threshold_mb * 1024 * 1024

    if file_size < threshold:
        raise ValueError(f"File size {file_size} below threshold {threshold}")

    # Get number of workers
    num_workers = config.io_workers
    num_chunks = calculate_num_chunks(file_size, num_workers)

    if num_chunks < 2:
        raise ValueError("Not enough chunks for parallel benefit")

    # Read header to get column info
    header_df = pd.read_csv(filepath_or_buffer, nrows=0, **kwargs)
    columns = list(header_df.columns)
    dtype = kwargs.get('dtype')

    # Find chunk boundaries
    boundaries = _find_chunk_boundaries(filepath_or_buffer, num_chunks)

    if len(boundaries) < 2:
        raise ValueError("Could not create multiple chunks")

    # Create chunk reading tasks
    # Remove parameters that don't apply to chunk reading
    chunk_kwargs = {k: v for k, v in kwargs.items()
                    if k not in ('names', 'header', 'skiprows', 'nrows')}

    tasks = [
        lambda b=boundary: _read_csv_chunk(
            filepath_or_buffer, b[0], b[1], columns, dtype, **chunk_kwargs
        )
        for boundary in boundaries
    ]

    # Execute in parallel
    chunks = execute_parallel(tasks, num_workers, ordered=True)

    # Concatenate results
    return pd.concat(chunks, ignore_index=True)


def apply_csv_patches() -> None:
    """Apply CSV reading patch to pandas.

    Patches pd.read_csv to use parallel reading for large local files,
    with automatic fallback to original pandas for unsupported cases.
    """
    from .._config import config

    def patched_read_csv(filepath_or_buffer, **kwargs):
        if not config.io_enabled:
            return _original_read_csv(filepath_or_buffer, **kwargs)

        try:
            return optimized_read_csv(filepath_or_buffer, **kwargs)
        except (ValueError, TypeError, OSError, IOError) as e:
            if config.warn_on_fallback:
                warnings.warn(
                    f"unlockedpd: Falling back to pandas read_csv: {e}",
                    RuntimeWarning,
                    stacklevel=2
                )
            return _original_read_csv(filepath_or_buffer, **kwargs)

    # Patch at module level
    pd.read_csv = patched_read_csv
