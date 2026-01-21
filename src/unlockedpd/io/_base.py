"""Base utilities for parallel IO operations."""

import os
from pathlib import Path
from typing import Union, Any, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

PathLike = Union[str, Path, os.PathLike]


def get_file_size(filepath: PathLike) -> int:
    """Get file size in bytes, returns 0 for non-local files."""
    try:
        return os.path.getsize(filepath)
    except (OSError, TypeError):
        return 0


def is_local_file(filepath: Any) -> bool:
    """Check if filepath is a local file path (not URL, not file-like)."""
    if not isinstance(filepath, (str, Path, os.PathLike)):
        return False
    path_str = str(filepath)
    # Check for URLs and cloud storage
    url_prefixes = ('http://', 'https://', 's3://', 'gs://', 'az://',
                    'hdfs://', 'abfs://', 'gcs://')
    if any(path_str.startswith(prefix) for prefix in url_prefixes):
        return False
    return True


def is_compressed(filepath: PathLike) -> bool:
    """Check if file has a compression extension."""
    path_str = str(filepath).lower()
    compressed_exts = ('.gz', '.bz2', '.zip', '.xz', '.zst', '.tar',
                       '.tgz', '.tar.gz', '.tar.bz2')
    return any(path_str.endswith(ext) for ext in compressed_exts)


def execute_parallel(
    tasks: List[Callable[[], Any]],
    max_workers: int,
    ordered: bool = True
) -> List[Any]:
    """Execute callable tasks in parallel using ThreadPoolExecutor.

    Args:
        tasks: List of no-argument callables to execute
        max_workers: Number of parallel workers
        ordered: If True, return results in input order; if False, return as completed

    Returns:
        List of results from each task
    """
    if not tasks:
        return []

    if ordered:
        results: List[Any] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(task): i
                for i, task in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results
    else:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
        return results
