"""Parallel Excel reading using sheet-level parallelism.

For multi-sheet Excel files, reads sheets in parallel using ThreadPoolExecutor.
Single-sheet files fall back to pandas.
"""

import os
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import warnings

import pandas as pd

from ._base import PathLike, is_local_file, get_file_size, execute_parallel
from ._thresholds import IO_WORKERS, EXCEL_PARALLEL_THRESHOLD_BYTES


# Store original for fallback
_original_read_excel = pd.read_excel


def optimized_read_excel(
    io: PathLike,
    sheet_name: Union[str, int, List, None] = 0,
    **kwargs
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Read Excel with parallel sheet reading.

    For multi-sheet Excel files where multiple sheets are requested,
    reads sheets in parallel using ThreadPoolExecutor.

    Args:
        io: Path to Excel file
        sheet_name: Sheet(s) to read. None means all sheets, list means specific sheets.
        **kwargs: Additional arguments for read_excel

    Returns:
        DataFrame for single sheet, or dict of DataFrames for multiple sheets

    Raises:
        ValueError: If parallel reading is not beneficial (triggers fallback)
    """
    from .._config import config

    # Check if parallel reading is applicable
    if not is_local_file(io):
        raise ValueError("Not a local file path")

    # Only parallelize when reading multiple sheets
    if sheet_name is not None and not isinstance(sheet_name, list):
        raise ValueError("Single sheet requested, no parallel benefit")

    # Check file size against threshold
    file_size = get_file_size(io)
    threshold = config.excel_threshold_mb * 1024 * 1024

    if file_size < threshold:
        raise ValueError(f"File size {file_size} below threshold {threshold}")

    # Get list of sheets to read
    try:
        xl = pd.ExcelFile(io)
        available_sheets = xl.sheet_names
    except Exception as e:
        raise ValueError(f"Could not read Excel file: {e}")

    if sheet_name is None:
        sheets_to_read = available_sheets
    else:
        # Validate requested sheets exist
        sheets_to_read = []
        for s in sheet_name:
            if isinstance(s, int):
                if 0 <= s < len(available_sheets):
                    sheets_to_read.append(available_sheets[s])
                else:
                    raise ValueError(f"Sheet index {s} out of range")
            else:
                if s in available_sheets:
                    sheets_to_read.append(s)
                else:
                    raise ValueError(f"Sheet '{s}' not found")

    if len(sheets_to_read) <= 1:
        raise ValueError("Single sheet, no parallel benefit")

    # Get number of workers (cap at number of sheets)
    num_workers = min(config.io_workers, len(sheets_to_read))

    # Create tasks for each sheet
    def read_sheet(sheet: str) -> tuple:
        df = _original_read_excel(io, sheet_name=sheet, **kwargs)
        return (sheet, df)

    tasks = [
        lambda s=sheet: read_sheet(s)
        for sheet in sheets_to_read
    ]

    # Execute in parallel
    results = execute_parallel(tasks, num_workers, ordered=True)

    # Build result dict
    return {sheet: df for sheet, df in results}


def apply_excel_patches() -> None:
    """Apply Excel reading patch to pandas.

    Patches pd.read_excel to use parallel sheet reading when multiple
    sheets are requested, with automatic fallback for other cases.
    """
    from .._config import config

    def patched_read_excel(io, sheet_name=0, **kwargs):
        if not config.io_enabled:
            return _original_read_excel(io, sheet_name=sheet_name, **kwargs)

        try:
            return optimized_read_excel(io, sheet_name=sheet_name, **kwargs)
        except ValueError as e:
            if config.warn_on_fallback:
                warnings.warn(
                    f"unlockedpd: Falling back to pandas read_excel: {e}",
                    RuntimeWarning,
                    stacklevel=2
                )
            return _original_read_excel(io, sheet_name=sheet_name, **kwargs)

    # Patch at module level
    pd.read_excel = patched_read_excel
