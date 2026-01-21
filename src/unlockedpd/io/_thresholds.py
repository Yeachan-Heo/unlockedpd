"""Thresholds and heuristics for parallel IO decisions."""

import os

# Number of CPUs for parallel operations
CPU_COUNT = os.cpu_count() or 4

# IO workers capped at 16 (diminishing returns for IO-bound work due to disk bandwidth)
IO_WORKERS = min(CPU_COUNT, 16)

# =============================================================================
# File Size Thresholds (bytes)
# =============================================================================

# CSV: Parallel reading benefits large files due to parsing overhead
CSV_PARALLEL_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB

# Parquet: Row groups enable natural parallelism, lower threshold
PARQUET_PARALLEL_THRESHOLD_BYTES = 20 * 1024 * 1024  # 20 MB

# Excel: Sheet parallelism works well, but openpyxl has overhead
EXCEL_PARALLEL_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10 MB

# =============================================================================
# CSV Chunking Parameters
# =============================================================================

# Target chunk size for CSV parallel reading (bytes)
CSV_TARGET_CHUNK_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB per chunk

# Minimum rows per chunk to avoid tiny chunk overhead
CSV_MIN_ROWS_PER_CHUNK = 10_000

# Maximum number of chunks (limits memory from concat overhead)
CSV_MAX_CHUNKS = 32

# =============================================================================
# Helper Functions
# =============================================================================

def get_threshold_bytes(format_type: str) -> int:
    """Get the parallel threshold in bytes for a given format."""
    thresholds = {
        'csv': CSV_PARALLEL_THRESHOLD_BYTES,
        'parquet': PARQUET_PARALLEL_THRESHOLD_BYTES,
        'excel': EXCEL_PARALLEL_THRESHOLD_BYTES,
    }
    return thresholds.get(format_type.lower(), CSV_PARALLEL_THRESHOLD_BYTES)


def calculate_num_chunks(
    file_size_bytes: int,
    num_workers: int,
    target_chunk_bytes: int = CSV_TARGET_CHUNK_SIZE_BYTES,
    max_chunks: int = CSV_MAX_CHUNKS
) -> int:
    """Calculate optimal number of chunks for parallel reading.

    Balances parallelism (more chunks) with overhead (fewer chunks).
    """
    # How many chunks based on target size
    size_based_chunks = max(1, file_size_bytes // target_chunk_bytes)

    # At least num_workers chunks for parallelism, but not more than max_chunks
    return max(num_workers, min(size_based_chunks, max_chunks))
