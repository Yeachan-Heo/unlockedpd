"""Tests for parallel IO operations."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import after setting up test environment
import unlockedpd
from unlockedpd.io import (
    read_files_parallel,
    read_csv_folder,
    read_parquet_folder,
)
from unlockedpd.io._base import (
    is_local_file,
    is_compressed,
    get_file_size,
    execute_parallel,
)
from unlockedpd.io._thresholds import (
    calculate_num_chunks,
    get_threshold_bytes,
)


class TestBaseUtilities:
    """Tests for _base.py utility functions."""

    def test_is_local_file_with_path(self):
        assert is_local_file("/path/to/file.csv")
        assert is_local_file(Path("/path/to/file.csv"))

    def test_is_local_file_with_urls(self):
        assert not is_local_file("http://example.com/file.csv")
        assert not is_local_file("https://example.com/file.csv")
        assert not is_local_file("s3://bucket/file.csv")
        assert not is_local_file("gs://bucket/file.csv")

    def test_is_local_file_with_non_path(self):
        from io import StringIO
        assert not is_local_file(StringIO("data"))
        assert not is_local_file(123)
        assert not is_local_file(None)

    def test_is_compressed(self):
        assert is_compressed("file.csv.gz")
        assert is_compressed("file.csv.bz2")
        assert is_compressed("file.csv.xz")
        assert is_compressed("file.zip")
        assert not is_compressed("file.csv")
        assert not is_compressed("file.parquet")

    def test_get_file_size(self, tmp_path):
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        size = get_file_size(test_file)
        assert size == 11

    def test_get_file_size_nonexistent(self):
        assert get_file_size("/nonexistent/path") == 0

    def test_execute_parallel_ordered(self):
        results = execute_parallel(
            [lambda i=i: i * 2 for i in range(5)],
            max_workers=2,
            ordered=True
        )
        assert results == [0, 2, 4, 6, 8]

    def test_execute_parallel_empty(self):
        results = execute_parallel([], max_workers=2)
        assert results == []


class TestThresholds:
    """Tests for _thresholds.py functions."""

    def test_get_threshold_bytes(self):
        assert get_threshold_bytes('csv') == 50 * 1024 * 1024
        assert get_threshold_bytes('parquet') == 20 * 1024 * 1024
        assert get_threshold_bytes('excel') == 10 * 1024 * 1024

    def test_calculate_num_chunks(self):
        # 100MB file with 4 workers
        chunks = calculate_num_chunks(100 * 1024 * 1024, 4)
        assert chunks >= 4  # At least num_workers
        assert chunks <= 32  # No more than max

    def test_calculate_num_chunks_small_file(self):
        # Small file should still get num_workers chunks
        chunks = calculate_num_chunks(10 * 1024 * 1024, 4)
        assert chunks >= 4


class TestMultiFileReading:
    """Tests for multi.py functions."""

    @pytest.fixture
    def csv_files(self, tmp_path):
        """Create multiple test CSV files."""
        files = []
        for i in range(3):
            df = pd.DataFrame({
                'a': [i * 10 + j for j in range(5)],
                'b': [f'row_{i}_{j}' for j in range(5)]
            })
            path = tmp_path / f"file_{i}.csv"
            df.to_csv(path, index=False)
            files.append(path)
        return files

    def test_read_files_parallel(self, csv_files):
        result = read_files_parallel(csv_files, pd.read_csv)

        assert len(result) == 15  # 3 files * 5 rows
        assert list(result.columns) == ['a', 'b']

    def test_read_files_parallel_empty_list(self):
        result = read_files_parallel([], pd.read_csv)
        assert len(result) == 0

    def test_read_files_parallel_with_source(self, csv_files):
        result = read_files_parallel(
            csv_files,
            pd.read_csv,
            add_source_column=True
        )

        assert '_source_file' in result.columns
        assert set(result['_source_file'].unique()) == {
            'file_0.csv', 'file_1.csv', 'file_2.csv'
        }

    def test_read_csv_folder(self, tmp_path, csv_files):
        # csv_files are already in tmp_path
        result = read_csv_folder(tmp_path, pattern="*.csv")

        assert len(result) == 15

    def test_read_csv_folder_empty(self, tmp_path):
        result = read_csv_folder(tmp_path, pattern="*.nonexistent")
        assert len(result) == 0


class TestCSVFallback:
    """Test that CSV reading falls back correctly."""

    def test_small_file_uses_pandas(self, tmp_path):
        """Small files should fall back to pandas (no error)."""
        df = pd.DataFrame({'a': range(10)})
        path = tmp_path / "small.csv"
        df.to_csv(path, index=False)

        # This should work (fallback to pandas)
        result = pd.read_csv(path)
        assert len(result) == 10

    def test_compressed_file_uses_pandas(self, tmp_path):
        """Compressed files should fall back to pandas."""
        df = pd.DataFrame({'a': range(10)})
        path = tmp_path / "test.csv.gz"
        df.to_csv(path, index=False, compression='gzip')

        # This should work (fallback to pandas)
        result = pd.read_csv(path)
        assert len(result) == 10


class TestConfiguration:
    """Test IO configuration."""

    def test_io_enabled_property(self):
        original = unlockedpd.config.io_enabled
        try:
            unlockedpd.config.io_enabled = False
            assert not unlockedpd.config.io_enabled

            unlockedpd.config.io_enabled = True
            assert unlockedpd.config.io_enabled
        finally:
            unlockedpd.config.io_enabled = original

    def test_io_workers_auto(self):
        unlockedpd.config.io_workers = 0
        workers = unlockedpd.config.io_workers
        # Auto should be between 1 and 16
        assert 1 <= workers <= 16

    def test_threshold_properties(self):
        assert unlockedpd.config.csv_threshold_mb >= 1
        assert unlockedpd.config.parquet_threshold_mb >= 1
        assert unlockedpd.config.excel_threshold_mb >= 1
