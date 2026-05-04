"""Optional native row-rank kernels.

The Numba rank implementation is portable and correct, but wide dense
``axis=1`` ranks spend most of their time in per-row ``argsort``.  This module
keeps a faster C++/pthread implementation optional and resource-clean: compile
into a user cache when a C++ compiler is available, create/join workers per
call, and fall back to Numba when anything is unsupported.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np


_SOURCE = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <pthread.h>
#include <stddef.h>
#include <vector>

struct task_t {
    const double* in;
    double* out;
    size_t rows;
    size_t cols;
    int threads;
    int tid;
    int ascending;
};

static void* rank_worker(void* arg) {
    task_t* task = (task_t*)arg;
    const double* in = task->in;
    double* out = task->out;
    size_t rows = task->rows;
    size_t cols = task->cols;
    size_t start = (rows * (size_t)task->tid) / (size_t)task->threads;
    size_t end = (rows * (size_t)(task->tid + 1)) / (size_t)task->threads;
    std::vector<uint32_t> idx(cols);

    for (size_t row = start; row < end; ++row) {
        const double* values = in + row * cols;
        double* ranks = out + row * cols;
        for (size_t col = 0; col < cols; ++col) {
            idx[col] = (uint32_t)col;
        }

        if (task->ascending) {
            std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
                return values[a] < values[b];
            });
        } else {
            std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
                return values[a] > values[b];
            });
        }

        size_t i = 0;
        while (i < cols) {
            size_t j = i;
            while (j + 1 < cols && values[idx[j]] == values[idx[j + 1]]) {
                ++j;
            }

            double average_rank = ((double)i + (double)j + 2.0) * 0.5;
            for (size_t k = i; k <= j; ++k) {
                ranks[idx[k]] = average_rank;
            }
            i = j + 1;
        }
    }
    return NULL;
}

extern "C" int upd_axis1_rank_average_no_nan_f64_pthread(
    const double* in,
    double* out,
    size_t rows,
    size_t cols,
    int threads,
    int ascending
) {
    if (rows == 0 || cols == 0) return 0;
    if (threads < 1) threads = 1;
    if ((size_t)threads > rows) threads = (int)rows;
    if (threads < 1) threads = 1;

    pthread_t* tids = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)threads);
    task_t* tasks = (task_t*)malloc(sizeof(task_t) * (size_t)threads);
    if (tids == NULL || tasks == NULL) {
        free(tids);
        free(tasks);
        return -1;
    }

    int started = 0;
    for (int i = 0; i < threads; ++i) {
        tasks[i] = (task_t){in, out, rows, cols, threads, i, ascending};
        if (pthread_create(&tids[i], NULL, rank_worker, &tasks[i]) != 0) {
            break;
        }
        ++started;
    }

    for (int i = 0; i < started; ++i) {
        pthread_join(tids[i], NULL);
    }
    free(tids);
    free(tasks);
    return started == threads ? 0 : -2;
}
"""


_LIB = None
_LOOKED_UP = False
_LOCK = threading.Lock()


def _cache_dir() -> Path:
    root = os.environ.get("UNLOCKEDPD_NATIVE_CACHE")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "unlockedpd" / "native"


def _library_path() -> Path:
    digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()[:16]
    return _cache_dir() / f"axis1_rank_pthread_{digest}.so"


def _compile_library(path: Path) -> bool:
    compiler = (
        os.environ.get("CXX")
        or shutil.which("c++")
        or shutil.which("g++")
        or shutil.which("clang++")
    )
    if not compiler:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unlockedpd-rank-native-") as tmp:
        source = Path(tmp) / "axis1_rank_pthread.cpp"
        output = Path(tmp) / path.name
        source.write_text(_SOURCE)
        cmd = [
            compiler,
            "-O3",
            "-march=native",
            "-funroll-loops",
            "-fPIC",
            "-shared",
            "-pthread",
            str(source),
            "-o",
            str(output),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except Exception:
            return False
        output.replace(path)
    return True


def _load_library():
    global _LIB, _LOOKED_UP

    if os.environ.get("UNLOCKEDPD_DISABLE_NATIVE_RANK", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None

    with _LOCK:
        if _LOOKED_UP:
            return _LIB
        _LOOKED_UP = True

        path = _library_path()
        if not path.exists() and not _compile_library(path):
            return None

        try:
            lib = ctypes.CDLL(str(path))
            fn = lib.upd_axis1_rank_average_no_nan_f64_pthread
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
            ]
            fn.restype = ctypes.c_int
        except Exception:
            return None

        _LIB = lib
        return _LIB


def native_axis1_rank_average_no_nan(
    arr: np.ndarray, *, ascending: bool, threads: int
) -> np.ndarray | None:
    """Run optional native dense axis=1 average rank, or ``None`` on fallback."""

    if arr.dtype != np.float64 or not arr.flags.c_contiguous:
        return None

    lib = _load_library()
    if lib is None:
        return None

    result = np.empty_like(arr)
    status = lib.upd_axis1_rank_average_no_nan_f64_pthread(
        arr.ctypes.data,
        result.ctypes.data,
        arr.shape[0],
        arr.shape[1],
        int(max(1, threads)),
        1 if ascending else 0,
    )
    if status != 0:
        return None
    return result
