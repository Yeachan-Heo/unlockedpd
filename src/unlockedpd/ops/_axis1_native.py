"""Optional native axis=1 transform kernels.

The pure-Python/Numba paths are resource-clean but still leave simple row-wise
copy transforms well below the 10x target.  This module keeps the native surface
optional: it compiles a tiny pthread-backed shared library into a user cache on
first use when a local C compiler is available, and callers fall back cleanly
when it is not.
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
#include <math.h>
#include <pthread.h>
#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#define UPD_IVDEP _Pragma("GCC ivdep")
#else
#define UPD_IVDEP
#endif

typedef struct {
    const double* in;
    double* out;
    size_t rows;
    size_t cols;
    size_t start;
    size_t end;
    long periods;
    int op;
} task_t;

static void* axis1_worker(void* arg) {
    task_t* task = (task_t*)arg;
    const double* restrict in = task->in;
    double* restrict out = task->out;
    size_t cols = task->cols;
    long periods = task->periods;
    size_t p = periods >= 0 ? (size_t)periods : (size_t)(-periods);

    if (periods == 0) {
        for (size_t row = task->start; row < task->end; ++row) {
            size_t base = row * cols;
            UPD_IVDEP
            for (size_t col = 0; col < cols; ++col) {
                double value = in[base + col];
                out[base + col] = task->op == 0 ? value - value : value / value - 1.0;
            }
        }
        return NULL;
    }

    if (p >= cols) {
        for (size_t row = task->start; row < task->end; ++row) {
            size_t base = row * cols;
            UPD_IVDEP
            for (size_t col = 0; col < cols; ++col) {
                out[base + col] = NAN;
            }
        }
        return NULL;
    }

    if (periods > 0) {
        for (size_t row = task->start; row < task->end; ++row) {
            size_t base = row * cols;
            for (size_t col = 0; col < p; ++col) {
                out[base + col] = NAN;
            }
            if (task->op == 0) {
                UPD_IVDEP
                for (size_t col = p; col < cols; ++col) {
                    out[base + col] = in[base + col] - in[base + col - p];
                }
            } else {
                UPD_IVDEP
                for (size_t col = p; col < cols; ++col) {
                    out[base + col] = in[base + col] / in[base + col - p] - 1.0;
                }
            }
        }
    } else {
        for (size_t row = task->start; row < task->end; ++row) {
            size_t base = row * cols;
            if (task->op == 0) {
                UPD_IVDEP
                for (size_t col = 0; col < cols - p; ++col) {
                    out[base + col] = in[base + col] - in[base + col + p];
                }
            } else {
                UPD_IVDEP
                for (size_t col = 0; col < cols - p; ++col) {
                    out[base + col] = in[base + col] / in[base + col + p] - 1.0;
                }
            }
            UPD_IVDEP
            for (size_t col = cols - p; col < cols; ++col) {
                out[base + col] = NAN;
            }
        }
    }
    return NULL;
}

static void run_axis1_transform(
    const double* restrict in,
    double* restrict out,
    size_t rows,
    size_t cols,
    long periods,
    int threads,
    int op
) {
    if (threads < 1) threads = 1;
    if ((size_t)threads > rows) threads = (int)rows;
    if (threads < 1) threads = 1;

    pthread_t tids[threads];
    task_t tasks[threads];
    size_t step = (rows + (size_t)threads - 1) / (size_t)threads;
    int actual_threads = 0;

    for (int i = 0; i < threads; ++i) {
        size_t start = (size_t)i * step;
        if (start >= rows) break;
        size_t end = start + step;
        if (end > rows) end = rows;
        tasks[i] = (task_t){in, out, rows, cols, start, end, periods, op};
        pthread_create(&tids[i], NULL, axis1_worker, &tasks[i]);
        actual_threads++;
    }

    for (int i = 0; i < actual_threads; ++i) {
        pthread_join(tids[i], NULL);
    }
}

void upd_axis1_diff_f64_pthread(
    const double* in,
    double* out,
    size_t rows,
    size_t cols,
    long periods,
    int threads
) {
    run_axis1_transform(in, out, rows, cols, periods, threads, 0);
}

void upd_axis1_pct_f64_pthread(
    const double* in,
    double* out,
    size_t rows,
    size_t cols,
    long periods,
    int threads
) {
    run_axis1_transform(in, out, rows, cols, periods, threads, 1);
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
    return _cache_dir() / f"axis1_pthread_{digest}.so"


def _compile_library(path: Path) -> bool:
    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unlockedpd-native-") as tmp:
        source = Path(tmp) / "axis1_pthread.c"
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

    if os.environ.get("UNLOCKEDPD_DISABLE_NATIVE_TRANSFORMS", "").lower() in {
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
            for name in ("upd_axis1_diff_f64_pthread", "upd_axis1_pct_f64_pthread"):
                fn = getattr(lib, name)
                fn.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_long,
                    ctypes.c_int,
                ]
                fn.restype = None
        except Exception:
            return None

        _LIB = lib
        return _LIB


def native_axis1_transform(
    arr: np.ndarray, periods: int, *, op: str, threads: int
) -> np.ndarray | None:
    """Run an optional native axis=1 transform, returning ``None`` on fallback."""

    if arr.dtype != np.float64 or not arr.flags.c_contiguous:
        return None

    lib = _load_library()
    if lib is None:
        return None

    result = np.empty_like(arr)
    fn = (
        lib.upd_axis1_diff_f64_pthread
        if op == "diff"
        else lib.upd_axis1_pct_f64_pthread
    )
    fn(
        arr.ctypes.data,
        result.ctypes.data,
        arr.shape[0],
        arr.shape[1],
        int(periods),
        int(max(1, threads)),
    )
    return result
