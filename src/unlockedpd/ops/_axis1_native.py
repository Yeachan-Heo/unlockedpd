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
#if defined(__linux__)
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#if defined(__AVX512F__)
#include <float.h>
#include <immintrin.h>
#endif

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

typedef struct {
    const double* in;
    double* out;
    size_t rows;
    size_t cols;
    size_t start;
    size_t end;
} flat_task_t;

static void advise_huge_pages(const void* ptr, size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (ptr == NULL || bytes == 0) return;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return;
    uintptr_t address = (uintptr_t)ptr;
    uintptr_t page_mask = (uintptr_t)page_size - 1;
    uintptr_t start = address & ~page_mask;
    uintptr_t end = (address + bytes + page_mask) & ~page_mask;
    if (end > start) {
        (void)madvise((void*)start, (size_t)(end - start), MADV_HUGEPAGE);
    }
#else
    (void)ptr;
    (void)bytes;
#endif
}

static void* axis1_worker_p1_flat(void* arg) {
    flat_task_t* task = (flat_task_t*)arg;
    const double* restrict in = task->in;
    double* restrict out = task->out;
    size_t rows = task->rows;
    size_t cols = task->cols;
    size_t start = task->start;
    size_t end = task->end;

    /*
     * pct_change(periods=1) is the large-frame division hot path.  Treat the
     * frame as one flat contiguous vector, compute the row-boundary cells too,
     * then overwrite exactly those first-column cells with NaN.  This preserves
     * pandas semantics while giving the compiler one long vectorizable loop
     * instead of restarting a 511-element inner loop for every row.
     */
    size_t compute_start = start == 0 ? 1 : start;
#if defined(__AVX512F__)
    size_t idx = compute_start;
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d two = _mm512_set1_pd(2.0);
    const __m512d zero = _mm512_setzero_pd();
    const __m512d max_value = _mm512_set1_pd(DBL_MAX);
    const __m512d min_value = _mm512_set1_pd(-DBL_MAX);

    for (; idx + 8 <= end; idx += 8) {
        __m512d numerator = _mm512_loadu_pd(in + idx);
        __m512d denominator = _mm512_loadu_pd(in + idx - 1);
        __mmask8 finite_denominator =
            _mm512_cmp_pd_mask(denominator, max_value, _CMP_LE_OQ) &
            _mm512_cmp_pd_mask(denominator, min_value, _CMP_GE_OQ);
        __mmask8 finite_numerator =
            _mm512_cmp_pd_mask(numerator, max_value, _CMP_LE_OQ) &
            _mm512_cmp_pd_mask(numerator, min_value, _CMP_GE_OQ);
        __mmask8 nonzero_denominator =
            _mm512_cmp_pd_mask(denominator, zero, _CMP_NEQ_OQ);
        __mmask8 fast_mask =
            finite_denominator & finite_numerator & nonzero_denominator;

        if (fast_mask) {
            __m512d reciprocal = _mm512_rcp14_pd(denominator);
            reciprocal = _mm512_mul_pd(
                reciprocal,
                _mm512_sub_pd(two, _mm512_mul_pd(denominator, reciprocal))
            );
            reciprocal = _mm512_mul_pd(
                reciprocal,
                _mm512_sub_pd(two, _mm512_mul_pd(denominator, reciprocal))
            );
            __m512d value = _mm512_sub_pd(
                _mm512_mul_pd(numerator, reciprocal),
                one
            );
            _mm512_mask_storeu_pd(out + idx, fast_mask, value);
        }
        if (fast_mask != 0xFF) {
            for (size_t lane = 0; lane < 8; ++lane) {
                if ((fast_mask & (1 << lane)) == 0) {
                    size_t scalar_idx = idx + lane;
                    out[scalar_idx] =
                        in[scalar_idx] / in[scalar_idx - 1] - 1.0;
                }
            }
        }
    }
    for (; idx < end; ++idx) {
        out[idx] = in[idx] / in[idx - 1] - 1.0;
    }
#else
    UPD_IVDEP
    for (size_t idx = compute_start; idx < end; ++idx) {
        out[idx] = in[idx] / in[idx - 1] - 1.0;
    }
#endif

    size_t first_row = (start + cols - 1) / cols;
    for (size_t row = first_row; row < rows; ++row) {
        size_t idx = row * cols;
        if (idx >= end) break;
        out[idx] = NAN;
    }
    return NULL;
}

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

static void run_axis1_transform_p1_flat(
    const double* restrict in,
    double* restrict out,
    size_t rows,
    size_t cols,
    int threads
) {
    if (threads < 1) threads = 1;
    if ((size_t)threads > rows) threads = (int)rows;
    if (threads < 1) threads = 1;

    pthread_t tids[threads];
    flat_task_t tasks[threads];
    pthread_attr_t attr;
    pthread_attr_t* attr_ptr = NULL;
    int attr_initialized = 0;
    if (pthread_attr_init(&attr) == 0) {
        attr_initialized = 1;
        if (pthread_attr_setstacksize(&attr, 256 * 1024) == 0) {
            attr_ptr = &attr;
        }
    }

    size_t total = rows * cols;
    size_t step = (total + (size_t)threads - 1) / (size_t)threads;
    int actual_threads = 0;

    for (int i = 0; i < threads; ++i) {
        size_t start = (size_t)i * step;
        if (start >= total) break;
        size_t end = start + step;
        if (end > total) end = total;
        tasks[i] = (flat_task_t){in, out, rows, cols, start, end};
        if (pthread_create(&tids[actual_threads], attr_ptr, axis1_worker_p1_flat, &tasks[i]) == 0) {
            actual_threads++;
        } else {
            axis1_worker_p1_flat(&tasks[i]);
        }
    }

    for (int i = 0; i < actual_threads; ++i) {
        pthread_join(tids[i], NULL);
    }
    if (attr_initialized) {
        pthread_attr_destroy(&attr);
    }
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

    size_t total_bytes = rows * cols * sizeof(double);
    advise_huge_pages(in, total_bytes);
    advise_huge_pages(out, total_bytes);

    if (periods == 1 && cols > 1 && op == 1) {
        run_axis1_transform_p1_flat(in, out, rows, cols, threads);
        return;
    }

    pthread_t tids[threads];
    task_t tasks[threads];
    pthread_attr_t attr;
    pthread_attr_t* attr_ptr = NULL;
    int attr_initialized = 0;
    if (pthread_attr_init(&attr) == 0) {
        attr_initialized = 1;
        /*
         * These workers keep only a tiny task pointer on their C stack.  The
         * libc default stack is commonly several MiB per pthread, which is
         * unnecessary virtual-memory pressure for short native bursts.
         */
        if (pthread_attr_setstacksize(&attr, 256 * 1024) == 0) {
            attr_ptr = &attr;
        }
    }
    size_t step = (rows + (size_t)threads - 1) / (size_t)threads;
    int actual_threads = 0;

    for (int i = 0; i < threads; ++i) {
        size_t start = (size_t)i * step;
        if (start >= rows) break;
        size_t end = start + step;
        if (end > rows) end = rows;
        tasks[i] = (task_t){in, out, rows, cols, start, end, periods, op};
        if (pthread_create(&tids[actual_threads], attr_ptr, axis1_worker, &tasks[i]) == 0) {
            actual_threads++;
        } else {
            axis1_worker(&tasks[i]);
        }
    }

    for (int i = 0; i < actual_threads; ++i) {
        pthread_join(tids[i], NULL);
    }
    if (attr_initialized) {
        pthread_attr_destroy(&attr);
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
