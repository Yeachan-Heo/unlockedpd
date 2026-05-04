import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import unlockedpd
from unlockedpd._resources import (
    get_last_selected_path,
    pairwise_rolling_memory_estimate,
    resolve_threadpool_workers,
    threadpool_chunks,
)


@pytest.fixture(autouse=True)
def restore_resource_config():
    old = {
        "threadpool_workers": unlockedpd.config.threadpool_workers,
        "max_memory_overhead": unlockedpd.config.max_memory_overhead,
        "max_cpu_overhead": unlockedpd.config.max_cpu_overhead,
        "warmup": unlockedpd.config.warmup,
        "warn_on_fallback": unlockedpd.config.warn_on_fallback,
    }
    yield
    unlockedpd.config.threadpool_workers = old["threadpool_workers"]
    unlockedpd.config.max_memory_overhead = old["max_memory_overhead"]
    unlockedpd.config.max_cpu_overhead = old["max_cpu_overhead"]
    unlockedpd.config.warmup = old["warmup"]
    unlockedpd.config.warn_on_fallback = old["warn_on_fallback"]


def test_threadpool_workers_are_separate_from_numba_threads():
    original_numba_threads = unlockedpd.config.num_threads

    unlockedpd.config.threadpool_workers = 2

    assert unlockedpd.config.num_threads == original_numba_threads
    assert resolve_threadpool_workers(99, operation="rolling") == 2
    workers, chunks = threadpool_chunks(5, operation="rolling")
    assert workers == 2
    assert chunks == [(0, 3), (3, 5)]


def test_auto_threadpool_workers_are_bounded_by_work_units():
    unlockedpd.config.threadpool_workers = 0

    assert resolve_threadpool_workers(3, operation="rolling") == 3
    assert resolve_threadpool_workers(99, operation="rolling") <= 8
    assert resolve_threadpool_workers(99, operation="pairwise") <= 4


def test_pairwise_memory_budget_fallback_warns_and_uses_pandas():
    rng = np.random.default_rng(123)
    df = pd.DataFrame(rng.standard_normal((32, 6)))
    unlockedpd.config.max_memory_overhead = 1.01
    unlockedpd.config.warn_on_fallback = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = df.rolling(5).corr()

    assert result.shape == (len(df) * df.shape[1], df.shape[1])
    assert get_last_selected_path() == "fallback"
    assert any("estimated RSS overhead" in str(w.message) for w in caught)


def test_pairwise_memory_estimate_includes_flat_and_full_outputs():
    estimate = pairwise_rolling_memory_estimate(64, 8)

    assert estimate.optimized_bytes > estimate.baseline_bytes
    assert estimate.ratio > 1.0
    assert estimate.ratio < unlockedpd.config.max_memory_overhead


def test_import_warmup_defaults_to_lazy_in_fresh_process():
    env = os.environ.copy()
    env.pop("UNLOCKEDPD_WARMUP", None)
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", "import unlockedpd; print(unlockedpd.config.warmup)"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=True,
    )

    assert proc.stdout.strip() == "lazy"
