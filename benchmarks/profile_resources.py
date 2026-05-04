#!/usr/bin/env python3
"""Resource profiling harness for unlockedpd.

Records pandas-vs-optimized wall time, process CPU seconds, RSS, and thread
observations in an isolated subprocess per implementation/repeat so patched
state does not contaminate comparisons.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import math
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "resource-profile-v1"
DEFAULT_CASES = (
    "import-only",
    "rolling-wide-10mb",
    "rolling-axis1-wide-32mb",
    "rolling-medium-100mb",
    "expanding-wide-10mb",
    "aggregation-wide-10mb",
    "aggregation-medium-100mb",
    "aggregation-axis1-wide-32mb",
    "cumulative-axis1-wide-32mb",
    "transform-axis1-wide-32mb",
    "rank-wide-1mb-control",
    "rank-axis1-wide-32mb",
    "pairwise-safe-rolling-corr",
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    operation: str
    shape: Optional[Tuple[int, int]]
    params: Dict[str, Any]
    parallelism_gate: bool = False

    def worker_payload(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "operation": self.operation,
            "shape": list(self.shape) if self.shape is not None else None,
            "params": self.params,
            "parallelism_gate": self.parallelism_gate,
        }


def _case_matrix() -> List[CaseSpec]:
    return [
        CaseSpec("import-only", "import_unlockedpd", None, {}, False),
        CaseSpec(
            "rolling-wide-10mb", "rolling_mean", (1280, 1024), {"window": 20}, True
        ),
        CaseSpec(
            "rolling-wide-10mb", "rolling_sum", (1280, 1024), {"window": 20}, True
        ),
        CaseSpec(
            "rolling-axis1-wide-32mb",
            "rolling_mean",
            (8192, 512),
            {"window": 20, "axis": 1},
            True,
        ),
        CaseSpec(
            "rolling-axis1-wide-32mb",
            "rolling_sum",
            (8192, 512),
            {"window": 20, "axis": 1},
            True,
        ),
        CaseSpec(
            "rolling-axis1-wide-32mb",
            "rolling_std",
            (8192, 512),
            {"window": 20, "axis": 1},
            True,
        ),
        CaseSpec(
            "rolling-axis1-wide-32mb",
            "rolling_var",
            (8192, 512),
            {"window": 20, "axis": 1},
            True,
        ),
        CaseSpec(
            "rolling-axis1-wide-32mb",
            "rolling_min",
            (8192, 512),
            {"window": 20, "axis": 1},
            True,
        ),
        CaseSpec(
            "rolling-axis1-wide-32mb",
            "rolling_max",
            (8192, 512),
            {"window": 20, "axis": 1},
            True,
        ),
        CaseSpec(
            "rolling-medium-100mb", "rolling_mean", (102400, 128), {"window": 20}, True
        ),
        CaseSpec(
            "rolling-medium-100mb", "rolling_std", (102400, 128), {"window": 20}, True
        ),
        CaseSpec("expanding-wide-10mb", "expanding_mean", (1280, 1024), {}, True),
        CaseSpec(
            "aggregation-wide-10mb", "dataframe_mean", (1280, 1024), {"axis": 0}, True
        ),
        CaseSpec(
            "aggregation-wide-10mb", "dataframe_sum", (1280, 1024), {"axis": 0}, True
        ),
        CaseSpec(
            "aggregation-medium-100mb",
            "dataframe_mean",
            (102400, 128),
            {"axis": 0},
            True,
        ),
        CaseSpec(
            "aggregation-medium-100mb",
            "dataframe_sum",
            (102400, 128),
            {"axis": 0},
            True,
        ),
        CaseSpec(
            "aggregation-axis1-wide-32mb",
            "dataframe_sum",
            (8192, 512),
            {"axis": 1},
            True,
        ),
        CaseSpec(
            "aggregation-axis1-wide-32mb",
            "dataframe_mean",
            (8192, 512),
            {"axis": 1},
            True,
        ),
        CaseSpec(
            "aggregation-axis1-wide-32mb",
            "dataframe_std",
            (8192, 512),
            {"axis": 1},
            True,
        ),
        CaseSpec(
            "aggregation-axis1-wide-32mb",
            "dataframe_var",
            (8192, 512),
            {"axis": 1},
            True,
        ),
        CaseSpec(
            "aggregation-axis1-wide-32mb",
            "dataframe_min",
            (8192, 512),
            {"axis": 1},
            True,
        ),
        CaseSpec(
            "aggregation-axis1-wide-32mb",
            "dataframe_max",
            (8192, 512),
            {"axis": 1},
            True,
        ),
        CaseSpec(
            "cumulative-axis1-wide-32mb",
            "dataframe_cumsum",
            (8192, 512),
            {"axis": 1},
            True,
        ),
        CaseSpec(
            "cumulative-axis1-wide-32mb",
            "dataframe_cumprod",
            (8192, 512),
            {"axis": 1},
            True,
        ),
        CaseSpec(
            "cumulative-axis1-wide-32mb",
            "dataframe_cummin",
            (8192, 512),
            {"axis": 1},
            True,
        ),
        CaseSpec(
            "cumulative-axis1-wide-32mb",
            "dataframe_cummax",
            (8192, 512),
            {"axis": 1},
            True,
        ),
        CaseSpec(
            "transform-axis1-wide-32mb",
            "dataframe_diff",
            (8192, 512),
            {"axis": 1},
            False,
        ),
        CaseSpec(
            "transform-axis1-wide-32mb",
            "dataframe_shift",
            (8192, 512),
            {"axis": 1},
            False,
        ),
        CaseSpec(
            "transform-axis1-wide-32mb",
            "dataframe_pct_change",
            (8192, 512),
            {"axis": 1, "fill_method": None},
            False,
        ),
        CaseSpec("rank-wide-1mb-control", "rank_axis1", (128, 1024), {"axis": 1}, False),
        CaseSpec("rank-axis1-wide-32mb", "rank_axis1", (8192, 512), {"axis": 1}, True),
        CaseSpec(
            "pairwise-safe-rolling-corr",
            "rolling_corr",
            (4096, 64),
            {"window": 20},
            True,
        ),
        CaseSpec(
            "pairwise-safe-rolling-corr",
            "rolling_cov",
            (4096, 64),
            {"window": 20},
            True,
        ),
    ]


def _matches_case(spec: CaseSpec, patterns: List[str]) -> bool:
    if not patterns:
        return True
    names = {spec.case_id, spec.operation, f"{spec.case_id}:{spec.operation}"}
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns for name in names)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module_path_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(_repo_root() / "src")
    env["PYTHONPATH"] = src + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def _rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


def _maxrss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(value if sys.platform == "darwin" else value * 1024)


def _thread_count() -> int:
    task_dir = Path("/proc/self/task")
    if task_dir.exists():
        try:
            return len(list(task_dir.iterdir()))
        except Exception:
            return threading.active_count()
    return threading.active_count()


class _ThreadSampler:
    def __init__(self, interval: float = 0.002) -> None:
        self.interval = interval
        self.peak_threads = _thread_count()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="resource-profiler-sampler", daemon=True
        )

    def __enter__(self) -> "_ThreadSampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_threads = max(self.peak_threads, _thread_count())
            time.sleep(self.interval)


def _resource_config_snapshot() -> Dict[str, Any]:
    try:
        import unlockedpd

        cfg = unlockedpd.config
        return {
            "num_threads": cfg.num_threads,
            "threadpool_workers": getattr(cfg, "threadpool_workers", 0) or "auto",
            "max_memory_overhead": getattr(cfg, "max_memory_overhead", 6.0),
            "max_cpu_overhead": getattr(cfg, "max_cpu_overhead", 6.0),
            "warmup": getattr(
                cfg, "warmup", os.environ.get("UNLOCKEDPD_WARMUP", "legacy_eager")
            ),
        }
    except Exception:
        return {
            "num_threads": int(os.environ.get("UNLOCKEDPD_NUM_THREADS", "0") or 0),
            "threadpool_workers": os.environ.get(
                "UNLOCKEDPD_THREADPOOL_WORKERS", "auto"
            ),
            "max_memory_overhead": float(
                os.environ.get("UNLOCKEDPD_MAX_MEMORY_OVERHEAD", "6.0") or 6.0
            ),
            "max_cpu_overhead": float(
                os.environ.get("UNLOCKEDPD_MAX_CPU_OVERHEAD", "6.0") or 6.0
            ),
            "warmup": os.environ.get("UNLOCKEDPD_WARMUP", "legacy_eager"),
        }


def _memory_total_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return 0


def _git_snapshot() -> Dict[str, Any]:
    def run(cmd: List[str]) -> str:
        try:
            return subprocess.check_output(
                cmd, cwd=_repo_root(), text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return ""

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "status", "--porcelain"])),
    }


def _library_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in ("pandas", "numpy", "numba", "unlockedpd"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except Exception:
            versions[name] = None
    return versions


def _dataframe(shape: tuple[int, int], seed: int):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(shape)
    return pd.DataFrame(arr, columns=[f"c{i}" for i in range(shape[1])])


def _run_operation(
    spec: Dict[str, Any],
    seed: int,
    implementation: str,
    df: Any = None,
) -> Tuple[Any, str]:
    operation = spec["operation"]
    shape = tuple(spec["shape"]) if spec.get("shape") else None
    params = spec.get("params") or {}

    if implementation == "optimized":
        import unlockedpd  # noqa: F401 - import applies patches
    else:
        import pandas as pd  # noqa: F401

    if operation == "import_unlockedpd":
        if implementation == "optimized":
            import unlockedpd  # noqa: F401

            return "imported", "optimized_import"
        import pandas as pd  # noqa: F401

        return "imported", "pandas"

    if shape is None:
        raise ValueError(f"shape required for {operation}")
    if df is None:
        df = _dataframe(shape, seed)

    if implementation == "optimized":
        selected_path = _infer_selected_path(operation, shape)
    else:
        selected_path = "pandas"

    if operation == "rolling_mean":
        result = df.rolling(
            params.get("window", 20), axis=params.get("axis", 0)
        ).mean()
    elif operation == "rolling_sum":
        result = df.rolling(
            params.get("window", 20), axis=params.get("axis", 0)
        ).sum()
    elif operation == "rolling_std":
        result = df.rolling(
            params.get("window", 20), axis=params.get("axis", 0)
        ).std()
    elif operation == "rolling_var":
        result = df.rolling(
            params.get("window", 20), axis=params.get("axis", 0)
        ).var()
    elif operation == "rolling_min":
        result = df.rolling(
            params.get("window", 20), axis=params.get("axis", 0)
        ).min()
    elif operation == "rolling_max":
        result = df.rolling(
            params.get("window", 20), axis=params.get("axis", 0)
        ).max()
    elif operation == "rolling_corr":
        result = df.rolling(params.get("window", 20)).corr()
    elif operation == "rolling_cov":
        result = df.rolling(params.get("window", 20)).cov()
    elif operation == "expanding_mean":
        result = df.expanding().mean()
    elif operation == "dataframe_mean":
        result = df.mean(axis=params.get("axis", 0))
    elif operation == "dataframe_sum":
        result = df.sum(axis=params.get("axis", 0))
    elif operation == "dataframe_std":
        result = df.std(axis=params.get("axis", 0))
    elif operation == "dataframe_var":
        result = df.var(axis=params.get("axis", 0))
    elif operation == "dataframe_min":
        result = df.min(axis=params.get("axis", 0))
    elif operation == "dataframe_max":
        result = df.max(axis=params.get("axis", 0))
    elif operation == "dataframe_cumsum":
        result = df.cumsum(axis=params.get("axis", 0))
    elif operation == "dataframe_cumprod":
        result = df.cumprod(axis=params.get("axis", 0))
    elif operation == "dataframe_cummin":
        result = df.cummin(axis=params.get("axis", 0))
    elif operation == "dataframe_cummax":
        result = df.cummax(axis=params.get("axis", 0))
    elif operation == "dataframe_diff":
        result = df.diff(axis=params.get("axis", 0), periods=params.get("periods", 1))
    elif operation == "dataframe_shift":
        result = df.shift(
            axis=params.get("axis", 0),
            periods=params.get("periods", 1),
            fill_value=params.get("fill_value", None),
        )
    elif operation == "dataframe_pct_change":
        result = df.pct_change(
            axis=params.get("axis", 0),
            periods=params.get("periods", 1),
            fill_method=params.get("fill_method", None),
        )
    elif operation == "rank_axis1":
        result = df.rank(axis=1)
    else:
        raise ValueError(f"unknown operation: {operation}")

    if implementation == "optimized":
        try:
            from unlockedpd._resources import get_last_selected_path

            selected_path = get_last_selected_path() or selected_path
        except Exception:
            pass

    return result, selected_path


def _checksum(result: Any) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    if isinstance(result, (pd.DataFrame, pd.Series)):
        values = result.to_numpy(dtype=float, copy=False)
        return {
            "shape": list(result.shape),
            "nanmean": float(np.nanmean(values)) if values.size else math.nan,
        }
    return {"repr": repr(result)[:80]}


def _infer_selected_path(operation: str, shape: tuple[int, int]) -> str:
    n = shape[0] * shape[1]
    if operation.startswith("rolling_") and operation not in {
        "rolling_corr",
        "rolling_cov",
    }:
        if n >= 10_000_000:
            return "threadpool"
        if n >= 500_000:
            return "parallel_numba"
        return "serial_numba"
    if operation.startswith("expanding_"):
        if n >= 10_000_000:
            return "threadpool"
        if n >= 500_000:
            return "parallel_numba"
        return "serial_numba"
    if operation.startswith("dataframe_"):
        if operation in {
            "dataframe_cumsum",
            "dataframe_cumprod",
            "dataframe_cummin",
            "dataframe_cummax",
        }:
            return "numpy_vectorized"
        if operation in {
            "dataframe_diff",
            "dataframe_shift",
            "dataframe_pct_change",
        }:
            return "pandas_native"
        if n >= 10_000_000:
            return "threadpool"
        if n >= 500_000:
            return "parallel_numba"
        return "serial_numba"
    if operation == "rank_axis1":
        if n < 500_000:
            return "pandas_native"
        return "parallel_numba"
    if operation in {"rolling_corr", "rolling_cov"}:
        return "threadpool"
    return "optimized"


def _worker_main(args: argparse.Namespace) -> int:
    spec = json.loads(args.worker_case)
    implementation = args.worker_implementation
    mode = args.worker_mode
    operation = str(spec.get("operation", ""))
    shape = tuple(spec["shape"]) if spec.get("shape") else None

    # Keep pandas baseline isolated from unlockedpd monkey patches.
    if implementation == "pandas":
        os.environ["UNLOCKEDPD_ENABLED"] = "false"

    prepared_df = None
    if operation != "import_unlockedpd":
        if shape is None:
            raise ValueError(f"shape required for {operation}")
        # Input construction can dominate sub-10ms DataFrame reductions. Build
        # data outside the measured section so speedups reflect the operation
        # backend rather than random-number/DataFrame setup cost.
        prepared_df = _dataframe(shape, args.worker_seed)

    if mode == "warm" and implementation == "optimized":
        import unlockedpd

        if getattr(unlockedpd.config, "warmup", "lazy") in {"eager", "full"}:
            unlockedpd._warmup_all()

    if mode == "warm":
        # Warm means the measured repeat excludes import/JIT/cache population for
        # this specific operation, not just that the package was imported.  Run
        # one unmeasured operation first in the same isolated worker process.
        _run_operation(spec, args.worker_seed, implementation, prepared_df)

    before_rss = _rss_bytes()
    before_ru = resource.getrusage(resource.RUSAGE_SELF)
    before_wall = time.perf_counter()
    selected_path = "pandas" if implementation == "pandas" else "optimized"
    error = None
    checksum: Any = None
    checksum_wall = 0.0
    checksum_cpu = 0.0
    try:
        with _ThreadSampler() as sampler:
            result, selected_path = _run_operation(
                spec, args.worker_seed, implementation, prepared_df
            )
        after_wall = time.perf_counter()
        after_ru = resource.getrusage(resource.RUSAGE_SELF)
        after_rss = _rss_bytes()
        checksum_before_wall = time.perf_counter()
        checksum_before_ru = resource.getrusage(resource.RUSAGE_SELF)
        checksum = _checksum(result)
        checksum_after_ru = resource.getrusage(resource.RUSAGE_SELF)
        checksum_wall = max(0.0, time.perf_counter() - checksum_before_wall)
        checksum_cpu = max(
            0.0,
            (checksum_after_ru.ru_utime - checksum_before_ru.ru_utime)
            + (checksum_after_ru.ru_stime - checksum_before_ru.ru_stime),
        )
        peak_threads = max(1, sampler.peak_threads - 1)  # subtract sampler thread
    except Exception as exc:  # worker reports failure as data so driver can continue
        error = f"{type(exc).__name__}: {exc}"
        peak_threads = max(1, _thread_count())
        if implementation == "optimized" and "ResourceBudgetExceeded" in error:
            selected_path = "fallback"
        after_wall = time.perf_counter()
        after_ru = resource.getrusage(resource.RUSAGE_SELF)
        after_rss = _rss_bytes()

    record = {
        "implementation": implementation,
        "selected_path": selected_path,
        "wall_seconds": max(0.0, after_wall - before_wall),
        "user_cpu_seconds": max(0.0, after_ru.ru_utime - before_ru.ru_utime),
        "system_cpu_seconds": max(0.0, after_ru.ru_stime - before_ru.ru_stime),
        "checksum_wall_seconds": checksum_wall,
        "checksum_cpu_seconds": checksum_cpu,
        "peak_rss_bytes": max(_maxrss_bytes(), before_rss, after_rss),
        "rss_delta_bytes": after_rss - before_rss,
        "peak_threads": int(peak_threads),
        "final_threads": int(_thread_count()),
        "checksum": checksum,
    }
    if error:
        record["error"] = error
    print(json.dumps(record, sort_keys=True))
    return 0


def _run_worker(
    spec: CaseSpec, seed: int, repeat: int, mode: str, implementation: str
) -> Dict[str, Any]:
    env = _module_path_env()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-case",
        json.dumps(spec.worker_payload(), sort_keys=True),
        "--_worker-seed",
        str(seed + repeat),
        "--_worker-mode",
        mode,
        "--_worker-implementation",
        implementation,
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=_repo_root(), env=env, text=True, capture_output=True
    )
    duration = time.perf_counter() - start
    if proc.returncode != 0:
        return {
            "implementation": implementation,
            "selected_path": "failed",
            "wall_seconds": duration,
            "user_cpu_seconds": 0.0,
            "system_cpu_seconds": 0.0,
            "peak_rss_bytes": 0,
            "rss_delta_bytes": 0,
            "peak_threads": 0,
            "final_threads": 0,
            "error": proc.stderr.strip()
            or proc.stdout.strip()
            or f"exit {proc.returncode}",
        }
    line = proc.stdout.strip().splitlines()[-1]
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {
            "implementation": implementation,
            "selected_path": "failed",
            "wall_seconds": duration,
            "user_cpu_seconds": 0.0,
            "system_cpu_seconds": 0.0,
            "peak_rss_bytes": 0,
            "rss_delta_bytes": 0,
            "peak_threads": 0,
            "final_threads": 0,
            "error": f"invalid worker JSON: {proc.stdout[-500:]} {proc.stderr[-500:]}",
        }


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _summarize(
    records: list[Dict[str, Any]],
    parallelism_gate: bool,
    max_memory: float,
    max_cpu: float,
) -> Dict[str, Any]:
    pandas = [
        r for r in records if r.get("implementation") == "pandas" and not r.get("error")
    ]
    opt = [
        r
        for r in records
        if r.get("implementation") == "optimized" and not r.get("error")
    ]

    def mean(items: List[float]) -> Optional[float]:
        return sum(items) / len(items) if items else None

    pandas_wall = mean([float(r["wall_seconds"]) for r in pandas])
    opt_wall = mean([float(r["wall_seconds"]) for r in opt])
    pandas_cpu = mean(
        [float(r["user_cpu_seconds"]) + float(r["system_cpu_seconds"]) for r in pandas]
    )
    opt_cpu = mean(
        [float(r["user_cpu_seconds"]) + float(r["system_cpu_seconds"]) for r in opt]
    )
    pandas_rss = mean(
        [
            max(
                1.0,
                abs(float(r.get("rss_delta_bytes", 0)))
                or float(r.get("peak_rss_bytes", 0)),
            )
            for r in pandas
        ]
    )
    opt_rss = mean(
        [
            max(
                1.0,
                abs(float(r.get("rss_delta_bytes", 0)))
                or float(r.get("peak_rss_bytes", 0)),
            )
            for r in opt
        ]
    )
    speedup = (
        _safe_ratio(pandas_wall or 0.0, opt_wall or 0.0)
        if pandas_wall is not None and opt_wall is not None
        else None
    )
    cpu_ratio = (
        _safe_ratio(opt_cpu or 0.0, pandas_cpu or 0.0)
        if pandas_cpu is not None and opt_cpu is not None
        else None
    )
    rss_ratio = (
        _safe_ratio(opt_rss or 0.0, pandas_rss or 0.0)
        if pandas_rss is not None and opt_rss is not None
        else None
    )
    selected_paths = sorted({str(r.get("selected_path", "unknown")) for r in opt})
    selected_parallel = any(
        p in {"threadpool", "parallel_numba", "numpy_vectorized"}
        for p in selected_paths
    )
    resource_ok = (cpu_ratio is None or cpu_ratio <= max_cpu) and (
        rss_ratio is None or rss_ratio <= max_memory
    )
    speedup_weighted_resource_limit = (
        speedup * 1.2 if speedup is not None and math.isfinite(speedup) else None
    )
    speedup_weighted_resource_ok = speedup_weighted_resource_limit is None or (
        (cpu_ratio is None or cpu_ratio <= speedup_weighted_resource_limit)
        and (rss_ratio is None or rss_ratio <= speedup_weighted_resource_limit)
    )
    return {
        "pandas_wall_seconds": pandas_wall,
        "optimized_wall_seconds": opt_wall,
        "speedup": speedup,
        "cpu_seconds_ratio": cpu_ratio,
        "rss_ratio": rss_ratio,
        "selected_path_optimized": selected_paths[0]
        if len(selected_paths) == 1
        else selected_paths,
        "pass_resource_budget": bool(resource_ok),
        "speedup_weighted_resource_limit": speedup_weighted_resource_limit,
        "pass_speedup_weighted_resource_budget": bool(speedup_weighted_resource_ok),
        "pass_parallelism_gate": bool(
            (not parallelism_gate) or selected_parallel or not resource_ok
        ),
        "fallback_reason": None
        if resource_ok
        else f"resource budget exceeded: cpu_ratio={cpu_ratio}, rss_ratio={rss_ratio}",
    }


def _load_comparisons(patterns: List[str]) -> list[Dict[str, Any]]:
    loaded = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                data = json.loads(Path(path).read_text())
                loaded.append(
                    {
                        "path": path,
                        "run_id": data.get("run_id"),
                        "schema_version": data.get("schema_version"),
                    }
                )
            except Exception as exc:
                loaded.append({"path": path, "error": str(exc)})
    return loaded


def _driver_main(args: argparse.Namespace) -> int:
    if not args.output:
        raise SystemExit("--output is required")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    patterns = args.case_filter or []
    selected_specs = [spec for spec in _case_matrix() if _matches_case(spec, patterns)]
    if args.max_case_elements:
        selected_specs = [
            spec
            for spec in selected_specs
            if spec.shape is None
            or spec.shape[0] * spec.shape[1] <= args.max_case_elements
        ]

    modes = ["cold", "warm"] if args.cold_and_warm else ["warm"]
    cases = []
    max_memory = float(os.environ.get("UNLOCKEDPD_MAX_MEMORY_OVERHEAD", "6.0") or 6.0)
    max_cpu = float(os.environ.get("UNLOCKEDPD_MAX_CPU_OVERHEAD", "6.0") or 6.0)

    for spec in selected_specs:
        for mode in modes:
            repeats: list[Dict[str, Any]] = []
            for repeat in range(args.repeats):
                for implementation in ("pandas", "optimized"):
                    record = _run_worker(spec, args.seed, repeat, mode, implementation)
                    record["repeat_index"] = repeat
                    repeats.append(record)
            case = spec.worker_payload()
            case["seed"] = args.seed
            case["mode"] = mode
            case["repeats"] = repeats
            case["summary"] = _summarize(
                repeats, spec.parallelism_gate, max_memory, max_cpu
            )
            cases.append(case)
            print(f"profiled {spec.case_id}:{spec.operation}:{mode}", file=sys.stderr)

    profile = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "git": _git_snapshot(),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_logical": os.cpu_count() or 0,
            "memory_total_bytes": _memory_total_bytes(),
        },
        "libraries": _library_versions(),
        "config": _resource_config_snapshot(),
        "seed": args.seed,
        "repeats_requested": args.repeats,
        "cold_and_warm": bool(args.cold_and_warm),
        "cases": cases,
    }
    if args.compare:
        profile["comparisons"] = _load_comparisons(args.compare)

    output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile unlockedpd resource overhead vs pandas"
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cold-and-warm", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--compare", nargs="*", default=[])
    parser.add_argument(
        "--case-filter",
        action="append",
        help="glob matched against case_id, operation, or case_id:operation",
    )
    parser.add_argument(
        "--max-case-elements",
        type=int,
        default=0,
        help="optional local-smoke size guard",
    )
    parser.add_argument("--_worker-case", dest="worker_case")
    parser.add_argument("--_worker-seed", dest="worker_seed", type=int, default=0)
    parser.add_argument("--_worker-mode", dest="worker_mode", choices=["cold", "warm"])
    parser.add_argument(
        "--_worker-implementation",
        dest="worker_implementation",
        choices=["pandas", "optimized"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.worker_case:
        return _worker_main(args)
    return _driver_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
