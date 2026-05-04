"""Tests for config and shared resource-policy helpers."""
import importlib

import numpy as np


def test_threadpool_workers_env_and_runtime_precedence(monkeypatch):
    monkeypatch.setenv("UNLOCKEDPD_THREADPOOL_WORKERS", "7")

    import unlockedpd._config as config_module

    cfg = config_module.UnlockedConfig()
    assert cfg.threadpool_workers == 7

    cfg.threadpool_workers = 3
    assert cfg.threadpool_workers == 3

    cfg.threadpool_workers = "auto"
    assert cfg.threadpool_workers == 0


def test_threadpool_workers_zero_auto_invalid_env(monkeypatch):
    import unlockedpd._config as config_module

    for value in ("0", "auto", "not-an-int", "-4"):
        monkeypatch.setenv("UNLOCKEDPD_THREADPOOL_WORKERS", value)
        assert config_module.UnlockedConfig().threadpool_workers == 0


def test_threadpool_workers_does_not_change_numba_threads(monkeypatch):
    import unlockedpd._config as config_module

    calls = []
    monkeypatch.setattr(config_module.numba, "set_num_threads", lambda value: calls.append(value))

    cfg = config_module.UnlockedConfig()
    cfg.threadpool_workers = 4
    assert calls == []

    cfg.num_threads = 2
    assert calls == [2]


def test_resource_budget_config_from_env(monkeypatch):
    monkeypatch.setenv("UNLOCKEDPD_MAX_MEMORY_OVERHEAD", "4.5")
    monkeypatch.setenv("UNLOCKEDPD_MAX_CPU_OVERHEAD", "5.25")

    import unlockedpd._config as config_module

    cfg = config_module.UnlockedConfig()
    assert cfg.max_memory_overhead == 4.5
    assert cfg.max_cpu_overhead == 5.25
    assert cfg.as_dict()["max_memory_overhead"] == 4.5


def test_resolve_threadpool_workers_clamps_all_caps(monkeypatch):
    import unlockedpd._resources as resources

    monkeypatch.setattr(resources.os, "cpu_count", lambda: 8)

    assert resources.resolve_threadpool_workers(
        work_units=100,
        configured_workers=12,
        operation_cap=5,
        memory_bandwidth_cap=6,
    ) == 5
    assert resources.resolve_threadpool_workers(
        work_units=3,
        configured_workers=0,
        memory_bandwidth_cap=32,
    ) == 3
    assert resources.resolve_threadpool_workers(
        work_units=None,
        configured_workers=0,
        memory_bandwidth_cap=32,
    ) == 8


def test_memory_estimates_and_budget_helpers():
    resources = importlib.import_module("unlockedpd._resources")

    arr = np.zeros((2, 3), dtype=np.float32)
    assert resources.estimate_array_nbytes(arr) == 24
    assert resources.estimate_array_nbytes((2, 3), dtype=np.float64) == 48
    assert resources.estimate_operation_nbytes((2, 3), dtype=np.float64, inputs=1, outputs=2) == 144

    assert resources.overhead_ratio(12, 3) == 4
    assert resources.check_resource_budget(
        memory_overhead=4,
        cpu_overhead=7,
        max_memory_overhead=6,
        max_cpu_overhead=6,
    ).pass_budget is False
