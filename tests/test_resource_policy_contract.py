"""Static contract tests for resource-policy integration.

These tests intentionally avoid importing ``unlockedpd`` so they can run in
minimal verification environments. They skip before the worker-2/worker-3
implementation slices are integrated, then enforce the coordination contract
once ``_resources.py`` and ``ops/_threadpool.py`` exist together.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(
    os.environ.get("UNLOCKEDPD_CONTRACT_REPO_ROOT", Path(__file__).resolve().parents[1])
)
RESOURCES_PATH = REPO_ROOT / "src" / "unlockedpd" / "_resources.py"
THREADPOOL_PATH = REPO_ROOT / "src" / "unlockedpd" / "ops" / "_threadpool.py"


@pytest.fixture(autouse=True)
def reset_unlockedpd():
    """Override the package-reset fixture; these tests read source text only."""

    yield


def _resource_policy_sources() -> tuple[str, str]:
    if not RESOURCES_PATH.exists() or not THREADPOOL_PATH.exists():
        pytest.skip("resource policy/threadpool integration slice not present yet")
    return (
        RESOURCES_PATH.read_text(encoding="utf-8"),
        THREADPOOL_PATH.read_text(encoding="utf-8"),
    )


def test_threadpool_worker_resolution_delegates_to_resources():
    resources_source, threadpool_source = _resource_policy_sources()

    assert "def resolve_threadpool_workers" in resources_source
    assert "def make_threadpool_chunks" in threadpool_source
    assert ".._resources" in threadpool_source
    assert (
        "resolve_threadpool_workers" in threadpool_source
        or "threadpool_chunks" in threadpool_source
    )


def test_threadpool_wrapper_does_not_duplicate_resource_budget_policy():
    resources_source, threadpool_source = _resource_policy_sources()

    assert "def check_resource_budget" in resources_source
    assert "max_memory_overhead" in resources_source
    assert "max_cpu_overhead" in resources_source

    forbidden_policy_terms = [
        "max_memory_overhead",
        "max_cpu_overhead",
        "UNLOCKEDPD_THREADPOOL_WORKERS",
        "UNLOCKEDPD_MAX_MEMORY_OVERHEAD",
        "UNLOCKEDPD_MAX_CPU_OVERHEAD",
        "config.threadpool_workers",
    ]
    for term in forbidden_policy_terms:
        assert term not in threadpool_source


def test_resources_defines_config_and_budget_contracts():
    resources_source, _threadpool_source = _resource_policy_sources()

    expected_definitions = [
        "def check_resource_budget",
        "def resolve_threadpool_workers",
    ]
    for definition in expected_definitions:
        assert definition in resources_source
