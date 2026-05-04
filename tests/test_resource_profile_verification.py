"""Tests for the resource profiling evidence verifier."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.compare_resource_profiles import (
    compare_profiles,
    _parse_project_dependencies,
    render_markdown_report,
    validate_profile_schema,
)


@pytest.fixture(autouse=True)
def reset_unlockedpd():
    """Override the package-reset fixture; these tests validate a stdlib tool."""

    yield


def _repeat(implementation: str, selected_path: str) -> dict:
    return {
        "implementation": implementation,
        "selected_path": selected_path,
        "wall_seconds": 1.0,
        "user_cpu_seconds": 0.8,
        "system_cpu_seconds": 0.1,
        "peak_rss_bytes": 100_000_000,
        "rss_delta_bytes": 10_000_000,
        "peak_threads": 4,
        "final_threads": 1,
    }


def _case(
    case_id: str,
    operation: str,
    *,
    selected_path: str = "threadpool",
    speedup: float = 4.5,
    cpu_ratio: float = 2.0,
    rss_ratio: float = 2.0,
    pass_resource_budget: bool = True,
    selected_path_optimized: bool = True,
    reason: str = "",
) -> dict:
    summary = {
        "speedup": speedup,
        "cpu_seconds_ratio": cpu_ratio,
        "rss_ratio": rss_ratio,
        "selected_path": selected_path,
        "selected_path_optimized": selected_path_optimized,
        "pass_resource_budget": pass_resource_budget,
        "pass_parallelism_gate": True,
    }
    if reason:
        summary["fallback_reason"] = reason

    return {
        "case_id": case_id,
        "operation": operation,
        "shape": [128, 64],
        "seed": 12345,
        "mode": "warm",
        "repeats": [
            _repeat("pandas", "pandas"),
            _repeat("optimized", selected_path),
        ],
        "summary": summary,
    }


def _profile(*cases: dict) -> dict:
    return {
        "schema_version": "resource-profile-v1",
        "run_id": "20260504T000000Z",
        "git": {"commit": "abc123", "dirty": False},
        "machine": {
            "platform": "linux",
            "python": sys.version.split()[0],
            "cpu_logical": 8,
            "memory_total_bytes": 16_000_000_000,
        },
        "config": {
            "num_threads": 0,
            "threadpool_workers": "auto",
            "max_memory_overhead": 6.0,
            "max_cpu_overhead": 6.0,
            "warmup": "lazy",
        },
        "cases": list(cases),
    }


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_profile_schema_requires_resource_metrics():
    profile = _profile(_case("rolling-wide-10mb", "rolling_mean"))

    issues = validate_profile_schema(profile, "sample")

    assert issues == []


def test_profile_comparison_passes_parallel_and_dependency_gates(tmp_path):
    baseline = _profile(
        _case("rolling-wide-10mb", "rolling_mean", speedup=2.0),
        _case(
            "import-only",
            "import",
            selected_path="pandas",
            selected_path_optimized=False,
        ),
    )
    after = _profile(
        _case(
            "rolling-wide-10mb", "rolling_mean", selected_path="threadpool", speedup=4.2
        ),
        _case(
            "import-only",
            "import",
            selected_path="pandas",
            selected_path_optimized=False,
        ),
    )
    baseline_path = _write_json(tmp_path / "baseline.json", baseline)
    after_path = _write_json(tmp_path / "after.json", after)
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        """
[project]
dependencies = [
    "numpy>=1.21",
    "pandas>=1.5,<3.0",
]
""",
        encoding="utf-8",
    )

    result = compare_profiles(baseline_path, after_path, pyproject_path)
    report = render_markdown_report(result)

    assert result.passed
    assert _parse_project_dependencies(pyproject_path) == [
        "numpy>=1.21",
        "pandas>=1.5,<3.0",
    ]
    assert "rolling-wide-10mb" in report
    assert "Warmup/import evidence" in report
    assert result.dependency_rows == [
        {
            "check": "mandatory runtime psutil",
            "path": str(pyproject_path),
            "pass": True,
        }
    ]


def test_profile_comparison_fails_mismatched_case_matrix(tmp_path):
    baseline_path = _write_json(
        tmp_path / "baseline.json",
        _profile(_case("rolling-wide-10mb", "rolling_mean")),
    )
    after_path = _write_json(
        tmp_path / "after.json",
        _profile(_case("expanding-wide-10mb", "expanding_mean")),
    )

    result = compare_profiles(baseline_path, after_path)

    assert not result.passed
    assert any("missing baseline cases" in issue.message for issue in result.issues)
    assert any("absent from baseline" in issue.message for issue in result.issues)


def test_profile_comparison_fails_unjustified_excessive_overhead(tmp_path):
    baseline_path = _write_json(
        tmp_path / "baseline.json",
        _profile(_case("rolling-wide-10mb", "rolling_mean")),
    )
    after_path = _write_json(
        tmp_path / "after.json",
        _profile(
            _case(
                "rolling-wide-10mb",
                "rolling_mean",
                speedup=3.0,
                cpu_ratio=6.5,
                rss_ratio=2.0,
            )
        ),
    )

    result = compare_profiles(baseline_path, after_path)

    assert not result.passed
    assert any("exceeds 6x overhead" in issue.message for issue in result.issues)


def test_profile_comparison_accepts_documented_budget_fallback(tmp_path):
    fallback_case = _case(
        "pairwise-safe-rolling-corr",
        "rolling_corr",
        selected_path="fallback",
        speedup=1.0,
        cpu_ratio=1.0,
        rss_ratio=7.5,
        pass_resource_budget=False,
        selected_path_optimized=False,
        reason="estimated pairwise output would exceed max_memory_overhead",
    )
    baseline_path = _write_json(tmp_path / "baseline.json", _profile(fallback_case))
    after_path = _write_json(tmp_path / "after.json", _profile(fallback_case))

    result = compare_profiles(baseline_path, after_path)

    assert result.passed
    assert result.excessive_overhead_rows[0]["fallback_documented"] is True


def test_profile_comparison_fails_mandatory_psutil_dependency(tmp_path):
    baseline_path = _write_json(
        tmp_path / "baseline.json",
        _profile(_case("rolling-wide-10mb", "rolling_mean")),
    )
    after_path = _write_json(
        tmp_path / "after.json",
        _profile(_case("rolling-wide-10mb", "rolling_mean")),
    )
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        """
[project]
dependencies = [
    "numpy>=1.21",
    "psutil>=5",
]
""",
        encoding="utf-8",
    )

    result = compare_profiles(baseline_path, after_path, pyproject_path)

    assert not result.passed
    assert any("psutil" in issue.message for issue in result.issues)


def test_compare_resource_profiles_cli_writes_markdown(tmp_path):
    baseline_path = _write_json(
        tmp_path / "baseline.json",
        _profile(_case("rolling-wide-10mb", "rolling_mean")),
    )
    after_path = _write_json(
        tmp_path / "after.json",
        _profile(_case("rolling-wide-10mb", "rolling_mean")),
    )
    markdown_path = tmp_path / "evidence.md"

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/compare_resource_profiles.py",
            "--baseline",
            str(baseline_path),
            "--after",
            str(after_path),
            "--markdown",
            str(markdown_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert markdown_path.exists()
    assert "Resource Profile Comparison: PASS" in markdown_path.read_text(
        encoding="utf-8"
    )
