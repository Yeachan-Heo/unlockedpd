"""Compare unlockedpd resource profile artifacts and emit evidence tables.

This verifier is intentionally lightweight and stdlib-only so it can run in CI
or on a developer machine after the profiling harness writes baseline/after
JSON.  It validates that the two artifacts are comparable, checks the resource
budget gates from the resource-leak optimization plan, and summarizes the
evidence needed for the final execution report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


MAX_ACCEPTABLE_OVERHEAD = 6.0
HIGH_OVERHEAD_FLOOR = 4.0
MIN_HIGH_OVERHEAD_SPEEDUP = 4.0

PARALLEL_PATHS = {
    "parallel",
    "parallel_numba",
    "threadpool",
    "thread_pool",
    "optimized_parallel",
}
FALLBACK_PATHS = {"fallback", "pandas", "rejected", "skipped"}
NAMED_PARALLELISM_CASES = {
    "rolling-wide-10mb",
    "rolling-medium-100mb",
    "expanding-wide-10mb",
    "aggregation-wide-10mb",
    "rank-wide-1mb-control",
    "pairwise-safe-rolling-corr",
}
TOP_LEVEL_REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "git",
    "machine",
    "config",
    "cases",
}
CASE_REQUIRED_FIELDS = {
    "case_id",
    "operation",
    "shape",
    "seed",
    "mode",
    "repeats",
    "summary",
}
SUMMARY_REQUIRED_FIELDS = {
    "speedup",
    "cpu_seconds_ratio",
    "rss_ratio",
    "selected_path_optimized",
    "pass_resource_budget",
}
REPEAT_REQUIRED_FIELDS = {
    "implementation",
    "selected_path",
    "wall_seconds",
    "user_cpu_seconds",
    "system_cpu_seconds",
    "peak_rss_bytes",
    "rss_delta_bytes",
    "peak_threads",
    "final_threads",
}


@dataclass(frozen=True, order=True)
class CaseKey:
    """Stable key used to compare baseline and after profile matrices."""

    case_id: str
    operation: str
    shape: tuple[Any, ...]
    mode: str

    @classmethod
    def from_case(cls, case: dict[str, Any]) -> "CaseKey":
        shape = case.get("shape", [])
        if isinstance(shape, list):
            normalized_shape = tuple(shape)
        elif isinstance(shape, tuple):
            normalized_shape = shape
        else:
            normalized_shape = (shape,)
        return cls(
            case_id=str(case.get("case_id", "")),
            operation=str(case.get("operation", "")),
            shape=normalized_shape,
            mode=str(case.get("mode", "")),
        )

    def label(self) -> str:
        shape = "x".join(str(part) for part in self.shape)
        return f"{self.case_id}/{self.operation}/{shape}/{self.mode}"


@dataclass
class ComparisonIssue:
    """A verification issue collected while comparing artifacts."""

    severity: str
    message: str


@dataclass
class ComparisonResult:
    """Structured result of a baseline/after resource profile comparison."""

    baseline_path: Path
    after_path: Path
    baseline: dict[str, Any]
    after: dict[str, Any]
    issues: list[ComparisonIssue] = field(default_factory=list)
    named_parallelism_rows: list[dict[str, Any]] = field(default_factory=list)
    high_overhead_rows: list[dict[str, Any]] = field(default_factory=list)
    excessive_overhead_rows: list[dict[str, Any]] = field(default_factory=list)
    warmup_rows: list[dict[str, Any]] = field(default_factory=list)
    dependency_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "FAIL" for issue in self.issues)


def load_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _missing_fields(mapping: dict[str, Any], required: Iterable[str]) -> list[str]:
    return [field for field in required if field not in mapping]


def validate_profile_schema(
    profile: dict[str, Any], label: str
) -> list[ComparisonIssue]:
    """Validate the minimum schema expected by the PRD/test spec."""

    issues: list[ComparisonIssue] = []
    missing_top = _missing_fields(profile, TOP_LEVEL_REQUIRED_FIELDS)
    if missing_top:
        issues.append(
            ComparisonIssue(
                "FAIL",
                f"{label} missing top-level fields: {', '.join(missing_top)}",
            )
        )
        return issues

    cases = profile.get("cases")
    if not isinstance(cases, list) or not cases:
        issues.append(
            ComparisonIssue("FAIL", f"{label} must contain non-empty cases[]")
        )
        return issues

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            issues.append(
                ComparisonIssue("FAIL", f"{label} cases[{index}] is not an object")
            )
            continue

        case_label = f"{label} {case.get('case_id', f'cases[{index}]')}"
        missing_case = _missing_fields(case, CASE_REQUIRED_FIELDS)
        if missing_case:
            issues.append(
                ComparisonIssue(
                    "FAIL",
                    f"{case_label} missing case fields: {', '.join(missing_case)}",
                )
            )

        summary = case.get("summary")
        if not isinstance(summary, dict):
            issues.append(
                ComparisonIssue("FAIL", f"{case_label} missing summary object")
            )
        else:
            missing_summary = _missing_fields(summary, SUMMARY_REQUIRED_FIELDS)
            if missing_summary:
                issues.append(
                    ComparisonIssue(
                        "FAIL",
                        f"{case_label} summary missing fields: {', '.join(missing_summary)}",
                    )
                )

        repeats = case.get("repeats")
        if not isinstance(repeats, list) or not repeats:
            issues.append(
                ComparisonIssue("FAIL", f"{case_label} must contain repeats[]")
            )
            continue

        for repeat_index, repeat in enumerate(repeats):
            if not isinstance(repeat, dict):
                issues.append(
                    ComparisonIssue(
                        "FAIL",
                        f"{case_label} repeats[{repeat_index}] is not an object",
                    )
                )
                continue
            missing_repeat = _missing_fields(repeat, REPEAT_REQUIRED_FIELDS)
            if missing_repeat:
                issues.append(
                    ComparisonIssue(
                        "FAIL",
                        f"{case_label} repeats[{repeat_index}] missing fields: "
                        f"{', '.join(missing_repeat)}",
                    )
                )

    return issues


def _case_map(profile: dict[str, Any]) -> dict[CaseKey, dict[str, Any]]:
    return {CaseKey.from_case(case): case for case in profile.get("cases", [])}


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _summary_path(case: dict[str, Any]) -> str:
    summary = case.get("summary", {})
    for field_name in (
        "selected_path",
        "optimized_path",
        "selected_path_optimized_name",
    ):
        value = summary.get(field_name)
        if value:
            return str(value)

    optimized_repeats = [
        repeat
        for repeat in case.get("repeats", [])
        if str(repeat.get("implementation", "")).lower() == "optimized"
    ]
    if optimized_repeats:
        return str(optimized_repeats[-1].get("selected_path", "unknown"))

    return "unknown"


def _fallback_reason(case: dict[str, Any]) -> str:
    summary = case.get("summary", {})
    for field_name in (
        "fallback_reason",
        "resource_budget_reason",
        "reason",
        "rejection_reason",
    ):
        value = summary.get(field_name)
        if value:
            return str(value)
    return ""


def _uses_fallback(case: dict[str, Any]) -> bool:
    path = _summary_path(case).lower()
    if path in FALLBACK_PATHS:
        return True
    summary = case.get("summary", {})
    return not _bool_value(summary.get("selected_path_optimized"), default=False)


def _uses_parallel_path(case: dict[str, Any]) -> bool:
    path = _summary_path(case).lower()
    return path in PARALLEL_PATHS or (
        _bool_value(case.get("summary", {}).get("selected_path_optimized"))
        and "parallel" in path
    )


def _case_ratios(case: dict[str, Any]) -> tuple[float, float, float]:
    summary = case.get("summary", {})
    return (
        _float_value(summary.get("rss_ratio")),
        _float_value(summary.get("cpu_seconds_ratio")),
        _float_value(summary.get("speedup")),
    )


def _resource_budget_passes(case: dict[str, Any]) -> bool:
    rss_ratio, cpu_ratio, _speedup = _case_ratios(case)
    summary = case.get("summary", {})
    explicit = summary.get("pass_resource_budget")
    if explicit is not None:
        return _bool_value(explicit)
    return rss_ratio <= MAX_ACCEPTABLE_OVERHEAD and cpu_ratio <= MAX_ACCEPTABLE_OVERHEAD


def _is_import_or_warmup_case(case: dict[str, Any]) -> bool:
    case_id = str(case.get("case_id", "")).lower()
    operation = str(case.get("operation", "")).lower()
    return (
        "import" in case_id or "warmup" in case_id or operation in {"import", "warmup"}
    )


def _compare_case_matrix(
    result: ComparisonResult,
    baseline_cases: dict[CaseKey, dict[str, Any]],
    after_cases: dict[CaseKey, dict[str, Any]],
) -> None:
    baseline_keys = set(baseline_cases)
    after_keys = set(after_cases)
    missing_after = sorted(baseline_keys - after_keys)
    missing_baseline = sorted(after_keys - baseline_keys)
    if missing_after:
        result.issues.append(
            ComparisonIssue(
                "FAIL",
                "after profile missing baseline cases: "
                + ", ".join(key.label() for key in missing_after),
            )
        )
    if missing_baseline:
        result.issues.append(
            ComparisonIssue(
                "FAIL",
                "after profile contains cases absent from baseline: "
                + ", ".join(key.label() for key in missing_baseline),
            )
        )


def _evaluate_after_cases(
    result: ComparisonResult,
    after_cases: dict[CaseKey, dict[str, Any]],
) -> None:
    for key, case in sorted(after_cases.items()):
        rss_ratio, cpu_ratio, speedup = _case_ratios(case)
        max_ratio = max(rss_ratio, cpu_ratio)
        selected_path = _summary_path(case)
        fallback_reason = _fallback_reason(case)
        uses_fallback = _uses_fallback(case)
        budget_passes = _resource_budget_passes(case)

        row = {
            "case": key.label(),
            "rss_ratio": rss_ratio,
            "cpu_seconds_ratio": cpu_ratio,
            "speedup": speedup,
            "selected_path": selected_path,
            "reason": fallback_reason,
        }

        if key.case_id in NAMED_PARALLELISM_CASES:
            parallel_gate = bool(budget_passes and _uses_parallel_path(case)) or (
                not budget_passes and uses_fallback and bool(fallback_reason)
            )
            parallel_row = dict(row)
            parallel_row["pass_parallelism_gate"] = parallel_gate
            result.named_parallelism_rows.append(parallel_row)
            if not parallel_gate:
                result.issues.append(
                    ComparisonIssue(
                        "FAIL",
                        f"{key.label()} did not preserve or justify the parallelism gate",
                    )
                )

        if max_ratio > MAX_ACCEPTABLE_OVERHEAD:
            excessive_row = dict(row)
            excessive_row["fallback_documented"] = uses_fallback and bool(
                fallback_reason
            )
            result.excessive_overhead_rows.append(excessive_row)
            if not excessive_row["fallback_documented"]:
                result.issues.append(
                    ComparisonIssue(
                        "FAIL",
                        f"{key.label()} exceeds {MAX_ACCEPTABLE_OVERHEAD:g}x overhead "
                        "without documented fallback/rejection",
                    )
                )
        elif max_ratio > HIGH_OVERHEAD_FLOOR:
            high_row = dict(row)
            high_row["pass_speedup_gate"] = speedup >= MIN_HIGH_OVERHEAD_SPEEDUP
            result.high_overhead_rows.append(high_row)
            if not high_row["pass_speedup_gate"]:
                result.issues.append(
                    ComparisonIssue(
                        "FAIL",
                        f"{key.label()} is in the 4-6x overhead band without "
                        f">={MIN_HIGH_OVERHEAD_SPEEDUP:g}x speedup",
                    )
                )

        if _is_import_or_warmup_case(case):
            result.warmup_rows.append(row)


def _parse_project_dependencies(pyproject_path: Path) -> list[str]:
    """Extract mandatory project.dependencies without third-party dependencies."""

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
        tomllib = None

    if tomllib is not None:
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        dependencies = data.get("project", {}).get("dependencies", [])
        return [str(dependency) for dependency in dependencies]

    text = pyproject_path.read_text(encoding="utf-8")
    in_project = False
    in_dependencies = False
    dependencies: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            in_dependencies = False
            continue

        if not in_project:
            continue

        if line.startswith("dependencies"):
            in_dependencies = "[" in line and "]" not in line
            remainder = line.split("[", 1)[-1] if "[" in line else ""
            if "]" in remainder:
                remainder = remainder.split("]", 1)[0]
                in_dependencies = False
            dependencies.extend(_extract_dependency_literals(remainder))
            continue

        if in_dependencies:
            if "]" in line:
                line = line.split("]", 1)[0]
                in_dependencies = False
            dependencies.extend(_extract_dependency_literals(line))

    return dependencies


def _extract_dependency_literals(text: str) -> list[str]:
    dependencies: list[str] = []
    for part in text.split(","):
        cleaned = part.strip().strip("'\"")
        if cleaned and not cleaned.startswith("#"):
            dependencies.append(cleaned)
    return dependencies


def check_runtime_dependencies(
    result: ComparisonResult,
    pyproject_path: Path | None,
) -> None:
    if pyproject_path is None:
        return

    dependencies = _parse_project_dependencies(pyproject_path)
    mandatory_psutil = any(_dependency_name(dep) == "psutil" for dep in dependencies)
    row = {
        "check": "mandatory runtime psutil",
        "path": str(pyproject_path),
        "pass": not mandatory_psutil,
    }
    result.dependency_rows.append(row)
    if mandatory_psutil:
        result.issues.append(
            ComparisonIssue(
                "FAIL", "psutil is listed as a mandatory runtime dependency"
            )
        )


def _dependency_name(dependency: str) -> str:
    return re.split(r"[<>=!~;\s\[]", dependency, maxsplit=1)[0].strip().lower()


def compare_profiles(
    baseline_path: Path,
    after_path: Path,
    pyproject_path: Path | None = None,
) -> ComparisonResult:
    baseline = load_profile(baseline_path)
    after = load_profile(after_path)
    result = ComparisonResult(
        baseline_path=baseline_path,
        after_path=after_path,
        baseline=baseline,
        after=after,
    )

    result.issues.extend(validate_profile_schema(baseline, "baseline"))
    result.issues.extend(validate_profile_schema(after, "after"))
    if any(issue.severity == "FAIL" for issue in result.issues):
        check_runtime_dependencies(result, pyproject_path)
        return result

    baseline_cases = _case_map(baseline)
    after_cases = _case_map(after)
    _compare_case_matrix(result, baseline_cases, after_cases)
    _evaluate_after_cases(result, after_cases)
    check_runtime_dependencies(result, pyproject_path)
    return result


def _markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_None._"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_markdown_report(result: ComparisonResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    config_after = result.after.get("config", {})
    warmup_default = config_after.get("warmup", "unknown")
    lines = [
        f"# Resource Profile Comparison: {status}",
        "",
        f"- Baseline artifact: `{result.baseline_path}`",
        f"- After artifact: `{result.after_path}`",
        f"- Warmup default/config in after profile: `{warmup_default}`",
        "",
        "## Issues",
        _markdown_table(
            ["severity", "message"],
            [
                {"severity": issue.severity, "message": issue.message}
                for issue in result.issues
            ],
        ),
        "",
        "## Named parallelism gate",
        _markdown_table(
            [
                "case",
                "rss_ratio",
                "cpu_seconds_ratio",
                "speedup",
                "selected_path",
                "pass_parallelism_gate",
                "reason",
            ],
            result.named_parallelism_rows,
        ),
        "",
        "## 4-6x overhead cases",
        _markdown_table(
            [
                "case",
                "rss_ratio",
                "cpu_seconds_ratio",
                "speedup",
                "selected_path",
                "pass_speedup_gate",
                "reason",
            ],
            result.high_overhead_rows,
        ),
        "",
        "## >6x overhead cases",
        _markdown_table(
            [
                "case",
                "rss_ratio",
                "cpu_seconds_ratio",
                "speedup",
                "selected_path",
                "fallback_documented",
                "reason",
            ],
            result.excessive_overhead_rows,
        ),
        "",
        "## Warmup/import evidence",
        _markdown_table(
            [
                "case",
                "rss_ratio",
                "cpu_seconds_ratio",
                "speedup",
                "selected_path",
                "reason",
            ],
            result.warmup_rows,
        ),
        "",
        "## Dependency gate",
        _markdown_table(["check", "path", "pass"], result.dependency_rows),
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare unlockedpd baseline/after resource profile artifacts."
    )
    parser.add_argument(
        "--baseline", required=True, type=Path, help="Baseline profile JSON"
    )
    parser.add_argument(
        "--after", required=True, type=Path, help="After-change profile JSON"
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("pyproject.toml"),
        help="pyproject.toml path for mandatory dependency checks",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Optional path to write the Markdown evidence report",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Emit the report but return success even when gates fail",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pyproject_path = args.project if args.project and args.project.exists() else None
    result = compare_profiles(args.baseline, args.after, pyproject_path)
    report = render_markdown_report(result)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")
    print(report)
    if result.passed or args.allow_fail:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
