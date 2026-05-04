import json
import os
import subprocess
import sys


def test_resource_profile_import_smoke_schema(tmp_path):
    output = tmp_path / "resource-profile.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "benchmarks/profile_resources.py",
            "--seed",
            "12345",
            "--repeats",
            "1",
            "--cold-and-warm",
            "--case-filter",
            "import-only",
            "--output",
            str(output),
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )

    data = json.loads(output.read_text())
    assert data["schema_version"] == "resource-profile-v1"
    assert data["seed"] == 12345
    assert data["machine"]["cpu_logical"] >= 1
    assert {case["mode"] for case in data["cases"]} == {"cold", "warm"}
    for case in data["cases"]:
        assert case["case_id"] == "import-only"
        assert "speedup" in case["summary"]
        assert "cpu_seconds_ratio" in case["summary"]
        assert "rss_ratio" in case["summary"]
        assert {repeat["implementation"] for repeat in case["repeats"]} == {"pandas", "optimized"}
