from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/results/manage_results.py"


def _run(result_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result-root",
            str(result_root),
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_result_management_cli_creates_valid_draft_and_catalog(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    created = _run(
        result_root,
        "create",
        "--result-id",
        "res_smoke",
        "--title",
        "Smoke result",
        "--experiment-type",
        "infrastructure_validation",
        "--method-family",
        "result_catalog",
        "--lifecycle-stage",
        "diagnostic",
        "--study-axis",
        "runtime_smoke",
        "--campaign-id",
        "cmp_smoke",
    )
    assert created.returncode == 0, created.stderr
    assert (result_root / "infrastructure_validation/result_catalog/res_smoke/result.yaml").is_file()

    validated = _run(result_root, "validate")
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout) == {"valid": True, "result_count": 1}

    generated = _run(result_root, "catalog")
    assert generated.returncode == 0, generated.stderr
    checked = _run(result_root, "catalog", "--check")
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["result_count"] == 1
