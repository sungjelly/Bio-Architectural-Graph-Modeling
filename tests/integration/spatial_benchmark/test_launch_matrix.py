from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "sweeps" / "launch_matrix.py"
SPEC = importlib.util.spec_from_file_location(
    "normal_core_launch_matrix",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
launch_matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launch_matrix
SPEC.loader.exec_module(launch_matrix)


def test_post_exit_completion_preserves_locked_authorization(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    authorization = {
        "lock_id": "lock-1",
        "artifact_id": "artifact-1",
        "final_matrix_file": "matrix.yaml",
        "final_matrix_sha256": "matrix-sha256",
        "canonical_job_hash": "job-sha256",
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "sealed_test_opened": True,
                "standards_lock": authorization,
            }
        ),
        encoding="utf-8",
    )
    job = launch_matrix.Job(
        job_id="job",
        values={},
        output=output,
        log=tmp_path / "job.log",
        authorization=authorization,
    )

    assert launch_matrix._job_succeeded(0, job)
    assert not launch_matrix._job_succeeded(1, job)

    mismatched = dict(authorization)
    mismatched["canonical_job_hash"] = "different-job"
    unauthorized_job = launch_matrix.Job(
        job_id="job",
        values={},
        output=output,
        log=tmp_path / "job.log",
        authorization=mismatched,
    )
    assert not launch_matrix._job_succeeded(0, unauthorized_job)
