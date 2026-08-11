"""Safety tests for the standalone four-GPU robustness coordinator.

The subprocesses in this file are tiny synthetic workers.  They never import
the scientific runner, inspect experiment artifacts, or touch a CUDA device.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import sys
import textwrap
import threading
import time
from typing import Any

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/train/launch_same_gene_robustness.py"
_SPEC = importlib.util.spec_from_file_location(
    "launch_same_gene_robustness_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_LAUNCHER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LAUNCHER
_SPEC.loader.exec_module(_LAUNCHER)

_WORKER = textwrap.dedent(
    r"""
    import argparse
    import fcntl
    import json
    import os
    from pathlib import Path
    import signal
    import sys
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--lock-dir", required=True)
    parser.add_argument("--sleep", type=float, required=True)
    parser.add_argument("--counter")
    parser.add_argument("--terminated")
    arguments = parser.parse_args()

    marker = Path(arguments.marker)
    record = Path(arguments.record)
    lock_dir = Path(arguments.lock_dir)
    for parent in (marker.parent, record.parent, lock_dir):
        parent.mkdir(parents=True, exist_ok=True)

    def on_sigterm(signum, _frame):
        if arguments.terminated:
            target = Path(arguments.terminated)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(signum), encoding="utf-8")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_sigterm)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "missing")
    lock_path = lock_dir / ("gpu-" + gpu + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lane_lock:
        try:
            fcntl.flock(lane_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record.write_text(
                json.dumps({"collision": True, "gpu": gpu}), encoding="utf-8"
            )
            raise SystemExit(71)
        record.write_text(
            json.dumps(
                {
                    "collision": False,
                    "gpu": gpu,
                    "omp": os.environ.get("OMP_NUM_THREADS"),
                    "mkl": os.environ.get("MKL_NUM_THREADS"),
                    "pythonpath": os.environ.get("PYTHONPATH"),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if arguments.counter:
            counter = Path(arguments.counter)
            value = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
            counter.write_text(str(value + 1), encoding="utf-8")
        time.sleep(arguments.sleep)
        marker.write_text("ok", encoding="utf-8")
    """
)

_VERIFY = textwrap.dedent(
    """
    from pathlib import Path
    import sys
    raise SystemExit(0 if Path(sys.argv[1]).read_text(encoding="utf-8") == "ok" else 9)
    """
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _make_plan(
    root: Path,
    *,
    gpus: list[int | str | None],
    sleep_seconds: float = 0.03,
    verify: bool = True,
    counter: Path | None = None,
    terminated: Path | None = None,
    data_preflight_exit: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    source_manifest = root / "authority/source-manifest.json"
    environment_lock = root / "authority/environment-lock.json"
    config = root / "configs/frozen.json"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    source_manifest.write_text('{"source":"frozen"}\n', encoding="utf-8")
    environment_lock.write_text('{"environment":"frozen"}\n', encoding="utf-8")
    config.write_text('{"config":"frozen"}\n', encoding="utf-8")

    jobs: list[dict[str, Any]] = []
    assets: dict[str, Any] = {
        "source_manifest": source_manifest,
        "environment_lock": environment_lock,
        "config": config,
        "markers": [],
        "records": [],
    }
    for index, gpu in enumerate(gpus):
        marker = root / f"outputs/job-{index}/_SUCCESS"
        record = root / f"state/records/job-{index}.json"
        argv = [
            sys.executable,
            "-c",
            _WORKER,
            "--config",
            str(config),
            "--marker",
            str(marker),
            "--record",
            str(record),
            "--lock-dir",
            str(root / "state/lane-locks"),
            "--sleep",
            str(sleep_seconds),
        ]
        if counter is not None:
            argv.extend(("--counter", str(counter)))
        if terminated is not None:
            argv.extend(("--terminated", str(terminated)))
        job: dict[str, Any] = {
            "job_id": f"job-{index}",
            "argv": argv,
            "gpu": gpu,
            "stdout_path": str(root / f"state/logs/job-{index}.stdout.log"),
            "stderr_path": str(root / f"state/logs/job-{index}.stderr.log"),
            "expected_success_marker": str(marker),
            "expected_config_sha256": _sha256(config),
        }
        if verify:
            job["verify_argv"] = [sys.executable, "-c", _VERIFY, str(marker)]
        jobs.append(job)
        assets["markers"].append(marker)
        assets["records"].append(record)

    plan_path = root / "plans/materialized.json"
    payload: dict[str, Any] = {
            "schema_version": 1,
            "plan_id": "synthetic-same-gene-robustness",
            "working_directory": ".",
            "source_manifest": {
                "path": str(source_manifest),
                "sha256": _sha256(source_manifest),
            },
            "environment_lock": {
                "path": str(environment_lock),
                "sha256": _sha256(environment_lock),
            },
            "disk_path": ".",
            "minimum_free_disk_gb": 40,
            "jobs": jobs,
    }
    if data_preflight_exit is not None:
        payload["data_preflight_argv"] = [
            sys.executable,
            "-c",
            f"raise SystemExit({int(data_preflight_exit)})",
        ]
    _write_json(plan_path, payload)
    assets["jobs"] = jobs
    return plan_path, assets


def _coordinator(root: Path, plan_path: Path) -> Any:
    plan = _LAUNCHER.load_plan(plan_path, project_root=root)
    return _LAUNCHER.FourGpuCoordinator(
        plan,
        ledger_path=root / "state/ledger.json",
        lock_path=root / "state/coordinator.lock",
        poll_seconds=0.005,
    )


@pytest.fixture(autouse=True)
def _ample_synthetic_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    gib = 1024**3
    monkeypatch.setenv("PYTHONPATH", "/synthetic/untrusted/inherited/path")
    monkeypatch.setattr(
        _LAUNCHER.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * gib, used=10 * gib, free=90 * gib),
    )
    monkeypatch.setattr(
        _LAUNCHER,
        "verify_live_environment",
        lambda _path, *, visibility_mode: {
            "verified": True,
            "visibility_mode": visibility_mode,
        },
    )


def test_four_lanes_are_isolated_and_children_never_use_a_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, assets = _make_plan(
        tmp_path, gpus=[0, 0, "auto", "auto", "auto", "auto"]
    )
    coordinator = _coordinator(tmp_path, plan_path)
    real_popen = _LAUNCHER.subprocess.Popen
    popen_shell_values: list[object] = []

    def recording_popen(*args: object, **kwargs: object) -> Any:
        popen_shell_values.append(kwargs.get("shell"))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(_LAUNCHER.subprocess, "Popen", recording_popen)

    assert coordinator.run() == 0
    records = [json.loads(path.read_text(encoding="utf-8")) for path in assets["records"]]
    assert {record["gpu"] for record in records} == {"0", "1", "2", "3"}
    assert all(record["collision"] is False for record in records)
    assert all(record["omp"] == "8" and record["mkl"] == "8" for record in records)
    assert all(record["pythonpath"] == str(tmp_path / "src") for record in records)
    assert popen_shell_values and all(value is False for value in popen_shell_values)

    ledger = json.loads((tmp_path / "state/ledger.json").read_text(encoding="utf-8"))
    assert ledger["status"] == "completed"
    assert ledger["status_counts"] == {"completed": 6}
    assert ledger["jobs"]["job-0"]["attempts"][0]["gpu"] == 0
    assert ledger["jobs"]["job-1"]["attempts"][0]["gpu"] == 0
    assert not list((tmp_path / "state").glob(".ledger.json.writing-*"))


def test_resume_skips_only_a_marker_that_passes_the_optional_verifier(
    tmp_path: Path,
) -> None:
    counter = tmp_path / "state/counter.txt"
    plan_path, assets = _make_plan(
        tmp_path, gpus=["auto"], counter=counter, verify=True
    )
    marker = assets["markers"][0]
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok", encoding="utf-8")

    # A stale marker is never trusted on the initial launch without a ledger.
    assert _coordinator(tmp_path, plan_path).run() == 0
    assert counter.read_text(encoding="utf-8") == "1"

    # On a true resume, marker + verifier is sufficient and creates no attempt.
    assert _coordinator(tmp_path, plan_path).run() == 0
    ledger = json.loads((tmp_path / "state/ledger.json").read_text(encoding="utf-8"))
    assert ledger["jobs"]["job-0"]["status"] == "skipped"
    assert len(ledger["jobs"]["job-0"]["attempts"]) == 1
    assert counter.read_text(encoding="utf-8") == "1"

    # A completed immutable attempt whose marker no longer verifies is never
    # overwritten or re-executed under the same attempt identity.
    marker.write_text("not-valid", encoding="utf-8")
    assert _coordinator(tmp_path, plan_path).run() == 1
    ledger = json.loads((tmp_path / "state/ledger.json").read_text(encoding="utf-8"))
    assert ledger["jobs"]["job-0"]["status"] == "retry_required"
    assert len(ledger["jobs"]["job-0"]["attempts"]) == 1
    assert counter.read_text(encoding="utf-8") == "1"


def test_success_verifier_reuses_the_completed_jobs_assigned_gpu(
    tmp_path: Path,
) -> None:
    plan_path, _assets = _make_plan(tmp_path, gpus=[3])
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    marker = Path(payload["jobs"][0]["expected_success_marker"])
    verification_record = tmp_path / "state/verifier-gpu.txt"
    payload["jobs"][0]["verify_argv"] = [
        sys.executable,
        "-c",
        "from pathlib import Path; import json, os, sys; "
        "Path(sys.argv[2]).write_text(json.dumps({"
        "'gpu': os.environ.get('CUDA_VISIBLE_DEVICES', ''), "
        "'pythonpath': os.environ.get('PYTHONPATH', '')}), encoding='utf-8'); "
        "raise SystemExit(0 if Path(sys.argv[1]).read_text(encoding='utf-8') == 'ok' else 9)",
        str(marker),
        str(verification_record),
    ]
    _write_json(plan_path, payload)

    assert _coordinator(tmp_path, plan_path).run() == 0
    assert json.loads(verification_record.read_text(encoding="utf-8")) == {
        "gpu": "3",
        "pythonpath": str(tmp_path / "src"),
    }


def test_generated_style_relative_scripts_import_src_under_clean_bound_environment(
    tmp_path: Path,
) -> None:
    plan_path, assets = _make_plan(tmp_path, gpus=[2])
    (tmp_path / "src/probe_module.py").write_text(
        "VALUE = 'imported'\n", encoding="utf-8"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    job_script = scripts / "job_probe.py"
    job_script.write_text(
        "import argparse, json, os\n"
        "from pathlib import Path\n"
        "from probe_module import VALUE\n"
        "p=argparse.ArgumentParser(); p.add_argument('--config'); "
        "p.add_argument('--marker'); p.add_argument('--record'); a=p.parse_args()\n"
        "Path(a.marker).parent.mkdir(parents=True, exist_ok=True)\n"
        "Path(a.record).parent.mkdir(parents=True, exist_ok=True)\n"
        "Path(a.record).write_text(json.dumps({'value': VALUE, 'pythonpath': "
        "os.environ.get('PYTHONPATH')}), encoding='utf-8')\n"
        "Path(a.marker).write_text('ok', encoding='utf-8')\n",
        encoding="utf-8",
    )
    verifier_script = scripts / "verify_probe.py"
    verifier_script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "from probe_module import VALUE\n"
        "Path(sys.argv[2]).write_text(json.dumps({'value': VALUE, 'pythonpath': "
        "os.environ.get('PYTHONPATH')}), encoding='utf-8')\n"
        "raise SystemExit(0 if Path(sys.argv[1]).read_text() == 'ok' else 9)\n",
        encoding="utf-8",
    )
    preflight_script = scripts / "preflight_probe.py"
    preflight_script.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "from probe_module import VALUE\n"
        "Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)\n"
        "Path(sys.argv[1]).write_text(VALUE + ':' + os.environ.get('PYTHONPATH', ''))\n",
        encoding="utf-8",
    )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    marker = assets["markers"][0]
    job_record = tmp_path / "state/generated-job.json"
    verifier_record = tmp_path / "state/generated-verifier.json"
    preflight_record = tmp_path / "state/generated-preflight.txt"
    payload["jobs"][0]["argv"] = [
        sys.executable,
        "scripts/job_probe.py",
        "--config",
        str(assets["config"]),
        "--marker",
        str(marker),
        "--record",
        str(job_record),
    ]
    payload["jobs"][0]["verify_argv"] = [
        sys.executable,
        "scripts/verify_probe.py",
        str(marker),
        str(verifier_record),
    ]
    payload["data_preflight_argv"] = [
        sys.executable,
        "scripts/preflight_probe.py",
        str(preflight_record),
    ]
    _write_json(plan_path, payload)

    assert _coordinator(tmp_path, plan_path).run() == 0
    expected_pythonpath = str(tmp_path / "src")
    assert preflight_record.read_text(encoding="utf-8") == (
        f"imported:{expected_pythonpath}"
    )
    assert json.loads(job_record.read_text(encoding="utf-8")) == {
        "value": "imported",
        "pythonpath": expected_pythonpath,
    }
    assert json.loads(verifier_record.read_text(encoding="utf-8")) == {
        "value": "imported",
        "pythonpath": expected_pythonpath,
    }


@pytest.mark.parametrize("drift_target", ["source_manifest", "config"])
def test_source_and_config_sha_drift_fail_before_launch(
    tmp_path: Path, drift_target: str
) -> None:
    plan_path, assets = _make_plan(tmp_path, gpus=["auto"])
    coordinator = _coordinator(tmp_path, plan_path)
    target = assets[drift_target]
    target.write_text("drifted\n", encoding="utf-8")

    with pytest.raises(_LAUNCHER.PreflightError, match="SHA"):
        coordinator.run()

    assert not assets["markers"][0].exists()
    ledger = json.loads((tmp_path / "state/ledger.json").read_text(encoding="utf-8"))
    assert ledger["status"] == "failed"
    assert ledger["jobs"]["job-0"]["attempts"] == []


def test_disk_floor_fails_closed_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, assets = _make_plan(tmp_path, gpus=["auto"])
    gib = 1024**3
    monkeypatch.setattr(
        _LAUNCHER.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * gib, used=61 * gib, free=39 * gib),
    )

    with pytest.raises(_LAUNCHER.PreflightError, match="below"):
        _coordinator(tmp_path, plan_path).run()
    assert not assets["markers"][0].exists()


def test_projected_outputs_must_leave_full_disk_floor(
    tmp_path: Path,
) -> None:
    plan_path, assets = _make_plan(tmp_path, gpus=["auto"])
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["projected_output_bytes"] = 51 * 1024**3
    _write_json(plan_path, payload)
    with pytest.raises(_LAUNCHER.PreflightError, match="projected outputs"):
        _coordinator(tmp_path, plan_path).run()
    assert not assets["records"][0].exists()


def test_sequential_jobs_subtract_only_the_remaining_output_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, assets = _make_plan(tmp_path, gpus=[0, 0])
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["projected_output_bytes"] = 20 * 1024**3
    _write_json(plan_path, payload)
    gib = 1024**3

    def shrinking_disk(_path: Path) -> SimpleNamespace:
        completed = sum(marker.is_file() for marker in assets["markers"])
        free = (60 - completed * 10) * gib
        return SimpleNamespace(total=100 * gib, used=100 * gib - free, free=free)

    monkeypatch.setattr(_LAUNCHER.shutil, "disk_usage", shrinking_disk)
    assert _coordinator(tmp_path, plan_path).run() == 0
    assert all(path.is_file() for path in assets["markers"])


def test_prepared_data_verifier_fails_before_any_gpu_child(tmp_path: Path) -> None:
    plan_path, assets = _make_plan(
        tmp_path,
        gpus=["auto"],
        data_preflight_exit=17,
    )
    with pytest.raises(_LAUNCHER.PreflightError, match="prepared-data"):
        _coordinator(tmp_path, plan_path).run()
    assert not assets["markers"][0].exists()
    assert not assets["records"][0].exists()


def test_global_flock_rejects_a_second_coordinator(tmp_path: Path) -> None:
    plan_path, _assets = _make_plan(tmp_path, gpus=["auto"])
    lock_path = tmp_path / "state/coordinator.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(_LAUNCHER.LauncherLockHeldError):
            _coordinator(tmp_path, plan_path).run()


def test_live_child_identity_blocks_unsafe_crash_recovery(tmp_path: Path) -> None:
    plan_path, assets = _make_plan(tmp_path, gpus=["auto"])
    coordinator = _coordinator(tmp_path, plan_path)
    coordinator._load_ledger()
    ledger_path = tmp_path / "state/ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ticks = _LAUNCHER._process_start_ticks(os.getpid())
    assert ticks is not None
    ledger["jobs"]["job-0"].update(
        {
            "status": "running",
            "pid": os.getpid(),
            "proc_start_ticks": ticks,
            "attempts": [
                {
                    "number": 1,
                    "status": "running",
                    "pid": os.getpid(),
                    "proc_start_ticks": ticks,
                }
            ],
        }
    )
    _LAUNCHER._atomic_json(ledger_path, ledger)

    with pytest.raises(_LAUNCHER.LiveChildError):
        _coordinator(tmp_path, plan_path).run()
    assert not assets["markers"][0].exists()


def test_failed_to_start_process_can_safely_relaunch_same_unclaimed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, assets = _make_plan(tmp_path, gpus=["auto"])
    real_popen = _LAUNCHER.subprocess.Popen

    def fail_start(*_args: object, **_kwargs: object) -> Any:
        raise OSError("synthetic launch failpoint")

    monkeypatch.setattr(_LAUNCHER.subprocess, "Popen", fail_start)
    with pytest.raises(OSError, match="synthetic"):
        _coordinator(tmp_path, plan_path).run()
    monkeypatch.setattr(_LAUNCHER.subprocess, "Popen", real_popen)

    assert _coordinator(tmp_path, plan_path).run() == 0
    ledger = json.loads((tmp_path / "state/ledger.json").read_text(encoding="utf-8"))
    state = ledger["jobs"]["job-0"]
    assert state["status"] == "completed"
    assert len(state["attempts"]) == 2
    assert state["attempts"][0]["status"] == "failed_to_start"
    assert assets["records"][0].exists()


def test_dead_running_attempt_can_reconcile_marker_without_second_launch(
    tmp_path: Path,
) -> None:
    plan_path, assets = _make_plan(tmp_path, gpus=["auto"])
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    marker = assets["markers"][0]
    payload["jobs"][0]["verify_argv"] = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; "
        "Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True); "
        "Path(sys.argv[1]).write_text('ok', encoding='utf-8')",
        str(marker),
    ]
    _write_json(plan_path, payload)
    coordinator = _coordinator(tmp_path, plan_path)
    coordinator._load_ledger()
    ledger_path = tmp_path / "state/ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["jobs"]["job-0"].update(
        {
                "status": "running",
                "gpu": 2,
                "pid": 999_999_999,
            "proc_start_ticks": 1,
            "attempts": [
                    {
                        "number": 1,
                        "status": "running",
                        "gpu": 2,
                    "pid": 999_999_999,
                    "proc_start_ticks": 1,
                }
            ],
        }
    )
    _LAUNCHER._atomic_json(ledger_path, ledger)

    assert _coordinator(tmp_path, plan_path).run() == 0
    reconciled = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert reconciled["jobs"]["job-0"]["status"] == "skipped"
    assert len(reconciled["jobs"]["job-0"]["attempts"]) == 1
    assert marker.read_text(encoding="utf-8") == "ok"
    assert not assets["records"][0].exists()


def test_dead_launcher_child_with_unclaimed_scientific_attempt_relaunches_safely(
    tmp_path: Path,
) -> None:
    plan_path, assets = _make_plan(tmp_path, gpus=["auto"])
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    marker = assets["markers"][0]
    payload["jobs"][0]["verify_argv"] = [
        sys.executable,
        "-c",
        "from pathlib import Path; import sys; "
        "raise SystemExit(0 if Path(sys.argv[1]).is_file() else 75)",
        str(marker),
    ]
    _write_json(plan_path, payload)
    coordinator = _coordinator(tmp_path, plan_path)
    coordinator._load_ledger()
    ledger_path = tmp_path / "state/ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["jobs"]["job-0"].update(
        {
                "status": "running",
                "gpu": 1,
                "pid": 999_999_999,
            "proc_start_ticks": 1,
            "attempts": [
                    {
                        "number": 1,
                        "status": "running",
                        "gpu": 1,
                    "pid": 999_999_999,
                    "proc_start_ticks": 1,
                }
            ],
        }
    )
    _LAUNCHER._atomic_json(ledger_path, ledger)

    assert _coordinator(tmp_path, plan_path).run() == 0
    relaunched = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert relaunched["jobs"]["job-0"]["status"] == "completed"
    assert len(relaunched["jobs"]["job-0"]["attempts"]) == 2
    assert assets["records"][0].exists()


def test_sigterm_is_forwarded_and_waited_for_without_launching_pending_job(
    tmp_path: Path,
) -> None:
    terminated = tmp_path / "state/terminated.txt"
    plan_path, assets = _make_plan(
        tmp_path,
        gpus=[0, 0],
        sleep_seconds=30.0,
        terminated=terminated,
    )
    coordinator = _coordinator(tmp_path, plan_path)
    outcome: list[int] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            outcome.append(coordinator.run())
        except BaseException as error:  # pragma: no cover - assertion reports it
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not assets["records"][0].exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert assets["records"][0].exists(), "synthetic child did not become ready"

    coordinator.request_stop(signal.SIGTERM)
    thread.join(timeout=5)

    assert not thread.is_alive(), "coordinator did not wait for SIGTERM exit"
    assert errors == []
    assert outcome == [128 + signal.SIGTERM]
    assert terminated.read_text(encoding="utf-8") == str(signal.SIGTERM)
    assert not assets["markers"][0].exists()
    assert not assets["records"][1].exists()
    ledger = json.loads((tmp_path / "state/ledger.json").read_text(encoding="utf-8"))
    assert ledger["status"] == "interrupted"
    assert ledger["jobs"]["job-0"]["status"] == "interrupted"
    assert ledger["jobs"]["job-1"]["status"] == "pending"


@pytest.mark.parametrize("bad_gpu", [{"unexpected": True}, 4, True])
def test_plan_rejects_invalid_gpu_without_type_leaks(
    tmp_path: Path, bad_gpu: object
) -> None:
    plan_path, _assets = _make_plan(tmp_path, gpus=["auto"])
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["jobs"][0]["gpu"] = bad_gpu
    _write_json(plan_path, payload)

    with pytest.raises(_LAUNCHER.PlanValidationError, match="must be"):
        _LAUNCHER.load_plan(plan_path, project_root=tmp_path)
