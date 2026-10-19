from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest


@pytest.fixture
def stage():
    path = Path(__file__).resolve().parents[3] / "scripts/operations/run_so2_bce_audit_stage.py"
    spec = importlib.util.spec_from_file_location("audit_stage_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job(label, gpu="0", *, lr=1e-4, **changes):
    value = {"label": label, "gpu": gpu, "profile": "screen", "batch_mode": "target_window",
             "batch_size": 1024, "learning_rate": lr, "dropout": 0.1,
             "weight_decay": 1e-5, "seed": 0, "smoke_run": "r_synthetic_window_smoke"}
    value.update(changes)
    return value


def plan():
    return {"stage_id": "grid_a", "allowed_gpus": ["0", "1"],
            "jobs": [job("a0"), job("b0", "1", lr=3e-5),
                     job("a1", lr=3e-4), job("b1", "1", batch_size=512)]}


@pytest.fixture
def simulation(stage, tmp_path, monkeypatch):
    paths = SimpleNamespace(project_root=tmp_path / "source", state_root=tmp_path / "state")
    paths.project_root.mkdir()
    simulation = SimpleNamespace(tick=0, launched=[], processes={}, results={}, signals=[],
                                 failures=set(), on_sleep=None, paths=paths)

    class Process:
        def __init__(self, label, gpu):
            self.pid = 1000 + len(simulation.launched)
            self.label, self.gpu = label, gpu
            duration, self.code = simulation.results.get(label, (1, 0))
            self.finish = simulation.tick + duration

        def poll(self):
            return self.code if simulation.tick >= self.finish else None

        def wait(self, timeout=None):
            simulation.tick = max(simulation.tick, self.finish)
            return self.code

    def popen(argv, **kwargs):
        label = Path(kwargs["stdout"].name).stem
        gpu = argv[argv.index("--gpu") + 1]
        if label in simulation.failures:
            raise OSError("synthetic launcher start failure")
        assert all(process.gpu != gpu or process.poll() is not None for process in simulation.processes.values())
        process = Process(label, gpu)
        simulation.launched.append({"label": label, "gpu": gpu, "tick": simulation.tick,
                                    "argv": argv, "kwargs": kwargs, "pid": process.pid})
        simulation.processes[process.pid] = process
        kwargs["stdout"].write("synthetic launcher output\n")
        kwargs["stdout"].flush()
        return process

    def sleep(_seconds):
        simulation.tick += 1
        if simulation.on_sleep is not None:
            simulation.on_sleep()

    def killpg(pid, number):
        assert pid in simulation.processes, "Signal targeted a process not launched by this stage"
        simulation.signals.append((pid, number))
        process = simulation.processes[pid]
        process.finish, process.code = simulation.tick, -number

    monkeypatch.setattr(stage.subprocess, "Popen", popen)
    monkeypatch.setattr(stage.time, "sleep", sleep)
    monkeypatch.setattr(stage.os, "killpg", killpg)
    return simulation


def state(simulation, stage_id="grid_a"):
    root = simulation.paths.state_root / "logs/so2_bce_optimization_audit" / stage_id
    return root, json.loads((root / "state.json").read_text())


def test_gpu_lanes_keep_order_and_persist_all_exit_codes(stage, simulation):
    simulation.results = {"a0": (1, 0), "a1": (2, 0), "b0": (2, 0), "b1": (1, 0)}
    declared = plan()
    original = deepcopy(declared)
    assert stage.run_stage(declared, simulation.paths) == 0
    assert declared == original
    assert [(row["label"], row["tick"]) for row in simulation.launched] == [
        ("a0", 0), ("b0", 0), ("a1", 1), ("b1", 2)]
    root, saved = state(simulation)
    assert saved["status"] == "completed"
    assert saved["ended_at"] is not None
    assert len(saved["plan_sha256"]) == 64
    assert json.loads((root / "plan.json").read_text()) == original
    assert all(row["status"] == "completed" and row["exit_code"] == 0 for row in saved["jobs"])
    for record in simulation.launched:
        assert isinstance(record["argv"], list)
        assert all(isinstance(value, str) for value in record["argv"])
        assert record["kwargs"]["start_new_session"] is True
        assert record["kwargs"]["cwd"] == simulation.paths.project_root
        assert not record["kwargs"].get("shell", False)
        assert (root / (record["label"] + ".log")).read_text() == "synthetic launcher output\n"
    assert not list(root.glob(".state-*"))


def test_nonzero_child_holds_every_pending_job_but_running_job_finishes(stage, simulation):
    simulation.results = {"a0": (1, 7), "b0": (3, 0)}
    assert stage.run_stage(plan(), simulation.paths) == 2
    assert [row["label"] for row in simulation.launched] == ["a0", "b0"]
    _, saved = state(simulation)
    rows = {row["label"]: row for row in saved["jobs"]}
    assert rows["a0"]["status"] == "failed" and rows["a0"]["exit_code"] == 7
    assert rows["b0"]["status"] == "completed" and rows["b0"]["exit_code"] == 0
    assert all(rows[label]["status"] == "held" and rows[label]["pid"] is None for label in ("a1", "b1"))
    assert simulation.tick == 3
    assert saved["hold_reason"] == "launcher_failed:a0"
    assert saved["status"] == "failed"
    assert not simulation.signals


def test_all_four_allowed_gpu_lanes_start_independent_jobs(stage, simulation):
    declared = plan()
    declared["allowed_gpus"] = ["0", "1", "2", "3"]
    declared["jobs"][2]["gpu"] = "2"
    declared["jobs"][3]["gpu"] = "3"
    assert stage.run_stage(declared, simulation.paths) == 0
    assert [(record["gpu"], record["tick"]) for record in simulation.launched] == [
        ("0", 0), ("1", 0), ("2", 0), ("3", 0)]


def test_operational_exception_stops_own_children_and_persists_failure(stage, simulation):
    simulation.results = {"a0": (100, 0), "b0": (100, 0)}

    def fail():
        raise RuntimeError("synthetic operational failure")

    simulation.on_sleep = fail
    with pytest.raises(RuntimeError, match="synthetic operational failure"):
        stage.run_stage(plan(), simulation.paths)
    _, saved = state(simulation)
    assert saved["status"] == "failed" and saved["hold_reason"] == "operational_exception"
    assert [row["status"] for row in saved["jobs"]] == ["failed", "failed", "held", "held"]
    assert [row["exit_code"] for row in saved["jobs"][:2]] == [-signal.SIGTERM, -signal.SIGTERM]
    assert saved["ended_at"] is not None


def test_launcher_spawn_failure_is_durable_and_does_not_retry(stage, simulation):
    simulation.failures.add("b0")
    assert stage.run_stage(plan(), simulation.paths) == 2
    assert [row["label"] for row in simulation.launched] == ["a0"]
    _, saved = state(simulation)
    rows = {row["label"]: row for row in saved["jobs"]}
    assert rows["a0"]["status"] == "completed"
    assert rows["b0"]["status"] == "launch_failed"
    assert rows["b0"]["exit_code"] is None and rows["b0"]["pid"] is None
    assert "synthetic" in rows["b0"]["error"]
    assert rows["a1"]["status"] == rows["b1"]["status"] == "held"


def test_sigterm_forwards_only_to_own_groups_holds_pending_and_restores_handler(stage, simulation):
    original = signal.getsignal(signal.SIGTERM)
    simulation.results = {"a0": (100, 0), "b0": (100, 0)}

    def interrupt():
        simulation.on_sleep = None
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    simulation.on_sleep = interrupt
    assert stage.run_stage(plan(), simulation.paths) == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is original
    assert sorted(simulation.signals) == [(1000, signal.SIGTERM), (1001, signal.SIGTERM)]
    _, saved = state(simulation)
    assert saved["status"] == "interrupted" and saved["stop_signal"] == signal.SIGTERM
    assert [row["status"] for row in saved["jobs"]] == ["failed", "failed", "held", "held"]
    assert [row["exit_code"] for row in saved["jobs"][:2]] == [-signal.SIGTERM, -signal.SIGTERM]


def test_stage_directory_is_never_overwritten_or_automatically_resumed(stage, simulation):
    assert stage.run_stage(plan(), simulation.paths) == 0
    root, _ = state(simulation)
    prior = {path.name: path.read_bytes() for path in root.iterdir()}
    launched = len(simulation.launched)
    with pytest.raises(FileExistsError):
        stage.run_stage(plan(), simulation.paths)
    assert len(simulation.launched) == launched
    assert {path.name: path.read_bytes() for path in root.iterdir()} == prior


def test_argv_matches_guarded_launcher_without_shell_or_paired_batch_argument(stage, tmp_path):
    window = job("window", learning_rate=3e-5)
    argv = stage.launcher_argv(window, tmp_path)
    assert argv == [stage.sys.executable, str(tmp_path / "scripts/train/launch_so2_bce_optimization_audit.py"),
                    "--profile", "screen", "--batch-mode", "target_window", "--gpu", "0",
                    "--learning-rate", "3e-05", "--dropout", "0.1", "--weight-decay", "1e-05",
                    "--seed", "0", "--target-batch-size", "1024", "--smoke-run", "r_synthetic_window_smoke"]
    paired = job("paired", profile="smoke", batch_mode="core_pair", batch_size=None, smoke_run=None)
    argv = stage.launcher_argv(paired, tmp_path)
    assert "--target-batch-size" not in argv
    assert "--smoke-run" not in argv
    assert "--attempt" not in argv and "--resume-checkpoint" not in argv


@pytest.mark.parametrize("problem", ["stage_path", "label_path", "label_duplicate", "gpu_denied",
    "gpu_duplicate", "job_duplicate", "seed", "nan", "regularization", "missing_smoke", "extra_field"])
def test_invalid_plan_is_rejected_before_any_operational_output(stage, simulation, problem):
    declared = plan()
    if problem == "stage_path":
        declared["stage_id"] = "../outside"
    elif problem == "label_path":
        declared["jobs"][0]["label"] = "../../x"
    elif problem == "label_duplicate":
        declared["jobs"][1]["label"] = "a0"
    elif problem == "gpu_denied":
        declared["jobs"][0]["gpu"] = "3"
    elif problem == "gpu_duplicate":
        declared["allowed_gpus"] = ["0", "0"]
    elif problem == "job_duplicate":
        declared["jobs"][1] = {**declared["jobs"][0], "label": "duplicate", "gpu": "1"}
    elif problem == "seed":
        declared["jobs"][0]["seed"] = 1
    elif problem == "nan":
        declared["jobs"][0]["learning_rate"] = float("nan")
    elif problem == "regularization":
        declared["jobs"][0].update(dropout=0.3, weight_decay=0.01)
    elif problem == "missing_smoke":
        declared["jobs"][0]["smoke_run"] = None
    else:
        declared["jobs"][0]["shell_command"] = "unauthorized"
    with pytest.raises(ValueError):
        stage.run_stage(declared, simulation.paths)
    assert not simulation.launched
    assert not simulation.paths.state_root.exists()
