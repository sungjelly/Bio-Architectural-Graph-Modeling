"""CPU-only admission and orchestration checks for the deferred LGR campaign."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from scripts.operations import run_local_gene_after_current as handoff


class MemoryRegistry:
    def __init__(self):
        self.campaign_stages = []
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE queue_jobs(job_id TEXT, campaign_id TEXT, status TEXT,
                run_id TEXT, canonical_config_json TEXT);
            CREATE TABLE runs(run_id TEXT, campaign_id TEXT, status TEXT,
                config_json TEXT, artifact_path TEXT);
        """)

    def connect(self):
        return self.connection

    def get_run(self, run_id):
        row = self.connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        return result

    def get_campaign(self, campaign_id):
        assert campaign_id == handoff.CAMPAIGN
        return {"config": {"protocol": "fixture"}}

    def update_campaign(self, campaign_id, **kwargs):
        assert campaign_id == handoff.CAMPAIGN
        assert kwargs["expected_config_sha256"] == handoff.canonical_sha256(kwargs["config"])
        self.campaign_stages.append(kwargs["status"])


def _registry(completed=9):
    registry = MemoryRegistry()
    for index, (method, seed) in enumerate(sorted(handoff.EXPECTED)):
        config = {"metadata": {"execution_role": "production"}, "trainer": {"max_epochs": 200},
                  "masking": {"method": method}, "seed": seed}
        registry.connection.execute("INSERT INTO queue_jobs VALUES(?,?,?,?,?)", (
            f"job-{index}", handoff.DEPENDENCY, "completed" if index < completed else "queued",
            f"run-{index}" if index < completed else None, json.dumps(config),
        ))
    # The predecessor's failed diagnostic must not become a failed tenth slot.
    registry.connection.execute("INSERT INTO queue_jobs VALUES(?,?,?,?,?)", (
        "diagnostic", handoff.DEPENDENCY, "failed", "failed-pilot",
        json.dumps({"metadata": {"execution_role": "pilot"}, "trainer": {"max_epochs": 2}}),
    ))
    registry.connection.commit()
    return registry


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setattr(handoff, "ROOT", root)
    deployed = tmp_path / "deployed-wrapper.sh"
    monkeypatch.setattr(handoff, "WRAPPER", deployed)
    # Keep plan preparation free of trainer/framework/GPU initialization.
    monkeypatch.setitem(sys.modules, "scripts.train.run_local_gene_early_stopping",
                        SimpleNamespace(SOURCE_FILES=("model.py",)))
    sources = {
        "model.py",
        "scripts/operations/run_local_gene_after_current.py",
        "scripts/analysis/report_local_gene_early_stopping.py",
        "scripts/analysis/report_local_gene_residual.py",
        "configs/dataset/so2_hurdle_nb_train12_val2_v1.yaml",
        "ops/supervisor/bagm-local-gene-early-stopping.sh",
        "ops/supervisor/bagm-local-gene-after-current.conf",
        f"experiments/campaigns/{handoff.CAMPAIGN}/frozen_task_contract.md",
    }
    for name in sources:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture: " + name + "\n")
    deployed.write_bytes((root / "ops/supervisor/bagm-local-gene-early-stopping.sh").read_bytes())
    registry = _registry()
    plan = handoff.build_plan(registry)
    directory = tmp_path / "controller"
    directory.mkdir()
    plan_path = directory / "plan.json"
    plan_path.write_text(json.dumps(plan))
    paths = SimpleNamespace(state_root=tmp_path / "state", report_root=tmp_path / "reports")
    return SimpleNamespace(registry=registry, plan=plan, path=plan_path, paths=paths, root=root)


def test_plan_pins_exactly_nine_production_slots_and_excludes_failed_pilot(prepared):
    handoff.validate_plan(prepared.plan)
    assert len(prepared.plan["dependencies"]) == 9
    assert {row["job_id"] for row in prepared.plan["dependencies"]} == {f"job-{i}" for i in range(9)}
    assert {(row["method"], row["seed"]) for row in prepared.plan["dependencies"]} == handoff.EXPECTED
    assert prepared.plan["automatic_retry"] is False
    status = handoff.dependency_status(prepared.registry, prepared.plan)
    assert status["ready"] and status["completed"] == 9


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "wrong_budget"])
def test_production_matrix_rejects_missing_duplicate_extra_or_changed_slots(mutation):
    registry = _registry()
    connection = registry.connection
    if mutation == "missing":
        connection.execute("DELETE FROM queue_jobs WHERE job_id='job-8'")
    elif mutation == "extra":
        connection.execute("INSERT INTO queue_jobs SELECT 'extra',campaign_id,status,run_id,canonical_config_json FROM queue_jobs WHERE job_id='job-0'")
    else:
        if mutation == "duplicate":
            value = connection.execute("SELECT canonical_config_json FROM queue_jobs WHERE job_id='job-0'").fetchone()[0]
        else:
            config = json.loads(connection.execute("SELECT canonical_config_json FROM queue_jobs WHERE job_id='job-1'").fetchone()[0])
            config["trainer"]["max_epochs"] = 199
            value = json.dumps(config)
        connection.execute("UPDATE queue_jobs SET canonical_config_json=? WHERE job_id='job-1'", (value,))
    with pytest.raises(RuntimeError):
        handoff.production_jobs(registry)


@pytest.mark.parametrize("status", ["failed", "cancelled", "pruned", "stale"])
def test_failed_production_dependency_holds_followup(prepared, status):
    prepared.registry.connection.execute("UPDATE queue_jobs SET status=? WHERE job_id='job-3'", (status,))
    with pytest.raises(RuntimeError, match="follow-up held"):
        handoff.dependency_status(prepared.registry, prepared.plan)


def test_configuration_drift_is_rejected_even_if_slot_identity_is_unchanged(prepared):
    row = handoff.production_jobs(prepared.registry)[0]
    row["config"]["trainer"]["learning_rate"] = 0.007
    prepared.registry.connection.execute("UPDATE queue_jobs SET canonical_config_json=? WHERE job_id=?",
                                         (json.dumps(row["config"]), row["job_id"]))
    with pytest.raises(RuntimeError, match="configuration changed"):
        handoff.dependency_status(prepared.registry, prepared.plan)


@pytest.mark.parametrize("drift", [None, "campaign", "registry_config", "bundle_config", "success_marker"])
def test_dependency_bundles_must_match_pinned_queue_config_except_worker_attempt(prepared, monkeypatch, drift):
    roots = {}
    for row in handoff.production_jobs(prepared.registry):
        root = prepared.path.parent / row["run_id"]
        (root / "provenance").mkdir(parents=True)
        (root / "_SUCCESS").write_text("success\n")
        (root / "summary.json").write_text("{}\n")
        (root / "provenance/artifact_checksums.json").write_text("{}\n")
        config = {**row["config"], "attempt": 1}
        (root / "config.resolved.yaml").write_text(json.dumps(config))
        prepared.registry.connection.execute("INSERT INTO runs VALUES(?,?,?,?,?)", (
            row["run_id"], handoff.DEPENDENCY, "completed", json.dumps(config), str(root),
        ))
        roots[row["run_id"]] = root
    if drift == "campaign":
        prepared.registry.connection.execute("UPDATE runs SET campaign_id='other-campaign' WHERE run_id='run-0'")
    elif drift in ("registry_config", "bundle_config"):
        config = json.loads((roots["run-0"] / "config.resolved.yaml").read_text())
        config["seed"] = 2
        if drift == "registry_config":
            prepared.registry.connection.execute("UPDATE runs SET config_json=? WHERE run_id='run-0'", (json.dumps(config),))
        else:
            (roots["run-0"] / "config.resolved.yaml").write_text(json.dumps(config))
    elif drift == "success_marker":
        (roots["run-0"] / "_SUCCESS").unlink()
    monkeypatch.setattr(handoff, "RunArchive", SimpleNamespace(artifact_path_for=lambda run_id, paths: roots[run_id]))
    verified = []
    monkeypatch.setattr(handoff, "verify_run_bundle", lambda root: verified.append(root))
    if drift:
        with pytest.raises(RuntimeError):
            handoff.verify_dependencies(prepared.registry, prepared.paths, prepared.plan, "plan-digest")
    else:
        receipt = handoff.verify_dependencies(prepared.registry, prepared.paths, prepared.plan, "plan-digest")
        assert len(receipt["dependencies"]) == len(verified) == 9
        assert receipt["plan_sha256"] == "plan-digest"
        assert {row["run_id"] for row in receipt["dependencies"]} == set(roots)


def test_nine_completed_slots_still_wait_for_active_nonproduction_dependency_job(prepared):
    prepared.registry.connection.execute("UPDATE queue_jobs SET status='queued' WHERE job_id='diagnostic'")
    status = handoff.dependency_status(prepared.registry, prepared.plan)
    assert status["completed"] == 9 and status["active_dependency_jobs"] == 1
    assert not status["ready"]


@pytest.mark.parametrize("drift", ["source", "wrapper", "retry"])
def test_frozen_sources_deployed_wrapper_and_retry_policy_are_enforced(prepared, drift):
    if drift == "source":
        (prepared.root / "model.py").write_text("changed\n")
    elif drift == "wrapper":
        handoff.WRAPPER.write_text("changed\n")
    else:
        prepared.plan["automatic_retry"] = True
    with pytest.raises(RuntimeError):
        handoff.validate_plan(prepared.plan)


def test_waiting_for_ninth_slot_never_checks_gpu_or_launches_any_child(prepared, monkeypatch):
    prepared.registry.connection.execute("UPDATE queue_jobs SET status='queued',run_id=NULL WHERE job_id='job-8'")
    controller = handoff.Controller(prepared.path, prepared.paths, prepared.registry, poll_seconds=1)

    class EndWaiting(Exception):
        pass

    def forbidden(*args, **kwargs):
        pytest.fail("A GPU/admission/launch action occurred while dependencies were incomplete")

    monkeypatch.setattr(handoff, "devices_available", forbidden)
    monkeypatch.setattr(handoff, "inspect_gpu", forbidden)
    monkeypatch.setattr(handoff, "verify_dependencies", forbidden)
    monkeypatch.setattr(handoff.subprocess, "Popen", forbidden)
    monkeypatch.setattr(handoff.time, "sleep", lambda seconds: (_ for _ in ()).throw(EndWaiting()))
    with pytest.raises(EndWaiting):
        controller.run()
    state = json.loads((prepared.path.parent / "state.json").read_text())
    assert state["status"] == "waiting_dependency"
    assert state["dependency"]["completed"] == 8
    assert state["children"] == []


@pytest.mark.parametrize("drift", ["plan", "source"])
def test_plan_or_source_change_while_waiting_is_caught_before_gpu_admission(prepared, monkeypatch, drift):
    prepared.registry.connection.execute("UPDATE queue_jobs SET status='queued' WHERE job_id='job-8'")
    controller = handoff.Controller(prepared.path, prepared.paths, prepared.registry, poll_seconds=1)

    def modify_during_wait(seconds):
        path = prepared.path if drift == "plan" else prepared.root / "model.py"
        path.write_text(path.read_text() + "\n")

    monkeypatch.setattr(handoff.time, "sleep", modify_during_wait)
    monkeypatch.setattr(handoff, "devices_available", lambda *args: pytest.fail("premature GPU admission"))
    monkeypatch.setattr(handoff.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("premature launch"))
    with pytest.raises(RuntimeError, match="Frozen handoff"):
        controller.run()


def test_devices_require_global_and_each_gpu_lock_and_release_all_on_failure(tmp_path, monkeypatch):
    paths = SimpleNamespace(state_root=tmp_path)
    held = set()
    inspected = []

    @contextmanager
    def lease(path):
        name = path.name
        if name.endswith("gpu-2.lock"):
            raise handoff.WorkerLockError("busy GPU 2")
        held.add(name)
        try:
            yield
        finally:
            held.remove(name)

    def inspect(gpu):
        assert "bagm-worker.lock" in held
        assert f"bagm-worker-gpu-{gpu}.lock" in held
        inspected.append(gpu)

    monkeypatch.setattr(handoff, "global_worker_lease", lease)
    monkeypatch.setattr(handoff, "ProjectWorkerLock", lease)
    monkeypatch.setattr(handoff, "inspect_gpu", inspect)
    with pytest.raises(handoff.WorkerLockError, match="busy GPU 2"):
        handoff.devices_available(paths)
    assert inspected == ["0", "1"]
    assert not held


def test_global_worker_lock_prevents_even_gpu_inspection(tmp_path, monkeypatch):
    @contextmanager
    def blocked(path):
        raise handoff.WorkerLockError("global worker busy")
        yield

    monkeypatch.setattr(handoff, "global_worker_lease", blocked)
    monkeypatch.setattr(handoff, "ProjectWorkerLock", lambda *args: pytest.fail("GPU lock before global admission"))
    monkeypatch.setattr(handoff, "inspect_gpu", lambda *args: pytest.fail("GPU inspection before global admission"))
    with pytest.raises(handoff.WorkerLockError, match="global worker busy"):
        handoff.devices_available(SimpleNamespace(state_root=tmp_path))


def test_interruption_signals_only_owned_running_children_and_blocks_next_launch(prepared, monkeypatch):
    controller = handoff.Controller(prepared.path, prepared.paths, prepared.registry)
    signaled = []
    running = SimpleNamespace(poll=lambda: None, terminate=lambda: signaled.append("owned-running"))
    finished = SimpleNamespace(poll=lambda: 0, terminate=lambda: pytest.fail("signaled completed child"))
    controller.children = [running, finished]
    monkeypatch.setattr(handoff.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("launch after interruption"))
    controller.stop()
    assert controller.interrupted and signaled == ["owned-running"]
    with pytest.raises(RuntimeError, match="interrupted before child launch"):
        controller.launch("not-started", ["irrelevant"], {})


def _simulate_children(prepared, monkeypatch, *, failed_child=None):
    launched = []
    gates = []

    def verify(registry, paths, plan, digest):
        assert handoff.dependency_status(registry, plan)["ready"]
        gates.append("dependencies")
        return {"plan_sha256": digest, "dependencies": [{"run_id": f"run-{i}"} for i in range(9)]}

    def devices(paths):
        assert gates == ["dependencies"]
        gates.append("devices")

    class Child:
        def __init__(self, index):
            self.pid = index + 100
            self.returncode = 2 if index == failed_child else 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

    def popen(command, **kwargs):
        assert gates == ["dependencies", "devices"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert Path(kwargs["env"]["BAGM_LOCAL_GENE_DEPENDENCY_RECEIPT"]).is_file()
        launched.append(command)
        return Child(len(launched) - 1)

    monkeypatch.setattr(handoff, "verify_dependencies", verify)
    monkeypatch.setattr(handoff, "devices_available", devices)
    monkeypatch.setattr(handoff, "inspect_gpu", lambda *args: pytest.fail("real GPU inspection"))
    monkeypatch.setattr(handoff.subprocess, "Popen", popen)
    monkeypatch.setattr(handoff, "find_completed_run", lambda registry, profile, variant, seed: f"{profile}-{variant}-{seed}")
    controller = handoff.Controller(prepared.path, prepared.paths, prepared.registry, poll_seconds=1)
    return controller, launched


def test_controller_runs_pilot_then_all_four_slots_then_one_complete_report(prepared, monkeypatch):
    controller, launched = _simulate_children(prepared, monkeypatch)
    controller.run()
    assert len(launched) == 6
    assert launched[0] == [str(handoff.WRAPPER), "0", "--profile", "smoke", "--variant", "local", "--seed", "0"]
    for command, (variant, seed, gpu) in zip(launched[1:5], handoff.SLOTS):
        assert command == [str(handoff.WRAPPER), gpu, "--profile", "full", "--variant", variant,
                           "--seed", str(seed), "--smoke-run", "smoke-local-0"]
    report = launched[-1]
    begin = report.index("--run-ids") + 1
    assert report[begin:begin+4] == ["full-local-0", "full-local-1", "full-self-0", "full-self-1"]
    assert "--publish-result" in report
    state = json.loads((prepared.path.parent / "state.json").read_text())
    assert state["status"] == "completed"
    assert prepared.registry.campaign_stages == ["pilot", "full", "complete"]
    assert all(row["return_code"] == 0 for row in state["children"])
    with pytest.raises(RuntimeError, match="execution already began"):
        handoff.Controller(prepared.path, prepared.paths, prepared.registry)


@pytest.mark.parametrize("failed_child,expected_launches", [(0, 1), (2, 5)])
def test_failed_child_prevents_later_stage_and_automatic_retry(prepared, monkeypatch, failed_child, expected_launches):
    controller, launched = _simulate_children(prepared, monkeypatch, failed_child=failed_child)
    with pytest.raises(RuntimeError, match="no automatic retry"):
        controller.run()
    assert len(launched) == expected_launches
    assert all("--publish-result" not in command for command in launched)
    with pytest.raises(RuntimeError, match="execution already began"):
        handoff.Controller(prepared.path, prepared.paths, prepared.registry)


def test_partial_launch_failure_drains_only_started_children_then_holds(prepared, monkeypatch):
    _, launched = _simulate_children(prepared, monkeypatch)
    previous_popen = handoff.subprocess.Popen
    completed = []

    class FinishingChild:
        pid = 101
        returncode = None
        polls = 0

        def poll(self):
            self.polls += 1
            if self.polls >= 3:
                self.returncode = 0
                completed.append(self.pid)
            return self.returncode

        def terminate(self):
            pytest.fail("A partial launch failure should drain an existing job")

    def partial_popen(command, **kwargs):
        if len(launched) == 2:
            raise OSError("simulated second full child launch failure")
        child = previous_popen(command, **kwargs)
        return FinishingChild() if len(launched) == 2 else child

    @contextmanager
    def lock(path):
        yield

    monkeypatch.setattr(handoff.subprocess, "Popen", partial_popen)
    monkeypatch.setattr(handoff, "current_paths", lambda: prepared.paths)
    monkeypatch.setattr(handoff, "Registry", lambda *args, **kwargs: prepared.registry)
    monkeypatch.setattr(handoff, "ProjectWorkerLock", lock)
    monkeypatch.setattr(handoff.signal, "signal", lambda *args: None)
    monkeypatch.setattr(handoff.time, "sleep", lambda seconds: None)
    status = handoff.main(["--plan", str(prepared.path), "--poll-seconds", "1"])
    assert status == 2
    assert len(launched) == 2 and completed == [101]
    state = json.loads((prepared.path.parent / "state.json").read_text())
    assert state["status"] == "held_failure"
    assert [child["label"] for child in state["children"]] == ["pilot", "local_s0"]
    assert all(child["return_code"] == 0 for child in state["children"])
    assert "second full child" in state["error"]
