from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import subprocess
import sys
from threading import Event

import pytest

from spatial_benchmark.registry import Registry

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("paper_controller", ROOT / "scripts/train/execute_so12_paper_split.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_jobs(tmp_path):
    registry = Registry(tmp_path / "registry.sqlite3")
    registry.create_campaign(MODULE.CAMPAIGN, name="approved")
    registry.create_campaign("cmp_unrelated", name="unrelated")
    own = registry.enqueue(campaign_id=MODULE.CAMPAIGN, configuration={"seed": 0},
        command=["synthetic"], requested_gpu="0,1,2,3")
    other = registry.enqueue(campaign_id="cmp_unrelated", configuration={"seed": 1},
        command=["synthetic"], requested_gpu="0,1,2,3", priority=100)
    return registry, own, other


def test_claim_only_approved_job_despite_other_priority(tmp_path):
    registry, own, other = fixture_jobs(tmp_path)
    claimed = MODULE.claim_exact_job(registry, own["job_id"], "approved-worker")
    assert claimed["status"] == "claimed"
    assert claimed["worker_id"] == "approved-worker"
    assert claimed["claimed_at"] and claimed["heartbeat_at"]
    assert registry.get_job(other["job_id"])["status"] == "queued"
    with pytest.raises(RuntimeError):
        MODULE.claim_exact_job(registry, other["job_id"], "approved-worker")


def test_concurrent_controllers_cannot_claim_same_job(tmp_path):
    registry, own, _ = fixture_jobs(tmp_path)
    def attempt(worker):
        try:
            return MODULE.claim_exact_job(registry, own["job_id"], worker)["worker_id"]
        except RuntimeError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ["worker-a", "worker-b"]))
    assert sum(result is not None for result in results) == 1


def test_interrupted_preflight_exits_before_control_returns():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    stop = Event()
    stop.set()
    with pytest.raises(InterruptedError):
        MODULE.wait_for_preflight(child, stop)
    assert child.poll() is not None


def test_launch_intent_prevents_duplicate_after_interruption(tmp_path):
    path = tmp_path / "launch_intent.json"
    MODULE.reserve_launch(path, {"seed": 0})
    saved = path.read_bytes()
    with pytest.raises(FileExistsError):
        MODULE.reserve_launch(path, {"seed": 1})
    assert path.read_bytes() == saved
