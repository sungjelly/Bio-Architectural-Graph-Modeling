"""A real subprocess catches launcher/logging integration failures before GPU work."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/analysis"))
from run_so12_hidden_leiden import jobs


def test_child_job_records_start_and_completion(tmp_path, monkeypatch):
    monkeypatch.setattr("run_so12_hidden_leiden.disk_gate", lambda _: 10 * 2**30)
    jobs(tmp_path, [("bounded_smoke", ["-c", "print('child completed')"])])
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [row["event"] for row in records] == ["job_started", "job_finished"]
    assert all(row["name"] == "bounded_smoke" for row in records)
    assert records[1]["exit_code"] == 0
    assert (tmp_path / "logs/bounded_smoke.log").read_text() == "child completed\n"
