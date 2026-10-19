import json
from scripts.diagnostics.watch_run_notification import terminal_event


def test_running_and_finalizing_are_not_completed(tmp_path):
    (tmp_path / '_SUCCESS').write_text(json.dumps({'run_id': 'r_test', 'status': 'success'}))
    for status in ('running', 'finalizing'):
        assert terminal_event({'run_id': 'r_test', 'status': status, 'artifact_path': str(tmp_path)}) is None


def test_completed_requires_matching_success_marker(tmp_path):
    run = {'run_id': 'r_test', 'status': 'completed', 'artifact_path': str(tmp_path)}
    assert terminal_event(run) is None
    (tmp_path / '_SUCCESS').write_text(json.dumps({'run_id': 'r_other', 'status': 'success'}))
    assert terminal_event(run) is None
    (tmp_path / '_SUCCESS').write_text(json.dumps({'run_id': 'r_test', 'status': 'success'}))
    assert terminal_event(run) == 'completed_pending_final_verification'


def test_failed_is_reported_as_failure_without_success_marker(tmp_path):
    assert terminal_event({'run_id': 'r_test', 'status': 'failed', 'artifact_path': str(tmp_path)}) == 'failed'
