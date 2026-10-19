#!/usr/bin/env python3
"""Notify an existing local Codex conversation after a registered run ends."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

from spatial_benchmark.paths import current_paths


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def get_run(database, run_id):
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=5) as con:
        con.row_factory = sqlite3.Row
        row = con.execute('SELECT run_id,status,artifact_path FROM runs WHERE run_id=?',
                          (run_id,)).fetchone()
    if row is None:
        raise ValueError('Unknown monitored run')
    return dict(row)


def check_daemon(codex):
    check = subprocess.run([codex, 'app-server', 'daemon', 'version'],
                           check=True, capture_output=True, text=True, timeout=15)
    daemon = json.loads(check.stdout)
    if daemon['status'] != 'running':
        raise RuntimeError('Codex app server is unavailable; retry before dispatch')
    return daemon


def terminal_event(run):
    status = run['status']
    if status == 'completed':
        marker = Path(run['artifact_path']) / '_SUCCESS'
        if marker.is_file():
            value = json.loads(marker.read_text())
            if value.get('run_id') == run['run_id'] and value.get('status') == 'success':
                return 'completed_pending_final_verification'
        return None
    if status in {'failed', 'pruned', 'cancelled'}:
        return status
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--thread-id', required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--codex', default='/root/.local/bin/codex')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    paths = current_paths()
    database = paths.state_root / 'tracking/bagm.sqlite3'
    run = get_run(database, args.run_id)
    if args.check:
        daemon = check_daemon(args.codex)
        print(json.dumps({'run_status': run['status'], 'thread_id': args.thread_id,
                          'app_server_version': daemon['appServerVersion'], 'daemon_verified': True}))
        return
    receipt = paths.state_root / 'notifications' / f'{args.run_id}.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    lock = receipt.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(receipt.read_text()) if receipt.exists() else {
        'run_id': args.run_id, 'thread_id': args.thread_id, 'created_at': now(), 'status': 'armed'}
    assert state['run_id'] == args.run_id and state['thread_id'] == args.thread_id
    if state['status'] in {'dispatched', 'dispatch_uncertain'}:
        print(json.dumps(state), flush=True)
        return
    save(receipt, state)
    print(json.dumps({'status': 'armed', 'run_id': args.run_id, 'poll_seconds': 60}), flush=True)
    stopped_checks = 0
    while True:
        try:
            run = get_run(database, args.run_id)
            event = terminal_event(run)
            worker = subprocess.run(['supervisorctl', 'status', args.worker],
                                    capture_output=True, text=True, timeout=10).stdout
            stopped_checks = stopped_checks + 1 if any(
                word in worker.split() for word in ('STOPPED', 'EXITED', 'FATAL', 'BACKOFF')) else 0
            if event is None and stopped_checks >= 3:
                event = 'worker_stopped_requires_status_check'
            state.update(last_checked_at=now(), last_run_status=run['status'])
            if event:
                state['event'] = event
                prompt = (
                    f'User-requested BAGM training completion notification. Monitor detected {event} '
                    f'for run {args.run_id}. Receipt: {receipt}. Verify the registry, final bundle '
                    f'checksums/completion marker and checkpoint metrics, then inform the user in this '
                    f'conversation whether training completed or failed/interrupted. Include final '
                    f'epochs, best validation score and saved checkpoint location if complete; compare '
                    f'to the existing zero/mean baseline report if available. Do not change or restart '
                    f'training. Record notification_reported_at in the receipt when reporting.'
                )
                check_daemon(args.codex)
                state.update(status='dispatch_uncertain', dispatch_started_at=now())
                save(receipt, state)
                # The supported CLI queues persistently and dispatches only once
                # the loaded root conversation is idle, without steering active work.
                response = subprocess.run([args.codex, 'queue', '--thread', args.thread_id,
                                           '--message', prompt], capture_output=True, text=True, timeout=30)
                if response.returncode:
                    # Only an explicit server rejection proves no prompt was
                    # accepted. Transport/reply loss remains ambiguous.
                    if '(code ' in response.stderr:
                        state['status'] = 'armed'
                    raise RuntimeError(response.stderr.strip()[-500:])
                state.update(status='dispatched', queue_receipt=response.stdout.strip(), dispatched_at=now())
                save(receipt, state)
                print(json.dumps(state), flush=True)
                return
            save(receipt, state)
        except Exception as error:
            state.update(last_error=repr(error), last_error_at=now())
            save(receipt, state)
            print(json.dumps({'error': repr(error), 'status': state['status']}), flush=True)
            if state['status'] == 'dispatch_uncertain':
                raise
        time.sleep(60)


if __name__ == '__main__':
    main()
