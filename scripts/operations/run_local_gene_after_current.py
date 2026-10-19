#!/usr/bin/env python3
"""Supervised, fail-closed handoff from the current nine-run GeneMAE matrix.

Contract: experiments/campaigns/cmp_20261005_local_gene_early_stopping/README.md.
Preparing/checking a plan is CPU-only. No GPU child is started until every pinned
production slot is successfully published and all four devices/locks are free.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.queueing import ProjectWorkerLock, WorkerLockError
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle
from scripts.diagnostics.run_so2_hurdle_gpu_trial import global_worker_lease, inspect_gpu

CAMPAIGN = 'cmp_20261005_local_gene_early_stopping'
DEPENDENCY = 'cmp_20261005_so2_genemae_masking'
EXPECTED = {(method, seed) for method in ('whole_cell', 'partial_gene', 'spatial_block') for seed in range(3)}
SLOTS = [('local', 0, '0'), ('local', 1, '1'), ('self', 0, '2'), ('self', 1, '3')]
WRAPPER = Path('/opt/supervisor-scripts/bagm-local-gene-early-stopping.sh')


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def production_jobs(registry):
    with registry.connect() as connection:
        records = [dict(row) for row in connection.execute(
            'SELECT * FROM queue_jobs WHERE campaign_id=? ORDER BY job_id', (DEPENDENCY,))]
    result = []
    for row in records:
        config = json.loads(row['canonical_config_json'])
        if config.get('metadata', {}).get('execution_role') == 'production':
            require(config['trainer']['max_epochs'] == 200, 'Dependency production epoch contract changed')
            result.append({**row, 'config': config, 'method': config['masking']['method'],
                           'seed': config['seed'], 'config_sha256': canonical_sha256(config)})
    require(len(result) == 9 and {(row['method'], row['seed']) for row in result} == EXPECTED,
            'Dependency must contain exactly the nine registered production slots')
    return result


def build_plan(registry):
    from scripts.train.run_local_gene_early_stopping import SOURCE_FILES
    files = set(SOURCE_FILES) | {
        'scripts/operations/run_local_gene_after_current.py',
        'scripts/analysis/report_local_gene_early_stopping.py',
        'scripts/analysis/report_local_gene_residual.py',
        'configs/dataset/so2_hurdle_nb_train12_val2_v1.yaml',
        'ops/supervisor/bagm-local-gene-early-stopping.sh',
        'ops/supervisor/bagm-local-gene-after-current.conf',
        f'experiments/campaigns/{CAMPAIGN}/frozen_task_contract.md',
    }
    return {'schema': 'local_gene_handoff_plan_v1', 'campaign_id': CAMPAIGN,
        'dependency_campaign_id': DEPENDENCY, 'created_at': utc(), 'project_root': str(ROOT),
        'source_hashes': {name: sha256_file(ROOT / name) for name in sorted(files)},
        'dependencies': [{key: row[key] for key in ('job_id', 'method', 'seed', 'config_sha256')}
                         for row in production_jobs(registry)],
        'slots': [{'variant': v, 'seed': s, 'gpu': g} for v, s, g in SLOTS],
        'report_version': 'v1', 'automatic_retry': False}


def validate_plan(plan):
    require(plan['schema'] == 'local_gene_handoff_plan_v1' and plan['campaign_id'] == CAMPAIGN
            and plan['dependency_campaign_id'] == DEPENDENCY, 'Unexpected handoff identity')
    require(plan['project_root'] == str(ROOT), 'Handoff source worktree changed')
    require(len(plan['dependencies']) == 9 and len({row['job_id'] for row in plan['dependencies']}) == 9
            and {(row['method'], row['seed']) for row in plan['dependencies']} == EXPECTED,
            'Pinned dependency coverage incomplete or duplicated')
    require(plan['slots'] == [{'variant': v, 'seed': s, 'gpu': g} for v, s, g in SLOTS], 'Follow-up matrix changed')
    require(plan['automatic_retry'] is False and plan['report_version'] == 'v1', 'Execution policy changed')
    for name, digest in plan['source_hashes'].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and sha256_file(path) == digest,
                f'Frozen handoff source drift: {name}')
    require(WRAPPER.is_file() and sha256_file(WRAPPER) == plan['source_hashes']['ops/supervisor/bagm-local-gene-early-stopping.sh'],
            'Deployed supervisor wrapper differs from frozen source')


def dependency_status(registry, plan):
    jobs = production_jobs(registry)
    pinned = {row['job_id']: row for row in plan['dependencies']}
    require({row['job_id'] for row in jobs} == set(pinned), 'Dependency job identities changed')
    for row in jobs:
        require(row['config_sha256'] == pinned[row['job_id']]['config_sha256'], 'Dependency configuration changed')
        require(row['status'] not in ('failed', 'cancelled', 'pruned', 'stale'),
                f"Dependency job {row['job_id']} is {row['status']}; follow-up held")
    completed = sum(row['status'] == 'completed' for row in jobs)
    with registry.connect() as connection:
        active = connection.execute("SELECT COUNT(*) FROM queue_jobs WHERE campaign_id=? AND status IN ('queued','claimed','running')", (DEPENDENCY,)).fetchone()[0]
    return {'ready': completed == 9 and active == 0, 'completed': completed,
            'total': 9, 'active_dependency_jobs': active,
            'jobs': [{'job_id': row['job_id'], 'status': row['status'], 'run_id': row['run_id']} for row in jobs]}


def verify_dependencies(registry, paths, plan, plan_digest):
    require(dependency_status(registry, plan)['ready'], 'Dependency matrix is not complete')
    rows = []
    for job in production_jobs(registry):
        record = registry.get_run(job['run_id'])
        require(record and record['status'] == 'completed' and record['campaign_id'] == DEPENDENCY,
                'Dependency run not completed in the pinned campaign')
        root = RunArchive.artifact_path_for(job['run_id'], paths)
        require(Path(record['artifact_path']).resolve() == root.resolve(), 'Dependency path is not canonical')
        verify_run_bundle(root)
        require((root / '_SUCCESS').is_file(), 'Dependency success marker missing')
        expected_config = dict(job['config']); expected_config.pop('attempt', None)
        run_config = record.get('config', record.get('config_json'))
        if isinstance(run_config, str):
            run_config = json.loads(run_config)
        for config in (run_config, yaml.safe_load((root / 'config.resolved.yaml').read_text())):
            observed_config = dict(config); observed_config.pop('attempt', None)
            require(observed_config == expected_config, 'Dependency run/bundle does not match its pinned queue slot')
        rows.append({'job_id': job['job_id'], 'run_id': job['run_id'], 'config_sha256': job['config_sha256'],
            'artifact_path': str(root), 'success_sha256': sha256_file(root / '_SUCCESS'),
            'summary_sha256': sha256_file(root / 'summary.json'),
            'checksum_manifest_sha256': sha256_file(root / 'provenance/artifact_checksums.json')})
    return {'schema': 'local_gene_dependency_receipt_v1', 'campaign_id': DEPENDENCY,
            'plan_sha256': plan_digest, 'verified_at': utc(), 'dependencies': rows}


def devices_available(paths):
    # These admission locks are released before children acquire their own locks.
    # A concurrent claimant makes a child fail closed; it cannot cause GPU sharing.
    with ExitStack() as stack:
        stack.enter_context(global_worker_lease(paths.state_root / 'locks/bagm-worker.lock'))
        for gpu in ('0', '1', '2', '3'):
            stack.enter_context(ProjectWorkerLock(paths.state_root / f'locks/bagm-worker-gpu-{gpu}.lock'))
            inspect_gpu(gpu)


def find_completed_run(registry, profile, variant, seed):
    with registry.connect() as connection:
        records = [dict(row) for row in connection.execute('SELECT * FROM runs WHERE campaign_id=?', (CAMPAIGN,))]
    matches = []
    for row in records:
        config = json.loads(row['config_json'])
        if config['metadata']['profile'] == profile and config['model']['variant'] == variant and config['seed'] == seed:
            matches.append(row)
    require(len(matches) == 1 and matches[0]['status'] == 'completed',
            f'Expected exactly one completed {profile}/{variant}/{seed} run; no retries are automatic')
    verify_run_bundle(Path(matches[0]['artifact_path']))
    return matches[0]['run_id']


class Controller:
    def __init__(self, plan_path, paths, registry, poll_seconds=30):
        self.plan_path, self.paths, self.registry = plan_path, paths, registry
        self.directory = plan_path.parent
        self.plan = json.loads(plan_path.read_text())
        self.plan_digest = sha256_file(plan_path)
        self.poll_seconds = poll_seconds
        self.children = []
        self.interrupted = False
        self.state = {'schema': 'local_gene_handoff_state_v1', 'plan_sha256': self.plan_digest,
                      'created_at': utc(), 'controller_pid': os.getpid(), 'children': []}
        state_path = self.directory / 'state.json'
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            require(previous['plan_sha256'] == self.plan_digest, 'Existing handoff plan identity changed')
            require(previous['status'] in ('waiting_dependency', 'waiting_devices'),
                    'An execution already began; inspect its registered runs before any restart')

    def record(self, status, **fields):
        previous = self.state.get('status')
        self.state.update(status=status, updated_at=utc(), **fields)
        atomic_json(self.directory / 'state.json', self.state)
        if previous != status:
            print(json.dumps({'status': status, 'updated_at': self.state['updated_at'], **fields}), flush=True)

    def campaign_stage(self, status):
        current = self.registry.get_campaign(CAMPAIGN)
        require(current is not None, 'Follow-up campaign is not registered')
        self.registry.update_campaign(CAMPAIGN, config=current['config'],
            expected_config_sha256=canonical_sha256(current['config']),
            reason=f'Supervised handoff reached {status}; fixed protocol unchanged',
            actor='local_gene_after_current', status=status)

    def stop(self, *_):
        self.interrupted = True
        for process in self.children:
            if process.poll() is None:
                process.terminate()

    def launch(self, label, command, environment):
        validate_plan(self.plan)
        require(not self.interrupted, 'Handoff interrupted before child launch')
        logfile = self.directory / (label + '.log')
        with logfile.open('x') as handle:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        self.children.append(child)
        self.state['children'].append({'label': label, 'pid': child.pid, 'command': command,
                                       'started_at': utc(), 'return_code': None})
        self.record(self.state['status'])
        return child

    def wait_children(self, children):
        while any(process.poll() is None for process in children):
            self.record(self.state['status'])
            time.sleep(min(5, self.poll_seconds))
        for child in children:
            row = next(row for row in self.state['children'] if row['pid'] == child.pid)
            row.update(return_code=child.returncode, ended_at=utc())
        self.record(self.state['status'])
        require(not self.interrupted and all(child.returncode == 0 for child in children),
                'A child failed or was interrupted; no automatic retry or later stage launch')

    def run(self):
        validate_plan(self.plan)
        while True:
            require(not self.interrupted, 'Handoff interrupted while waiting')
            require(sha256_file(self.plan_path) == self.plan_digest, 'Frozen handoff plan changed')
            validate_plan(self.plan)
            status = dependency_status(self.registry, self.plan)
            if status['ready']:
                break
            self.record('waiting_dependency', dependency=status)
            time.sleep(self.poll_seconds)
        receipt = verify_dependencies(self.registry, self.paths, self.plan, self.plan_digest)
        receipt_path = self.directory / 'dependency_receipt.json'
        if receipt_path.exists():
            old = json.loads(receipt_path.read_text())
            require(old['plan_sha256'] == receipt['plan_sha256'] and old['dependencies'] == receipt['dependencies'],
                    'Previously verified dependencies changed')
        else:
            atomic_json(receipt_path, receipt)
        while True:
            require(not self.interrupted, 'Handoff interrupted waiting for devices')
            validate_plan(self.plan)
            try:
                devices_available(self.paths)
                break
            except (WorkerLockError, RuntimeError) as error:
                self.record('waiting_devices', device_wait_reason=str(error))
                time.sleep(self.poll_seconds)
        environment = dict(os.environ, BAGM_LOCAL_GENE_DEPENDENCY_RECEIPT=str(receipt_path),
                           CUDA_VISIBLE_DEVICES='', PYTHONPATH=f'{ROOT / "src"}:{ROOT}')
        self.campaign_stage('pilot')
        self.record('pilot_running', dependency_receipt=str(receipt_path))
        pilot = self.launch('pilot', [str(WRAPPER), '0', '--profile', 'smoke', '--variant', 'local', '--seed', '0'], environment)
        self.wait_children([pilot])
        pilot_id = find_completed_run(self.registry, 'smoke', 'local', 0)
        self.campaign_stage('full')
        self.record('full_running', pilot_run_id=pilot_id)
        children = []
        for variant, seed, gpu in SLOTS:
            children.append(self.launch(f'{variant}_s{seed}', [str(WRAPPER), gpu, '--profile', 'full',
                '--variant', variant, '--seed', str(seed), '--smoke-run', pilot_id], environment))
        self.wait_children(children)
        full_ids = [find_completed_run(self.registry, 'full', variant, seed) for variant, seed, _ in SLOTS]
        self.record('report_running', full_run_ids=full_ids)
        report = self.launch('report', [sys.executable, str(ROOT / 'scripts/analysis/report_local_gene_early_stopping.py'),
            '--run-ids', *full_ids, '--version', self.plan['report_version'], '--publish-result'], environment)
        self.wait_children([report])
        self.campaign_stage('complete')
        self.record('completed', report_path=str(self.paths.report_root / 'local_gene_early_stopping' / self.plan['report_version']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--prepare-plan', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args(argv)
    require(1 <= args.poll_seconds <= 60, 'Poll interval must be 1..60 seconds')
    require(not (args.prepare_plan and args.check_only), 'Choose preparation or checking')
    paths = current_paths(); registry = Registry(paths.state_root / 'tracking/bagm.sqlite3', initialize=False)
    args.plan = args.plan.resolve()
    if args.prepare_plan:
        require(not args.plan.exists(), 'Frozen handoff plan already exists')
        plan = build_plan(registry); validate_plan(plan)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        with args.plan.open('x') as handle:
            json.dump(plan, handle, indent=2, sort_keys=True); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        print(json.dumps({'plan': str(args.plan), 'dependency_jobs': len(plan['dependencies'])})); return 0
    plan = json.loads(args.plan.read_text()); validate_plan(plan)
    if args.check_only:
        print(json.dumps(dependency_status(registry, plan), sort_keys=True)); return 0
    with ProjectWorkerLock(paths.state_root / 'locks/local-gene-after-current-controller.lock'):
        controller = Controller(args.plan, paths, registry, args.poll_seconds)
        signal.signal(signal.SIGTERM, controller.stop); signal.signal(signal.SIGINT, controller.stop)
        try:
            controller.run(); return 0
        except BaseException as error:
            # A partial launch must not orphan already-started model processes.
            # Let only this controller's existing jobs finish unless termination
            # was explicitly requested; never signal other project jobs.
            active = [child for child in controller.children if child.poll() is None]
            if active:
                controller.record('draining_owned_children', error=str(error))
                try:
                    controller.wait_children(active)
                except BaseException:
                    pass
            controller.record('held_failure', error=str(error), traceback=traceback.format_exc())
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            return 2


if __name__ == '__main__':
    raise SystemExit(main())
