"""Registered, reproducible follow-up selecting at most two complete clusters."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.fingerprints import sha256_file

EVALUATION = 'ev_so12_cancer_limited_union_s0_20261018'
CAMPAIGN = 'cmp_20261018_so12_cancer_limited_union'
RUN = 'r_20261017T044837Z_caf76350_s000_f00_a01_12df41be'
PARENT_ID = 'ev_so12_cancer_comparison_s0_20261018'
PARENT_SHA = 'e9a345642c160602ddd6b4c5c377c415377a6489453b8d60c82f34108a60c2b0'
METHODS = ['h4', 'h3', 'h2', 'h1', 'h0', 'pca', 'scvi']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def record(path):
    path = Path(path)
    return {'sha256': sha256_file(path), 'size_bytes': path.stat().st_size}


def locations():
    p = current_paths()
    return (p, p.report_root / 'analyses/so12_cancer_comparison' / PARENT_ID,
            p.scratch_root / 'active_runs' / EVALUATION,
            p.report_root / 'analyses/so12_cancer_limited_union' / EVALUATION)


def verify_inventory(root, expected_sha=None):
    manifest = read(root / 'manifest.json')
    digest = sha256_file(root / 'manifest.json')
    require(expected_sha is None or digest == expected_sha, 'Source manifest identity changed')
    require(read(root / '_SUCCESS')['manifest_sha256'] == digest, 'Completion marker changed')
    entries = list(root.rglob('*'))
    require(not any(p.is_symlink() for p in entries), 'Symlinks are not permitted in an immutable bundle')
    actual = {str(p.relative_to(root)) for p in entries if p.is_file()} - {'manifest.json', '_SUCCESS'}
    require(set(manifest['files']) == actual, 'Incomplete immutable inventory')
    for relative, expected in manifest['files'].items():
        name = Path(relative)
        require(not name.is_absolute() and '..' not in name.parts, 'Unsafe inventory path')
        require(record(root / name) == expected, f'Immutable source changed: {relative}')
    return manifest


def initialize():
    p, parent, out, final = locations()
    require(not final.exists(), 'Evaluation is already published')
    require(shutil.disk_usage(p.project_root).free >= 5.3 * 2**30, 'Insufficient disk reserve')
    before = verify_inventory(parent, PARENT_SHA)
    registry = Registry(p.state_root / 'tracking/bagm.sqlite3')
    with registry.connect() as connection:
        existing = connection.execute('SELECT * FROM evaluations WHERE evaluation_id=?', (EVALUATION,)).fetchone()
    if existing is not None:
        require(existing['run_id'] == RUN and Path(existing['artifact_path']) == out, 'Evaluation identity collision')
        require(read(out / 'config.resolved.json')['parent_manifest_sha256'] == PARENT_SHA, 'Config identity differs')
        return out
    require(not out.exists(), 'Unregistered active output exists')
    out.mkdir(parents=True)
    contract = p.project_root / f'experiments/campaigns/{CAMPAIGN}/README.md'
    config = {'schema': 'so12_cancer_limited_union_v1', 'evaluation_id': EVALUATION,
        'campaign_id': CAMPAIGN, 'source_run_id': RUN, 'parent_evaluation_id': PARENT_ID,
        'parent_path': str(parent), 'parent_manifest_sha256': PARENT_SHA,
        'methods': METHODS, 'resolutions': [.25, .5, 1., 2.], 'seeds': [0, 1],
        'primary_resolution': 1., 'primary_seed': 0, 'n_cells': 81607, 'cores': [15, 23],
        'selection_modes': ['best_single', 'best_pair', 'best_at_most_two'],
        'selection': 'exact exhaustive footprint F1; ties fewer clusters then lexicographic IDs',
        'all_evaluable_pixels_retained': True, 'exploratory_annotation_selected': True,
        'split_id': 'so12_paper_21train_5val_2test_v1', 'dataset_id': 'so12_paper_core_split',
        'minimum_free_gib': 5, 'incremental_budget_mib': 300,
        'contract_sha256_at_start': sha256_file(contract), 'started_at': datetime.now(timezone.utc).isoformat()}
    write(out / 'config.resolved.json', config)
    shutil.copyfile(contract, out / 'task_contract_at_start.md')
    write(out / 'provenance/parent_verification_before.json', {'passed': True,
        'manifest_sha256': PARENT_SHA, 'files_verified': len(before['files']), 'path': str(parent)})
    write(out / 'provenance/environment.json', {'python': sys.version, 'platform': platform.platform(),
        'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn', 'Pillow']},
        'compute': 'CPU-only integer subset enumeration and raster rendering; no GPU training or kernels',
        'thread_limit': 4, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=p.project_root, text=True).strip(),
        'free_bytes_at_start': shutil.disk_usage(out).free})
    (out / 'provenance/git_status_at_start.txt').write_text(subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=p.project_root, text=True))
    (out / 'provenance/git_diff_at_start.patch').write_text(subprocess.check_output(['git', 'diff', '--', 'scripts/analysis', 'experiments/campaigns', 'ops/supervisor', 'tests'], cwd=p.project_root, text=True))
    registry.create_campaign(CAMPAIGN, name='One or two cluster union region agreement',
        scientific_question='How well can one or two complete clusters cover outlined cancer regions?',
        config={'contract': str(contract.relative_to(p.project_root)), 'evaluation_id': EVALUATION})
    registry.create_evaluation(EVALUATION, run_id=RUN, checkpoint_name='best',
        dataset_id=config['dataset_id'], split_id=config['split_id'], status='pending',
        artifact_path=out, metrics={'campaign_id': CAMPAIGN, 'stage': 'initialized'})
    print(json.dumps({'initialized': EVALUATION, 'parent_files_verified': len(before['files'])}), flush=True)
    return out


def source_paths(p):
    return sorted({*p.project_root.glob('scripts/analysis/*so12_cancer_limited*.py'),
        *p.project_root.glob('tests/**/test_so12_cancer_limited*.py'),
        *p.project_root.glob('ops/supervisor/*so12-cancer-limited*'),
        p.project_root / 'scripts/analysis/so12_cancer_metrics.py',
        p.project_root / 'scripts/analysis/so12_cancer_figures.py',
        p.project_root / 'scripts/analysis/so12_cancer_registration.py'})


def execute():
    p, parent, out, final = locations()
    initialize()
    registry = Registry(p.state_root / 'tracking/bagm.sqlite3')
    lockpath = p.state_root / 'locks/so12-cancer-limited-union.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        binding = {}
        for source in source_paths(p):
            relative = str(source.relative_to(p.project_root))
            binding[relative] = record(source)
            target = out / 'provenance/execution_sources' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        write(out / 'provenance/execution_source_binding.json', binding)
        with registry.transaction(immediate=True) as c:
            c.execute("UPDATE evaluations SET status='running' WHERE evaluation_id=?", (EVALUATION,))
        commands = [
            [sys.executable, '-u', 'scripts/analysis/so12_cancer_limited_metrics.py', '--parent', str(parent), '--output', str(out)],
            [sys.executable, '-u', 'scripts/analysis/audit_so12_cancer_limited_union.py', '--parent', str(parent), '--output', str(out)],
            [sys.executable, '-u', 'scripts/analysis/so12_cancer_limited_figures.py', '--parent', str(parent), '--output', str(out)],
            [sys.executable, '-u', 'scripts/analysis/so12_cancer_limited_report.py', '--parent', str(parent), '--output', str(out)],
        ]
        write(out / 'provenance/commands.json', {'cwd': str(p.project_root), 'commands': commands})
        logdir = out / 'logs'
        logdir.mkdir(exist_ok=True)
        try:
            for index, command in enumerate(commands):
                name = Path(command[2]).stem
                print(json.dumps({'stage': name, 'status': 'running'}), flush=True)
                step_start = time.monotonic()
                with (logdir / f'{index}_{name}.log').open('a') as log:
                    subprocess.run(command, cwd=p.project_root, stdout=log, stderr=subprocess.STDOUT, check=True)
                write(out / f'stages/{name}.json', {'passed': True, 'status': 'complete', 'runtime_seconds': time.monotonic() - step_start})
                require(shutil.disk_usage(out).free >= 5 * 2**30, 'Disk reserve breached')
                print(json.dumps({'stage': name, 'status': 'complete'}), flush=True)
            for relative, expected in binding.items():
                require(record(p.project_root / relative) == expected, f'Execution source changed: {relative}')
            verify_inventory(parent, PARENT_SHA)
            write(out / 'provenance/parent_verification_after.json', {'passed': True, 'manifest_sha256': PARENT_SHA})
            write(out / 'execution.json', {'passed': True, 'status': 'complete', 'runtime_seconds': time.monotonic() - started,
                'child_peak_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                'free_bytes': shutil.disk_usage(out).free, 'incremental_bytes': sum(x.stat().st_size for x in out.rglob('*') if x.is_file()),
                'compute': 'CPU only, four threads; no training or GPU work', 'failures': []})
        except BaseException as exc:
            write(out / f'failures/{time.time_ns()}.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            raise


def publish():
    p, parent, out, final = locations()
    registry = Registry(p.state_root / 'tracking/bagm.sqlite3')
    if not final.exists():
        require(read(out / 'execution.json')['passed'], 'Execution is incomplete')
        require(read(out / 'manual_visual_review.json')['passed'], 'Visual review is required')
        require(read(out / 'independent_visual_review.json')['passed'], 'Independent visual review is required')
        require(read(out / 'independent_metrics_review.json')['passed'], 'Independent metric review is required')
        require(read(out / 'focused_tests.json')['passed'], 'Focused tests must pass')
        require(read(out / 'figures/figure_manifest.json')['passed'], 'Figures are incomplete')
        require(read(out / 'report_manifest.json')['passed'], 'Report is incomplete')
        metrics = read(out / 'metrics/limited_union.json')
        require(metrics['passed'] and len(metrics['results']) == 56 and metrics['all_evaluable_pixels_retained'], 'Metric coverage differs')
        for relative, digest in read(out / 'limited_metrics_summary.json')['output_sha256'].items():
            require(sha256_file(out / relative) == digest, f'Metric output changed: {relative}')
        audit = read(out / 'independent_metrics_review.json')
        require(audit['metrics_sha256'] == sha256_file(out / 'metrics/limited_union.json'), 'Audit metric binding changed')
        for outputs in read(out / 'figures/figure_manifest.json')['figures'].values():
            for item in outputs.values():
                require(record(out / 'figures' / item['path']) == {'sha256': item['sha256'], 'size_bytes': item['size_bytes']}, 'Figure output changed')
        report = read(out / 'report_manifest.json')
        require(record(out / report['report_path']) == {'sha256': report['sha256'], 'size_bytes': report['size_bytes']}, 'Report output changed')
        for relative, digest in read(out / 'summary.json')['source_sha256'].items():
            require(sha256_file(out / relative) == digest, f'Summary binding changed: {relative}')
        require(shutil.disk_usage(out).free >= 5 * 2**30, 'Disk reserve breached')
        require(sum(x.stat().st_size for x in out.rglob('*') if x.is_file()) < 300 * 2**20, 'Incremental budget exceeded')
        verify_inventory(parent, PARENT_SHA)
        for relative, expected in read(out / 'provenance/execution_source_binding.json').items():
            require(record(p.project_root / relative) == expected, f'Execution source changed: {relative}')
        shutil.copyfile(p.project_root / f'experiments/campaigns/{CAMPAIGN}/README.md', out / 'task_contract_at_publication.md')
        for source in source_paths(p):
            target = out / 'provenance/final_sources' / source.relative_to(p.project_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        files = {str(x.relative_to(out)): record(x) for x in sorted(out.rglob('*')) if x.is_file() and str(x.relative_to(out)) not in ['manifest.json', '_SUCCESS']}
        manifest = {'evaluation_id': EVALUATION, 'campaign_id': CAMPAIGN, 'source_run_id': RUN,
            'status': 'complete', 'passed': True, 'parent_manifest_sha256': PARENT_SHA,
            'summary': read(out / 'summary.json'), 'files': files}
        write(out / 'manifest.json', manifest)
        write(out / '_SUCCESS', {'manifest_sha256': sha256_file(out / 'manifest.json')})
        final.parent.mkdir(parents=True, exist_ok=True)
        out.rename(final)
    require(not out.exists(), 'Active and final output coexist')
    manifest = verify_inventory(final)
    require(manifest['evaluation_id'] == EVALUATION and manifest['source_run_id'] == RUN, 'Publication identity changed')
    now = datetime.now(timezone.utc).isoformat()
    with registry.transaction(immediate=True) as c:
        rows = c.execute('SELECT sha256 FROM artifacts WHERE evaluation_id=? AND path=?', (EVALUATION, str(final / 'manifest.json'))).fetchall()
        require(all(x['sha256'] == sha256_file(final / 'manifest.json') for x in rows), 'Catalog digest changed')
        if not rows:
            c.execute('INSERT INTO artifacts(run_id,evaluation_id,kind,path,sha256,size_bytes,status,created_at) VALUES(?,?,?,?,?,?,?,?)',
                (RUN, EVALUATION, 'so12_cancer_limited_union', str(final / 'manifest.json'), sha256_file(final / 'manifest.json'),
                 (final / 'manifest.json').stat().st_size, 'present', now))
        changed = c.execute("UPDATE evaluations SET status='completed',artifact_path=?,metrics_json=?,finished_at=? WHERE evaluation_id=? AND run_id=?",
            (str(final), json.dumps(manifest['summary'], allow_nan=False), now, EVALUATION, RUN))
        require(changed.rowcount == 1, 'Missing evaluation registry row')
        c.execute("UPDATE campaigns SET status='completed',updated_at=? WHERE campaign_id=?", (now, CAMPAIGN))
    print(json.dumps({'published': str(final), 'manifest_sha256': sha256_file(final / 'manifest.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['initialize', 'execute', 'publish', 'verify'], required=True)
    args = parser.parse_args()
    if args.stage == 'initialize': initialize()
    elif args.stage == 'execute': execute()
    elif args.stage == 'publish': publish()
    else:
        print(json.dumps({'files_verified': len(verify_inventory(locations()[3])['files'])}))
