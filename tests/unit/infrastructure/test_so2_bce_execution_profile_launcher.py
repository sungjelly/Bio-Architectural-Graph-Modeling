"""Admission and ordinary predictive archive requirements for runtime profiling."""
from __future__ import annotations

from copy import deepcopy
import pytest

from scripts.analysis import launch_so2_bce_execution_profile as launcher
from scripts.analysis import so2_bce_execution_profile_contract as contract
from spatial_benchmark.checkpoint_catalog import index_checkpoint_catalog
from spatial_benchmark.configuration import ConfigurationError
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256, create_run_id, scientific_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import WorkerSettings, _analysis_only_artifact_contract
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, RunValidationError, verify_run_bundle


@pytest.fixture
def setup(tmp_path, monkeypatch):
    paths = ProjectPaths.from_environment({'BAGM_ROOT': str(tmp_path)})
    parent = {'seed': 0, 'fold': 0, 'attempt': 1, 'campaign': {'campaign_id': 'cmp_parent'},
        'classification': {'lifecycle_stage': 'pilot', 'study_axis': 'optimization', 'scientific_variant': 'parent'},
        'experiment': {}, 'evaluation': {'protocol': 'parent_predictive'}, 'model': {'name': 'synthetic_bce'},
        'dataset': {'dataset_id': 'synthetic', 'version': 'v1', 'dataset_fingerprint': 'dataset-sha',
            'preprocessing_version': 'prep1', 'split_id': 'split1', 'split_fingerprint': 'split-sha'},
        'metadata': {'execution_profile': 'smoke', 'frozen_task_contract_sha256': 'a'*64},
        'trainer': {'max_epochs': 2, 'optimizer_updates_per_global_epoch': 209},
        'launcher': {'requested_gpu': '0', 'disk_safety_min_free_gb': 5}}
    monkeypatch.setattr(contract, 'PARENT_CONFIG_SHA', canonical_sha256(parent))
    monkeypatch.setattr(launcher, 'PARENT_CONFIG_SHA', canonical_sha256(parent))
    monkeypatch.setattr(contract, 'validate_audit', lambda _: None)
    for relative in launcher.INSTRUMENTATION_FILES:
        path = paths.project_root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source fixture\n')
    frozen = paths.project_root/'experiments/campaigns'/contract.CAMPAIGN/'frozen_task_contract.md'
    frozen.parent.mkdir(parents=True)
    frozen.write_text('Synthetic declared profile contract\n')
    reference = {'run_id': contract.PARENT_RUN, 'artifact_path': '/synthetic/parent',
        'configuration_sha256': contract.PARENT_CONFIG_SHA, 'success_marker_sha256': contract.PARENT_SUCCESS_SHA,
        'receipt_path': '/synthetic/parent/diagnostics/optimization_audit_preflight.json',
        **{key: 'b'*64 for key in ('config_file_sha256', 'manifest_sha256', 'checksum_manifest_sha256',
            'epoch_metrics_sha256', 'receipt_sha256', 'code_fingerprint_sha256')}}
    config = contract.build_configuration(parent, reference, gpu='0', task_contract_sha256=sha256_file(frozen),
        instrumentation_hashes=launcher.instrumentation_hashes(paths))
    db = Registry(paths.state_root/'tracking/bagm.sqlite3')
    db.create_campaign(contract.CAMPAIGN, name='Synthetic profile fixture')
    db.register_dataset('synthetic', 'v1', display_name='Synthetic', processed_fingerprint='dataset-sha',
                        preprocessing_version='prep1')
    db.register_split('split1', dataset_id='synthetic', dataset_version='v1', method='fixed',
                      unit='sample', fingerprint='split-sha')
    monkeypatch.setattr(launcher, 'load_parent', lambda *_: (parent, reference))
    return paths, db, parent, reference, config


def test_exact_guarded_command_gpu_data_and_instrumentation_admission(setup):
    paths, db, _, _, config = setup
    worker = launcher.ExecutionProfileWorker(db, paths=paths,
        settings=WorkerSettings(worker_id='fixture', gpu='0', parallel_gpu_workers=True, min_free_gb=5))
    command = contract.command_for_configuration(config, paths.project_root)
    assert '--nproc-per-node=1' in command
    assert command[-4:] == ['--config', '{run_scratch}/config.resolved.yaml', '--run-scratch', '{run_scratch}']
    worker._validate_job(config, queued_command=command, requested_gpu='0')
    with pytest.raises(ConfigurationError, match='argv'):
        worker._validate_job(config, queued_command=['unapproved'], requested_gpu='0')
    with pytest.raises(ConfigurationError, match='GPU'):
        worker._validate_job(config, queued_command=command, requested_gpu='1')
    for field in ('dataset_fingerprint', 'split_fingerprint', 'preprocessing_version'):
        changed = deepcopy(config)
        changed['dataset'][field] = 'changed'
        with pytest.raises(ConfigurationError):
            launcher.validate_registered_inputs(db, paths, changed)
    (paths.project_root/launcher.INSTRUMENTATION_FILES[0]).write_text('# changed instrumentation\n')
    with pytest.raises(ConfigurationError, match='instrumentation changed'):
        worker._validate_job(config, queued_command=command, requested_gpu='0')


def test_parent_and_contract_are_rechecked_before_execution(setup, monkeypatch):
    paths, db, parent, reference, config = setup
    worker = launcher.ExecutionProfileWorker(db, paths=paths,
        settings=WorkerSettings(worker_id='fixture', gpu='0', parallel_gpu_workers=True, min_free_gb=5))
    command = contract.command_for_configuration(config, paths.project_root)
    altered = {**reference, 'epoch_metrics_sha256': 'f'*64}
    monkeypatch.setattr(launcher, 'load_parent', lambda *_: (parent, altered))
    with pytest.raises(ConfigurationError, match='parent provenance'):
        worker._validate_job(config, queued_command=command, requested_gpu='0')
    monkeypatch.setattr(launcher, 'load_parent', lambda *_: (parent, reference))
    frozen = paths.project_root/'experiments/campaigns'/contract.CAMPAIGN/'frozen_task_contract.md'
    frozen.write_text('Changed contract\n')
    with pytest.raises(ConfigurationError, match='frozen contract changed'):
        worker._validate_job(config, queued_command=command, requested_gpu='0')


def make_archive(paths, config, *, checkpoint=True, predictions=True):
    run_id = create_run_id(seed=0, fold=0, attempt=1, scientific_id_value=scientific_id(config))
    archive = RunArchive.create(run_id, paths=paths, resolved_config=config,
        manifest={'lifecycle_status_source': 'registry_and_completion_marker'})
    metric = 'val/target_only_masked_binary_cross_entropy'
    archive.write_summary({'status': 'completed', 'primary_metric_name': metric, 'primary_metric_value': .25,
                           'best_epoch': 1, 'final_epoch': 1})
    archive.append_metric_event({'name': metric, 'value': .25, 'step': 1})
    archive.write_json('metrics/final.json', {metric: .25})
    archive.write_table('metrics/history', [{'epoch': 1, metric: .25}])
    archive.prepare_log_files()
    for name in ('git', 'hardware', 'data_fingerprints', 'split_fingerprint'):
        archive.write_json('provenance/'+name+'.json', {'synthetic': True})
    archive.write_text('provenance/uncommitted_changes.patch', '')
    archive.write_text('provenance/environment.txt', 'synthetic CPU fixture\n')
    archive.write_text('provenance/command.txt', '{"argv":["fixture"],"cwd":"/synthetic"}\n')
    if checkpoint:
        archive.write_bytes('checkpoints/best.ckpt', b'synthetic checksum lifecycle fixture')
    if predictions:
        archive.write_predictions('validation', [{'run_id': run_id, 'sample_key': 'sk_0123456789abcdef',
            'graph_id': 'synthetic', 'dataset_id': 'synthetic', 'split': 'validation', 'fold': 0,
            'y_true': [0., 1.], 'y_pred': [.1, .8], 'sample_loss': .25, 'node_count': 2,
            'edge_count': 2, 'effective_neighbor_count': 1., 'effective_mask_rate': .5}])
    return archive


def test_profile_retains_predictive_success_and_verified_checkpoint_catalog(setup):
    paths, db, _, _, config = setup
    assert not _analysis_only_artifact_contract(config)
    archive = make_archive(paths, config)
    final = archive.finalize_success()
    assert verify_run_bundle(final)['status'] == 'success'
    sid = scientific_id(config)
    db.register_variant(sid, campaign_id=contract.CAMPAIGN, configuration=config)
    db.create_run(archive.run_id, campaign_id=contract.CAMPAIGN, scientific_id=sid, repro_id='synthetic',
        seed=0, fold=0, attempt=1, configuration=config, status='completed', artifact_path=final,
        primary_metric_name='val/target_only_masked_binary_cross_entropy', primary_metric_value=.25)
    checkpoint = final/'checkpoints/best.ckpt'
    db.record_artifacts(archive.run_id, [{'kind': 'checkpoint_best', 'path': checkpoint,
        'sha256': sha256_file(checkpoint), 'size_bytes': checkpoint.stat().st_size}])
    indexed = index_checkpoint_catalog(db, paths, run_reference=archive.run_id, verify=True)
    assert indexed['indexed_checkpoints'] == 1
    entry, = db.list_checkpoint_catalog(run_id=archive.run_id, role='best', verification_status='verified')
    assert entry['sha256'] == sha256_file(checkpoint)


@pytest.mark.parametrize('omitted', ['checkpoint', 'predictions'])
def test_profile_cannot_bypass_predictive_archive_requirements(setup, omitted):
    paths, _, _, _, config = setup
    archive = make_archive(paths, config, checkpoint=omitted!='checkpoint', predictions=omitted!='predictions')
    with pytest.raises(RunValidationError):
        archive.finalize_success()


def test_cli_requires_explicit_gpu_and_positive_attempt():
    with pytest.raises(SystemExit):
        launcher.main([])
    with pytest.raises(ConfigurationError, match='positive'):
        launcher.main(['--gpu', '0', '--attempt', '0', '--check-only'])
