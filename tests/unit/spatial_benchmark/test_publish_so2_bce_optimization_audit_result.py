"""Curated promotion must preserve seed coverage, negative results and sources."""
from __future__ import annotations

import json
import csv
from types import SimpleNamespace

import pytest

from scripts.analysis import publish_so2_bce_optimization_audit_result as publisher
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.result_catalog import create_result_scaffold, sha256_file, validate_result_manifest
from spatial_benchmark.identifiers import create_run_id
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle


def rows(effects=(.003, .002), late=(.0, .0, .0, .0)):
    result = []
    for index, (variant, seed) in enumerate((('incumbent', 0), ('incumbent', 1), ('comparator', 0), ('comparator', 1))):
        endpoint = .24 if variant == 'incumbent' else .24-effects[seed]
        result.append({'run_id': f'r_test_{variant}_{seed}', 'profile': 'confirmation', 'status': 'completed',
            'batch_mode': 'target_window', 'target_batch_size': 1024,
            'learning_rate': 1e-4 if variant == 'incumbent' else 3e-4,
            'dropout': .1, 'weight_decay': 1e-5, 'seed': seed, 'fold': 0,
            'final_epoch': 32, 'endpoint_epochs': [29, 30, 31, 32],
            'endpoint_bce': endpoint, 'previous_window_bce': endpoint+late[index]})
    return result


def coverage(observed):
    return [{**r, 'resolved': True, 'completed_run_id': r['run_id'], 'accepted_failure_run_ids': []} for r in observed]


def test_known_paired_effect_requires_material_mean_and_both_seed_directions():
    result = publisher.analyze_confirmation(rows())
    assert result['mean_paired_bce_reduction'] == pytest.approx(.0025)
    assert [r['bce_reduction'] for r in result['paired_seed_effects']] == pytest.approx([.003, .002])
    assert result['fixed_budget_outcome'] == 'supported'
    # Large mean alone is insufficient when one seed moves adversely.
    assert publisher.analyze_confirmation(rows((.01, -.001)))['fixed_budget_outcome'] == 'negative'
    assert publisher.analyze_confirmation(rows((.001, .001)))['fixed_budget_outcome'] == 'negative'


def test_unconverged_endpoint_does_not_claim_eventual_negative_result():
    result = publisher.analyze_confirmation(rows((.001, .001), (.001, 0., 0., 0.)))
    assert result['fixed_budget_outcome'] == 'negative'
    assert result['any_endpoint_unconverged'] is True
    assert result['eventual_performance_outcome'] == 'inconclusive'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate_seed', 'screen', 'wrong_endpoint', 'same_variant', 'nonfinite'])
def test_rejects_incomplete_or_ineligible_confirmation(mutation):
    observed = rows()
    if mutation == 'missing':
        observed.pop()
    elif mutation == 'duplicate_seed':
        observed[-1]['seed'] = 0
    elif mutation == 'screen':
        observed[-1]['profile'] = 'screen'
    elif mutation == 'wrong_endpoint':
        observed[-1]['endpoint_epochs'] = [1]
    elif mutation == 'same_variant':
        for row in observed:
            row['learning_rate'] = 1e-4
    else:
        observed[-1]['endpoint_bce'] = float('nan')
    with pytest.raises(ValueError):
        publisher.analyze_confirmation(observed)


def test_verified_failure_keeps_planned_seed_and_suppresses_numeric_comparison():
    observed = rows()
    planned = coverage(observed)
    failed = observed.pop()
    planned[-1].update(completed_run_id=None, accepted_failure_run_ids=[failed['run_id']])
    decision = publisher.analyze_complete_coverage(observed, planned, [{'run_id': failed['run_id'], 'reason': 'GPU resource gate'}])
    assert decision['fixed_budget_outcome'] == 'inconclusive'
    assert decision['mean_paired_bce_reduction'] is None
    assert decision['paired_seed_effects'] == []
    assert 'does not establish predictive inferiority' in decision['failure_interpretation']
    with pytest.raises(ValueError, match='verified accepted failure'):
        publisher.analyze_complete_coverage(observed, planned, [])
    planned[-1]['resolved'] = False
    with pytest.raises(ValueError, match='fully resolved'):
        publisher.analyze_complete_coverage(observed, planned, [{'run_id': failed['run_id']}])


def test_publication_validates_schema_preserves_other_records_and_refuses_overwrite(tmp_path, monkeypatch):
    paths = ProjectPaths.from_environment({'BAGM_ROOT': str(tmp_path)})
    paths.report_root.mkdir(parents=True)
    source = paths.report_root/'synthetic_summary.json'
    source.write_text('{}\n')
    sources = [{'kind': 'report', 'root': 'report', 'path': source.name,
                'sha256': sha256_file(source), 'verification_status': 'verified'}]
    observed = rows()
    run_sources = []
    for row in observed:
        artifact = paths.artifact_root/(row['run_id']+'.json')
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('{}\n')
        run_sources.append({'kind': 'run_artifact', 'root': 'artifact', 'path': artifact.name,
            'run_id': row['run_id'], 'sha256': sha256_file(artifact), 'verification_status': 'verified'})
    report = {**publisher.analyze_confirmation(observed), 'expected_coverage': coverage(observed), 'git_commit': 'a'*40}
    monkeypatch.setattr(publisher, 'load_report_chain', lambda *args: (report, sources))
    monkeypatch.setattr(publisher, 'verify_sources', lambda *args: (observed, run_sources, []))
    old = create_result_scaffold(result_id='res_unrelated', title='Unrelated draft', experiment_type='data_qc',
        method_family='other', lifecycle_stage='planned', study_axis='qc', campaign_ids=['cmp_other'],
        paths=paths, git_commit='a'*40)
    original = (old/'result.yaml').read_bytes()
    checked = publisher.publish(source, 'res_unit_optimization', paths=paths, check_only=True)
    assert checked['published'] is False
    assert not (paths.result_root/'predictive_benchmark/relative_qkv/res_unit_optimization').exists()
    result = publisher.publish(source, 'res_unit_optimization', paths=paths)
    manifest = paths.result_root/'predictive_benchmark/relative_qkv/res_unit_optimization/result.yaml'
    validated = validate_result_manifest(manifest, paths=paths, verify_payloads=True, verify_sources=True)
    assert result['published'] is True
    assert validated.payload['outcome'] == 'supported'
    assert validated.payload['evidence']['patient_replication']['status'] == 'not_tested'
    assert 'single fixed split' in validated.payload['method']['evaluation_design']
    assert (old/'result.yaml').read_bytes() == original
    assert json.loads((paths.result_root/'catalog.json').read_text())['result_count'] == 2
    with pytest.raises(ValueError, match='already exists'):
        publisher.publish(source, 'res_unit_optimization', paths=paths)


def test_report_stage_requires_checksum_and_cannot_promote_grid(tmp_path):
    paths = ProjectPaths.from_environment({'BAGM_ROOT': str(tmp_path)})
    folder = paths.report_root/'synthetic'
    folder.mkdir(parents=True)
    report = {'schema': publisher.SCHEMA, 'campaign_id': publisher.CAMPAIGN,
        'stage': 'grid', 'required_coverage_complete': True}
    (folder/'summary.json').write_text(json.dumps(report))
    (folder/'report.md').write_text('Synthetic report\n')
    (folder/'report_builder.py').write_text('# synthetic fixture\n')
    manifest = {'files': {p.name: sha256_file(p) for p in folder.iterdir()}}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='stage is incomplete'):
        publisher.load_report(folder/'summary.json', paths, 'confirmation')
    (folder/'report.md').write_text('Changed report\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        publisher.load_report(folder/'summary.json', paths, 'confirmation')


def test_real_archive_success_maps_to_completed_registry_status(tmp_path):
    paths = ProjectPaths.from_environment({'BAGM_ROOT': str(tmp_path)})
    run_id = create_run_id(seed=0, fold=0, attempt=1, scientific_id_value='sci_1234567890abcdef')
    archive = RunArchive.create(run_id, paths=paths, resolved_config={'model': {'name': 'synthetic'}, 'seed': 0},
        manifest={'lifecycle_status_source': 'registry_and_completion_marker'})
    archive.write_summary({'status': 'completed', 'primary_metric_name': 'val/loss', 'primary_metric_value': .25})
    archive.append_metric_event({'name': 'val/loss', 'value': .25, 'step': 1})
    archive.write_json('metrics/final.json', {'val/loss': .25})
    archive.write_table('metrics/history', [{'epoch': 1, 'val/loss': .25}])
    archive.prepare_log_files()
    for name in ('git', 'hardware', 'data_fingerprints', 'split_fingerprint'):
        archive.write_json('provenance/'+name+'.json', {'synthetic': True})
    archive.write_text('provenance/uncommitted_changes.patch', '')
    archive.write_text('provenance/environment.txt', 'synthetic CPU fixture\n')
    archive.write_text('provenance/command.txt', '{"argv":["fixture"],"cwd":"/synthetic"}\n')
    archive.write_bytes('checkpoints/best.ckpt', b'synthetic checksum fixture only')
    archive.write_predictions('validation', [{'run_id': run_id, 'sample_key': 'sk_0123456789abcdef',
        'graph_id': 'synthetic', 'dataset_id': 'synthetic', 'split': 'validation', 'fold': 0,
        'y_true': [0., 1.], 'y_pred': [.1, .8], 'sample_loss': .25, 'node_count': 2,
        'edge_count': 2, 'effective_neighbor_count': 1., 'effective_mask_rate': .5}])
    root = archive.finalize_success()
    assert verify_run_bundle(root)['status'] == 'success'
    record = {'status': 'completed', 'campaign_id': publisher.CAMPAIGN, 'artifact_path': str(root)}
    registry = SimpleNamespace(get_run=lambda _: record)
    publisher.verify_execution_identity(run_id, 'completed', root, registry)
    record['status'] = 'running'
    with pytest.raises(ValueError, match='Registry lifecycle'):
        publisher.verify_execution_identity(run_id, 'completed', root, registry)


def test_early_regimen_effect_is_preserved_separately_from_tuning_outcome(tmp_path):
    paths = ProjectPaths.from_environment({'BAGM_ROOT': str(tmp_path)})
    run_ids = [create_run_id(seed=0, fold=0, attempt=1, scientific_id_value='sci_1234567890abcdef') for _ in range(2)]
    for run_id, steps, bce, f1 in zip(run_ids, (6, 209), (.25, .24), (.10, .15)):
        path = RunArchive.artifact_path_for(run_id, paths)/'results/epoch_metrics.csv'
        path.parent.mkdir(parents=True)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['global_epoch', publisher.METRIC, 'validation_detection_f1',
                'cumulative_optimizer_updates', 'train_target_cells'])
            writer.writeheader()
            writer.writerows({'global_epoch': epoch, publisher.METRIC: bce, 'validation_detection_f1': f1,
                'cumulative_optimizer_updates': epoch*steps, 'train_target_cells': 208696} for epoch in range(1, 9))
    report = {'early_update_regime_comparison': {'paired_core_run_id': run_ids[0],
        'target_window_incumbent_run_id': run_ids[1], 'bce_reduction_with_target_windows': .01,
        'f1_increase_with_target_windows': .05}}
    early = publisher.early_update_comparison(report, paths)
    assert early['bce_reduction_with_target_windows'] == pytest.approx(.01)
    assert early['f1_increase_with_target_windows'] == pytest.approx(.05)
    decision = publisher.analyze_confirmation(rows((.001, .001)))
    decision['early_update_regime_comparison'] = early
    record = publisher.result_record('res_negative_tuning', {'git_commit': 'a'*40}, decision, [], [])
    assert record['outcome'] == 'negative'
    assert 'eight-epoch target-window incumbent reduced BCE' in record['conclusion']['observed_result']
    assert 'does not establish eventual superiority' in record['conclusion']['observed_result']
