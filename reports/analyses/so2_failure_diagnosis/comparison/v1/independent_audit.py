#!/usr/bin/env python3
"""Independent SO2 audit; no inference, training, or source/registry mutation.

Run from repository root with PYTHONPATH=src. Default is read-only and prints
checks. --write-receipt writes beside this script only while below scratch and
before publication. Hashes, fixed masks, raw support counts, target transforms,
constant objectives, metric arithmetic, summaries, and registry are checked.
Full sparse neighborhood predictions are not independently recomputed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from spatial_benchmark.adjacency_ablation import sample_uniform_mask_numpy
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry

RUNS = (
    'r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6',
    'r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2',
    'r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf',
    'r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf',
)
CORES = tuple(range(15, 29))
NAMES = ('gene_mean', 'zero_count', 'huber_constant', 'local_log_mean',
         'local_count_mean', 'full_graph_log_mean')
BINS = ('count_1', 'count_2', 'count_3', 'count_4_7', 'count_8_plus')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def close(actual, expected, *, atol=1e-12, rtol=1e-10):
    actual, expected = float(actual), float(expected)
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual - expected) <= atol + rtol * max(abs(actual), abs(expected)), (actual, expected)


def csv_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def audit(receipt):
    p = current_paths()
    b = p.report_root / 'analyses/so2_failure_diagnosis/baselines/v1'
    checks, pins = receipt['checks'], receipt['source_sha256']

    def pin(path, expected=None):
        checksum = sha(path)
        if expected is not None:
            assert checksum == expected, str(path)
        try:
            key = str(path.relative_to(p.project_root))
        except ValueError:
            key = str(path)
        pins[key] = checksum
        return checksum

    def bundle(root, expected=None):
        m = read(root / 'manifest.json')
        checksum = pin(root / 'manifest.json', expected)
        assert read(root / '_SUCCESS')['manifest_sha256'] == checksum
        for rel, item in m['files'].items():
            f = root / rel
            assert sha(f) == item['sha256'], str(f)
            assert f.stat().st_size == item['size_bytes'], str(f)
        return m

    manifest = bundle(b)
    cfg, summary = read(b / 'config.resolved.json'), read(b / 'summary.json')
    assert manifest['cores'] == list(CORES) == cfg['cores']
    assert cfg['pilot_cores'] == [21, 23]
    assert cfg['local_k'] == 16 and cfg['local_radius_um'] == 75 and cfg['huber_delta'] == 1
    assert cfg['classification']['model_seed'] == 0 and cfg['classification']['fold_known'] is False
    assert cfg['device'] == 'cpu' and cfg['analysis_design'] == 'exploratory_transductive_all_fit'
    checks['baseline_bundle_files'] = len(manifest['files'])
    prior, prior_files = {}, 0
    for rel, expected in cfg['source_report_manifests'].items():
        root = (p.report_root / rel).parent
        prior_files += len(bundle(root, expected)['files'])
        prior[root.parent.name] = {c: read(root / f'core_{c}.json') for c in CORES}
    assert set(prior) == set(RUNS)
    checks['four_prior_pinned_payload_files'] = prior_files
    for rel, expected in cfg['code_sha256'].items():
        assert sha(b / 'source' / rel) == expected
    assert sha(b / 'frozen_task_contract.md') == cfg['contract_sha256']
    # The deterministic mask generator is the one frozen into the baseline bundle.
    import spatial_benchmark.adjacency_ablation as mask_module
    assert sha(Path(mask_module.__file__)) == cfg['code_sha256']['src/spatial_benchmark/adjacency_ablation.py']
    checks['frozen_code_and_contract_verified'] = True
    cohort = p.data_root / 'processed/so2_14core_relative_qkv_v1'
    graphs = p.data_root / 'processed/so2_14core_relative_qkv_graphs_v1'
    pin(cohort / 'manifest.json', cfg['dataset']['cohort_manifest_file_sha256'])
    pin(graphs / 'manifest.json', cfg['dataset']['graph_manifest_file_sha256'])
    cm, gm = read(cohort / 'manifest.json'), read(graphs / 'manifest.json')
    graph_records = {v['alias']: v for v in gm['cores']}
    assert sha(cohort / 'cohort_statistics.npz') == cm['files']['cohort_statistics.npz']
    fit = read(b / 'constant_fit.json')
    assert sha(b / 'constants.npz') == fit['constants_sha256']
    with np.load(b / 'constants.npz', allow_pickle=False) as a:
        means, scales, constants = a['means'], a['scales'], a['huber_z']
    with np.load(cohort / 'cohort_statistics.npz', allow_pickle=False) as a:
        assert np.array_equal(means, a['expression_mean'])
        assert np.array_equal(scales, a['expression_scale'])
    assert all(x.shape == (1000,) and np.isfinite(x).all() for x in (means, scales, constants))
    assert np.all(scales > 0)
    histograms = [np.zeros(1, dtype=np.float64) for _ in range(1000)]
    receipts, metric_count, transform_error = {}, 0, 0.0
    for core in CORES:
        r = read(b / f'core_{core}.json')
        receipts[core] = r
        alias, n = f'SO2-C{core}', r['n_cells']
        assert r['core_alias'] == alias and r['completed_at']
        rel = f'cores/{alias}.npz'
        assert sha(cohort / rel) == r['source_npz_sha256'] == cm['files'][rel]
        edge_path = graphs / 'cores' / alias / 'edge_index.npy'
        assert sha(edge_path) == r['edge_sha256'] == graph_records[alias]['files']['edge_index.npy']
        edges = np.load(edge_path, mmap_mode='r')
        assert edges.shape == (2, r['full_edges']) and np.issubdtype(edges.dtype, np.integer)
        assert edges.min() >= 0 and edges.max() < n and not np.any(edges[0] == edges[1])
        with np.load(cohort / rel, allow_pickle=False) as a:
            raw, target, coords = a['expression_counts'], a['target_expression'], a['coordinates_um']
        assert raw.shape == target.shape == (n, 1000) and coords.shape == (n, 2)
        assert np.issubdtype(raw.dtype, np.integer) and raw.min() >= 0
        realization = sample_uniform_mask_numpy(n, 1000, seed=r['mask_seed'])
        mask, counts = realization.mask, realization.masked_gene_counts
        assert realization.checksum == r['mask_checksum'] and int(mask.sum()) == r['n_masked_entries']
        close(np.square(counts).sum() / (1000 * mask.sum()), r['entry_weighted_mask_fraction'])
        assert int(counts[counts >= 900].sum()) == r['scored_entries_from_ge900_masked']
        assert 0 <= r['local_edges'] <= 16 * n
        assert all(0 <= v <= r['n_masked_entries'] for v in r['fallback_masked_entries'].values())
        for sources in prior.values():
            old = sources[core]
            for key in ('n_cells', 'mask_seed', 'mask_checksum', 'n_masked_entries'):
                assert r[key] == old[key], (core, key)
            assert old['source_files'][f'cohort/{rel}'] == r['source_npz_sha256']
            assert old['source_files'][f'graph/cores/{alias}/edge_index.npy'] == r['edge_sha256']
        support = {key: 0 for key in ('all', 'zero', 'positive') + BINS}
        for start in range(0, 1000, 64):
            sl = slice(start, min(start + 64, 1000))
            z = (np.log1p(raw[:, sl].astype(np.float64)) - means[sl]) / scales[sl]
            np.testing.assert_allclose(target[:, sl], z, atol=2e-5, rtol=2e-6)
            transform_error = max(transform_error, float(np.max(np.abs(target[:, sl] - z))))
            values = raw[:, sl][mask[:, sl]]
            support['all'] += len(values)
            support['zero'] += int(np.count_nonzero(values == 0))
            support['positive'] += int(np.count_nonzero(values > 0))
            for count in (1, 2, 3):
                support[f'count_{count}'] += int(np.count_nonzero(values == count))
            support['count_4_7'] += int(np.count_nonzero((values >= 4) & (values < 8)))
            support['count_8_plus'] += int(np.count_nonzero(values >= 8))
        for gene in range(1000):
            h = np.bincount(raw[:, gene]).astype(np.float64) / (14 * n)
            if len(h) > len(histograms[gene]):
                histograms[gene] = np.pad(histograms[gene], (0, len(h) - len(histograms[gene])))
            histograms[gene][:len(h)] += h
        assert set(r['metrics']) == set(NAMES)
        for name, result in r['metrics'].items():
            strata = result['strata']
            assert set(strata) == set(support)
            assert result['huber_delta'] == 1
            for stratum, value in strata.items():
                assert value['n'] == support[stratum] > 0
                assert 0 <= value['n_exact_count'] <= value['n']
                for key, total in value['sums'].items():
                    close(value[key], total / value['n'])
                    metric_count += 1
                close(value['exact_count_accuracy'], value['n_exact_count'] / value['n'])
            for parent, children in (('all', ('zero', 'positive')), ('positive', BINS)):
                assert strata[parent]['n'] == sum(strata[s]['n'] for s in children)
                assert strata[parent]['n_exact_count'] == sum(strata[s]['n_exact_count'] for s in children)
                for key, total in strata[parent]['sums'].items():
                    # A near-zero signed sum needs a tolerance scaled by absolute errors.
                    scale = strata[parent]['sums']['mae_standardized'] if key == 'bias_standardized' else abs(total)
                    close(total, sum(strata[s]['sums'][key] for s in children), atol=1e-10 * max(scale, 1))
            d = result['detection']
            tp, fp = d['true_positive_count'], d['false_positive_count']
            tn, fn = d['true_negative_count'], d['false_negative_count']
            assert min(tp, fp, tn, fn) >= 0 and tp + fn == support['positive'] and tn + fp == support['zero']
            assert d['continuous_count_threshold'] == .5
            close(d['positive_recall'], tp / (tp + fn))
            close(d['zero_specificity'], tn / (tn + fp))
            close(d['balanced_accuracy'], .5 * (tp / (tp + fn) + tn / (tn + fp)))
            if tp + fp:
                close(d['positive_precision'], tp / (tp + fp))
            else:
                assert d['positive_precision'] is None
            if name in ('gene_mean', 'zero_count'):
                for stratum, value in strata.items():
                    old = prior[RUNS[-1]][core]['metrics'][name]['strata'][stratum]
                    assert value['n'] == old['n']
                    for key in tuple(value['sums']) + ('exact_count_accuracy',):
                        close(value[key], old[key], atol=2e-6)
        assert 0 < r['peak_host_rss_gib'] < 12 and r['runtime_seconds'] > 0
        if core in (21, 23):
            assert r['runtime_seconds'] < 300
        assert r['baseline_replay_max_difference'] <= 2e-6
        print(json.dumps({'audited_core': core, 'masked_entries': r['n_masked_entries']}), flush=True)
    checks.update(all14_core_mask_input_graph_and_source_identities=True,
                  raw_count_stratum_supports_independently_recomputed=True,
                  reconstructed_metric_values=metric_count, max_target_transform_abs_error=transform_error,
                  all14_resource_limits_passed=True, total_cells=sum(r['n_cells'] for r in receipts.values()))
    assert checks['total_cells'] == 246063
    constant_rows = csv_rows(b / 'constant_by_gene.csv')
    assert len(constant_rows) == 1000
    residuals, risks, improvements = [], [], []
    for gene, weights in enumerate(histograms):
        close(weights.sum(), 1)
        logs = np.log1p(np.arange(len(weights)))
        close(weights @ logs, means[gene], atol=2e-7)
        y = (logs - means[gene]) / scales[gene]
        c = constants[gene]
        psi = float(weights @ np.clip(c - y, -1, 1))
        assert abs(psi) < 1e-9
        def risk(pred):
            error = np.abs(pred - y)
            quadratic = np.minimum(error, 1)
            return float(weights @ (.5 * quadratic**2 + error - quadratic))
        rh, rm, rz = risk(c), risk(0), risk(y[0])
        assert rh <= min(rm, rz) + 1e-10
        row = constant_rows[gene]
        assert int(row['gene_index']) == gene
        expected = dict(zero_fraction=weights[0], mean_log1p=means[gene], scale_log1p=scales[gene],
                        huber_constant_z=c, huber_constant_log1p=c * scales[gene] + means[gene],
                        huber_stationarity_residual=psi, huber_risk_constant=rh,
                        huber_risk_gene_mean=rm, huber_risk_zero_count=rz)
        for key, value in expected.items():
            close(row[key], value)
        residuals.append(abs(psi)); risks.append(rh); improvements.append(min(rm, rz) - rh)
    assert fit['genes'] == 1000 and fit['below_mean_count'] == int(np.count_nonzero(constants < -1e-9))
    close(fit['max_stationarity_error'], max(residuals))
    close(fit['mean_fitted_huber_risk'], np.mean(risks))
    close(fit['mean_huber_z'], np.mean(constants)); close(fit['median_huber_z'], np.median(constants))
    checks['independent_equal_core_histogram_and_constant_risks'] = dict(
        genes=1000, max_stationarity_residual=max(residuals),
        minimum_risk_improvement_over_both_constants=min(improvements), below_mean_genes=fit['below_mean_count'])
    per_core = csv_rows(b / 'per_core.csv')
    assert len(per_core) == 14 * 6 * 8
    assert len({(r['core'], r['predictor'], r['stratum']) for r in per_core}) == len(per_core)
    by_alias = {r['core_alias']: r for r in receipts.values()}
    for row in per_core:
        value = by_alias[row['core']]['metrics'][row['predictor']]['strata'][row['stratum']]
        for key in row.keys() - {'core', 'predictor', 'stratum'}:
            close(row[key], value[key])
    table = csv_rows(b / 'summary.csv')
    assert len(table) == len(summary['summary']) == 6 * 8
    for rows in (table, summary['summary']):
        assert len({(r['predictor'], r['stratum']) for r in rows}) == 48
        for row in rows:
            assert int(row['cores']) == 14
            for key in row.keys() - {'predictor', 'stratum', 'cores'}:
                values = [r['metrics'][row['predictor']]['strata'][row['stratum']][key] for r in receipts.values()]
                close(row[key], np.mean(values))
    assert summary['status'] == 'completed' and summary['cores'] == 14
    assert summary['evaluation_id'] == cfg['evaluation_id'] == manifest['evaluation_id']
    assert summary['masked_entries'] == sum(r['n_masked_entries'] for r in receipts.values())
    close(summary['runtime_seconds'], sum(r['runtime_seconds'] for r in receipts.values()))
    close(summary['peak_host_rss_gib'], max(r['peak_host_rss_gib'] for r in receipts.values()))
    close(summary['replay_max_difference'], max(r['baseline_replay_max_difference'] for r in receipts.values()))
    assert summary['constant_fit'] == fit
    checks['all_per_core_and_equal_core_summary_values_recomputed'] = True
    metrics = (('all', 'mse_standardized'), ('all', 'huber_standardized'),
               ('positive', 'mse_standardized'), ('positive', 'mse_log1p'))
    comparisons = []
    for rid in RUNS:
        for baseline in NAMES[2:]:
            for stratum, key in metrics:
                model = np.array([prior[rid][c]['metrics']['model']['strata'][stratum][key] for c in CORES])
                base = np.array([receipts[c]['metrics'][baseline]['strata'][stratum][key] for c in CORES])
                delta = float(model.mean() - base.mean())
                comparisons.append(dict(model_run_id=rid, baseline=baseline, stratum=stratum, metric=key,
                    model_equal_core_mean=float(model.mean()), baseline_equal_core_mean=float(base.mean()),
                    model_minus_baseline=delta, model_relative_error_percent=100 * delta / float(base.mean()),
                    model_lower_error_cores=int(np.count_nonzero(model < base)),
                    model_higher_error_cores=int(np.count_nonzero(model > base)), cores=14))
    receipt['paired_comparisons'] = comparisons
    with Registry(initialize=False).connect() as connection:
        rows = connection.execute('SELECT * FROM evaluations WHERE evaluation_id=?', (cfg['evaluation_id'],)).fetchall()
        assert len(rows) == 1
        evaluation = dict(rows[0])
        assert evaluation['status'] == 'completed' and evaluation['run_id'] == RUNS[-1] and evaluation['finished_at']
        assert evaluation['artifact_path'] == str(b) and json.loads(evaluation['metrics_json']) == summary
        rows = connection.execute('SELECT * FROM artifacts WHERE evaluation_id=? AND path=?',
                                  (cfg['evaluation_id'], str(b / 'manifest.json'))).fetchall()
        assert len(rows) == 1
        artifact = dict(rows[0])
        assert artifact['sha256'] == sha(b / 'manifest.json') and artifact['status'] == 'present'
        assert artifact['size_bytes'] == (b / 'manifest.json').stat().st_size and artifact['run_id'] == RUNS[-1]
    checks['registry_completed_path_metrics_and_artifact_verified'] = True
    receipt['scope'] = ('Independent source/payload identity, mask reconstruction, raw support counts, normalization, '
        'graph identity and index checks, constant stationarity/risk, metric arithmetic/support partitions, '
        'equal-core aggregation, paired comparisons, and registry audit. Full sparse neighborhood predictions '
        'are not independently recomputed; source orientation/observed-only fallback contract was reviewed '
        'before evaluation and covered by the reported synthetic tests. This is an exploratory all-fit single-seed '
        'comparison, without a held-out generalization, neural mechanism, or biological replication claim.')
    receipt['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-receipt', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve()
    output = source.with_name('independent_verification.json')
    if args.write_receipt:
        assert source.is_relative_to(current_paths().scratch_root.resolve()), 'Refuse writes outside mutable scratch'
        assert not (source.parent / '_SUCCESS').exists(), 'Refuse altering a successful bundle'
    started = time.monotonic()
    receipt = dict(schema='so2_failure_diagnosis_independent_verification_v1', status='running',
        auditor='comparison_audit', created_at=datetime.now(timezone.utc).isoformat(),
        audit_implementation_sha256=sha(source), checks={}, source_sha256={}, paired_comparisons=[])
    try:
        audit(receipt)
    except Exception:
        receipt['status'] = 'failed'
        receipt['error'] = traceback.format_exc()
        raise
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - started
        if args.write_receipt:
            tmp = output.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
            os.replace(tmp, output)
        print(json.dumps({k: v for k, v in receipt.items() if k not in ('paired_comparisons', 'source_sha256')}, indent=2))


if __name__ == '__main__':
    main()
