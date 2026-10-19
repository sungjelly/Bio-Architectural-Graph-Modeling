"""Histogram operating-point bounds must cover every unresolved bin allocation."""
from __future__ import annotations

import json

import pytest

from scripts.analysis import report_so2_bce_threshold_sensitivity as report
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import ProjectPaths


def histogram():
    positive = [0]*4096
    negative = [0]*4096
    positive[100], positive[409], positive[3000] = 1, 2, 2
    negative[100], negative[409], negative[3000] = 1, 3, 1
    return {'schema': 'binary_expression_detection_curve_histogram_v1', 'bin_count': 4096,
        'bin_width': 1/4096,
        'bin_index_definition': 'floor(nonzero_probability * bin_count), clamped to [0, bin_count - 1]',
        'positive_counts_by_ascending_probability_bin': positive,
        'negative_counts_by_ascending_probability_bin': negative,
        'masked_entries': 10, 'positive_entries': 5}


def test_known_aligned_operating_point_and_requested_threshold_bounds():
    first = report.threshold_rows(histogram())[0]
    assert first['requested_threshold'] == .1
    assert first['reported_aligned_threshold'] == 410/4096
    assert first['lower_aligned_threshold'] == 409/4096
    assert first['exact_at_requested_threshold'] is False
    assert first['true_positive'] == 2 and first['false_positive'] == 1
    assert first['precision'] == pytest.approx(2/3)
    assert first['recall'] == .4
    assert first['f1'] == .5
    assert first['predicted_positive_rate'] == .3
    # All feasible placements of the ambiguous bin's positives/negatives.
    for positive_included in range(3):
        for negative_included in range(4):
            exact = report.metrics(2+positive_included, 1+negative_included, 5, 5)
            for metric, (lower, upper) in first['requested_threshold_bounds'].items():
                assert lower <= exact[metric] <= upper
    # Using only the two aligned precision endpoints as bounds would fail.
    assert first['requested_threshold_bounds']['precision'] == pytest.approx([1/3, 4/5])


def test_primary_half_cutoff_is_exact_and_integer_confusion_gate_is_strict():
    rows = report.threshold_rows(histogram())
    half = rows[-1]
    assert half['requested_threshold'] == half['reported_aligned_threshold'] == .5
    assert half['exact_at_requested_threshold'] is True
    assert all(bounds == [half[key], half[key]] for key, bounds in half['requested_threshold_bounds'].items())
    prefix = 'val/unseen_donor/'
    final = {prefix+'detection_threshold': .5, prefix+'predicted_positive_class_rate': half['predicted_positive_rate']}
    final.update({prefix+'detection_'+key: half[key] for key in ('true_positive', 'true_negative', 'false_positive',
        'false_negative', 'precision', 'recall', 'f1')})
    report.check_primary_metrics(rows, final)
    final[prefix+'detection_true_negative'] += 1
    with pytest.raises(ValueError, match='confusion counts'):
        report.check_primary_metrics(rows, final)


@pytest.mark.parametrize('change', ['fractional_count', 'negative_count', 'wrong_support', 'wrong_bins', 'missing_class'])
def test_histogram_contract_rejects_invalid_counts_and_support(change):
    value = histogram()
    if change == 'fractional_count':
        value['positive_counts_by_ascending_probability_bin'][0] = .5
    elif change == 'negative_count':
        value['negative_counts_by_ascending_probability_bin'][0] = -1
    elif change == 'wrong_support':
        value['masked_entries'] += 1
    elif change == 'wrong_bins':
        value['bin_count'] = 2048
    else:
        value['positive_counts_by_ascending_probability_bin'] = [0]*4096
        value['positive_entries'] = 0
        value['masked_entries'] = 5
    with pytest.raises(ValueError):
        report.threshold_rows(value)


def test_empty_predicted_positive_set_has_explicit_zero_rates():
    value = histogram()
    value['positive_counts_by_ascending_probability_bin'][0] = 5
    value['negative_counts_by_ascending_probability_bin'][0] = 5
    for key in ('positive_counts_by_ascending_probability_bin', 'negative_counts_by_ascending_probability_bin'):
        value[key][1:] = [0]*4095
    rows = report.threshold_rows(value)
    assert all(row[key] == 0 for row in rows for key in report.METRICS)


def test_count_derived_classifiers_expose_rare_positive_f1_tradeoff():
    value = report.constant_classifier_references(8, 100)
    positive, negative = value['classifiers']
    assert (positive['true_positive'], positive['false_positive'], positive['true_negative'], positive['false_negative']) == (8, 92, 0, 0)
    assert positive['precision'] == .08 and positive['recall'] == 1
    assert positive['f1'] == pytest.approx(16/108)
    assert positive['predicted_positive_rate'] == 1
    assert (negative['true_positive'], negative['false_positive'], negative['true_negative'], negative['false_negative']) == (0, 0, 92, 8)
    assert negative['f1'] == negative['recall'] == negative['predicted_positive_rate'] == 0
    assert negative['precision'] == 0 and negative['precision_defined'] is False
    assert value['constant_score_average_precision'] == .08
    # A high-precision, low-recall operating point can have lower F1 than
    # predicting everything positive. This does not make precision equivalent.
    selective = report.metrics(5, 2, 80, 920)
    assert selective['precision'] > .7 and selective['recall'] == .0625
    assert positive['f1'] > selective['f1']
    assert report.THRESHOLDS == (.1, .2, .3, .5)


@pytest.mark.parametrize('positive,total', [(0, 5), (5, 5), (1, 1), (1, 1000000)])
def test_count_references_handle_support_boundaries_without_division_errors(positive, total):
    value = report.constant_classifier_references(positive, total)
    all_positive, all_negative = value['classifiers']
    assert all_positive['f1'] == pytest.approx(2*positive/(total+positive))
    assert all_negative['f1'] == 0
    assert all_negative['precision_defined'] is False
    assert all_positive['recall_defined'] is (positive > 0)
    for row in value['classifiers']:
        assert sum(row[k] for k in ('true_positive', 'false_positive', 'true_negative', 'false_negative')) == total
        assert all(0 <= row[key] <= 1 for key in report.METRICS)


@pytest.mark.parametrize('positive,total', [(0, 0), (-1, 10), (11, 10), (1.5, 10), (1, 10.5), (True, 10)])
def test_count_references_reject_invalid_support(positive, total):
    with pytest.raises(ValueError, match='integer counts'):
        report.constant_classifier_references(positive, total)


@pytest.mark.parametrize('stage,complete', [('grid', False), ('confirmation', False)])
def test_report_refuses_transient_or_incomplete_parent(tmp_path, stage, complete):
    paths = ProjectPaths.from_environment({'BAGM_ROOT': str(tmp_path)})
    folder = paths.report_root/'fixture'
    folder.mkdir(parents=True)
    source = folder/'summary.json'
    source.write_text(json.dumps({'schema': 'so2_bce_optimization_audit_report_v1',
        'campaign_id': report.CAMPAIGN, 'stage': stage, 'complete': complete, 'required_coverage_complete': complete}))
    (folder/'manifest.json').write_text(json.dumps({'files': {'summary.json': sha256_file(source)}}))
    with pytest.raises(ValueError, match='incomplete'):
        report.load_sources(source, paths)
