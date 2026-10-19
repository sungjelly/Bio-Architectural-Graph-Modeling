"""Promote verified pooled-core backward-removal evidence as a separate result."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import yaml

from spatial_benchmark.registry import Registry
from spatial_benchmark.result_catalog import validate_result_manifest
from run_so12_cancer_pooled_union import (
    CAMPAIGN, EVALUATION, METHODS, PARENT_SHA, RUN,
    locations, read, record, require, verify_inventory,
)


RESULT_ID = 'res_so12_test_cancer_pooled_retention'


def _number(value, digits=4):
    return 'undefined' if value is None else f'{value:.{digits}f}'


def _verify_coverage(metrics, summary, config):
    expected = {(method, resolution, seed) for method in METHODS
                for resolution in (.25, .5, 1., 2.) for seed in (0, 1)}
    actual = {(row['method'], float(row['resolution']), int(row['seed']))
              for row in metrics['results']}
    require(metrics['passed'] and len(metrics['results']) == 56 and actual == expected,
            'Incomplete pooled method/resolution/seed coverage')
    require(config['primary_resolution'] == 1. and config['primary_seed'] == 0
            and config['n_cells'] == 81607 and set(config['cores']) == {15, 23},
            'Primary setting or pooled cell population differs from the contract')
    scores = summary['primary_scores']
    require(len(scores) == 7 and {row['method'] for row in scores} == set(METHODS),
            'Primary pooled score summary is incomplete')
    require(all('core' not in row for row in scores),
            'Primary scores must be pooled, not independently selected per core')


def _observed_text(summary):
    by_method = {row['method']: row for row in summary['primary_scores']}
    descriptions = []
    for method in METHODS:
        row = by_method[method]
        descriptions.append(
            f"{method}: pooled micro F1={_number(row['best_f1'])}, "
            f"clusters={row['k']}/{row['initial_clusters']} "
            f"({_number(row['cluster_retained_pct'], 1)}%), "
            f"in-region cells={row['roi_cells_retained']}/{row['roi_cells_total']} "
            f"({_number(row['roi_cells_retained_pct'], 1)}%), "
            f"F1 at ≥90% retention={_number(row['retention90_f1'])}, "
            f"F1 at ≥95% retention={_number(row['retention95_f1'])}")
    return ('Fixed resolution 1, Leiden seed 0; one shared global cluster-ID union '
            'applies to both cores 15 and 23. Annotation-selected backward path maxima: '
            + ' | '.join(descriptions)
            + '. All 56 partitions and complete pooled paths are retained.')


def promote():
    p, parent, _, final = locations()
    manifest = verify_inventory(final)
    verify_inventory(parent, PARENT_SHA)
    require(manifest['evaluation_id'] == EVALUATION
            and manifest['source_run_id'] == RUN
            and manifest['parent_manifest_sha256'] == PARENT_SHA
            and manifest['passed'], 'Evaluation identity or verification differs')
    registry = Registry(p.state_root / 'tracking/bagm.sqlite3')
    with registry.connect() as connection:
        row = connection.execute(
            'SELECT run_id,status,artifact_path FROM evaluations WHERE evaluation_id=?',
            (EVALUATION,)).fetchone()
    require(row is not None and row['run_id'] == RUN and row['status'] == 'completed'
            and Path(row['artifact_path']) == final, 'Registry publication is incomplete')
    summary = manifest['summary']
    _verify_coverage(read(final / 'metrics/pooled_union.json'), summary,
                     read(final / 'config.resolved.json'))
    destination = (p.result_root / 'descriptive_analysis/representation_comparison'
                   / RESULT_ID)
    require(not destination.exists(),
            'Curated result already exists; do not overwrite a verified record')
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    observed_text = _observed_text(summary)
    scvi_run = read(parent / 'scvi_run_registration.json')['derived_run_id']
    result = {
        'schema_name': 'bagm_result_record', 'schema_version': 1, 'revision': 1,
        'result_id': RESULT_ID,
        'title': 'Pooled-core backward cluster removal: micro F1 and in-region cell retention',
        'status': 'verified', 'outcome': 'inconclusive',
        'experiment_type': 'descriptive_analysis', 'method_family': 'representation_comparison',
        'lifecycle_stage': 'exploratory_screen', 'study_axis': 'pooled_cluster_retention',
        'campaign_ids': [CAMPAIGN],
        'method': {
            'name': 'Pooled backward removal of shared global clusters for frozen h0-h4, PCA and scVI',
            'model_family': 'Relative-QKV graph encoder, training-fitted PCA50 and negative-binomial scVI50',
            'graph_context': 'Unchanged parent joint 81607-test-cell cosine k30 unweighted undirected graph per representation; Leiden RBConfiguration resolutions .25,.5,1,2 and seeds 0,1',
            'masking_or_perturbation': 'No refitting. Start every global cluster overlapping either annotation; use the same retained cluster IDs on both cores. Remove one cluster maximizing the remaining pooled pixel F1, with exact ties resolved by lexicographic retained IDs; continue to empty.',
            'evaluation_design': 'Annotation-selected shared union on cores 15 and 23, optimizing micro Dice/F1 computed from summed pixel TP, FP and FN. The path maximum breaks ties by fewest clusters then IDs. Full paths, nondominated points and best F1 at ≥90%/≥95% pooled unweighted cell retention are reported. Primary resolution 1 seed 0 stays fixed; all settings are retained.',
            'implementation_version': 'so12_cancer_pooled_union_v1 with immutable code, environment and parent source binding',
        },
        'conclusion': {
            'question': 'How does pooled annotated-region micro F1 change as one shared set of global clusters is removed across both test cores?',
            'estimand': 'Micro pixel Dice/F1 = 2ΣTP/(2ΣTP+ΣFP+ΣFN) along one deterministic shared-cluster deletion path, versus pooled unweighted retention of cells with at least one valid owned pixel inside their core annotation. Cell denominators stay fixed; cluster retention is relative to initial global positive-overlap clusters. The pooled metric is not the mean of core F1 scores or the combination of independently optimized core unions.',
            'observed_result': observed_text,
            'strongest_alternative_explanation': 'Annotation prevalence, unequal core pixel contributions, fragmented cellular states, approximate registration and annotation-selected overfitting can improve pooled overlap while hiding core heterogeneity and without independent predictive superiority.',
            'controls': [
                'Identical canonical cells, frozen parent partitions and registration, complete immutable parent checksum verification.',
                'One shared global cluster-ID set at every step; pooled confusion counts are summed across cores before computing F1.',
                'Independent native-raster audit of deletion steps, fixed denominators, operating points and exact pooled unrestricted reference.',
                'Selected outside pixels remain FP and omitted inside pixels remain FN; full-region scores include uncovered positive pixels.',
                'All 56 partitions and paths through empty, both Leiden seeds and all four resolutions; primary resolution 1 seed 0 fixed in advance.',
                'Unweighted pooled cell retention with fixed denominators; majority-inside sensitivity requires at least half the valid owned cell area inside its annotation.',
                'Full F1-retention curves and nondominated points, plus illustrative ≥90% and ≥95% retention constraints; no combined weighted objective or fixed-cardinality optimality claim.',
            ],
            'remaining_uncertainty': summary['limitations'],
            'maximum_defensible_claim': summary['maximum_defensible_claim'],
        },
        'evidence': {
            'predictive_gain': {'status': 'not_tested', 'summary': 'Annotation-selected pooled overlap is not independent predictive accuracy or graph-specific expression gain.'},
            'stability': {'status': 'mixed', 'summary': 'All four resolutions, both Leiden seeds and majority-inside cell sensitivity are retained; there is one fitted encoder seed.'},
            'faithfulness': {'status': 'not_applicable', 'summary': 'No attribution or cell-cell mechanism inference.'},
            'null_calibration': {'status': 'not_tested', 'summary': 'The empty endpoint and exact pooled unrestricted reference characterize the path; no new biological null calibration or population p-value is established.'},
            'patient_replication': {'status': 'not_tested', 'summary': 'Two cores, donor overlap permitted and one fitted encoder seed; pooling pixels or cells does not create independent biological replicates.'},
            'external_support': {'status': 'mixed', 'summary': 'Supplied hand-outlined histology uses approximate-section registration and uncertain annotation exhaustiveness; the annotations also select clusters.'},
            'perturbation_support': {'status': 'not_tested', 'summary': 'Computational cluster removal is not a controlled biological perturbation.'},
        },
        'provenance': {
            'created_at_utc': now, 'updated_at_utc': now,
            'git_commit': read(final / 'provenance/environment.json')['git_commit'],
            'run_ids': [RUN, scvi_run], 'expected_seeds': [0, 1], 'included_seeds': [0, 1],
            'expected_folds': [0], 'included_folds': [0], 'failed_or_excluded_runs': [],
            'sources': [
                {'kind': 'evaluation', 'root': 'report',
                 'path': str((final / 'manifest.json').relative_to(p.report_root)),
                 'sha256': record(final / 'manifest.json')['sha256'],
                 'verification_status': 'verified', 'run_id': RUN},
                {'kind': 'evaluation', 'root': 'report',
                 'path': str((parent / 'manifest.json').relative_to(p.report_root)),
                 'sha256': PARENT_SHA, 'verification_status': 'verified', 'run_id': RUN},
                {'kind': 'report', 'root': 'report',
                 'path': str((parent / 'scvi_run_registration.json').relative_to(p.report_root)),
                 'sha256': record(parent / 'scvi_run_registration.json')['sha256'],
                 'verification_status': 'verified', 'run_id': scvi_run},
            ],
        },
        'files': [],
    }
    destination.mkdir(parents=True)
    (destination / 'result.yaml').write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    (destination / 'README.md').write_text(
        '# ' + result['title'] + '\n\n' + observed_text + '\n\n'
        'One shared set of global cluster IDs applies to both cores. Pooled micro F1 '
        'uses summed TP, FP and FN pixel counts before calculating the ratio. It is '
        'not an average of core F1 values or a combination of independently optimized '
        'core selections. Each selected cluster contributes its entire evaluable '
        'footprint in both cores; outside pixels remain false positives and omitted '
        'inside pixels remain false negatives.\n\n'
        'An in-region cell has at least one valid owned pixel inside its core annotation. '
        'Pooled retention sums these unweighted cell counts across cores over a fixed '
        'denominator. Cluster retention uses the initial global overlapping clusters. '
        'Majority-inside sensitivity requires at least half the valid owned cell area '
        'inside the annotation.\n\n'
        'Every greedy removal maximizes remaining pooled F1 with exact lexicographic '
        'ties and continues through the empty union. The path maximum breaks ties by '
        'fewest clusters then IDs. Comparison with the exact pooled unrestricted '
        'reference does not establish optimality at each cluster count. Full curves, '
        'nondominated points and best path F1 subject to ≥90%/≥95% pooled cell retention '
        'are retained; the thresholds illustrate the tradeoff.\n\n'
        + summary['limitations'] + '\n\n' + summary['maximum_defensible_claim'] + '\n\n'
        + f'Full report, figures and metrics: `{final.relative_to(p.project_root)}`. '
        'This separate follow-up preserves the earlier per-core evaluations.\n')
    validate_result_manifest(destination / 'result.yaml', paths=p,
                             verify_payloads=True, verify_sources=True)
    command = [sys.executable, 'scripts/results/manage_results.py']
    # Other historical records have unavailable payloads; retain their existing
    # structural validation policy while strictly checking this new source set.
    for arguments in (['validate'], ['catalog'], ['catalog', '--check']):
        subprocess.run(command + arguments, cwd=p.project_root, check=True)
    print(destination)


if __name__ == '__main__':
    promote()
