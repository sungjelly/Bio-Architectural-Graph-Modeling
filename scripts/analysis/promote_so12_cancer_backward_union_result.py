"""Promote the verified backward-removal analysis without changing its parents."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import yaml

from spatial_benchmark.registry import Registry
from spatial_benchmark.result_catalog import validate_result_manifest
from run_so12_cancer_backward_union import (
    CAMPAIGN, EVALUATION, METHODS, PARENT_SHA, RUN,
    locations, read, record, require, verify_inventory,
)


RESULT_ID = 'res_so12_test_cancer_backward_retention'


def _number(value, digits=4):
    return 'undefined' if value is None else f'{value:.{digits}f}'


def _verify_coverage(metrics, summary):
    expected = {(method, resolution, seed) for method in METHODS
                for resolution in (.25, .5, 1., 2.) for seed in (0, 1)}
    actual = {(row['method'], float(row['resolution']), int(row['seed']))
              for row in metrics['results']}
    require(metrics['passed'] and len(metrics['results']) == 56 and actual == expected,
            'Incomplete method/resolution/seed coverage')
    require(metrics['partition_count'] == 56 and metrics['core_path_count'] == 112,
            'Incomplete core-path coverage')
    require(metrics['primary_resolution'] == 1. and metrics['primary_seed'] == 0,
            'Primary setting differs from the task contract')
    require(metrics['n_cells'] == 81607 and metrics['all_evaluable_pixels_retained']
            and metrics['cell_retention_unweighted']
            and metrics['cell_retention_denominators_fixed'],
            'Cell population or retention estimand changed')
    for row in metrics['results']:
        require(set(row['cores']) == {'15', '23'}, 'Core coverage differs')
        for curve in row['cores'].values():
            require(curve['status'] == 'complete'
                    and len(curve['steps']) == len(curve['initial_overlap_cluster_ids']) + 1
                    and curve['steps'][-1]['selected_clusters'] == [],
                    'Backward path did not continue through the empty union')
            require(curve['optimal_fixed_cardinality_subset_claimed'] is False,
                    'Greedy deletion cannot establish fixed-cardinality optimality')
    scores = summary['primary_scores']
    require(len(scores) == 14 and {(r['method'], int(r['core'])) for r in scores}
            == {(method, core) for method in METHODS for core in (15, 23)},
            'Primary score summary is incomplete')


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
    _verify_coverage(read(final / 'metrics/backward_union.json'), summary)
    destination = (p.result_root / 'descriptive_analysis/representation_comparison'
                   / RESULT_ID)
    require(not destination.exists(),
            'Curated result already exists; do not overwrite a verified record')
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    observed = []
    for method in METHODS:
        rows = sorted((r for r in summary['primary_scores'] if r['method'] == method),
                      key=lambda r: int(r['core']))
        observed.append(method + ': ' + '; '.join(
            f"core {r['core']} best F1={_number(r['best_f1'])}, "
            f"clusters={r['k']}/{r['initial_clusters']} "
            f"({_number(r['cluster_retained_pct'], 1)}%), "
            f"in-region cells={r['roi_cells_retained']}/{r['roi_cells_total']} "
            f"({_number(r['roi_cells_retained_pct'], 1)}%), "
            f"F1 at ≥90% retention={_number(r['retention90_f1'])}, "
            f"F1 at ≥95% retention={_number(r['retention95_f1'])}"
            for r in rows))
    observed_text = ('Fixed resolution 1, Leiden seed 0, annotation-selected backward '
                     'path maxima: ' + ' | '.join(observed) +
                     '. All 56 partitions and 112 complete core paths are retained.')
    scvi_run = read(parent / 'scvi_run_registration.json')['derived_run_id']
    result = {
        'schema_name': 'bagm_result_record', 'schema_version': 1, 'revision': 1,
        'result_id': RESULT_ID,
        'title': 'Backward cluster removal: region F1 and unweighted in-region cell retention',
        'status': 'verified', 'outcome': 'inconclusive',
        'experiment_type': 'descriptive_analysis', 'method_family': 'representation_comparison',
        'lifecycle_stage': 'exploratory_screen', 'study_axis': 'backward_cluster_retention',
        'campaign_ids': [CAMPAIGN],
        'method': {
            'name': 'Greedy backward complete-cluster removal for frozen h0-h4, PCA and scVI',
            'model_family': 'Relative-QKV graph encoder, training-fitted PCA50 and negative-binomial scVI50',
            'graph_context': 'Unchanged parent joint 81607-test-cell cosine k30 unweighted undirected graph per representation; Leiden RBConfiguration resolutions .25,.5,1,2 and seeds 0,1',
            'masking_or_perturbation': 'No model or registration refit. Start all positive-overlap clusters; remove the cluster giving highest remaining pixel F1, with exact ties resolved by lexicographic retained IDs; continue to empty.',
            'evaluation_design': 'Annotation-selected per-core paths on unchanged segmented-cell support. Highest path F1 breaks ties by fewest clusters then IDs. Show complete curves, nondominated points, and best path F1 at ≥90%/≥95% unweighted in-region-cell retention. Primary resolution 1 seed 0 remains fixed; both seeds and all resolutions are retained.',
            'implementation_version': 'so12_cancer_backward_union_v1 with immutable source, environment and parent checksum binding',
        },
        'conclusion': {
            'question': 'How does annotated-region pixel F1 change as complete clusters and in-region cells are removed?',
            'estimand': 'Pixel Dice/F1 along deterministic backward-deletion paths versus retained unweighted cells with at least one valid owned pixel inside the region. Cell denominators stay fixed; cluster retention is relative to the initial overlapping clusters. The path does not establish optimal subsets at each cardinality.',
            'observed_result': observed_text,
            'strongest_alternative_explanation': 'Annotation prevalence, fragmented cell states, approximate spatial alignment and annotation-selected overfitting can create favorable overlap without independent predictive or population-level superiority.',
            'controls': [
                'Identical canonical cells, frozen parent partitions and registration, complete immutable parent checksum verification.',
                'Independent native-raster audit of every step, selected operating point, fixed denominators and parent unrestricted upper bound.',
                'Selected outside pixels remain FP and omitted inside pixels remain FN; full-region scores additionally include uncovered positive pixels.',
                'All 56 partitions and 112 paths through empty, both Leiden seeds and all four resolutions; fixed primary resolution 1 seed 0.',
                'Majority-inside-cell sensitivity uses at least half valid owned area inside the region; unweighted counts remain distinct from area-weighted pixel recall.',
                'Full F1-retention curves and nondominated points; ≥90% and ≥95% retention are illustrative supplementary constraints without a combined weighted objective.',
            ],
            'remaining_uncertainty': summary['limitations'],
            'maximum_defensible_claim': summary['maximum_defensible_claim'],
        },
        'evidence': {
            'predictive_gain': {'status': 'not_tested', 'summary': 'Annotation-selected overlap is not independent predictive accuracy or graph-specific expression gain.'},
            'stability': {'status': 'mixed', 'summary': 'All four resolutions and both Leiden seeds plus majority-inside-cell boundary sensitivity are reported; one fitted encoder seed remains.'},
            'faithfulness': {'status': 'not_applicable', 'summary': 'No attribution or cell-cell mechanism inference.'},
            'null_calibration': {'status': 'not_tested', 'summary': 'The empty endpoint and parent unrestricted oracle bound characterize the deletion path; no new biological null calibration or population p-value is established.'},
            'patient_replication': {'status': 'not_tested', 'summary': 'Only two cores, donor overlap permitted and one fitted encoder seed; cells and pixels are not independent biological replicates.'},
            'external_support': {'status': 'mixed', 'summary': 'Supplied hand-outlined histology uses approximate-section registration and uncertain annotation exhaustiveness; these annotations also select clusters.'},
            'perturbation_support': {'status': 'not_tested', 'summary': 'Cluster removal is an analysis operation, not a controlled biological perturbation.'},
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
        'An in-region cell has at least one valid owned pixel inside the supplied region. '
        'Retention is an unweighted cell count over a fixed denominator. Cluster retention '
        'uses the initial positive-overlap clusters as its denominator. Majority-inside '
        'retention requires at least half the valid owned cell area inside and is a sensitivity analysis.\n\n'
        'Every selected cluster contributes its full evaluable footprint: outside pixels remain '
        'false positives and omitted inside pixels remain false negatives. Each removal '
        'maximizes the next remaining union F1 with exact lexicographic ties and continues to '
        'empty. The best path point breaks ties by fewest clusters then IDs. This greedy path '
        'is compared with the exact parent unrestricted maximum; it does not guarantee '
        'optimality at each cluster count. Full curves, nondominated points and best F1 '
        'subject to ≥90%/≥95% cell retention are included in the report.\n\n'
        + summary['limitations'] + '\n\n' + summary['maximum_defensible_claim'] + '\n\n'
        + f'Full report, figures and metrics: `{final.relative_to(p.project_root)}`. '
        'This separate follow-up preserves both earlier evaluations.\n')
    validate_result_manifest(destination / 'result.yaml', paths=p,
                             verify_payloads=True, verify_sources=True)
    command = [sys.executable, 'scripts/results/manage_results.py']
    for arguments in (['validate', '--verify-payloads', '--verify-sources'],
                      ['catalog', '--verify-payloads', '--verify-sources'],
                      ['catalog', '--check', '--verify-payloads', '--verify-sources']):
        subprocess.run(command + arguments, cwd=p.project_root, check=True)
    print(destination)


if __name__ == '__main__':
    promote()
