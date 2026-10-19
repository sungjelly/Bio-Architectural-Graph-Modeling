"""Index the restricted-union descriptive result without changing its parent."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import yaml
from spatial_benchmark.result_catalog import validate_result_manifest
from spatial_benchmark.registry import Registry
from run_so12_cancer_limited_union import (EVALUATION, CAMPAIGN, RUN, METHODS,
    locations, read, require, verify_inventory, record)


def promote():
    p, parent, _, final = locations()
    manifest = verify_inventory(final)
    require(manifest['evaluation_id'] == EVALUATION, 'Evaluation identity differs')
    registry = Registry(p.state_root / 'tracking/bagm.sqlite3')
    with registry.connect() as c:
        row = c.execute('SELECT run_id,status,artifact_path FROM evaluations WHERE evaluation_id=?', (EVALUATION,)).fetchone()
    require(row is not None and row['run_id'] == RUN and row['status'] == 'completed' and Path(row['artifact_path']) == final,
            'Registry publication is incomplete')
    summary = manifest['summary']
    identifier = 'res_so12_test_cancer_limited_cluster_agreement'
    destination = p.result_root / 'descriptive_analysis/representation_comparison' / identifier
    require(not destination.exists(), 'Curated result already exists; do not overwrite a verified record')
    destination.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    observed = []
    for method in METHODS:
        rows = [r for r in summary['primary_scores'] if r['method'] == method]
        observed.append(method + ': ' + '; '.join(f"core {r['core']} one={r['single_f1']:.4f}, ≤2={r['limited_f1']:.4f}, IDs [{r['selected_ids']}]" for r in rows))
    observed_text = 'Fixed resolution 1, seed 0 footprint Dice/F1: ' + ' | '.join(observed) + '. All 56 partitions retained.'
    scvi_run = read(parent / 'scvi_run_registration.json')['derived_run_id']
    result = {'schema_name': 'bagm_result_record', 'schema_version': 1, 'revision': 1,
        'result_id': identifier, 'title': 'One or two complete Leiden clusters against outlined regions on SO12 test cores',
        'status': 'verified', 'outcome': 'inconclusive', 'experiment_type': 'descriptive_analysis',
        'method_family': 'representation_comparison', 'lifecycle_stage': 'exploratory_screen',
        'study_axis': 'limited_cluster_region_agreement', 'campaign_ids': [CAMPAIGN],
        'method': {'name': 'Exact best single, pair and at-most-two cluster unions for frozen h0-h4, PCA and scVI',
            'model_family': 'Relative-QKV graph encoder, training-fitted PCA50 and negative-binomial scVI50',
            'graph_context': 'Unchanged parent joint 81607-test-cell cosine k30 unweighted undirected graph per representation; Leiden RBConfiguration resolutions .25,.5,1,2 and seeds0,1',
            'masking_or_perturbation': 'No refitting; fixed-union 50um boundary sensitivity; 20 within-core label shuffles at each primary partition',
            'evaluation_design': 'Per-core annotation-selected exhaustive singleton/pair search over unchanged segmented-cell support; opposite-core selection and full-region scores secondary. Resolution1 seed0 figures remain fixed. Parent fit uses train21/val5/test2.',
            'implementation_version': 'so12_cancer_limited_union_v1 with immutable code, environment and parent source binding'},
        'conclusion': {'question': 'Can one or two complete clusters delineate the supplied annotated regions while penalizing spillover?',
            'estimand': 'Maximum footprint Dice/F1 under a one/two-cluster cardinality restriction per test core; annotation-selected descriptive agreement',
            'observed_result': observed_text,
            'strongest_alternative_explanation': 'The reference spans several cellular states, so a compact subset may omit much of the region; annotation-selected grouping, core-specific partitions and approximate alignment also influence scores.',
            'controls': ['Identical canonical cells, fixed registration, unchanged parent labels and complete parent checksum verification before and after.',
                'Exact integer enumeration; independent native-pixel audit; no reference-based cropping of selected clusters; FP outside and FN omitted inside remain.',
                'All four resolutions and both seeds; fixed primary figures; one ≤ at-most-two ≤ unrestricted oracle invariants.',
                'Separate exact pair, opposite-core transfer, full-region support, all-positive/empty baselines, fixed-union boundary sensitivity and within-core shuffle controls.',
                'High-contrast prediction groups and TP/FP/FN/TN maps with native-pixel count verification and independent visual review.'],
            'remaining_uncertainty': summary['limitations'],
            'maximum_defensible_claim': summary['maximum_defensible_claim']},
        'evidence': {
            'predictive_gain': {'status': 'not_tested', 'summary': 'Annotation-selected region overlap is not independent predictive accuracy or graph-specific expression gain.'},
            'stability': {'status': 'mixed', 'summary': 'Both Leiden seeds, all four resolutions and fixed-union boundary sensitivity reported without selection of a favorable primary setting.'},
            'faithfulness': {'status': 'not_applicable', 'summary': 'No attribution or cell-cell mechanism inference.'},
            'null_calibration': {'status': 'mixed', 'summary': 'All-positive, empty and within-core shuffled-label controls included; no biological spatial null or population p-value.'},
            'patient_replication': {'status': 'not_tested', 'summary': 'Only two cores, donor overlap permitted and one fitted encoder seed.'},
            'external_support': {'status': 'mixed', 'summary': 'Supplied hand-outlined histology is compared using approximate-section registration and uncertain annotation exhaustiveness.'},
            'perturbation_support': {'status': 'not_tested', 'summary': 'No controlled biological perturbation.'}},
        'provenance': {'created_at_utc': now, 'updated_at_utc': now,
            'git_commit': read(final / 'provenance/environment.json')['git_commit'], 'run_ids': [RUN, scvi_run],
            'expected_seeds': [0, 1], 'included_seeds': [0, 1], 'expected_folds': [0], 'included_folds': [0],
            'failed_or_excluded_runs': [], 'sources': [
                {'kind': 'evaluation', 'root': 'report', 'path': str((final / 'manifest.json').relative_to(p.report_root)),
                 'sha256': record(final / 'manifest.json')['sha256'], 'verification_status': 'verified', 'run_id': RUN},
                {'kind': 'evaluation', 'root': 'report', 'path': str((parent / 'manifest.json').relative_to(p.report_root)),
                 'sha256': record(parent / 'manifest.json')['sha256'], 'verification_status': 'verified', 'run_id': RUN},
                {'kind': 'report', 'root': 'report', 'path': str((parent / 'scvi_run_registration.json').relative_to(p.report_root)),
                 'sha256': record(parent / 'scvi_run_registration.json')['sha256'], 'verification_status': 'verified', 'run_id': scvi_run}]},
        'files': []}
    (destination / 'result.yaml').write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    (destination / 'README.md').write_text('# ' + result['title'] + '\n\n' + observed_text + '\n\n' +
        summary['limitations'] + '\n\nThe restricted positive group contains at most two whole clusters. '
        'All other evaluable pixels remain in the FP/FN denominators. Selected clusters retain their color outside '
        'the outline; no error pixels are hidden. This is a separate follow-up and does not replace the unrestricted result.\n\n' +
        f'Full report, source-bound images and metrics: `{final.relative_to(p.project_root)}`.\n')
    validate_result_manifest(destination / 'result.yaml', paths=p, verify_payloads=True, verify_sources=True)
    subprocess.run([str(Path('/venv/main/bin/python')), 'scripts/results/manage_results.py', 'validate'], cwd=p.project_root, check=True)
    subprocess.run([str(Path('/venv/main/bin/python')), 'scripts/results/manage_results.py', 'catalog'], cwd=p.project_root, check=True)
    print(destination)


if __name__ == '__main__':
    promote()
