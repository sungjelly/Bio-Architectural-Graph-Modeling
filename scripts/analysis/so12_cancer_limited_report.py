"""Portable figure-led report for exact one/two-cluster selection."""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from run_so12_cancer_limited_union import (EVALUATION, CAMPAIGN, RUN, PARENT_SHA,
    METHODS, read, write, record, require, locations)

DISPLAY = {'pca': 'PCA', 'scvi': 'scVI', **{f'h{i}': f'h{i}' for i in range(5)}}
LIMITATION = ('These are annotation-selected descriptive F1 scores, not independently evaluated '
    'cancer-classifier accuracy. Region selection uses the scored core. The opposite-core transfer '
    'results are separately reported and remain exploratory because both cores were jointly clustered. '
    'Registration is approximate; section identity and annotation exhaustiveness are unverified. '
    'Only 69.5% and 65.0% of annotated pixels have segmented-cell coverage on cores 15 and 23. '
    'Two cores and one fitted encoder seed do not support population superiority claims.')


def primary_rows(metrics):
    rows = []
    primary = {r['method']: r for r in metrics['results'] if r['resolution'] == 1 and r['seed'] == 0}
    require(set(primary) == set(METHODS), 'Primary partition coverage incomplete')
    for method in METHODS:
        partition = primary[method]
        for core in [15, 23]:
            item = partition['cores'][str(core)]
            chosen = item['best_at_most_two']
            score = chosen['footprint']
            other = 23 if core == 15 else 15
            rows.append({'method': method, 'core': core,
                'single_ids': ','.join(map(str, item['best_single']['selected_clusters'])),
                'single_f1': item['best_single']['footprint']['dice_f1'],
                'pair_ids': ','.join(map(str, item['best_pair']['selected_clusters'])),
                'pair_f1': (item['best_pair']['footprint'] or {}).get('dice_f1'),
                'selected_ids': ','.join(map(str, chosen['selected_clusters'])),
                'selected_count': len(chosen['selected_clusters']),
                'limited_f1': score['dice_f1'], 'precision': score['precision'], 'recall': score['recall'],
                'tp': score['tp'], 'fp': score['fp'], 'fn': score['fn'],
                'full_region_f1': chosen['full_region']['dice_f1'],
                'transfer_f1': partition['transfer'][f'{other}_to_{core}']['best_at_most_two']['footprint']['dice_f1'],
                'unrestricted_f1': item['unrestricted']['footprint']['dice_f1']})
    return rows


def resolution_plot(output, metrics):
    fig, axes = plt.subplots(2, 7, figsize=(23, 7.2), sharex=True, sharey=True)
    for column, method in enumerate(METHODS):
        for row, core in enumerate([15, 23]):
            axis = axes[row, column]
            for mode, color, label in [('best_single', '#0072B2', 'Best one'), ('best_at_most_two', '#D55E00', 'Best ≤2')]:
                vals = {seed: [] for seed in [0, 1]}
                resolutions = [.25, .5, 1., 2.]
                for seed in [0, 1]:
                    for res in resolutions:
                        record_ = next(r for r in metrics['results'] if r['method'] == method and r['resolution'] == res and r['seed'] == seed)
                        vals[seed].append(record_['cores'][str(core)][mode]['footprint']['dice_f1'])
                axis.plot(resolutions, vals[0], color=color, marker='o', label=label + ', seed 0')
                axis.plot(resolutions, vals[1], color=color, linestyle='--', alpha=.7, label=label + ', seed 1')
            axis.axvline(1, color='#444444', lw=.7, alpha=.5)
            axis.set_title(f'Core {core} · {DISPLAY[method]}')
            axis.set_xticks([.25, .5, 1, 2], ['.25', '.5', '1', '2'])
            axis.set_ylim(0, 1)
            axis.grid(axis='y', alpha=.2)
            if column == 0: axis.set_ylabel('Annotation-selected F1')
            if row == 1: axis.set_xlabel('Leiden resolution')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('One- and two-cluster unions · every resolution and both Leiden seeds', fontsize=16)
    fig.tight_layout(rect=(0, .075, 1, .96))
    output.mkdir(exist_ok=True)
    records = {}
    for extension in ['png', 'pdf']:
        path = output / f'resolution_agreement.{extension}'
        fig.savefig(path, dpi=190, facecolor='white', bbox_inches='tight')
        records[extension] = {'path': path.name, **record(path)}
    plt.close(fig)
    return records


def render(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    p, expected_parent, active, _ = locations()
    require(parent == expected_parent and output == active, 'Report requires the registered active evaluation')
    metrics = read(output / 'metrics/limited_union.json')
    require(len(metrics['results']) == 56, 'Incomplete metrics')
    rows = primary_rows(metrics)
    frame = pd.DataFrame(rows)
    primary_partition = next(r for r in metrics['results'] if r['resolution'] == 1 and r['seed'] == 0)
    all_positive = {core: primary_partition['cores'][str(core)]['all_positive']['footprint']['dice_f1'] for core in [15, 23]}
    frame.to_csv(output / 'metrics/primary_selection_table.csv', index=False)
    figures = read(output / 'figures/figure_manifest.json')
    figures['figures']['resolution_agreement'] = resolution_plot(output / 'figures', metrics)
    write(output / 'figures/figure_manifest.json', figures)
    primary_table = frame[['method', 'core', 'single_f1', 'limited_f1', 'selected_ids', 'precision', 'recall', 'fp', 'fn']].copy()
    primary_table.columns = ['Method', 'Core', 'Best one F1', 'Best ≤2 F1', 'Selected IDs', 'Precision', 'Recall', 'FP pixels', 'FN pixels']
    secondary = frame[['method', 'core', 'full_region_f1', 'transfer_f1', 'unrestricted_f1']].copy()
    secondary.columns = ['Method', 'Core', 'Full-region F1, same union', 'Other-core transfer F1', 'Unrestricted oracle F1']
    summary = {'evaluation_id': EVALUATION, 'campaign_id': CAMPAIGN, 'source_run_id': RUN,
        'status': 'complete', 'passed': True, 'parent_manifest_sha256': PARENT_SHA,
        'n_cells': 81607, 'test_cores': [15, 23], 'partitions': 56,
        'primary_resolution': 1., 'primary_seed': 0, 'primary_scores': rows,
        'all_positive_footprint_f1': all_positive,
        'prediction_group': 'Exact best union of at most two whole visible clusters; all evaluable pixels retained',
        'maximum_defensible_claim': 'Exploratory approximate-section agreement of one/two annotation-selected clusters on two test cores; no independent classifier accuracy or population superiority.',
        'limitations': LIMITATION,
        'source_sha256': {'metrics/limited_union.json': record(output / 'metrics/limited_union.json')['sha256'],
            'figures/figure_manifest.json': record(output / 'figures/figure_manifest.json')['sha256']}}
    write(output / 'summary.json', summary)
    captions = {
        'core_15_five_panel': 'Core 15: supplied histology and best ≤2 cluster groups at fixed resolution 1, seed 0.',
        'core_23_five_panel': 'Core 23: supplied histology and best ≤2 cluster groups at fixed resolution 1, seed 0.',
        'combined_five_panel': 'Both cores, identical selected-versus-other color meanings across all representations.',
        'limited_union_errors': 'Best ≤2 cluster union errors. Green: TP; vermillion: FP; blue: FN; light gray: TN. White is outside evaluated segmented support.',
        'best_single_errors': 'Best single-cluster errors, on exactly the same pixel support as the ≤2 comparison.',
        'resolution_agreement': 'All four resolutions and both seeds. Solid: seed 0; dashed: seed 1. No setting is removed based on its score.',
    }
    ordered = ['combined_five_panel', 'limited_union_errors', 'best_single_errors',
               'core_15_five_panel', 'core_23_five_panel', 'all_representations_2x7', 'resolution_agreement']
    ordered += [name for name in figures['figures'] if name not in ordered]
    rendered = []
    embedded = []
    for name in ordered:
        if name not in figures['figures']: continue
        item = figures['figures'][name]['png']
        imagepath = output / 'figures' / item['path']
        require(record(imagepath)['sha256'] == item['sha256'], 'Figure bytes changed')
        encoded = base64.b64encode(imagepath.read_bytes()).decode('ascii')
        caption = captions.get(name, name.replace('_', ' ') + ' · Fixed resolution 1, seed 0; selected groups use the scored annotation.')
        rendered.append(f'<figure><img alt="{html.escape(caption, quote=True)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        embedded.append({'name': name, **record(imagepath)})
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>One- and two-cluster region agreement</title><style>
body{font:16px/1.5 system-ui,sans-serif;margin:28px auto;max-width:1500px;padding:0 24px;color:#17212b;background:white}
h1,h2{line-height:1.2}p{max-width:1100px}figure{margin:30px 0}img{max-width:100%;height:auto}figcaption{color:#334155}
table{border-collapse:collapse;font-size:14px;margin:20px 0;width:100%}th,td{padding:7px 10px;border:1px solid #cbd5e1;text-align:right}th{background:#edf2f7}
.note{padding:14px;background:#fff4dc;border-left:4px solid #D55E00}code{overflow-wrap:anywhere}
@media print{body{margin:0;font-size:11px}figure{break-inside:avoid}table{font-size:10px}h2{break-after:avoid}}
</style><main><h1>One- and two-cluster region agreement</h1>
<p>Each representation now selects its best single cluster and best pair. The main maps show whichever has the higher F1 (at most two clusters). Every selected cluster is shown in full, including its pixels outside the annotated region.</p>
<p>Other cells remain in the evaluation. An outlined pixel assigned to an unselected cluster is a false negative; a selected pixel outside the outline is a false positive. F1 = 2TP / (2TP + FP + FN). Empty-background gaps are excluded from primary footprint scoring and penalized in the secondary full-region score.</p>
<p class="note">''' + html.escape(LIMITATION) + '''</p>
<h2>Fixed resolution 1, seed 0</h2>''' + primary_table.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0) + '<p>The all-positive footprint baseline is ' + '/'.join(f'{all_positive[core]:.4f}' for core in [15,23]) + ' (cores 15/23). Its broad coverage can yield high F1 when the annotation occupies most of the core; compact selected groups trade that coverage for specificity.</p>' + ''.join(rendered) + '''
<h2>Coverage, transfer and unrestricted comparison</h2>
<p>Full-region scoring retains the same selected group and adds uncovered annotated pixels as missed pixels. Transfer selects the group on the opposite core and freezes its IDs. The unrestricted parent union is an annotation-selected upper bound, often using many more clusters.</p>''' + secondary.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0) + '''
<h2>Reproducibility</h2><p>All 56 existing partitions were reused: seven representations, four resolutions and two Leiden seeds. No training, preprocessing refit, registration refit or reclustering was performed. Best groups are selected separately per core and partition, using exact integer overlap enumeration. The supplied annotation is not used to color predictions by location. Source and image checksums, all settings, sensitivity scores, within-core shuffle controls and independent pixel checks accompany this report.</p>
<p>Machine-readable metrics: <a href="../metrics/limited_union.csv">all settings CSV</a> and <a href="../metrics/primary_selection_table.csv">primary selected groups CSV</a>. The embedded figures are portable offline; linked companion files require the bundle directory.</p>
<p>Evaluation <code>''' + EVALUATION + '</code>; parent manifest <code>' + PARENT_SHA + '</code>.</p></main></html>'
    reportdir = output / 'report'
    reportdir.mkdir(exist_ok=True)
    (reportdir / 'index.html').write_text(page)
    write(output / 'report_manifest.json', {'passed': True, 'status': 'complete', 'report_path': 'report/index.html',
        **record(reportdir / 'index.html'), 'embedded_pngs': embedded})
    print({'report': str(reportdir / 'index.html'), 'primary_rows': len(rows)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    render(args.parent, args.output)
