"""High-contrast selected-cluster groups and native-pixel union errors."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from so12_cancer_figures import load_partitions
from so12_cancer_registration import WINDOWS, BAR_PIXELS
from run_so12_cancer_limited_union import read, write, record, require, locations, METHODS

PRIMARY = ('gt', 'h4', 'h0', 'pca', 'scvi')
DISPLAY = {'gt': 'Annotated histology', 'pca': 'PCA', 'scvi': 'scVI',
           **{f'h{i}': f'BAGM h{i}' for i in range(5)}}
GROUP_COLORS = {'selected': '#E66101', 'other': '#2858A4'}
ERROR_COLORS = {'tp': '#009E73', 'fp': '#D55E00', 'fn': '#0072B2', 'tn': '#CBD5E1'}


def rgb(hexcode):
    return np.array([int(hexcode[i:i+2], 16) for i in (1, 3, 5)], dtype=np.uint8)


def predictions(owner, valid, labels, selected):
    owner, valid, labels = np.asarray(owner), np.asarray(valid), np.asarray(labels)
    require(owner.shape == valid.shape and owner.dtype.kind == 'i' and valid.dtype == np.bool_, 'Invalid owner/valid raster')
    require(owner.min() >= -1 and owner.max() < len(labels), 'Owner index outside canonical labels')
    require(len(selected) == len(set(selected)) and all(0 <= x <= labels.max() for x in selected), 'Invalid selected cluster IDs')
    support = valid & (owner >= 0)
    prediction = np.zeros(owner.shape, dtype=bool)
    prediction[support] = np.isin(labels[owner[support]], selected)
    return support, prediction


def group_image(owner, valid, labels, selected):
    """Color solely by cluster membership, never by the reference ROI."""
    support, prediction = predictions(owner, valid, labels, selected)
    image = np.full((*owner.shape, 3), 255, dtype=np.uint8)
    image[support & ~prediction] = rgb(GROUP_COLORS['other'])
    image[prediction] = rgb(GROUP_COLORS['selected'])
    return image


def error_image(owner, valid, roi, labels, selected, expected):
    support, prediction = predictions(owner, valid, labels, selected)
    require(roi.shape == owner.shape and roi.dtype == np.bool_, 'Invalid reference mask')
    masks = {'tp': support & prediction & roi, 'fp': support & prediction & ~roi,
             'fn': support & ~prediction & roi, 'tn': support & ~prediction & ~roi}
    image = np.full((*owner.shape, 3), 255, dtype=np.uint8)
    counts = {}
    for key, mask in masks.items():
        counts[key] = int(mask.sum())
        require(counts[key] == expected[key], f'Plotted {key} count differs from metric')
        image[mask] = rgb(ERROR_COLORS[key])
    require(sum(counts.values()) == int(support.sum()), 'Error categories do not partition the support')
    return image, counts


def panel(axis, image, roi, core, title):
    x0, y0, x1, y1 = WINDOWS[core]
    crop = np.s_[y0:y1, x0:x1]
    axis.imshow(image[crop], interpolation='nearest')
    # A white halo and black line remain visible on all prediction/error colors.
    axis.contour(roi[crop], levels=[.5], colors=['white'], linewidths=2.3)
    axis.contour(roi[crop], levels=[.5], colors=['#111111'], linewidths=.9)
    axis.set_xlim(0, x1-x0); axis.set_ylim(y1-y0, 0)
    axis.set_aspect('equal'); axis.axis('off'); axis.set_title(title, fontsize=10, pad=9)
    x, y = 35, y1-y0-35
    length = BAR_PIXELS[core]
    axis.plot([x, x+length], [y, y], color='white', linewidth=5)
    axis.plot([x, x+length], [y, y], color='#111111', linewidth=2.5)
    axis.text(x+length/2, y-14, '500 µm', ha='center', va='bottom', fontsize=8,
              bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 1.5})


def title(method, item, core=None, errors=False):
    if method == 'gt': return DISPLAY[method]
    score = item['footprint']
    clusters = ', '.join(map(str, item['selected_clusters']))
    prefix = f'Core {core} · ' if core else ''
    text = f'{prefix}{DISPLAY[method]} · clusters {clusters}\nF1 {score["dice_f1"]:.3f} · P {score["precision"]:.3f} · R {score["recall"]:.3f}'
    if errors: text += f'\nFP {score["fp"]:,} · FN {score["fn"]:,} pixels'
    return text


def save(fig, directory, name):
    result = {}
    for ext in ['png', 'pdf']:
        path = directory / f'{name}.{ext}'
        fig.savefig(path, dpi=200, facecolor='white', bbox_inches='tight')
        result[ext] = {'path': path.name, **record(path)}
    plt.close(fig)
    return result


def group_legend(fig):
    handles = [Patch(color=GROUP_COLORS['selected'], label='Selected cluster group'),
        Patch(color=GROUP_COLORS['other'], label='All other cells'),
        Line2D([0], [0], color='#111111', lw=1.5, label='Supplied annotation boundary')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .022))
    fig.text(.5, .006, 'Best ≤2 clusters selected using each core’s annotation · Resolution 1, seed 0 · Approximate section registration', ha='center', fontsize=9)


def render(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    p, expected_parent, active, _ = locations()
    require(parent == expected_parent and output == active, 'Figures require the registered active evaluation')
    metrics = read(output / 'metrics/limited_union.json')
    require(metrics['passed'] and metrics['all_evaluable_pixels_retained'], 'Metric support is incomplete')
    primary = {row['method']: row for row in metrics['results'] if (row['resolution'], row['seed']) == (1., 0)}
    require(set(primary) == set(METHODS), 'Primary methods incomplete')
    partitions = load_partitions(parent)
    registration = read(parent / 'registration/registration_manifest.json')
    require(registration['passed'] and registration['status'] == 'verified_approximate_section', 'Reviewed registration required')
    figures = output / 'figures'; figures.mkdir(exist_ok=True)
    sources, rois, masks, maps, scores = {}, {}, {}, {}, {}
    checks, records = [], {}
    for row in registration['cores']:
        core = row['core_number']; path = parent / row['raster_path']
        require(record(path)['sha256'] == row['raster_sha256'], 'Owner raster changed')
        with np.load(path, allow_pickle=False) as arrays:
            owner, valid, roi = arrays['owner'], arrays['valid'], arrays['roi']
        rois[core] = roi; masks[core] = (owner, valid)
        source = Path(row['image']['path'])
        require(record(source)['sha256'] == row['image']['sha256'], 'Clinical input changed')
        sources[str(core)] = {'raster': row['raster_path'], 'raster_sha256': row['raster_sha256'], 'image': row['image']}
        maps[core] = {'gt': np.asarray(Image.open(source).convert('RGB'))}
        scores[core] = {}
        for method in METHODS:
            item = primary[method]['cores'][str(core)]['best_at_most_two']
            scores[core][method] = item
            maps[core][method] = group_image(owner, valid, partitions[method][0], item['selected_clusters'])
        fig, axes = plt.subplots(1, 5, figsize=(18, 6.5))
        for axis, method in zip(axes, PRIMARY, strict=True):
            panel(axis, maps[core][method], roi, core, title(method, scores[core].get(method)))
        fig.suptitle(f'Core {core} · best one- or two-cluster groups', fontsize=16, y=.985)
        fig.subplots_adjust(wspace=.045, top=.86, bottom=.11)
        group_legend(fig)
        records[f'core_{core}_five_panel'] = save(fig, figures, f'core_{core}_five_panel')
    for name, methods, figsize in [('combined_five_panel', PRIMARY, (19, 12.2)),
                                   ('all_representations_2x7', METHODS, (25, 11.8))]:
        fig, axes = plt.subplots(2, len(methods), figsize=figsize)
        for r, core in enumerate([15, 23]):
            for c, method in enumerate(methods):
                label = f'Core {core} · Annotated histology' if method == 'gt' else title(method, scores[core][method], core)
                panel(axes[r, c], maps[core][method], rois[core], core, label)
        fig.subplots_adjust(wspace=.04, hspace=.16, top=.95, bottom=.07)
        group_legend(fig)
        records[name] = save(fig, figures, name)
    for mode, name, label in [('best_at_most_two', 'limited_union_errors', 'Best ≤2 clusters'),
                             ('best_single', 'best_single_errors', 'Best single cluster')]:
        fig, axes = plt.subplots(2, 7, figsize=(25, 12.4))
        for r, core in enumerate([15, 23]):
            owner, valid = masks[core]
            for c, method in enumerate(METHODS):
                item = primary[method]['cores'][str(core)][mode]
                error, counts = error_image(owner, valid, rois[core], partitions[method][0], item['selected_clusters'], item['footprint'])
                checks.append({'core': core, 'method': method, 'selection': mode, 'counts': counts, 'passed': True})
                panel(axes[r, c], error, rois[core], core, title(method, item, core, errors=True))
        legend_names = {'tp': 'TP · selected inside', 'fp': 'FP · selected outside', 'fn': 'FN · missed inside', 'tn': 'TN · other outside'}
        fig.legend(handles=[Patch(color=color, label=legend_names[key]) for key, color in ERROR_COLORS.items()],
                   loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .022))
        fig.suptitle(f'{label} · annotation-selected union errors', fontsize=17, y=.99)
        fig.text(.5, .006, 'Black/white: supplied boundary · White gaps: outside evaluated segmented support · Resolution 1, seed 0', ha='center', fontsize=9)
        fig.subplots_adjust(wspace=.045, hspace=.22, top=.89, bottom=.07)
        records[name] = save(fig, figures, name)
    result = {'schema': 'so12_cancer_limited_figures_v1', 'passed': True, 'status': 'complete',
        'resolution': 1., 'seed': 0, 'n_cells': 81607, 'selection': 'best_at_most_two',
        'group_colors': GROUP_COLORS, 'error_colors': ERROR_COLORS, 'prediction_colors_depend_on_roi': False,
        'shared_registration': True, 'annotations_shifted': False, 'scale_bar_um': 500,
        'metric_sha256': record(output / 'metrics/limited_union.json')['sha256'],
        'registration_manifest_sha256': record(parent / 'registration/registration_manifest.json')['sha256'],
        'sources': sources, 'partitions': {key: item for key, (_, item) in partitions.items()},
        'native_pixel_checks': checks, 'figures': records}
    write(figures / 'figure_manifest.json', result)
    print({'figures': list(records), 'native_error_maps_checked': len(checks)}, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    render(args.parent, args.output)
