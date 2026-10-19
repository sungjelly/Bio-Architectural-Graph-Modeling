"""Portable, figure-led report for the fixed SO12 representation comparison."""
from __future__ import annotations
import argparse
import base64
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spatial_benchmark.paths import current_paths
from spatial_benchmark.fingerprints import sha256_file
from run_so12_paper_test_h0_leiden import read, write

METHODS=['h4','h3','h2','h1','h0','pca','scvi']
DISPLAY={name:name for name in METHODS}|{'pca':'PCA','scvi':'scVI'}


def primary_rows(metrics):
    selected=[r for r in metrics['results'] if r['resolution']==1 and r['seed']==0]
    assert {r['method'] for r in selected}==set(METHODS)
    rows=[]
    for item in selected:
        for core in [15,23]:
            result=item['evaluation']['cores'][str(core)]
            assert result['status']=='complete'
            source=23 if core==15 else 15
            transfer=item['evaluation']['transfer'][f'{source}_to_{core}']
            assert transfer['status']=='complete'
            for name,value in [('oracle',result['oracle']),('cross_core_transfer',transfer)]:
                for support in ['footprint','full_region']:
                    rows.append({'method':item['method'],'core':core,'estimand':name,'support':support,
                        'resolution':1.,'seed':0,'total_clusters':item['cluster_count'],
                        'selected_clusters':','.join(map(str,value['selected_clusters'])),
                        'selected_cluster_count':len(value['selected_clusters']),**value[support]})
    return pd.DataFrame(rows)


def resolution_plot(metrics,folder):
    colors=dict(zip(METHODS,plt.get_cmap('tab10').colors))
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
    for row,core in enumerate([15,23]):
        source=23 if core==15 else 15
        for col,kind in enumerate(['oracle','cross_core_transfer']):
            ax=axes[row,col]
            for method in METHODS:
                for seed in [0,1]:
                    items=sorted((r for r in metrics['results'] if r['method']==method and r['seed']==seed),key=lambda r:r['resolution'])
                    values=[(r['evaluation']['cores'][str(core)]['oracle'] if kind=='oracle' else
                        r['evaluation']['transfer'][f'{source}_to_{core}'])['footprint']['dice_f1'] for r in items]
                    ax.plot([r['resolution'] for r in items],values,'o-' if seed==0 else '--',
                        color=colors[method],alpha=1 if seed==0 else .38,lw=1.8 if seed==0 else 1,
                        label=DISPLAY[method] if seed==0 else None,ms=4)
            control=metrics['results'][0]['evaluation']['cores'][str(core)]['all_positive']['footprint']['dice_f1']
            ax.axhline(control,c='black',ls=':',lw=1.2,label='All cells positive')
            ax.set_title(f'Core {core}: '+('annotation-selected union' if kind=='oracle' else f'union selected on core {source}'))
            ax.set_ylim(0,1.02); ax.set_xticks([.25,.5,1,2]); ax.grid(alpha=.2)
            if row==1: ax.set_xlabel('Leiden resolution')
            if col==0: ax.set_ylabel('Pixel Dice / F1 on cell footprint')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=8,fontsize=9)
    fig.suptitle('All fixed resolutions and both Leiden seeds (solid: seed 0; dashed: seed 1)',fontsize=13)
    for ext in ['png','pdf']: fig.savefig(folder/f'resolution_agreement.{ext}',dpi=180)
    plt.close(fig)


def image_tag(path,caption):
    data=base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def union_error_maps(output,metrics,registration):
    """Show primary oracle-union errors on exactly the scored cell footprint."""
    from matplotlib.patches import Patch
    clustering=read(output/'clustering_summary.json')
    partitions={r['method']:r['labels']['path'] for r in clustering['partitions'] if r['resolution']==1 and r['seed']==0}
    evaluations={r['method']:r['evaluation'] for r in metrics['results'] if r['resolution']==1 and r['seed']==0}
    fig,axes=plt.subplots(2,7,figsize=(23,12))
    for row,core in enumerate([15,23]):
        record=next(r for r in registration['cores'] if r['core_number']==core)
        with np.load(output/record['raster_path'],allow_pickle=False) as arrays:
            owner,roi,valid=arrays['owner'],arrays['roi'],arrays['valid']
        footprint=(owner>=0)&valid
        yy,xx=np.where(valid);bounds=(xx.min(),xx.max()+1,yy.max()+1,yy.min())
        for col,method in enumerate(METHODS):
            labels=np.load(output/partitions[method],allow_pickle=False)
            result=evaluations[method]['cores'][str(core)]['oracle']
            positive=np.zeros(owner.shape,bool)
            positive[footprint]=np.isin(labels[owner[footprint]],result['selected_clusters'])
            measured={'tp':int(np.count_nonzero(footprint&roi&positive)),
                'fp':int(np.count_nonzero(footprint&~roi&positive)),
                'fn':int(np.count_nonzero(footprint&roi&~positive))}
            if any(measured[key]!=result['footprint'][key] for key in measured):
                raise RuntimeError('Rendered error pixels differ from scored counts')
            rgb=np.full((*owner.shape,3),255,np.uint8)
            rgb[footprint]=[224,228,233]
            rgb[footprint&roi&positive]=[32,156,106]
            rgb[footprint&~roi&positive]=[220,70,62]
            rgb[footprint&roi&~positive]=[60,110,210]
            ax=axes[row,col];ax.imshow(rgb,interpolation='nearest');ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.axis('off')
            ax.set_title(f'Core {core} · {DISPLAY[method]}\nF1 {result["footprint"]["dice_f1"]:.3f}',fontsize=11)
    fig.legend(handles=[Patch(color=np.array(color)/255,label=label) for label,color in
        [('True positive',[32,156,106]),('False positive (outside outline)',[220,70,62]),
         ('False negative (missed outline)',[60,110,210]),('True negative',[224,228,233])]],
        loc='lower center',ncol=4,frameon=False,fontsize=11)
    fig.suptitle('Annotation-selected union errors · resolution 1, seed 0 · descriptive oracle, not unbiased accuracy',fontsize=14)
    fig.subplots_adjust(wspace=.02,hspace=.08,top=.95,bottom=.045)
    for ext in ['png','pdf']:fig.savefig(output/f'figures/oracle_union_errors.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def report(output):
    paths=current_paths(); output=Path(output).resolve()
    assert output.is_relative_to(paths.scratch_root.resolve()/'active_runs')
    metrics=read(output/'metrics/pixel_agreement.json')
    assert metrics['passed'] and metrics['pixel_scores_available']
    registration=read(output/'registration/registration_manifest.json')
    mixing_receipt=read(output/'core_mixing_summary.json')
    assert mixing_receipt['passed'] and mixing_receipt['status']=='complete'
    mixing=pd.read_csv(output/'metrics/core_mixing.csv')
    config=read(output/'config.resolved.json')
    pca=read(output/'pca_manifest.json'); scvi=read(output/'scvi_manifest.json')
    assert pca['passed'] and scvi['passed']
    rows=primary_rows(metrics)
    rows.to_csv(output/'metrics/primary_resolution1_seed0.csv',index=False)
    figures=output/'figures'; figures.mkdir(exist_ok=True)
    resolution_plot(metrics,figures)
    union_error_maps(output,metrics,registration)
    compact=[]
    for method in METHODS:
        row={'Method':DISPLAY[method]}
        for estimand,prefix in [('oracle','Oracle'),('cross_core_transfer','Transfer')]:
            for core in [15,23]:
                value=rows.loc[(rows.method==method)&(rows.core==core)&(rows.estimand==estimand)&(rows.support=='footprint'),'dice_f1'].item()
                row[f'{prefix} core {core}']=value
        compact.append(row)
    table=pd.DataFrame(compact)
    table.to_csv(output/'metrics/primary_summary.csv',index=False)
    coverage=[]
    first=metrics['results'][0]['evaluation']['cores']
    for core in [15,23]:
        c=first[str(core)]['coverage']
        coverage.append({'core':core,'cell_footprint_pixels':c['footprint_pixels'],
            'annotated_region_pixels':c['full_region_positive_pixels'],'annotated_coverage':c['positive_region_coverage'],
            'full_region_dice_ceiling':c['full_region_pixel_coverage_dice_ceiling'],
            'all_positive_footprint_dice':first[str(core)]['all_positive']['footprint']['dice_f1']})
    pd.DataFrame(coverage).to_csv(output/'metrics/coverage.csv',index=False)
    summary={'schema':'so12_cancer_comparison_summary_v1','evaluation_id':config['evaluation_id'],
        'source_run_id':config['source_run_id'],'status':'complete','passed':True,'exploratory':True,
        'n_cells':81607,'test_cores':[15,23],'methods':METHODS,'resolutions':config['resolutions'],
        'seeds':[0,1],'primary_resolution':1.,'primary_seed':0,'primary_scores':compact,'coverage':coverage,
        'pca_training_explained_variance':float(np.sum(pca['explained_variance_ratio'])),
        'scvi_selected_epoch_zero_based':scvi['selected_epoch_zero_based'],'scvi_validation_elbo':scvi['validation_elbo'],
        'maximum_defensible_claim':'Exploratory approximate-section cluster-union agreement in two cores; no unbiased cancer classifier accuracy or population superiority claim.',
        'metric':'Dice/F1=2TP/(2TP+FP+FN); all predicted footprint outside outlined regions is FP',
        'selection':'Oracle selects on the scored core; cross-core transfer selects on the other core. All resolutions and seeds retained.',
        'fit_policy':config['fit_policy'],'registration':'Geometry-only similarity, independently visually reviewed; approximate, not pixel-exact',
        'source_sha256':{name:sha256_file(output/name) for name in ['config.resolved.json','layers_manifest.json','pca_manifest.json',
            'scvi_manifest.json','clustering_summary.json','metrics/pixel_agreement.json','registration/registration_manifest.json','figures/figure_manifest.json','core_mixing_summary.json']}}
    write(output/'summary.json',summary)
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO12 core 15 and 23 representation comparison</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#172033;max-width:1500px;margin:32px auto;padding:0 24px}h1{font-size:28px}h2{font-size:21px;margin-top:32px}
img{max-width:100%;height:auto}figure{margin:20px 0}figcaption{color:#475569;font-size:14px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;border-bottom:1px solid #cbd5e1;text-align:right}th:first-child,td:first-child{text-align:left}.note{padding:16px;background:#f1f5f9;border-left:4px solid #64748b}@media print{body{margin:0;font-size:11px}figure,table{break-inside:avoid}img{max-height:90vh}}
</style><h1>Core 15 and 23: frozen representations and annotated regions</h1>
<p>Ground truth, h4, h0, PCA50 and scVI50 are shown on matched geometry. All main maps use Leiden resolution 1, seed 0, fixed before cancer scoring. Cluster colors and IDs are shared between cores within a method; IDs do not correspond between methods.</p>
<p class="note">This is an exploratory comparison to the supplied blue outlines. The H&amp;E screenshots were aligned using tissue geometry alone. Alignment is approximate, especially around the internal tissue gap in core 23. Outlined regions are the user-designated reference; unoutlined tissue is reference-negative for this exercise, not independently verified cancer-free tissue.</p>'''
    text+=image_tag(figures/'combined_five_panel.png','Fixed-resolution comparison: supplied annotated histology, h4, h0, PCA and scVI. Blue outlines retain the original annotation under the same reviewed transform for every method.')
    text+='<h2>Pixel agreement at the fixed primary setting</h2><p>Dice/F1 rewards overlap and penalizes both missed annotated pixels and predicted pixels outside the outline. Multiple clusters may be selected. Primary scores use the segmented-cell footprint; complete-region scores additionally count annotated pixels without segmented cell coverage as false negatives.</p>'
    text+='<p><strong>Oracle</strong> is the exact best union selected and scored on the same core, a descriptive upper bound. <strong>Transfer</strong> selects the union on the other core and then scores this core; the two cores were jointly clustered, so this remains an exploratory transductive analysis. Neither is a trained cancer-classifier accuracy estimate.</p>'
    text+=table.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)
    text+='<p>Predicting every segmented cell positive gives the prevalence baseline shown below. High F1 alone is insufficient when most of the footprint is annotated.</p>'
    text+=pd.DataFrame(coverage).to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)
    text+=image_tag(figures/'resolution_agreement.png','All four prespecified resolutions and both seeds. No resolution or seed was chosen for better annotation agreement. Black dotted lines predict all cells positive.')
    text+=image_tag(figures/'oracle_union_errors.png','Exact annotation-selected union at the primary setting: green overlap, red spill outside the outline, blue missed annotated footprint. Multiple clusters may form each union; the same annotation selects and scores these oracle maps.')
    text+='<h2>Core mixing and transfer limitations</h2><p>Shared cluster IDs do not guarantee that the same clusters occur in both cores. This table measures core composition without using the annotations. A core-dominated cluster has more than 90% of its cells from one core; this may reflect tissue composition, technical differences or spatial context. Poor transfer can coexist with good within-core oracle overlap.</p>'
    mixing_table=mixing[['method','cross_core_edge_fraction','total_clusters','clusters_over_90pct_one_core',
        'core_15_cell_fraction_unseen_in_23','core_23_cell_fraction_unseen_in_15']].copy()
    mixing_table['method']=mixing_table['method'].map(DISPLAY)
    text+=mixing_table.to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0)
    text+='<h2>All graph depths and baselines</h2>'+image_tag(figures/'all_representations_2x7.png','h0 through h4, PCA50 and scVI50, same resolution 1 and seed 0. The trained h0 already incorporates expression and morphology; PCA/scVI are expression-only comparators.')
    text+='<h2>Fit population and reproducibility</h2><p>The latest BAGM model used 21 training cores, five validation cores and held-out cores 15/23. PCA was centered/fitted on those same 264,491 training cells using frozen Pearson features; scVI used the same training cells and the five validation cores for stopping, raw counts of the same 1,000 genes, known slide batches, and frozen posterior-mean inference on test cells. Cancer outlines were excluded from all encoder fitting and clustering.</p>'
    text+=f'<p>PCA50 retains {100*summary["pca_training_explained_variance"]:.2f}% of training feature variance. scVI selected epoch {scvi["selected_epoch_zero_based"]+1} (one-based), validation negative-ELBO loss {scvi["validation_elbo"]:.4f}. BAGM uses frozen epoch 194. All methods use cosine k=30, row L2 normalization, an unweighted undirected union graph and converged RBConfiguration Leiden.</p>'
    text+='<p>The complete machine-readable metrics contain precision, recall, TP/FP/FN, IoU, selected cluster IDs, seed ARI, 20 within-core shuffled-label controls and fixed-union 50-µm boundary sensitivity. Pixel counts are not independent biological replicates. With two cores and one fitted encoder seed, this does not establish population-level superiority or causality.</p>'
    for core in [15,23]: text+=image_tag(output/f'registration/core_{core}_registration_overlay.png',f'Core {core} geometry-only registration audit. Tissue-outline Dice is an alignment diagnostic, not a cancer score.')
    text+='<h2>Methods references</h2><p><a href="https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html">scVI model and frozen latent inference</a>; <a href="https://docs.scvi-tools.org/en/stable/api/reference/scvi.dataloaders.DataSplitter.html">explicit training/validation indices</a>. Software versions, source snapshots, data/checkpoint hashes, registration matrices and logs are retained locally in this bundle.</p></html>'
    target=output/'report/index.html'; target.parent.mkdir(exist_ok=True); target.write_text(text)
    write(output/'report_manifest.json',{'status':'complete','passed':True,'primary_table':'metrics/primary_summary.csv',
        'report_path':'report/index.html','report_sha256':sha256_file(target),'code_sha256':sha256_file(Path(__file__))})
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.output)
