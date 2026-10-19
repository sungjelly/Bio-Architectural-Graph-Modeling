"""Report pixel F1 alongside constant-denominator unweighted cell retention."""
from __future__ import annotations
import argparse
import base64
import html
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from run_so12_cancer_backward_union import locations,read,write,record,require,EVALUATION,CAMPAIGN,RUN,METHODS

DISPLAY={'pca':'PCA','scvi':'scVI',**{f'h{i}':f'h{i}' for i in range(5)}}
LIMITATIONS=('All group choices use the scored annotation and are descriptive, not independent cancer-classifier accuracy. '
    'Greedy removal chooses the best next deletion, not necessarily the globally best subset at every cluster count. '
    'The path maximum is checked against the parent exact unrestricted optimum. Registration is approximate; '
    'annotation exhaustiveness and section identity are unverified. Pixels/cells are observations from only two cores, not independent patients.')


def rows(metrics):
    result=[]
    for method in METHODS:
        row=next(x for x in metrics['results'] if (x['method'],x['resolution'],x['seed'])==(method,1.,0))
        for core in [15,23]:
            path=row['cores'][str(core)];best=path['steps'][path['best_step_index']]
            result.append({'method':method,'core':core,'best_f1':best['footprint']['dice_f1'],
                'k':best['k'],'initial_clusters':best['initial_overlap_clusters'],'cluster_retained_pct':best['cluster_retained_pct'],
                'roi_cells_retained':best['roi_cells_retained'],'roi_cells_total':best['roi_cells_total'],
                'roi_cells_retained_pct':best['roi_cells_retained_pct'],'majority_cells_retained_pct':best['majority_cells_retained_pct'],
                'precision':best['footprint']['precision'],'recall':best['footprint']['recall'],
                'selected_clusters':best['selected_clusters'],
                'retention90_f1':path['steps'][path['best_retention90_step_index']]['footprint']['dice_f1'],
                'retention95_f1':path['steps'][path['best_retention95_step_index']]['footprint']['dice_f1'],
                'greedy_gap_to_parent_unrestricted_f1':path['greedy_gap_to_parent_unrestricted_f1']})
    return result


def save(fig,output,name):
    result={}
    for ext in ['png','pdf']:
        file=output/f'figures/{name}.{ext}';fig.savefig(file,dpi=190,facecolor='white',bbox_inches='tight')
        result[ext]={'path':str(file.relative_to(output)),**record(file)}
    plt.close(fig);return result


def render(parent,output):
    parent,output=Path(parent).resolve(),Path(output).resolve();p,expected_parent,active,_=locations()
    require(parent==expected_parent and output==active,'Requires the registered active evaluation')
    metrics=read(output/'metrics/backward_union.json');require(metrics['passed'] and len(metrics['results'])==56,'Incomplete metrics')
    primary=rows(metrics);pd.DataFrame(primary).to_csv(output/'metrics/primary_report_table.csv',index=False)
    (output/'figures').mkdir(exist_ok=True);figures={}
    fig,axes=plt.subplots(2,7,figsize=(24,8),sharex=True,sharey=True)
    for column,method in enumerate(METHODS):
        part=next(x for x in metrics['results'] if (x['method'],x['resolution'],x['seed'])==(method,1.,0))
        for row,core in enumerate([15,23]):
            axis=axes[row,column];path=part['cores'][str(core)];steps=path['steps'];best=steps[path['best_step_index']]
            axis.plot([s['roi_cells_retained_pct'] for s in steps],[s['footprint']['dice_f1'] for s in steps],color='#334155',marker='o',ms=3,lw=1.3)
            frontier=[s for s in steps if s['pareto_nondominated']]
            axis.scatter([s['roi_cells_retained_pct'] for s in frontier],[s['footprint']['dice_f1'] for s in frontier],color='#009E73',s=35,zorder=3)
            axis.scatter([best['roi_cells_retained_pct']],[best['footprint']['dice_f1']],color='#D55E00',marker='*',s=160,zorder=5)
            axis.set_title(f'Core {core} · {DISPLAY[method]}\nBest F1 {best["footprint"]["dice_f1"]:.3f}; {best["k"]}/{best["initial_overlap_clusters"]} clusters\n{best["roi_cells_retained_pct"]:.1f}% in-region cells',fontsize=10)
            for x in [90,95]:axis.axvline(x,color='#64748b',ls=':',lw=.8,alpha=.6)
            axis.set_xlim(0,103);axis.set_ylim(0,1.02);axis.set_xticks([0,25,50,75,100]);axis.grid(alpha=.2)
            if column==0:axis.set_ylabel('Pixel Dice/F1')
            if row==1:axis.set_xlabel('In-region cells retained (%)')
    handles=[Line2D([0],[0],color='#334155',marker='o',label='Every removal step'),
        Line2D([0],[0],color='#009E73',marker='o',ls='',label='No other step improves both F1 and retention'),
        Line2D([0],[0],color='#D55E00',marker='*',ls='',markersize=12,label='Best F1 on path')]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
    fig.suptitle('Backward removal · fixed resolution 1, seed 0 · read each path from 100% toward 0%',fontsize=16)
    fig.text(.5,.002,'Cell denominator is fixed: every cell with at least one valid owned pixel inside the annotation. Dotted lines: 90% and 95% retention.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.085,1,.95));figures['f1_vs_cell_retention']=save(fig,output,'f1_vs_cell_retention')
    fig,axes=plt.subplots(2,7,figsize=(24,7.2),sharex=True,sharey=True)
    colors={.25:'#0072B2',.5:'#009E73',1.:'#D55E00',2.:'#CC79A7'}
    for column,method in enumerate(METHODS):
        for row,core in enumerate([15,23]):
            axis=axes[row,column]
            for part in metrics['results']:
                if part['method']!=method:continue
                path=part['cores'][str(core)];best=path['steps'][path['best_step_index']]
                axis.scatter(best['roi_cells_retained_pct'],best['footprint']['dice_f1'],color=colors[part['resolution']],marker='o' if part['seed']==0 else 'x',s=75)
            axis.set_title(f'Core {core} · {DISPLAY[method]}');axis.set_xlim(0,103);axis.set_ylim(0,1.02);axis.grid(alpha=.2)
            if column==0:axis.set_ylabel('Best pixel Dice/F1 on path')
            if row==1:axis.set_xlabel('In-region cells retained (%)')
    handles=[Line2D([0],[0],color=color,marker='o',ls='',label=f'Resolution {res:g}') for res,color in colors.items()]
    handles += [Line2D([0],[0],color='#444444',marker=m,ls='',label=f'Seed {s}') for s,m in [(0,'o'),(1,'x')]]
    fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False)
    fig.suptitle('Best path operating points · every resolution and both Leiden seeds',fontsize=16)
    fig.tight_layout(rect=(0,.08,1,.94));figures['all_settings_best_points']=save(fig,output,'all_settings_best_points')
    table=[]
    for row in primary:
        table.append({'Method':DISPLAY[row['method']],'Core':row['core'],'Best F1':row['best_f1'],
            'Clusters kept':f'{row["k"]}/{row["initial_clusters"]} ({row["cluster_retained_pct"]:.1f}%)',
            'In-region cells kept':f'{row["roi_cells_retained"]:,}/{row["roi_cells_total"]:,} ({row["roi_cells_retained_pct"]:.1f}%)',
            'F1 with ≥90% cells':row['retention90_f1'],'F1 with ≥95% cells':row['retention95_f1']})
    maximum_gap=max(c['greedy_gap_to_parent_unrestricted_f1'] for r in metrics['results'] for c in r['cores'].values())
    summary={'evaluation_id':EVALUATION,'campaign_id':CAMPAIGN,'source_run_id':RUN,'passed':True,'status':'complete',
        'n_cells':81607,'partitions':56,'paths':112,'primary_resolution':1.,'primary_seed':0,'primary_scores':primary,
        'maximum_greedy_gap_to_exact_unrestricted':maximum_gap,'limitations':LIMITATIONS,
        'maximum_defensible_claim':'Exploratory F1/cluster-count/cell-retention tradeoffs on two approximately aligned annotated test cores; no independent diagnostic accuracy or population superiority.',
        'cell_retention_definition':'Fixed denominator of cells with at least one valid owned pixel inside annotation; unweighted count.',
        'cluster_retention_definition':'Fixed denominator of clusters with at least one valid owned pixel inside annotation at step zero.'}
    write(output/'summary.json',summary)
    images=[]
    for name,files in figures.items():
        raw=(output/files['png']['path']).read_bytes();encoded=base64.b64encode(raw).decode()
        caption='Every deletion step at the fixed primary setting.' if name=='f1_vs_cell_retention' else 'All 56 partitions retained; each point is its own annotation-selected path maximum.'
        images.append(f'<figure><img alt="{html.escape(caption,quote=True)}" src="data:image/png;base64,{encoded}"><figcaption>{caption}</figcaption></figure>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Backward cluster removal and cell retention</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#17212b;background:white;max-width:1500px;margin:25px auto;padding:0 24px}p{max-width:1100px}table{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}td,th{padding:8px;border:1px solid #cbd5e1;text-align:right}th{background:#edf2f7}figure{margin:30px 0}img{max-width:100%;height:auto}.note{border-left:4px solid #D55E00;padding:12px;background:#fff4dc}@media print{figure{break-inside:avoid}table{font-size:10px}}</style><main>
<h1>Backward cluster removal and cell retention</h1><p>Start with every cluster overlapping the annotation. At each step try removing each remaining cluster, choose the removal giving the highest remaining pixel F1, and continue to zero clusters. Whole clusters are retained: selected pixels outside the outline remain false positives and omitted pixels inside remain false negatives.</p>
<p><strong>Two different percentages:</strong> clusters kept / starting overlapping clusters, and in-region cells kept / all in-region cells. An in-region cell has at least one valid owned pixel inside the annotation. These denominators stay fixed; cells are counted once without area weighting. Pixel recall is a separate area-weighted measure. Majority-inside cell counts are retained as boundary sensitivity in the full tables.</p>
<p>There is no arbitrary combined score. The best-F1 result is shown alongside the best available F1 on the removal path with at least90% or95% cell retention. These are illustrative coverage constraints. Green curve points have no other step with both better F1 and equal-or-better retention, or better retention and equal-or-better F1.</p>
<p class="note">'''+html.escape(LIMITATIONS)+'''</p><h2>Fixed resolution 1, seed 0</h2>'''+pd.DataFrame(table).to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)+''.join(images)+f'<p>Largest path-maximum shortfall from the parent exact unrestricted optimum across112 paths: {maximum_gap:.12g}. A zero gap verifies the best F1 for these data, but does not certify every intermediate fixed-cardinality group as globally optimal.</p>'+'''
<h2>Complete numeric outputs</h2><p><a href="../metrics/backward_steps.csv">Every removal step</a> · <a href="../metrics/backward_best_summary.csv">Best and constrained points at all settings</a> · <a href="../metrics/primary_report_table.csv">Compact primary table</a>. Figures are embedded for offline use; linked numeric files require the companion bundle directory.</p>
<p>Registration, cluster labels and representations are unchanged; no model fitting or additional data copies. Only two cores contribute biological observations. The existing train/validation/test split remains unchanged.</p></main></html>'''
    (output/'report').mkdir(exist_ok=True);(output/'report/index.html').write_text(page)
    write(output/'report_manifest.json',{'passed':True,'status':'complete','report_path':'report/index.html','report_record':record(output/'report/index.html'),
        'figures':figures,'metrics_sha256':record(output/'metrics/backward_union.json')['sha256'],'embedded_pngs':2})
    print({'primary_rows':len(primary),'maximum_greedy_gap':maximum_gap,'figures':list(figures)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();render(args.parent,args.output)
