"""Render pooled backward-removal F1 and constant-denominator cell retention."""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from run_so12_cancer_pooled_union import locations, read, write, record, require, EVALUATION, CAMPAIGN, RUN, METHODS

DISPLAY = {'pca':'PCA', 'scvi':'scVI', **{f'h{i}':f'h{i}' for i in range(5)}}
COLORS = ['#255B91','#3572A5','#4A85B4','#639BC2','#82AFCF','#278573','#A46A40']
LIMITATIONS = ('Cluster groups are selected using both scored annotations. Scores are descriptive agreement, '
    'not independent cancer-classifier accuracy. Registration is approximate, annotation exhaustiveness is '
    'unverified, and only two cores contribute biological observations. Pooling weights native evaluable '
    'pixels equally, not cores or calibrated physical area. Greedy removal does not certify globally '
    'optimal subsets at every cluster count.')


def save(fig, output, name):
    files = {}
    for ext in ['png','pdf','svg']:
        file = output/f'figures/{name}.{ext}'
        fig.savefig(file,dpi=300,facecolor='white')
        files[ext]={'path':str(file.relative_to(output)),**record(file)}
    plt.close(fig)
    return files


def render(parent, output):
    p, expected, active, _ = locations()
    parent, output = Path(parent).resolve(), Path(output).resolve()
    require(parent == expected and output == active, 'Expected registered active evaluation')
    metrics=read(output/'metrics/pooled_union.json')
    require(metrics['passed'] and len(metrics['results']) == 56, 'Incomplete metric grid')
    rows=[]
    for method in METHODS:
        part=next(r for r in metrics['results'] if (r['method'],r['resolution'],r['seed']) == (method,1.,0))
        curve=part['pooled'];best=curve['steps'][curve['best_step_index']]
        rows.append({'method':method,'best_f1':best['footprint']['dice_f1'],'k':best['k'],
            'initial_clusters':best['initial_overlap_clusters'],'cluster_retained_pct':best['cluster_retained_pct'],
            'roi_cells_retained':best['roi_cells_retained'],'roi_cells_total':best['roi_cells_total'],
            'roi_cells_retained_pct':best['roi_cells_retained_pct'],'majority_cells_retained_pct':best['majority_cells_retained_pct'],
            'precision':best['footprint']['precision'],'recall':best['footprint']['recall'],
            'selected_clusters':best['selected_clusters'],
            'tp':best['footprint']['tp'],'fp':best['footprint']['fp'],'fn':best['footprint']['fn'],
            'retention90_f1':curve['steps'][curve['best_retention90_step_index']]['footprint']['dice_f1'],
            'retention95_f1':curve['steps'][curve['best_retention95_step_index']]['footprint']['dice_f1'],
            'greedy_gap_to_parent_unrestricted_f1':curve['greedy_gap_to_parent_unrestricted_f1']})
    pd.DataFrame(rows).to_csv(output/'metrics/primary_report_table.csv',index=False)
    require(all(abs(r['best_f1']-2*r['tp']/(2*r['tp']+r['fp']+r['fn']))<1e-15 for r in rows),'Invalid pooled F1')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#243447',
        'axes.labelcolor':'#243447','xtick.color':'#243447','ytick.color':'#243447','pdf.fonttype':42,'svg.fonttype':'none'})
    (output/'figures').mkdir(exist_ok=True);figures={}
    values=np.array([r['best_f1'] for r in rows]);lo=float(np.floor(values.min()*100)/100-.01);hi=float(np.ceil(values.max()*100)/100+.01)
    require(0 < lo < values.min() and hi > values.max(),'Invalid zoom range')
    fig,axis=plt.subplots(figsize=(10.5,5.9));fig.subplots_adjust(left=.095,right=.97,bottom=.26,top=.76)
    fig.text(.095,.94,'Cancer-region agreement · cores 15 + 23',fontsize=19,weight='bold')
    fig.text(.095,.882,'One shared cluster group per representation · best pooled pixel F1',fontsize=12)
    fig.text(.095,.835,'Backward cluster removal · Leiden resolution 1, seed 0',fontsize=10.5,color='#526376')
    bars=axis.bar(np.arange(7),values-lo,bottom=lo,width=.66,color=COLORS,edgecolor='white',linewidth=.8,zorder=3)
    require(np.allclose([b.get_y()+b.get_height() for b in bars],values,rtol=0,atol=1e-15),'Bar tops differ')
    for x,value in enumerate(values):axis.text(x,value+(hi-lo)*.023,f'{value:.4f}',ha='center',va='bottom',fontsize=12)
    axis.set(ylim=(lo,hi),xlim=(-.65,6.65),xticks=np.arange(7),xticklabels=[DISPLAY[m] for m in METHODS],ylabel='Pooled pixel F1 (Dice)')
    axis.yaxis.set_major_locator(MultipleLocator(.01 if hi-lo<=.09 else .02));axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.grid(axis='y',color='#DDE3EA',lw=.8,zorder=0);axis.spines[['top','right']].set_visible(False)
    for spine in ['left','bottom']:axis.spines[spine].set_color('#AAB5C0')
    axis.tick_params(axis='x',length=0,pad=8)
    for y in [.012,.034]:axis.plot([-.008,.008],[y-.012,y+.012],transform=axis.transAxes,color='#243447',lw=1.3,clip_on=False,zorder=5)
    axis.text(.5,-.19,f'Zoomed y axis: {lo:.2f}–{hi:.2f} (zero omitted)',transform=axis.transAxes,ha='center',fontsize=10,color='#526376')
    fig.text(.095,.11,'F1 = 2ΣTP / (2ΣTP + ΣFP + ΣFN); the same selected cluster IDs apply to both cores.',fontsize=10)
    fig.text(.095,.063,'Each evaluable native pixel has equal weight. Exact labels show scores on the truncated axis.',fontsize=9.5,color='#526376')
    fig.text(.095,.025,'Annotation-selected descriptive agreement; tissue-section alignment is approximate.',fontsize=9.5,color='#526376')
    figures['pooled_f1_bar']=save(fig,output,'pooled_f1_bar')

    table_rows=[]
    for r in rows:
        table_rows.append([DISPLAY[r['method']],f"{r['best_f1']:.4f}",f"{r['k']}/{r['initial_clusters']} ({r['cluster_retained_pct']:.1f}%)",
            f"{r['roi_cells_retained']:,}/{r['roi_cells_total']:,} ({r['roi_cells_retained_pct']:.1f}%)"])
    fig,axis=plt.subplots(figsize=(10.5,5.0));axis.set_axis_off();fig.subplots_adjust(left=.055,right=.97,bottom=.18,top=.78)
    fig.text(.055,.93,'Pooled F1 and region coverage · cores 15 + 23',fontsize=18,weight='bold')
    fig.text(.055,.865,'One shared selected union · fixed Leiden resolution 1, seed 0',fontsize=11)
    table=axis.table(cellText=table_rows,colLabels=['Method','Pooled F1','Clusters kept','In-region cells kept'],cellLoc='center',colWidths=[.14,.16,.28,.42],bbox=(0,0,1,1))
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (r,c),cell in table.get_celld().items():
        cell.set_linewidth(.6);cell.set_edgecolor('#DDE3EA')
        if r==0:cell.set_facecolor('#243F59');cell.set_text_props(color='white',weight='bold')
        else:
            cell.set_facecolor('#F0F4F8' if r%2 else 'white')
            if c==0:cell.set_text_props(color=COLORS[r-1],weight='bold')
    fig.text(.055,.105,'Clusters: selected / initially overlapping global clusters, counted once across cores.',fontsize=9.5)
    fig.text(.055,.058,'Cells: selected / all cells with any valid pixel inside their annotation; both cores combined.',fontsize=9.5)
    fig.text(.055,.022,'Approximate alignment; annotation-selected descriptive agreement.',fontsize=9,color='#526376')
    figures['pooled_f1_retention_table']=save(fig,output,'pooled_f1_retention_table')

    fig,axes=plt.subplots(2,7,figsize=(22,7.8),sharex=True,sharey=True)
    colors={.25:'#0072B2',.5:'#009E73',1.:'#D55E00',2.:'#CC79A7'}
    for column,method in enumerate(METHODS):
        primary=next(x for x in metrics['results'] if (x['method'],x['resolution'],x['seed'])==(method,1.,0))['pooled']
        steps=primary['steps'];best=steps[primary['best_step_index']]
        axes[0,column].plot([s['roi_cells_retained_pct'] for s in steps],[s['footprint']['dice_f1'] for s in steps],marker='o',ms=3,color=COLORS[column],lw=1.3)
        axes[0,column].scatter(best['roi_cells_retained_pct'],best['footprint']['dice_f1'],s=120,marker='*',color='#D55E00',zorder=5)
        axes[0,column].set_title(f"{DISPLAY[method]} · F1 {best['footprint']['dice_f1']:.3f}\n{best['roi_cells_retained_pct']:.1f}% cells")
        for part in metrics['results']:
            if part['method'] != method:continue
            path=part['pooled'];point=path['steps'][path['best_step_index']]
            axes[1,column].scatter(point['roi_cells_retained_pct'],point['footprint']['dice_f1'],color=colors[part['resolution']],marker='o' if part['seed']==0 else 'x',s=65)
        for row in [0,1]:
            ax=axes[row,column];ax.set_xlim(0,103);ax.set_ylim(0,1.02);ax.grid(alpha=.2);ax.set_xticks([0,25,50,75,100])
            if column==0:ax.set_ylabel('Pooled pixel F1')
            if row==1:ax.set_xlabel('In-region cells kept (%)')
    fig.suptitle('Pooled F1–retention tradeoffs · top: resolution 1, seed 0 paths; bottom: all setting maxima',fontsize=15)
    fig.text(.5,.035,'Bottom colors: blue r0.25, green r0.5, orange r1, purple r2. Circle: seed 0; cross: seed 1. Top star: best path F1.',ha='center',fontsize=10)
    fig.text(.5,.01,'Shared cluster groups are selected using the pooled annotations. Fixed cell denominator; no optimal fixed-cardinality subset claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94));figures['pooled_f1_retention_paths']=save(fig,output,'pooled_f1_retention_paths')

    summary={'evaluation_id':EVALUATION,'campaign_id':CAMPAIGN,'source_run_id':RUN,'passed':True,'status':'complete',
        'n_cells':81607,'cores':[15,23],'partitions':56,'paths':56,'primary_resolution':1.,'primary_seed':0,'primary_scores':rows,
        'aggregation':'pooled_native_pixel_micro_F1_with_shared_cluster_selection',
        'maximum_greedy_gap_to_exact_unrestricted':max(r['pooled']['greedy_gap_to_parent_unrestricted_f1'] for r in metrics['results']),
        'limitations':LIMITATIONS,'maximum_defensible_claim':'Exploratory pooled overlap and cell-retention tradeoffs for one shared cluster group across two approximately aligned test cores; no independent accuracy or population superiority.',
        'cell_retention_definition':'Fixed pooled denominator of cells with at least one valid owned pixel inside their annotation; unweighted count.',
        'cluster_retention_definition':'Fixed denominator of unique global clusters overlapping either annotation at step zero.',
        'bar_axis_limits':[lo,hi]}
    write(output/'summary.json',summary)
    images=[]
    for name,files in figures.items():
        encoded=base64.b64encode((output/files['png']['path']).read_bytes()).decode()
        images.append(f'<figure><img alt="{name.replace("_"," ")}" src="data:image/png;base64,{encoded}"></figure>')
    display_table=pd.DataFrame(table_rows,columns=['Method','Pooled F1','Clusters kept','In-region cells kept'])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pooled-core F1</title><style>body{font:16px/1.5 system-ui;color:#243447;max-width:1400px;margin:24px auto;padding:0 24px}img{max-width:100%}figure{margin:24px 0}td,th{padding:9px;border:1px solid #ddd;text-align:right}table{border-collapse:collapse}p{max-width:1100px}@media print{figure{break-inside:avoid}}</style><main><h1>Pooled-core backward cluster removal</h1><p>One global cluster-ID union is applied to both cores. At each step, remove the cluster giving the highest remaining pooled pixel F1. Sum TP, FP and FN before computing F1; do not average per-core scores or independently optimize core masks. All selected cluster pixels outside either annotation count as false positives.</p><p>Cluster retention counts each global cluster once. Cell retention sums disjoint cells across both cores with a fixed denominator. Each native evaluable pixel receives equal weight; neither cores nor physical area are equally weighted.</p><p>'''+html.escape(LIMITATIONS)+'''</p>'''+display_table.to_html(index=False,border=0)+''.join(images)+'''<p><a href="../metrics/primary_report_table.csv">Primary full precision CSV</a> · <a href="../metrics/pooled_steps.csv">All deletion steps</a> · <a href="../metrics/pooled_best_summary.csv">All settings and retention operating points</a></p></main></html>'''
    (output/'report').mkdir(exist_ok=True);(output/'report/index.html').write_text(page)
    write(output/'report_manifest.json',{'passed':True,'status':'complete','report_path':'report/index.html','report_record':record(output/'report/index.html'),'figures':figures,'metrics_sha256':record(output/'metrics/pooled_union.json')['sha256'],'embedded_pngs':3})
    print({'primary_rows':len(rows),'axis_limits':[lo,hi],'figures':list(figures)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.parent,args.output)
