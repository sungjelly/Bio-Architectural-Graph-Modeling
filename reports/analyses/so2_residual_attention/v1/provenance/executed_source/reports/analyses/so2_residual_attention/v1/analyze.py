"""Read-only checkpoint activation diagnostic; see adjacent README.md."""
import argparse
import csv
import importlib.util
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from spatial_benchmark.paths import current_paths
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.relative_qkv_post_training import fixed_inference_mask
from spatial_benchmark.so2_recurrent_hl_clustering import _load_core_batch

PATHS = current_paths()
OUT = PATHS.report_root / 'analyses/so2_residual_attention/v1'
PREVIOUS = PATHS.report_root / 'analyses/so2_attention_tau_beta/v1'
RUN = 'r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf'
SHA = 'aba6a1d910e60f6a30bb4b27b15dbd1c30199da41a15460c23d994079b67543b'
RATIO_BINS = np.r_[0., np.geomspace(1e-4, 1e3, 141), np.inf]

def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

def load_model():
    spec = importlib.util.spec_from_file_location('tau_beta_source', PREVIOUS/'analyze.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.load_model(torch.device('cpu'))

def distribution(x):
    assert np.isfinite(x).all()
    return dict(mean=float(np.mean(x)), std=float(np.std(x)),
                min=float(np.min(x)), max=float(np.max(x)),
                q01=float(np.quantile(x,.01)), q10=float(np.quantile(x,.1)),
                q25=float(np.quantile(x,.25)), median=float(np.median(x)),
                q75=float(np.quantile(x,.75)), q90=float(np.quantile(x,.9)),
                q99=float(np.quantile(x,.99)))

def vector_metrics(h,u,f,y):
    h,u,f,y = [np.asarray(x,dtype=np.float64) for x in (h,u,f,y)]
    a = h+u
    hn,un,an,fn,yn = [np.linalg.norm(x,axis=1) for x in (h,u,a,f,y)]
    assert np.all(hn>0) and np.all(an>0)
    hu = np.sum(h*u,axis=1)
    cosine = np.divide(hu,hn*un,out=np.zeros_like(hu),where=un>0)
    hc,uc = [x-x.mean(axis=1,keepdims=True) for x in (h,u)]
    hcn,ucn = [np.linalg.norm(x,axis=1) for x in (hc,uc)]
    assert np.all(hcn>0)
    ratios = dict(attention_to_residual=un/hn, ffn_to_post_attention=fn/an,
                  post_attention_to_residual=an/hn, output_to_residual=yn/hn,
                  feature_centered_attention_to_residual=ucn/hcn)
    vectors = dict(residual=h,attention=u,post_attention=a,ffn=f,output=y)
    ms = {name:float(np.mean(np.sum(x*x,axis=1))) for name,x in vectors.items()}
    centered_ms = {name:float(np.mean(np.sum((x-x.mean(0))**2,axis=1)))
                   for name,x in vectors.items()}
    # Exact double-precision algebra; y has actual FP32 addition rounding.
    identity = float(np.max(np.abs(an**2-(hn**2+un**2+2*hu)))/max(1.,float(np.max(an**2))))
    assert identity < 1e-12 and np.max(np.abs(cosine)) <= 1+1e-12
    return dict(distributions={k:distribution(v) for k,v in ratios.items()},
                cosine=distribution(cosine), mean_squared_norms=ms,
                between_cell_mean_squared_norms=centered_ms,
                mean_dot_residual_attention=float(hu.mean()),
                fraction_attention_larger=float(np.mean(un>hn)),
                fraction_opposing=float(np.mean(hu<0)),
                zero_attention_cells=int(np.sum(un==0)),
                energy_identity_relative_error=identity,
                rms_attention_to_residual=float(np.sqrt(ms['attention']/ms['residual'])),
                between_cell_rms_attention_to_residual=float(np.sqrt(centered_ms['attention']/centered_ms['residual'])),
                rms_ffn_to_post_attention=float(np.sqrt(ms['ffn']/ms['post_attention']))), ratios

def self_check():
    h=np.array([[1.,-1.],[2.,-2.]])
    for factor in (0.,.5,-.5):
        u=factor*h; f=np.zeros_like(h)
        m,_=vector_metrics(h,u,f,h+u)
        assert abs(m['distributions']['attention_to_residual']['mean']-abs(factor))<1e-12
        if factor: assert abs(m['cosine']['mean']-np.sign(factor))<1e-12
    u=np.array([[1.,1.],[2.,2.]])
    m,_=vector_metrics(h,u,np.zeros_like(h),h+u)
    assert abs(m['cosine']['mean'])<1e-12
    assert abs(m['distributions']['post_attention_to_residual']['mean']-np.sqrt(2))<1e-12

@torch.no_grad()
def analyze_core(model,config,core,pilot):
    began=time.time(); alias=f'SO2-C{core}'
    cohort=PATHS.data_root/'processed/so2_14core_relative_qkv_v1'
    graphs=PATHS.data_root/'processed/so2_14core_relative_qkv_graphs_v1'
    assert sha256_file(cohort/'manifest.json')==config['dataset']['cohort_manifest_file_sha256']
    assert sha256_file(graphs/'manifest.json')==config['dataset']['graph_manifest_file_sha256']
    cm=json.loads((cohort/'manifest.json').read_text());gm=json.loads((graphs/'manifest.json').read_text())
    cr=next(r for r in cm['cores'] if r['alias']==alias);gr=next(r for r in gm['cores'] if r['alias']==alias)
    record=dict(cr,directed_edges=gr['graph']['qc']['n_directed_edges'],
                receiver_major_canonical_order=gr['graph']['qc']['receiver_major_canonical_order'])
    inputs=[cohort/'cores'/f'{alias}.npz',graphs/'cores'/alias/'edge_index.npy',graphs/'cores'/alias/'relative_geometry.npy']
    hashes={str(p.relative_to(PATHS.project_root)):sha256_file(p) for p in inputs}
    for p in inputs:
        expected=cm['files'][str(p.relative_to(cohort))] if p.is_relative_to(cohort) else gr['files'][p.name]
        assert hashes[str(p.relative_to(PATHS.project_root))]==expected
    batch,_=_load_core_batch(cohort_dir=cohort,graph_dir=graphs,record=record)
    mask=fixed_inference_mask(batch);mt=torch.from_numpy(mask.mask.copy())
    prior=json.loads((PATHS.report_root/f'analyses/so2_distance_attention/v1/core_{core}.json').read_text())
    assert mask.checksum_sha256==prior['mask_sha256'] and hashes==prior['input_files']
    edges,geometry=batch.edge_index,batch.relative_geometry
    layout=model._receiver_layout(edges,num_nodes=batch.n_nodes);assert layout.edge_order is None
    h,_=model._encode_and_targets(batch.target_expression,mt,batch.node_covariates,None)
    rows=[];histograms=[];reconstruction=0.;direct_check=0.
    for layer,block in enumerate(model.blocks):
        q,k,v=block.project_nodes(h);outputs=[];us=[];fs=[];seen=0
        handles=[block.attention_output_projection.register_forward_hook(lambda mod,inp,out:us.append(out)),
                 block.ffn_output.register_forward_hook(lambda mod,inp,out:fs.append(out))]
        try:
            for start,stop in model._receiver_ranges(layout,num_nodes=batch.n_nodes):
                lo,hi=layout.receiver_ptr[start],layout.receiver_ptr[stop]
                source,recv=edges[0,lo:hi],edges[1,lo:hi]
                result=model._attention_partition(block,q[start:stop],k,v,h[start:stop],source,recv,
                                                  geometry[lo:hi],receiver_start=start)
                y,attention,_,_,_=result
                assert len(us)==len(outputs)+1 and len(fs)==len(outputs)+1
                reconstructed=(h[start:stop]+us[-1])+fs[-1]
                err=float((reconstructed-y).abs().max());reconstruction=max(reconstruction,err)
                assert err==0
                # Independently reconstruct weighted values for the first partition of each block.
                if not outputs:
                    aggregated=torch.zeros((stop-start,block.attention_heads,block.attention_head_dim))
                    aggregated.index_add_(0,recv-start,v.index_select(0,source)*attention.unsqueeze(-1))
                    independent=torch.nn.functional.linear(aggregated.flatten(1),block.attention_output_projection.weight)
                    e=float((independent-us[-1]).abs().max());direct_check=max(direct_check,e)
                    assert e==0
                outputs.append(y);seen+=hi-lo
        finally:
            for handle in handles:handle.remove()
        y=torch.cat(outputs);u=torch.cat(us);f=torch.cat(fs)
        assert seen==batch.n_edges and y.shape==h.shape==u.shape==f.shape
        m,ratios=vector_metrics(h.numpy(),u.numpy(),f.numpy(),y.numpy())
        rows.append(dict(core=core,block=layer+1,cells=batch.n_nodes,**m))
        histograms.append(np.array([np.histogram(x,RATIO_BINS)[0] for x in ratios.values()]))
        h=y
        print(json.dumps(dict(core=core,block=layer+1,seconds=time.time()-began)),flush=True)
    replay=None
    if pilot:
        targets=torch.linspace(0,batch.n_nodes-1,8).long();pred=model.decoder(h[targets])
        public=model(input_expression=batch.target_expression,gene_mask=mt,edge_index=edges,
                     relative_geometry=geometry,node_covariates=batch.node_covariates,target_nodes=targets)
        replay=float((pred-public.prediction).abs().max())
        assert torch.allclose(pred,public.prediction,atol=1e-6,rtol=1e-6)
    np.savez_compressed(OUT/f'core_{core}_histograms.npz',counts=np.array(histograms),
                        ratio_bins=RATIO_BINS,metrics=np.array(list(ratios)))
    sources=[Path(__file__),PREVIOUS/'analyze.py',*[PATHS.project_root/'src/spatial_benchmark'/n for n in
        ('geometry_modulated_relative_qkv_graph_transformer.py','relative_qkv_graph_transformer.py','models.py',
         'relative_qkv_post_training.py','so2_recurrent_hl_clustering.py')]]
    write(OUT/f'core_{core}.json',dict(run_id=RUN,checkpoint_sha256=SHA,core=core,cells=batch.n_nodes,
        edges=batch.n_edges,mask_seed=mask.seed,mask_sha256=mask.checksum_sha256,input_files=hashes,
        code_hashes={str(p.relative_to(PATHS.project_root)):sha256_file(p) for p in sources},
        histograms_sha256=sha256_file(OUT/f'core_{core}_histograms.npz'),runtime_seconds=time.time()-began,
        device='cpu',threads=torch.get_num_threads(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
        peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        reconstruction_max_abs_error=reconstruction,independent_attention_update_max_abs_error=direct_check,
        pilot_public_forward_max_abs_error=replay,rows=rows))

def summarize():
    receipts=[json.loads((OUT/f'core_{c}.json').read_text()) for c in range(15,29)]
    assert all(c['checkpoint_sha256']==SHA for c in receipts)
    assert sum(c['cells'] for c in receipts)==246063
    rows=[r for c in receipts for r in c['rows']];blocks=[]
    for block in range(1,5):
        sub=[r for r in rows if r['block']==block];assert len(sub)==14
        ms={k:float(np.mean([r['mean_squared_norms'][k] for r in sub])) for k in sub[0]['mean_squared_norms']}
        cms={k:float(np.mean([r['between_cell_mean_squared_norms'][k] for r in sub])) for k in sub[0]['mean_squared_norms']}
        result=dict(block=block,mean_squared_norms=ms,between_cell_mean_squared_norms=cms,
                    rms_attention_to_residual=float(np.sqrt(ms['attention']/ms['residual'])),
                    between_cell_rms_attention_to_residual=float(np.sqrt(cms['attention']/cms['residual'])),
                    rms_ffn_to_post_attention=float(np.sqrt(ms['ffn']/ms['post_attention'])),
                    mean_cosine=float(np.mean([r['cosine']['mean'] for r in sub])),
                    fraction_attention_larger=float(np.mean([r['fraction_attention_larger'] for r in sub])),
                    fraction_opposing=float(np.mean([r['fraction_opposing'] for r in sub])))
        for metric in sub[0]['distributions']:
            vals=[r['distributions'][metric] for r in sub]
            result[metric]=dict(equal_core_mean=float(np.mean([v['mean'] for v in vals])),
                core_mean_min=min(v['mean'] for v in vals),core_mean_max=max(v['mean'] for v in vals),
                mean_core_median=float(np.mean([v['median'] for v in vals])),
                mean_core_q10=float(np.mean([v['q10'] for v in vals])),
                mean_core_q90=float(np.mean([v['q90'] for v in vals])))
        blocks.append(result)
    summary=dict(run_id=RUN,checkpoint_sha256=SHA,total_cells=sum(c['cells'] for c in receipts),
                 total_edges=sum(c['edges'] for c in receipts),weighting='equal cell within core, equal core',
                 quantiles='exact within core; cross-core averages are not pooled quantiles',per_block=blocks)
    write(OUT/'summary.json',summary)
    flat=[]
    for r in rows:
        item={k:r[k] for k in ('core','block','cells','rms_attention_to_residual','between_cell_rms_attention_to_residual',
                              'rms_ffn_to_post_attention','fraction_attention_larger','fraction_opposing')}
        item['mean_cosine']=r['cosine']['mean']
        for metric,stats in r['distributions'].items():
            for k,v in stats.items():item[metric+'_'+k]=v
        flat.append(item)
    with (OUT/'per_core_block.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cores',nargs='+',type=int)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--pilot',action='store_true')
    p.add_argument('--summarize',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest');self_check()
    if args.summarize:summarize()
    else:
        model,config=load_model()
        for core in args.cores:
            assert 15<=core<=28 and not (OUT/f'core_{core}.json').exists()
            analyze_core(model,config,core,args.pilot)
