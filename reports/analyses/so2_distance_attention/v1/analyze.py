"""Full-graph descriptive distance-attention diagnostic; see README.md."""
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
from spatial_benchmark.paths import current_paths
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.relative_qkv_post_training import fixed_inference_mask
from spatial_benchmark.so2_recurrent_hl_clustering import _load_core_batch

PATHS=current_paths()
OUT=PATHS.report_root/'analyses/so2_distance_attention/v1'
PREVIOUS=PATHS.report_root/'analyses/so2_attention_tau_beta/v1'
RUN='r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf'
SHA='aba6a1d910e60f6a30bb4b27b15dbd1c30199da41a15460c23d994079b67543b'
BINS=np.linspace(0,500,51)
CHANNELS=('attention','degree_scaled_attention','inverse_square_attention',
          'degree_scaled_inverse_square','score','content','beta')

def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def load_model():
    spec=importlib.util.spec_from_file_location('tau_beta_source',PREVIOUS/'analyze.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.load_model(torch.device('cpu'))

def centered_moments(distance,score,receiver,lower,upper):
    use=(distance>=lower)&(distance<=upper)
    r,s,recv=distance[use],score[use],receiver[use]
    if len(r)==0:
        return np.zeros((8,5)),0
    _,start,degree=np.unique(recv,return_index=True,return_counts=True)
    log=np.log(r)
    x=log-np.repeat(np.add.reduceat(log,start)/degree,degree)
    z=r-np.repeat(np.add.reduceat(r,start)/degree,degree)
    y=s-np.repeat(np.add.reduceat(s,start,axis=0)/degree[:,None],degree,axis=0)
    weights=1/np.repeat(degree,degree)
    # Columns: E[x²], E[z²], E[y²], E[xy], E[zy], summed per receiver.
    moments=np.stack([np.full(8,np.sum(weights*x*x)),np.full(8,np.sum(weights*z*z)),
        np.sum(weights[:,None]*y*y,axis=0),np.sum((weights*x)[:,None]*y,axis=0),
        np.sum((weights*z)[:,None]*y,axis=0)],axis=1)
    return moments,len(degree)

def fit_moments(m):
    xx,zz,yy,xy,zy=map(float,m)
    slope=xy/xx
    rate=zy/zz
    residual=yy-xy*xy/xx
    exp_residual=yy-zy*zy/zz
    return dict(power_exponent=-slope,power_r2=xy*xy/(xx*yy),
        exponential_rate_per_um=-rate,exponential_length_um=(-1/rate if rate<0 else None),
        exponential_r2=zy*zy/(zz*yy),flat_mse=yy,power_mse=max(0.,residual),
        inverse_square_mse=yy+4*xx+4*xy,exponential_mse=max(0.,exp_residual),
        inverse_square_r2=1-(yy+4*xx+4*xy)/yy)

def self_check():
    r=np.array([10.,20.,40.,30.,60.,120.]);recv=np.array([0,0,0,1,1,1])
    s=np.repeat((-2*np.log(r)+np.repeat([13.,-7.],3))[:,None],8,axis=1)
    m,n=centered_moments(r,s,recv,0,500)
    for row in m/n:
        fit=fit_moments(row)
        assert abs(fit['power_exponent']-2)<1e-12 and abs(fit['power_r2']-1)<1e-12
    s=np.repeat((-r/80+np.repeat([13.,-7.],3))[:,None],8,axis=1)
    m,n=centered_moments(r,s,recv,0,500)
    for row in m/n:
        fit=fit_moments(row)
        assert abs(fit['exponential_length_um']-80)<1e-10
        assert abs(fit['exponential_r2']-1)<1e-12

@torch.no_grad()
def analyze_core(model,config,core,pilot):
    began=time.time();alias=f'SO2-C{core}'
    cohort=PATHS.data_root/'processed/so2_14core_relative_qkv_v1'
    graphs=PATHS.data_root/'processed/so2_14core_relative_qkv_graphs_v1'
    assert sha256_file(cohort/'manifest.json')==config['dataset']['cohort_manifest_file_sha256']
    assert sha256_file(graphs/'manifest.json')==config['dataset']['graph_manifest_file_sha256']
    cm=json.loads((cohort/'manifest.json').read_text());gm=json.loads((graphs/'manifest.json').read_text())
    cr=next(r for r in cm['cores'] if r['alias']==alias);gr=next(r for r in gm['cores'] if r['alias']==alias)
    record=dict(cr,directed_edges=gr['graph']['qc']['n_directed_edges'],
        receiver_major_canonical_order=gr['graph']['qc']['receiver_major_canonical_order'])
    inputs=[cohort/'cores'/f'{alias}.npz',graphs/'cores'/alias/'edge_index.npy',graphs/'cores'/alias/'relative_geometry.npy']
    input_hashes={str(p.relative_to(PATHS.project_root)):sha256_file(p) for p in inputs}
    for p in inputs:
        expected=cm['files'][str(p.relative_to(cohort))] if p.is_relative_to(cohort) else gr['files'][p.name]
        assert input_hashes[str(p.relative_to(PATHS.project_root))]==expected
    batch,coords=_load_core_batch(cohort_dir=cohort,graph_dir=graphs,record=record)
    mask=fixed_inference_mask(batch);mask_tensor=torch.from_numpy(mask.mask.copy())
    edges,geometry=batch.edge_index,batch.relative_geometry
    layout=model._receiver_layout(edges,num_nodes=batch.n_nodes)
    assert layout.edge_order is None
    h,_=model._encode_and_targets(batch.target_expression,mask_tensor,batch.node_covariates,None)
    all_stats=[];all_bins=[];rows=[];max_distance_error=0.;max_softmax_error=0.
    for layer,block in enumerate(model.blocks):
        q,k,v=block.project_nodes(h);outputs=[]
        sums=np.zeros((7,50,8));counts=np.zeros(50,np.int64)
        moments=np.zeros((2,8,5));receiver_counts=np.zeros(2,np.int64)
        ref_moments=np.zeros((2,8,5));seen=0
        for start,stop in model._receiver_ranges(layout,num_nodes=batch.n_nodes):
            lo,hi=layout.receiver_ptr[start],layout.receiver_ptr[stop]
            source,recv=edges[0,lo:hi],edges[1,lo:hi]
            geo=geometry[lo:hi]
            result=model._attention_partition(block,q[start:stop],k,v,h[start:stop],source,recv,geo,receiver_start=start)
            output,attention,content,beta,score=result;outputs.append(output)
            a=attention.numpy().astype(np.float64);c=content.numpy().astype(np.float64)
            b=beta.numpy().astype(np.float64);s=score.numpy().astype(np.float64)
            assert np.isfinite(s).all() and np.isfinite(a).all() and np.all(a>0)
            assert np.max(np.abs(s-(c+b)))<3e-7
            receiver=recv.numpy();sender=source.numpy()
            r=np.linalg.norm(coords[sender]-coords[receiver],axis=1)
            assert np.all(r>0) and np.max(r)<=500+1e-6
            distance_error=float(np.max(np.abs(r-geo[:,-1].numpy().astype(np.float64)*500)))
            assert distance_error<4e-5
            max_distance_error=max(max_distance_error,distance_error)
            _,offset,degree=np.unique(receiver,return_index=True,return_counts=True)
            n=np.repeat(degree,degree)
            softmax_error=float(np.max(np.abs(np.add.reduceat(a,offset,axis=0)-1)))
            assert softmax_error<2e-6
            max_softmax_error=max(max_softmax_error,softmax_error)
            raw=r**-2
            ref=raw/np.repeat(np.add.reduceat(raw,offset),degree)
            assert np.max(np.abs(np.add.reduceat(ref,offset)-1))<1e-12
            ref_score=np.repeat(np.log(ref)[:,None],8,axis=1)
            for fit_index,(lower,upper) in enumerate(((10.,450.),(0.,500.000001))):
                m,nrec=centered_moments(r,s,receiver,lower,upper)
                rm,rn=centered_moments(r,ref_score,receiver,lower,upper)
                assert nrec==rn
                moments[fit_index]+=m;ref_moments[fit_index]+=rm;receiver_counts[fit_index]+=nrec
            bid=np.minimum((r/10).astype(int),49)
            counts+=np.bincount(bid,minlength=50)
            vals=(a,a*n[:,None],np.repeat(ref[:,None],8,axis=1),
                np.repeat((ref*n)[:,None],8,axis=1),s,c,b)
            for ci,value in enumerate(vals):
                for head in range(8):
                    sums[ci,:,head]+=np.bincount(bid,weights=value[:,head],minlength=50)
            seen+=len(r)
        h=torch.cat(outputs,0)
        assert seen==batch.n_edges and counts.sum()==seen
        all_bins.append(sums);all_stats.append(counts)
        for fi,label in enumerate(('10_450_um','all_edges')):
            normalized=moments[fi]/receiver_counts[fi]
            control=ref_moments[fi]/receiver_counts[fi]
            for head in range(8):
                f=fit_moments(normalized[head]);cf=fit_moments(control[head])
                assert abs(cf['power_exponent']-2)<1e-10 and abs(cf['power_r2']-1)<1e-10
                rows.append(dict(core=core,block=layer+1,head=head+1,range=label,
                    receivers=int(receiver_counts[fi]),moments=normalized[head].tolist(),
                    reference_exponent=cf['power_exponent'],reference_r2=cf['power_r2'],**f))
        print(json.dumps({'core':core,'block':layer+1,'seconds':time.time()-began}),flush=True)
    replay=None
    if pilot:
        targets=torch.linspace(0,batch.n_nodes-1,8).long()
        predicted=model.decoder(h[targets])
        public=model(input_expression=batch.target_expression,gene_mask=mask_tensor,
            edge_index=edges,relative_geometry=geometry,node_covariates=batch.node_covariates,target_nodes=targets)
        replay=float((predicted-public.prediction).abs().max())
        assert torch.allclose(predicted,public.prediction,atol=1e-6,rtol=1e-6)
    np.savez_compressed(OUT/f'core_{core}_bins.npz',sums=np.array(all_bins),counts=np.array(all_stats),bin_edges=BINS)
    source_files=[Path(__file__),PREVIOUS/'analyze.py',*[PATHS.project_root/'src/spatial_benchmark'/n for n in
        ('geometry_modulated_relative_qkv_graph_transformer.py','relative_qkv_graph_transformer.py','models.py',
         'relative_qkv_post_training.py','so2_recurrent_hl_clustering.py')]]
    write(OUT/f'core_{core}.json',dict(run_id=RUN,checkpoint_sha256=SHA,core=core,
        cells=batch.n_nodes,edges=batch.n_edges,mask_seed=mask.seed,mask_sha256=mask.checksum_sha256,
        input_files=input_hashes,code_hashes={str(p.relative_to(PATHS.project_root)):sha256_file(p) for p in source_files},
        bins_sha256=sha256_file(OUT/f'core_{core}_bins.npz'),runtime_seconds=time.time()-began,
        device='cpu',threads=torch.get_num_threads(),torch_version=torch.__version__,
        distance_cache_max_error_um=max_distance_error,attention_sum_max_error=max_softmax_error,
        pilot_public_forward_max_abs_error=replay,rows=rows))

def summarize():
    receipts=[json.loads((OUT/f'core_{c}.json').read_text()) for c in range(15,29)]
    rows=[r for c in receipts for r in c['rows']]
    assert len(rows)==896 and all(c['checkpoint_sha256']==SHA for c in receipts)
    curves=[]
    for c in receipts:
        p=OUT/f'core_{c["core"]}_bins.npz';assert sha256_file(p)==c['bins_sha256']
        with np.load(p) as d:
            counts=d['counts'];sums=d['sums']
            assert np.all(counts.sum(-1)==c['edges'])
            curve=np.divide(sums,counts[:,None,:,None],out=np.full_like(sums,np.nan),where=counts[:,None,:,None]>0)
            curves.append(curve)
    curves=np.array(curves)
    means=np.nanmean(curves,axis=0);lo=np.nanmin(curves,axis=0);hi=np.nanmax(curves,axis=0)
    np.savez_compressed(OUT/'curves.npz',bin_edges=BINS,mean=means,core_min=lo,core_max=hi,per_core=curves,channels=np.array(CHANNELS))
    heads=[];blocks=[]
    for label in ('10_450_um','all_edges'):
        for block in range(1,5):
            for head in range(1,9):
                sub=[r for r in rows if r['block']==block and r['head']==head and r['range']==label]
                m=np.mean([r['moments'] for r in sub],axis=0)
                heads.append(dict(block=block,head=head,range=label,moments=m.tolist(),
                    core_exponent_min=min(r['power_exponent'] for r in sub),core_exponent_max=max(r['power_exponent'] for r in sub),
                    **fit_moments(m)))
            sub=[r for r in heads if r['block']==block and r['range']==label]
            m=np.mean([r['moments'] for r in sub],axis=0)
            blocks.append(dict(block=block,range=label,moments=m.tolist(),**fit_moments(m)))
    for name,data in [('per_core_head',rows),('per_head',heads),('per_block',blocks)]:
        cleaned=[{k:v for k,v in r.items() if k!='moments'} for r in data]
        with (OUT/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(cleaned[0]));w.writeheader();w.writerows(cleaned)
    write(OUT/'summary.json',dict(run_id=RUN,checkpoint_sha256=SHA,
        total_cells=sum(c['cells'] for c in receipts),total_edges=sum(c['edges'] for c in receipts),
        primary_fit_range_um=[10,450],mask='same fixed per-cell mask as tau/beta analysis',
        regression_weighting='equal edge within receiver, equal receiver within core, equal core',
        curve_weighting='equal edge within each core/bin, equal core',per_head=heads,per_block=blocks))
    print(json.dumps(blocks,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cores',nargs='+',type=int)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--pilot',action='store_true');p.add_argument('--summarize',action='store_true')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads);torch.set_float32_matmul_precision('highest');self_check()
    if args.summarize:summarize()
    else:
        model,config=load_model()
        for core in args.cores:
            assert 15<=core<=28 and not (OUT/f'core_{core}.json').exists()
            analyze_core(model,config,core,args.pilot)
