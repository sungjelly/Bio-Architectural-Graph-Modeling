"""Full-edge geometry-only modulation diagnostic; see README.md."""
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

PATHS = current_paths()
OUT = PATHS.report_root/'analyses/so2_geometry_modulation/v1'
PREVIOUS = PATHS.report_root/'analyses/so2_attention_tau_beta/v1'
RUN = 'r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf'
SHA = 'aba6a1d910e60f6a30bb4b27b15dbd1c30199da41a15460c23d994079b67543b'
BINS = np.linspace(0,3,301)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def load_model(device):
    path = PREVIOUS/'analyze.py'
    spec = importlib.util.spec_from_file_location('tau_beta_source',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_model(device)


@torch.no_grad()
def analyze(model, config, core, device):
    began = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    graph_root = PATHS.data_root/'processed/so2_14core_relative_qkv_graphs_v1'
    assert sha256_file(graph_root/'manifest.json') == config['dataset']['graph_manifest_file_sha256']
    manifest = json.loads((graph_root/'manifest.json').read_text())
    alias = f'SO2-C{core}'
    record = next(r for r in manifest['cores'] if r['alias']==alias)
    path = graph_root/'cores'/alias/'relative_geometry.npy'
    assert sha256_file(path) == record['files'][path.name]
    geometry = np.load(path, mmap_mode='r')
    edge_count = record['graph']['qc']['n_directed_edges']
    assert geometry.shape == (edge_count,70)
    selected = np.sort(np.random.default_rng(2026090500+core).choice(edge_count,4096,replace=False))
    rows, histograms = [], []
    for bi, block in enumerate(model.blocks):
        n = 0
        means = np.zeros((8,32),np.float64)
        m2 = np.zeros((8,32),np.float64)
        minima,maxima = np.full(8,np.inf),np.full(8,-np.inf)
        within,beyond = np.zeros(8,np.int64),np.zeros(8,np.int64)
        hist = np.zeros((8,300),np.int64)
        mean_error = 0.0
        for start in range(0,edge_count,32768):
            stop = min(edge_count,start+32768)
            x = torch.from_numpy(np.array(geometry[start:stop],copy=True)).to(device)
            g,_ = block.geometry_encoder(x)
            assert torch.isfinite(g).all() and (g>0).all()
            error = float((g.mean(-1)-1).abs().max())
            mean_error=max(mean_error,error)
            assert error < 5e-7
            var,mean = torch.var_mean(g,dim=0,correction=0)
            v,m = var.cpu().numpy().astype(np.float64),mean.cpu().numpy().astype(np.float64)
            chunk_n=stop-start
            delta=m-means
            m2 += v*chunk_n+delta*delta*n*chunk_n/(n+chunk_n)
            means += delta*chunk_n/(n+chunk_n)
            n+=chunk_n
            minima=np.minimum(minima,g.amin(dim=(0,2)).cpu().numpy())
            maxima=np.maximum(maxima,g.amax(dim=(0,2)).cpu().numpy())
            diff=(g-1).abs()
            within+=(diff<=0.1).sum(dim=(0,2)).cpu().numpy()
            beyond+=(diff>0.25).sum(dim=(0,2)).cpu().numpy()
            ids=selected[np.searchsorted(selected,start):np.searchsorted(selected,stop)]-start
            sample=g[torch.from_numpy(ids).to(device)].cpu().numpy()
            for head in range(8):
                h,_=np.histogram(sample[:,head,:],bins=BINS)
                hist[head]+=h
        assert n==edge_count and np.all(hist.sum(1)==4096*32)
        variance=m2/n
        assert np.all(variance>=0)
        for head in range(8):
            rows.append(dict(core=core,block=bi+1,head=head+1,edge_count=edge_count,
                dimension_means=means[head].tolist(),dimension_variances=variance[head].tolist(),
                total_ms=float(np.mean(variance[head]+(means[head]-1)**2)),
                min_g=float(minima[head]),max_g=float(maxima[head]),mean_g=float(means[head].mean()),
                fraction_within_10pct=float(within[head]/(n*32)),
                fraction_beyond_25pct=float(beyond[head]/(n*32)),mean_one_max_error=mean_error))
        histograms.append(hist)
        print(json.dumps({'core':core,'block':bi+1,'edges':n,'seconds':time.time()-began}),flush=True)
    source_paths=[Path(__file__),PREVIOUS/'analyze.py',
        PATHS.project_root/'src/spatial_benchmark/geometry_modulated_relative_qkv_graph_transformer.py']
    source_hashes={str(p.relative_to(PATHS.project_root)):sha256_file(p) for p in source_paths}
    np.savez_compressed(OUT/f'core_{core}_histogram.npz',counts=np.array(histograms),bin_edges=BINS)
    write(OUT/f'core_{core}.json',dict(run_id=RUN,checkpoint_sha256=SHA,core=core,
        cells=record['n_cells'],edges=edge_count,geometry_sha256=record['files'][path.name],
        source_code_sha256=source_hashes,precision='FP32; TF32 and AMP disabled',
        gpu=torch.cuda.get_device_name(device),device=str(device),torch_version=torch.__version__,
        cuda_version=torch.version.cuda,runtime_seconds=time.time()-began,
        peak_vram_gib=torch.cuda.max_memory_allocated(device)/2**30,
        histogram_file_sha256=sha256_file(OUT/f'core_{core}_histogram.npz'),
        histogram_edges_sampled=4096,histogram_seed=2026090500+core,
        histogram_edge_ids_sha256=__import__('hashlib').sha256(selected.tobytes()).hexdigest(),rows=rows))


def add_metrics(row):
    row['dynamic_ms']=row['within_core_ms']+row['between_core_ms']
    assert np.isclose(row['total_ms'],row['fixed_ms']+row['dynamic_ms'],atol=1e-12,rtol=1e-10)
    for name in ('total','fixed','dynamic'):
        row[name+'_rms']=float(np.sqrt(row[name+'_ms']))
    row['dynamic_fraction']=row['dynamic_ms']/row['total_ms'] if row['total_ms'] else 0.0
    return row


def summarize():
    receipts=[json.loads((OUT/f'core_{c}.json').read_text()) for c in range(15,29)]
    rows=[r for receipt in receipts for r in receipt['rows']]
    assert len(rows)==448
    assert all(r['checkpoint_sha256']==SHA for r in receipts)
    probabilities=[]
    for r in receipts:
        p=OUT/f'core_{r["core"]}_histogram.npz'
        assert sha256_file(p)==r['histogram_file_sha256']
        with np.load(p) as d:
            counts=d['counts']
            assert counts.shape==(4,8,300)
            probabilities.append(counts/counts.sum(-1,keepdims=True))
    np.savez_compressed(OUT/'distributions.npz',bin_edges=BINS,probability=np.mean(probabilities,axis=0))
    heads=[]
    for block in range(1,5):
        for head in range(1,9):
            sub=[r for r in rows if r['block']==block and r['head']==head]
            mu=np.array([r['dimension_means'] for r in sub])
            variance=np.array([r['dimension_variances'] for r in sub])
            heads.append(add_metrics(dict(block=block,head=head,total_ms=float(np.mean(variance+(mu-1)**2)),
                fixed_ms=float(np.mean((mu.mean(0)-1)**2)),within_core_ms=float(variance.mean()),
                between_core_ms=float(mu.var(0).mean()),mean_g=float(mu.mean()),
                min_g=min(r['min_g'] for r in sub),max_g=max(r['max_g'] for r in sub),
                fraction_within_10pct=float(np.mean([r['fraction_within_10pct'] for r in sub])),
                fraction_beyond_25pct=float(np.mean([r['fraction_beyond_25pct'] for r in sub])))))
    blocks=[]
    for block in range(1,5):
        sub=[r for r in heads if r['block']==block]
        row={k:float(np.mean([r[k] for r in sub])) for k in ('total_ms','fixed_ms','within_core_ms',
             'between_core_ms','mean_g','fraction_within_10pct','fraction_beyond_25pct')}
        row.update(block=block,min_g=min(r['min_g'] for r in sub),max_g=max(r['max_g'] for r in sub))
        blocks.append(add_metrics(row))
    for name,data in [('per_head',heads),('per_block',blocks)]:
        with (OUT/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(OUT/'summary.json',dict(run_id=RUN,checkpoint_sha256=SHA,
        total_cells=sum(r['cells'] for r in receipts),total_edges=sum(r['edges'] for r in receipts),
        weighting='equal core, equal edge within core, equal dimensions and heads',
        moment_scope='every edge',histogram_scope='4096 sampled edges per core, every dimension',
        per_head=heads,per_block=blocks))
    print(json.dumps(blocks,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cores',nargs='+',type=int)
    p.add_argument('--device',default='cuda:0');p.add_argument('--summarize',action='store_true')
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.summarize:
        summarize()
    else:
        device=torch.device(a.device);model,config=load_model(device)
        for core in a.cores:
            assert 15<=core<=28 and not (OUT/f'core_{core}.json').exists()
            analyze(model,config,core,device)
