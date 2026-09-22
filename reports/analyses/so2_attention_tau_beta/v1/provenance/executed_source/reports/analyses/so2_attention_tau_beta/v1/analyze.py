"""Descriptive, immutable-source attention scale inspection; see README.md."""
import argparse
import csv
import gc
import hashlib
import inspect
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from spatial_benchmark.paths import current_paths
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.geometry_modulated_relative_qkv_graph_transformer import (
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer as Model,
)
from spatial_benchmark.relative_qkv_post_training import fixed_inference_mask
from spatial_benchmark.so2_recurrent_hl_clustering import _load_core_batch

RUN = 'r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf'
SHA = 'aba6a1d910e60f6a30bb4b27b15dbd1c30199da41a15460c23d994079b67543b'
PATHS = current_paths()
OUT = PATHS.report_root / 'analyses/so2_attention_tau_beta/v1'
BUNDLE = PATHS.artifact_root / 'runs/2026/09' / RUN


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_model(device):
    checkpoint = BUNDLE / 'checkpoints/last.ckpt'
    assert sha256_file(checkpoint) == SHA
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert payload['run_id'] == RUN and payload['completed_global_epochs'] == 200
    args = payload['model_construction']
    assert args['class'] == Model.__name__
    source_file = PATHS.project_root/'src/spatial_benchmark/geometry_modulated_relative_qkv_graph_transformer.py'
    untracked = json.loads((BUNDLE/'provenance/untracked_files.json').read_text())
    entries = untracked if isinstance(untracked,list) else untracked['files']
    original = next(r for r in entries if r['path']==str(source_file.relative_to(PATHS.project_root)))
    assert sha256_file(source_file) == original['sha256'], 'Attention implementation differs from training'
    model = Model(**{k: args[k] for k in inspect.signature(Model).parameters})
    model.load_state_dict(payload['model_state_dict'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 5134088
    for key, value in model.state_dict().items():
        assert torch.equal(value, payload['model_state_dict'][key])
    return model.to(device).eval(), payload['resolved_config']


@torch.no_grad()
def analyze_core(model, config, core, device, pilot):
    started = time.time()
    code_files = [Path(__file__), *[PATHS.project_root/'src/spatial_benchmark'/name for name in (
        'geometry_modulated_relative_qkv_graph_transformer.py','relative_qkv_graph_transformer.py',
        'models.py','relative_qkv_post_training.py','so2_recurrent_hl_clustering.py')]]
    code_hashes = {str(p.relative_to(PATHS.project_root)):sha256_file(p) for p in code_files}
    alias = f'SO2-C{core}'
    cohort = PATHS.data_root / 'processed/so2_14core_relative_qkv_v1'
    graphs = PATHS.data_root / 'processed/so2_14core_relative_qkv_graphs_v1'
    assert sha256_file(cohort/'manifest.json') == config['dataset']['cohort_manifest_file_sha256']
    assert sha256_file(graphs/'manifest.json') == config['dataset']['graph_manifest_file_sha256']
    cm = json.loads((cohort/'manifest.json').read_text())
    gm = json.loads((graphs/'manifest.json').read_text())
    cr = next(c for c in cm['cores'] if c['alias'] == alias)
    gr = next(c for c in gm['cores'] if c['alias'] == alias)
    record = dict(cr, directed_edges=gr['graph']['qc']['n_directed_edges'],
                  receiver_major_canonical_order=gr['graph']['qc']['receiver_major_canonical_order'])
    batch, _ = _load_core_batch(cohort_dir=cohort, graph_dir=graphs, record=record)
    source_files = [cohort/'cores'/f'{alias}.npz', graphs/'cores'/alias/'edge_index.npy',
                    graphs/'cores'/alias/'relative_geometry.npy']
    sources = {str(p.relative_to(PATHS.project_root)): sha256_file(p) for p in source_files}
    for path in source_files:
        expected = (cm['files'][str(path.relative_to(cohort))]
                    if path.is_relative_to(cohort) else gr['files'][path.name])
        assert sources[str(path.relative_to(PATHS.project_root))] == expected
    mask = fixed_inference_mask(batch)
    expression = batch.target_expression.to(device)
    covariates = batch.node_covariates.to(device)
    device_mask = torch.from_numpy(mask.mask.copy()).to(device)
    edges, geometry = batch.edge_index, batch.relative_geometry
    layout = model._receiver_layout(edges, num_nodes=batch.n_nodes)
    assert layout.edge_order is None
    h, _ = model._encode_and_targets(expression, device_mask, covariates, None)
    torch.cuda.reset_peak_memory_stats(device)
    rows = []
    for layer, block in enumerate(model.blocks):
        q, k, v = block.project_nodes(h)
        outputs = []
        sums = np.zeros((8, 8), dtype=np.float64)
        minima = np.full(8, np.inf)
        maxima = np.full(8, -np.inf)
        count = 0
        max_identity_error = 0.0
        shift_checked = False
        for start, stop in model._receiver_ranges(layout, num_nodes=batch.n_nodes):
            lo, hi = layout.receiver_ptr[start], layout.receiver_ptr[stop]
            source = edges[0,lo:hi].to(device)
            receiver = edges[1,lo:hi].to(device)
            geo = geometry[lo:hi].to(device=device, dtype=h.dtype)
            result = model._attention_partition(block, q[start:stop], k, v,
                h[start:stop], source, receiver, geo, receiver_start=start)
            output, attention, content, beta, combined = result
            outputs.append(output)
            assert torch.isfinite(combined).all() and torch.isfinite(attention).all()
            error = (content + beta - combined).abs().max().item()
            max_identity_error = max(max_identity_error, error)
            assert error == 0
            receiver_np = edges[1,lo:hi].numpy()
            _, offsets, degree = np.unique(receiver_np, return_index=True, return_counts=True)
            c = content.cpu().numpy().astype(np.float64)
            b = beta.cpu().numpy().astype(np.float64)
            cmean = np.add.reduceat(c, offsets, axis=0) / degree[:,None]
            bmean = np.add.reduceat(b, offsets, axis=0) / degree[:,None]
            cc = c - np.repeat(cmean, degree, axis=0)
            bb = b - np.repeat(bmean, degree, axis=0)
            sums += np.array([b.sum(0), (b*b).sum(0), np.abs(b).sum(0),
                (c*c).sum(0), (bb*bb).sum(0), (cc*cc).sum(0),
                (bb*cc).sum(0), (np.abs(b)>0.95).sum(0)])
            minima = np.minimum(minima, b.min(0))
            maxima = np.maximum(maxima, b.max(0))
            count += len(b)
            if not shift_checked:
                d = int(degree[0])
                centered_logits = torch.from_numpy((cc[:d]+bb[:d]).astype(np.float32)).to(device)
                assert torch.allclose(torch.softmax(centered_logits, dim=0), attention[:d], atol=2e-6, rtol=2e-5)
                shift_checked = True
        h = torch.cat(outputs, dim=0)
        assert count == batch.n_edges
        m = sums / count
        tau = block.attention_logit_scale().cpu().numpy()
        for head in range(8):
            rows.append(dict(core=core, block=layer+1, head=head+1, edges=count,
                tau=float(tau[head]), beta_mean=float(m[0,head]), beta_ms=float(m[1,head]),
                beta_mean_abs=float(m[2,head]), content_ms=float(m[3,head]),
                beta_centered_ms=float(m[4,head]), content_centered_ms=float(m[5,head]),
                centered_cross_moment=float(m[6,head]), beta_saturation_fraction=float(m[7,head]),
                beta_min=float(minima[head]), beta_max=float(maxima[head])))
        print(json.dumps({'core':core,'block':layer+1,'edges':count,'elapsed_s':time.time()-started}), flush=True)
    replay_error = None
    if pilot:
        targets = torch.linspace(0,batch.n_nodes-1,8,device=device).long()
        predicted = model.decoder(h[targets])
        public = model(input_expression=expression, gene_mask=device_mask,
            edge_index=edges, relative_geometry=geometry, node_covariates=covariates,
            target_nodes=targets)
        replay_error = float((predicted-public.prediction).abs().max())
        assert torch.allclose(predicted, public.prediction, atol=1e-6, rtol=1e-6)
    receipt = dict(run_id=RUN, checkpoint_sha256=SHA, core=core, cells=batch.n_nodes,
        edges=batch.n_edges, mask_seed=mask.seed, mask_sha256=mask.checksum_sha256,
        input_files=sources, input_core_record=cr, input_graph_record=gr,
        precision='FP32, AMP disabled', device=str(device),
        source_code_sha256=code_hashes, tf32_enabled=torch.backends.cuda.matmul.allow_tf32,
        gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, runtime_seconds=time.time()-started,
        peak_vram_gib=torch.cuda.max_memory_allocated(device)/2**30,
        pilot_public_forward_max_abs_error=replay_error,
        score_identity_verified=True, softmax_shift_invariance_verified=True, rows=rows)
    write_json(OUT/f'core_{core}.json',receipt)
    del batch, h, q, k, v, outputs, result
    gc.collect()
    torch.cuda.empty_cache()


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    receipts = [json.loads((OUT/f'core_{c}.json').read_text()) for c in range(15,29)]
    assert all(r['checkpoint_sha256']==SHA for r in receipts)
    assert sum(r['cells'] for r in receipts) == 246063
    all_rows = [row for receipt in receipts for row in receipt['rows']]
    assert len(all_rows)==448
    with (OUT/'per_core_head.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    aggregates=[]
    for block in range(1,5):
        for head in range(1,9):
            rows=[r for r in all_rows if r['block']==block and r['head']==head]
            means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('core','block','head','edges')}
            out=dict(block=block,head=head,**means)
            for key in ('beta','content','beta_centered','content_centered'):
                out[key+'_rms']=float(np.sqrt(out[key+'_ms']))
            out['beta_to_tau_rms_ratio']=out['beta_rms']/out['tau']
            out['centered_beta_to_content_ratio']=out['beta_centered_rms']/out['content_centered_rms']
            out['centered_channel_correlation']=out['centered_cross_moment']/(out['beta_centered_rms']*out['content_centered_rms'])
            out['beta_min']=min(r['beta_min'] for r in rows)
            out['beta_max']=max(r['beta_max'] for r in rows)
            aggregates.append(out)
    with (OUT/'per_head.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregates[0])); w.writeheader(); w.writerows(aggregates)
    summary=dict(run_id=RUN,checkpoint_sha256=SHA,weighting='equal core; equal edge within each core',
        mask='one fixed uniform per-cell masking realization per core',
        total_cells=sum(r['cells'] for r in receipts),total_edges=sum(r['edges'] for r in receipts),
        per_head=aggregates,per_block=[])
    for block in range(1,5):
        rows=[r for r in aggregates if r['block']==block]
        row=dict(block=block,tau_mean=float(np.mean([r['tau'] for r in rows])))
        for key in ('beta','content','beta_centered','content_centered'):
            row[key+'_rms']=float(np.sqrt(np.mean([r[key+'_ms'] for r in rows])))
        row['centered_beta_to_content_ratio']=row['beta_centered_rms']/row['content_centered_rms']
        summary['per_block'].append(row)
    write_json(OUT/'summary.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    panels=[('tau','Learned scale τ','Blues',1.85,2.0),
            ('beta_rms','Geometry bias β: RMS','YlOrBr',0,1),
            ('centered_beta_to_content_ratio','Neighbor-varying β / content RMS','viridis',0,None)]
    for ax,(key,title,cmap,vmin,vmax) in zip(axs,panels):
        a=np.array([r[key] for r in aggregates]).reshape(4,8)
        im=ax.imshow(a,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
        ax.set_xticks(range(8),range(1,9));ax.set_yticks(range(4),[f'Block {i}' for i in range(1,5)])
        ax.set_xlabel('Attention head');ax.set_title(title,pad=12)
        for i in range(4):
            for j in range(8):
                norm=im.norm(a[i,j]);color='white' if norm>0.65 and cmap!='YlOrBr' else '#202020'
                ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',color=color,fontsize=9)
        fig.colorbar(im,ax=ax,shrink=0.75)
    fig.suptitle('Latest trained model: attention scale and geometry bias',fontsize=16)
    fig.supxlabel('200 epochs · seed 0 · all 14 full-core graphs · FP32 · fixed masking\n'
        'Right: subtract each receiver/head’s mean before comparing score variation; equal-core aggregation.',fontsize=10)
    fig.savefig(OUT/'tau_beta.png',dpi=180)
    fig.savefig(OUT/'tau_beta.pdf')
    plt.close(fig)
    print(json.dumps(summary['per_block'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cores',nargs='+',type=int)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True)
    if args.summarize:
        summarize()
    else:
        model,config=load_model(torch.device(args.device))
        for core in args.cores:
            assert not (OUT/f'core_{core}.json').exists(), 'Refusing to overwrite completed receipt'
            analyze_core(model,config,core,torch.device(args.device),args.pilot)
