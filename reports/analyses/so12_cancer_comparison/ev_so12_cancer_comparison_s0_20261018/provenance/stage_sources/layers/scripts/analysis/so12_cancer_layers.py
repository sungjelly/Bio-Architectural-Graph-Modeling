"""Audited h1-h3 extraction; existing h0/h4 bundles remain immutable references."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import so12_paper_test_extraction as base
import so12_paper_test_h0_extraction as h0
from run_so12_paper_test_h0_leiden import verify_published as verify_h0
from spatial_benchmark.paths import current_paths


def extract_layers(model, core, normalization, device, reference_h0, reference_h4):
    states, encoders = [], []
    original = model._chunked_layer
    previous = model.__dict__.get('_chunked_layer')
    had_override = '_chunked_layer' in model.__dict__
    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        states.append(value[0].detach().clone())
        return value
    hook = model.encoder.register_forward_hook(lambda mod, args, value: encoders.append(value.detach().clone()))
    model._chunked_layer = capture
    try:
        final, audit = base.extract_core_array(model, core, normalization, device, verify_invariance=True)
    finally:
        hook.remove()
        if had_override:
            model._chunked_layer = previous
        else:
            delattr(model, '_chunked_layer')
    base._require(len(states) == 8 and len(encoders) == 2, 'Incorrect layer capture count')
    base._require(all(torch.equal(states[i], states[i+4]) for i in range(4)), 'Layer states changed between forwards')
    base._require(torch.equal(encoders[0], encoders[1]), 'Encoder changed between forwards')
    base._require(np.array_equal(encoders[0].cpu().numpy(), reference_h0), 'Published h0 mismatch')
    base._require(np.array_equal(final, reference_h4), 'Published h4 mismatch')
    arrays = {f'h{i+1}': states[i].cpu().numpy().copy() for i in range(3)}
    base._require(all(a.shape == reference_h4.shape and a.dtype == np.float32 and np.isfinite(a).all()
                      and (np.linalg.norm(a, axis=1)>0).all() for a in arrays.values()), 'Invalid intermediate states')
    audit.update(published_h0_exact=True, published_h4_exact=True, intermediate_states_exact=True)
    return arrays, audit


def extract(output, device='cuda:0'):
    paths = current_paths()
    output = Path(output).resolve()
    base._require(output.is_relative_to(paths.scratch_root.resolve()/'active_runs'), 'Active output required')
    base._require(shutil.disk_usage(output).free > 5.3 * 2**30, 'Insufficient disk reserve')
    started = time.monotonic()
    source = h0.verify_source(paths.artifact_root/'runs/2026/10'/base.SOURCE_RUN_ID, paths)
    ref4_path = paths.report_root/'analyses/so12_paper_test_leiden'/h0.REFERENCE_EVALUATION_ID
    ref4, cells, h4_array = h0.verify_reference(ref4_path, source, paths)
    ref0_path = paths.report_root/'analyses/so12_paper_test_h0_leiden/ev_so12_paper_test_h0_s0_20261018'
    verify_h0(ref0_path, base._read(ref0_path/'config.resolved.json'))
    base._require(base.file_hash(ref0_path/'cells.csv.gz') == ref4['cells_sha256'], 'Reference cell order differs')
    h0_array = np.load(ref0_path/'embeddings/h0.npy', mmap_mode='r', allow_pickle=False)
    base._write(output/'provenance/source_verified.json', source)
    base._write(output/'provenance/references.json', {'h4': ref4, 'h0': {'path':str(ref0_path),
        'manifest_sha256':base.file_hash(ref0_path/'manifest.json')}})
    execution = base.configure_deterministic_inference()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    target = torch.device(device)
    if target.type == 'cuda':
        torch.cuda.set_device(target)
        torch.cuda.reset_peak_memory_stats(target)
    data = base.load_so12_paper_data(source['config'], paths, roles=('test',), verify_checksums=True, load_graphs=True)
    new_cells = base.prepare_cells(data, output, paths)
    base._require(new_cells.equals(cells), 'Extracted cell order changed')
    model = base.build_model(source['config'], genes=len(data.genes), covariates=data.cores[0].node_covariates.shape[1]).to(target).float().eval()
    checkpoint = Path(source['source_run_dir'])/source['checkpoint_relative_path']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state_dict'], strict=True)
    del state
    normalization = base.normalization_tensors(data, target)
    (output/'embeddings').mkdir(exist_ok=True)
    matrices = {}
    for name in ('h1','h2','h3'):
        path = output/f'embeddings/{name}.npy'
        base._require(not path.exists(), f'Refusing overwrite {name}')
        matrices[name] = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(81607,256))
    audits = []
    for core in sorted(data.cores, key=lambda c:-c.n_cells):
        selected = cells.loc[cells.alias==core.alias, 'row_index'].to_numpy()
        start, stop = int(selected[0]),int(selected[-1])+1
        arrays, audit = extract_layers(model,core,normalization,target,h0_array[start:stop],h4_array[start:stop])
        for name, value in arrays.items():
            matrices[name][start:stop] = value
            matrices[name].flush()
        receipt = {'core':core.alias,'rows':[start,stop],'n_cells':core.n_cells,'audit':audit}
        base._write(output/f'embeddings/{core.alias}_layers_receipt.json',receipt)
        audits.append(receipt)
        print(json.dumps(receipt), flush=True)
    matrices.clear()
    base._require(base.file_hash(checkpoint)==source['checkpoint_sha256'] and
        base.training_source_binding()==source['binding']['source_sha256'], 'Frozen source changed')
    result = {'status':'complete','passed':True,'source_run_id':base.SOURCE_RUN_ID,'checkpoint_epoch':194,
        'checkpoint_sha256':source['checkpoint_sha256'],'n_cells':81607,'test_core_numbers':[15,23],
        'dimensions':256,'cells_sha256':base.file_hash(output/'cells.csv.gz'),'execution':execution,
        'fitted_parameters_sha256':source['fitted_parameters_sha256'],'normalization_refit':False,
        'elapsed_seconds':time.monotonic()-started,'cores':audits,
        'embeddings':{name:{'path':f'embeddings/{name}.npy','sha256':base.file_hash(output/f'embeddings/{name}.npy')}
                      for name in ('h1','h2','h3')},
        'references':{'h0':str(ref0_path),'h4':str(ref4_path)}}
    base._write(output/'layers_manifest.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(extract(args.output,args.device)),flush=True)
