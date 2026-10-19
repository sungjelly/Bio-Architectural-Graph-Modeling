"""CPU representations and frozen-graph Leiden analysis of fully observed SO12 cells.

No GPU, source dataset, or active model is modified. This is exploratory,
transductive clustering, not held-out biological validation.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from spatial_benchmark.paths import current_paths
from spatial_benchmark.identifiers import canonical_sha256


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def event(out, stage, **values):
    row = {'time': time.time(), 'stage': stage, **values}
    with (out/'analysis_metrics.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row), flush=True)


def kernel_fit_transform(z, train, cores, config):
    """Centered RBF kernel PCA on core-balanced training landmarks, then extension."""
    rng = np.random.default_rng(config.get('seed', 0))
    groups = np.unique(cores[train])
    target = min(config.get('kernel_landmarks', 512), int(train.sum()))
    chosen = []
    for group in groups:
        rows = np.flatnonzero(train & (cores == group))
        take = min(len(rows), target//len(groups))
        chosen.extend(rng.choice(rows, take, replace=False).tolist())
    remaining = np.setdiff1d(np.flatnonzero(train), chosen)
    if len(chosen) < target:
        chosen.extend(rng.choice(remaining, target-len(chosen), replace=False).tolist())
    landmark_indices = np.array(chosen, dtype=np.int64)
    landmarks = z[landmark_indices].astype('float64')
    d2 = cdist(landmarks, landmarks, 'sqeuclidean')
    pair_distances = d2[np.triu_indices(target, 1)]
    bandwidth = float(np.median(pair_distances[pair_distances > 0]))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Degenerate RBF bandwidth')
    gamma = 1.0/bandwidth
    k = np.exp(-gamma*d2)
    col_mean, grand_mean = k.mean(0), float(k.mean())
    kc = k-col_mean[None, :]-col_mean[:, None]+grand_mean
    eig, vec = np.linalg.eigh(kc)
    take = np.argsort(eig)[::-1][:config.get('n_components', 32)]
    values, vectors = eig[take], vec[:, take]
    if np.any(values <= 1e-10):
        raise ValueError('Insufficient positive kernel eigenvalues')
    projection = vectors / np.sqrt(values)[None, :]
    emb = np.empty((len(z), len(values)), dtype=np.float32)
    for start in range(0, len(z), 4096):
        chunk = np.exp(-gamma*cdist(z[start:start+4096], landmarks, 'sqeuclidean'))
        chunk -= chunk.mean(1)[:, None]
        chunk -= col_mean[None, :]
        chunk += grand_mean
        emb[start:start+4096] = chunk @ projection
    params = dict(landmark_indices=landmark_indices, landmarks=landmarks,
                  gamma=np.array(gamma), kernel_column_mean=col_mean,
                  kernel_grand_mean=np.array(grand_mean), projection=projection,
                  eigenvalues=values)
    details = {'landmarks': target, 'gamma': gamma,
               'landmark_variance_retained': float(values.sum()/eig[eig>0].sum()),
               'landmarks_train_only': bool(train[landmark_indices].all()),
               'method': 'centered_RBF_kernel_PCA_training_landmark_extension'}
    return emb, details, params


def spatial_neighbors(xy, cores, config):
    """Directed Gaussian kNN within each core and a hard physical distance limit."""
    k = config.get('spatial_k', 15)
    radius = config.get('spatial_radius_um', 50.0)
    idx = np.full((len(xy), k), -1, dtype=np.int32)
    distances = np.full((len(xy), k), np.inf, dtype=np.float32)
    for core in np.unique(cores):
        rows = np.flatnonzero(cores == core)
        kk = min(k+1, len(rows))
        d, j = cKDTree(xy[rows]).query(xy[rows], k=kk, workers=1)
        d, j = np.asarray(d).reshape(len(rows), kk), np.asarray(j).reshape(len(rows), kk)
        # Explicit self-removal also handles zero-distance coordinate duplicates.
        valid = (j != np.arange(len(rows))[:, None]) & (d <= radius)
        order = np.argsort(~valid, axis=1, kind='stable')[:, :min(k, kk)]
        di, ji = np.take_along_axis(d, order, 1), np.take_along_axis(j, order, 1)
        vi = np.take_along_axis(valid, order, 1)
        idx[rows, :order.shape[1]] = np.where(vi, rows[ji], -1)
        distances[rows, :order.shape[1]] = np.where(vi, di, np.inf)
    valid = idx >= 0
    sigma = np.max(np.where(valid, distances, 0), axis=1)
    sigma = np.maximum(sigma, 1e-6)
    weights = np.exp(-np.square(np.where(valid, distances, 0)/sigma[:, None]))*valid
    weights /= np.maximum(weights.sum(1)[:, None], 1e-12)
    rr = np.repeat(np.arange(len(xy), dtype=np.int32), k)[valid.ravel()]
    cc = idx[valid]
    assert np.all(cores[rr] == cores[cc]) and not np.any(rr == cc)
    w = sparse.csr_matrix((weights[valid].astype('float32'), (rr, cc)), shape=(len(xy), len(xy)))
    isolated = ~valid.any(1)
    # No eligible neighbors: retain self signal instead of inventing a tissue edge.
    w += sparse.diags(isolated.astype('float32'))
    return w.tocsr(), idx, distances, {'directed_edges': int(valid.sum()),
        'isolated_cells_self_fallback': int(isolated.sum()), 'cross_core_edges': 0,
        'max_distance_um': float(distances[valid].max()) if valid.any() else 0.,
        'radius_um': float(radius), 'k': int(k)}


def spatial_fit_transform(z, w, train, config):
    neighbor = np.asarray(w @ z)
    means, scales, blocks = [], [], []
    weight = config.get('spatial_weight', 0.3)
    for block, importance in ((z, 1-weight), (neighbor, weight)):
        mean = block[train].mean(0, dtype=np.float64)
        # Scalar block RMS preserves relative variance of the original PCA axes.
        scale = float(np.sqrt(np.mean((block[train]-mean)**2)))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError('Degenerate spatial block scale')
        blocks.append(((block-mean)*(np.sqrt(importance)/scale)).astype('float32'))
        means.append(mean)
        scales.append(scale)
    x = np.concatenate(blocks, 1)
    mean = x[train].mean(0, dtype=np.float64)
    xt = x[train].astype('float64')-mean
    cov = xt.T@xt/(len(xt)-1)
    values, vectors = np.linalg.eigh(cov)
    select = np.argsort(values)[::-1][:config.get('n_components', 32)]
    comp = vectors[:, select]
    emb = ((x-mean)@comp).astype('float32')
    return emb, {'spatial_weight': weight, 'variance_retained': float(values[select].sum()/values.sum()),
                 'fit_training_cells': int(train.sum()), 'block_scaling': 'train_scalar_RMS'}, {
        'block_means': np.array(means), 'block_scales': np.array(scales),
        'spatial_weight': np.array(weight), 'pca_mean': mean, 'components': comp,
        'eigenvalues': values[select]}


def knn_graph(z, config, out=None):
    import faiss
    faiss.omp_set_num_threads(config.get('threads', 8))
    z = np.ascontiguousarray(z, dtype=np.float32)
    n, d = z.shape
    k = min(config.get('knn_k', 30), n-1)
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_L2)
    index.hnsw.efConstruction = 120
    index.hnsw.efSearch = 128
    index.add(z)
    query_idx = np.random.default_rng(config.get('seed', 0)+19).choice(n, min(256, n), replace=False)
    exact = faiss.IndexFlatL2(d)
    exact.add(z)
    _, truth = exact.search(z[query_idx], k+1)

    def remove_self(indices, dist, rows):
        bad = indices == rows[:, None]
        order = np.argsort(bad, axis=1, kind='stable')[:, :k]
        return np.take_along_axis(indices, order, 1), np.take_along_axis(dist, order, 1)

    true_ids, _ = remove_self(truth, np.zeros_like(truth), query_idx)
    for ef in (128, 256, 512):
        index.hnsw.efSearch = ef
        dd, jj = index.search(z[query_idx], k+1)
        jj, _ = remove_self(jj, dd, query_idx)
        recalls = np.array([len(set(a)&set(b))/k for a, b in zip(jj, true_ids)])
        if recalls.mean() >= .90:
            break
    if recalls.mean() < .90:
        raise ValueError(f'HNSW exact-recall guard failed: {recalls.mean():.4f}')
    neighbors = np.empty((n, k), dtype='int32')
    dist = np.empty((n, k), dtype='float32')
    for start in range(0, n, 8192):
        dd, jj = index.search(z[start:start+8192], k+1)
        jj, dd = remove_self(jj, dd, np.arange(start, min(start+8192, n)))
        neighbors[start:start+8192], dist[start:start+8192] = jj, np.maximum(dd, 0)
    assert np.all(neighbors >= 0) and np.all(neighbors < n)
    assert not np.any(neighbors == np.arange(n)[:, None])
    assert np.isfinite(dist).all()
    if out:
        np.save(out/'knn_neighbors.npy', neighbors)
        np.save(out/'knn_sqdist.npy', dist)
    sigma2 = np.maximum(dist[:, -1], 1e-10)
    weights = np.exp(-dist/sigma2[:, None]).astype('float32')
    rows = np.repeat(np.arange(n, dtype='int32'), k)
    a = sparse.csr_matrix((weights.ravel(), (rows, neighbors.ravel())), shape=(n, n))
    a = ((a+a.T)*0.5).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    return a, neighbors, {'k': k, 'metric': 'euclidean', 'hnsw_m': 32,
        'ef_construction': 120, 'ef_search': ef,
        'exact_recall_queries': len(query_idx), 'mean_knn_recall': float(recalls.mean()),
        'min_query_recall': float(recalls.min()), 'edges': int(a.nnz//2),
        'threaded_HNSW_insertion_nondeterministic': True,
        'weight': 'exp(-d_squared/kth_neighbor_d_squared); arithmetic_mean_symmetrization'}


def leiden_labels(a, resolution, seed):
    import igraph as ig
    import leidenalg
    upper = sparse.triu(a, k=1, format='coo')
    edges = np.column_stack([upper.row, upper.col])
    graph = ig.Graph(n=a.shape[0], edges=edges, directed=False)
    part = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
        weights=upper.data, resolution_parameter=float(resolution), seed=int(seed), n_iterations=2)
    labels = np.asarray(part.membership, dtype=np.int32)
    return labels, {'n_clusters': int(len(np.unique(labels))), 'quality': float(part.quality()),
                    'resolution': float(resolution), 'seed': int(seed), 'iterations': 2}


def _leiden_worker_init():
    global _worker_thread_limit
    _worker_thread_limit = threadpool_limits(limits=1)


def leiden_grid(a, config, progress=None):
    """Independent partitions in bounded spawn workers; same frozen graph/settings."""
    settings = [(float(r), int(s)) for r in config.get('leiden_resolutions',[.5,1.,1.5])
                for s in (config.get('leiden_seeds',[0,1]) if r == 1. else [config.get('seed',0)])]
    workers = min(config.get('runtime', {}).get('leiden_parallel_workers', 4), len(settings))
    results = {}
    if workers == 1:
        for r,s in settings:
            results[(r,s)] = leiden_labels(a,r,s)
            if progress:
                progress(r,s,results[(r,s)][1])
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_leiden_worker_init) as pool:
            pending = {pool.submit(leiden_labels,a,r,s):(r,s) for r,s in settings}
            for future in as_completed(pending):
                r,s = pending[future]
                results[(r,s)] = future.result()
                if progress:
                    progress(r,s,results[(r,s)][1])
    labels = pd.DataFrame({'row_index':np.arange(a.shape[0])})
    metrics = {}
    for r,s in settings:
        key = f'leiden_r{r:.1f}_s{s}'
        labels[key], metrics[key] = results[(r,s)]
    return labels, metrics


def spatial_agreement(labels, spatial_idx, cores, seed=0, permutations=20):
    valid = spatial_idx >= 0
    rows = np.repeat(np.arange(len(labels), dtype='int32'), spatial_idx.shape[1])[valid.ravel()]
    cols = spatial_idx[valid]
    observed = float(np.mean(labels[rows] == labels[cols]))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cores == c) for c in np.unique(cores)]
    nulls = []
    shuffled = labels.copy()
    for _ in range(permutations):
        for group in groups:
            shuffled[group] = rng.permutation(labels[group])
        nulls.append(float(np.mean(shuffled[rows] == shuffled[cols])))
    null = float(np.mean(nulls))
    return {'observed': observed, 'within_core_label_permutation_mean': null,
        'permutations': permutations, 'excess_over_null': observed-null,
        'adjusted_agreement': (observed-null)/max(1-null, 1e-12)}


def load_pca(out, config):
    paths = current_paths()
    data = paths.data_root/'processed/so12_pearson_integration_v1'
    manifest = json.loads((data/'manifest.json').read_text())
    obs = pd.read_csv(data/'observations.csv.gz')
    assert len(obs) == manifest['cells'] == 407659
    assert not obs.duplicated(['slide','fov','cell_ID']).any()
    assert obs.train.sum() == 301198
    expected_train = obs.core_number.isin(config.get('train_cores', list(range(1,11))+list(range(15,25)))).to_numpy()
    np.testing.assert_array_equal(obs.train.to_numpy(dtype=bool), expected_train)
    assert obs.groupby('core').train.nunique().eq(1).all()
    assert obs.groupby('core').slide.nunique().eq(1).all()
    assert np.isfinite(obs[['x_um','y_um','depth']].to_numpy()).all()
    genes = json.loads((data/'genes.json').read_text())
    assert len(genes) == 1000 and not any(g.startswith(('Negative','SystemControl')) for g in genes)
    assert canonical_sha256(genes) == manifest['provenance']['genes_sha256']
    with np.load(data/'pearson/fitted_parameters.npz') as fit:
        mean, comp = fit['pca_mean'].astype('float32'), fit['components'].astype('float32')
        ratio = fit['explained_variance_ratio']
        parameters = {name: fit[name] for name in fit.files}
    assert comp.shape == (1000, config.get('n_components',32))
    z = np.empty((len(obs), comp.shape[1]), dtype='float32')
    alignment = {r['core']: r for r in manifest['provenance']['alignment_checks']}
    receipt = []
    covered = np.zeros(len(obs), dtype=bool)
    for entry in manifest['matrices']:
        if entry['method'] != 'pearson':
            continue
        p = data/entry['path']
        if sha(p) != entry['sha256']:
            raise ValueError('Pearson matrix checksum mismatch: '+entry['core'])
        a = alignment[entry['core']]
        lo, hi = a['row_start'], a['row_stop']
        assert 0 <= lo < hi <= len(obs) and not covered[lo:hi].any()
        covered[lo:hi] = True
        group = obs.iloc[lo:hi]
        assert (group.core == entry['core']).all() and len(group) == entry['rows']
        x = np.load(p, mmap_mode='r')
        assert x.shape == (len(group),1000) and np.isfinite(x).all()
        slide = int(group.core_number.iloc[0] > 14)+1
        core_path = paths.data_root/f'processed/so{slide}_14core_relative_qkv_v1/cores/{entry["core"]}.npz'
        with np.load(core_path, allow_pickle=False) as raw:
            np.testing.assert_allclose(group[['x_um','y_um']].to_numpy(), raw['coordinates_um'], rtol=0, atol=1e-8)
        for start in range(lo, hi, 8192):
            stop = min(hi,start+8192)
            z[start:stop] = (x[start-lo:stop-lo]-mean)@comp
        receipt.append({'core': entry['core'], 'rows': hi-lo, 'sha256': entry['sha256'],
                        'coordinates_match': True})
    assert covered.all() and len(receipt) == 28 and np.isfinite(z).all()
    write_cols = ['core','slide','fov','cell_ID','train','qc_pass','depth','x_um','y_um']
    cells = obs[write_cols].copy()
    cells.insert(0, 'row_index', np.arange(len(obs)))
    cells.to_csv(out/'cells.csv.gz', index=False)
    dump(out/'input_verification.json', {'matrices':receipt, 'cells':len(obs),
        'observations_sha256':sha(data/'observations.csv.gz'), 'manifest_sha256':sha(data/'manifest.json'),
        'parameters_sha256':sha(data/'pearson/fitted_parameters.npz'),
        'genes_sha256':sha(data/'genes.json'), 'core_count':int(obs.core.nunique()),
        'excluded_unmapped_cells':340, 'source_unchanged':True})
    return z, obs, parameters, {'variance_retained':float(ratio.sum()),
        'upstream_run_id':manifest['run_id'], 'fit_training_cells':int(obs.train.sum()),
        'method':'frozen_training_Pearson_PCA'}


def run_analysis(out: Path, config: dict) -> dict:
    t0 = time.monotonic()
    methods = config.get('methods',['pca','kernel_pca','spatial_pca'])
    with threadpool_limits(limits=config.get('threads',8)):
        event(out, 'load_verify_and_project')
        z, obs, params, info = load_pca(out, config)
        train, cores = obs.train.to_numpy(dtype=bool), obs.core.to_numpy()
        xy = obs[['x_um','y_um']].to_numpy()
        event(out, 'spatial_neighbors')
        w, spatial_idx, spatial_dist, spatial_info = spatial_neighbors(xy, cores, config)
        np.save(out/'spatial_neighbors.npy', spatial_idx)
        np.save(out/'spatial_distances_um.npy', spatial_dist)
        dump(out/'spatial_graph.json', spatial_info)
        del spatial_dist
        results, primary = {}, {}
        for method in methods:
            mt = time.monotonic()
            folder = out/'methods'/method
            folder.mkdir(parents=True)
            event(out, 'embedding', method=method)
            if method == 'pca':
                emb, detail, fitted = z, info, params
            elif method == 'kernel_pca':
                emb, detail, fitted = kernel_fit_transform(z, train, cores, config)
            elif method == 'spatial_pca':
                emb, detail, fitted = spatial_fit_transform(z, w, train, config)
            else:
                raise ValueError('Unknown method '+method)
            assert emb.shape == (len(obs),config.get('n_components',32)) and np.isfinite(emb).all()
            np.save(folder/'embedding.npy', emb)
            np.savez(folder/'encoder_parameters.npz', **fitted)
            event(out, 'knn_graph', method=method)
            a, neighbors, graph_info = knn_graph(emb, config, folder)
            event(out,'leiden_grid',method=method,workers=config.get('runtime',{}).get('leiden_parallel_workers',4))
            labels, partitions = leiden_grid(a, config, lambda r,s,m: event(out,'leiden_complete',
                method=method,resolution=r,seed=s,clusters=m['n_clusters']))
            labels.to_csv(folder/'clusters.csv.gz',index=False)
            main = labels['leiden_r1.0_s0'].to_numpy()
            primary[method] = main.copy()
            size_frame = pd.crosstab(main, obs.slide).rename_axis('cluster').reset_index()
            size_frame['total'] = size_frame.drop(columns='cluster').sum(axis=1)
            size_frame.to_csv(folder/'cluster_sizes.csv', index=False)
            pd.crosstab(obs.core,main).to_csv(folder/'clusters_by_core.csv')
            spatial_metric = spatial_agreement(main, spatial_idx, cores, permutations=20)
            diag = {'encoder':detail, 'graph':graph_info, 'partitions':partitions,
                'n_cells':len(obs), 'n_components':emb.shape[1], 'n_clusters':int(len(np.unique(main))),
                'smallest_cluster':int(np.bincount(main).min()), 'largest_cluster':int(np.bincount(main).max()),
                'seed_ari':float(adjusted_rand_score(main,labels['leiden_r1.0_s1'])),
                'resolution_ari_0.5':float(adjusted_rand_score(main,labels['leiden_r0.5_s0'])),
                'resolution_ari_1.5':float(adjusted_rand_score(main,labels['leiden_r1.5_s0'])),
                'slide_nmi':float(normalized_mutual_info_score(obs.slide,main)),
                'cross_slide_neighbor_fraction':float(np.mean(obs.slide.to_numpy()[:,None] != obs.slide.to_numpy()[neighbors])),
                'spatial_agreement':spatial_metric,
                'runtime_seconds':time.monotonic()-mt,
                'graph_sha256':sha(folder/'knn_neighbors.npy')}
            dump(folder/'diagnostics.json',diag)
            results[method] = diag
            event(out,'method_complete', method=method, clusters=diag['n_clusters'], seed_ari=diag['seed_ari'],
                  recall=graph_info['mean_knn_recall'], runtime_seconds=diag['runtime_seconds'])
            del a, neighbors, emb, labels
            gc.collect()
        comparison = {f'{a}__{b}':float(adjusted_rand_score(primary[a],primary[b]))
                      for i,a in enumerate(methods) for b in methods[i+1:]}
        summary = {'n_cells':len(obs),'n_genes':1000,'n_cores':28,'n_slides':2,
            'method_metrics':results, 'pairwise_cluster_ari':comparison, 'spatial_graph':spatial_info,
            'runtime_seconds':time.monotonic()-t0, 'gpu_used':False,
            'claim':'exploratory_transductive_representation_clustering_only',
            'limitations':['PCA32 retains only 8.22 percent of normalized-feature variance.',
                'Slide and tissue composition are confounded; mixing is not an optimization target.',
                'Spatial smoothing builds local continuity into the representation.',
                'Radius-only neighborhoods do not model tissue barriers or segmentation spillover.',
                'No independent biological cell-type or spatial-domain validation.',
                'Kernel PCA is a training-landmark approximation, not exact full-cohort kernel PCA.',
                'HNSW insertion is threaded and may vary; saved neighbors fix the analyzed graphs.']}
        dump(out/'analysis_summary.json', summary)
        event(out,'report')
        from so12_embedding_report import write_report
        write_report(out, config, summary)
        event(out,'analysis_complete', runtime_seconds=time.monotonic()-t0)
        return summary


def smoke(config):
    """Planted signal, split isolation, geometry bounds, and real Leiden checks."""
    rng = np.random.default_rng(7)
    n = 160
    z = rng.normal(size=(n,8)).astype('float32')
    truth = np.repeat([0,1], n//2)
    z[:,0] += truth*12
    train = np.arange(n)%4 != 0
    cores = np.tile(np.repeat(['a','b'],40),2)
    xy = np.column_stack([np.arange(n)%40, truth*100.])
    cfg = {**config,'n_components':4,'kernel_landmarks':40,'knn_k':10,'spatial_k':4,
           'spatial_radius_um':5.,'threads':1}
    w,idx,dist,receipt = spatial_neighbors(xy,cores,cfg)
    assert receipt['cross_core_edges']==0 and receipt['max_distance_um'] <= 5
    np.testing.assert_allclose(np.asarray(w.sum(1)).ravel(),1,atol=1e-6)
    s,_,p = spatial_fit_transform(z,w,train,cfg)
    kernel,_,kp = kernel_fit_transform(z,train,cores,cfg)
    altered = z.copy(); altered[~train] += 100
    _,_,kp2 = kernel_fit_transform(altered,train,cores,cfg)
    np.testing.assert_array_equal(kp['landmarks'],kp2['landmarks'])
    a,nn,diagnostics = knn_graph(z,cfg)
    lab,_ = leiden_labels(a,.2,0)
    lab2,_ = leiden_labels(a,.2,0)
    assert adjusted_rand_score(lab,lab2)==1
    assert adjusted_rand_score(lab,truth) > .9
    assert np.isfinite(s).all() and np.isfinite(kernel).all()
    assert np.all(train[kp['landmark_indices']])
    assert a.shape == (n,n) and (a-a.T).nnz==0
    return {'passed':True,'planted_cluster_ari':float(adjusted_rand_score(lab,truth)),
            'leiden_reproducible_on_saved_graph':True,'kernel_landmarks_train_only':True,
            'spatial_graph_core_and_radius_checks':True,'mean_knn_recall':diagnostics['mean_knn_recall']}
