# SO1 14-core model-embedding direct clustering

This completed post-training report is locked to run `r_20260826T204925Z_06d12943_s000_f00_a01_94691119` and its
immutable `last.ckpt`. No retraining, preprocessing refit, batch correction,
neighbor sampling, or GPU execution occurred.

## Representation and clustering method

`h0` is the exact all-node output of the trained NodeEncoder before graph
attention. `hL` is the exact all-node output after the fourth Relative-Geometric
QKV graph layer and immediately before the expression decoder. For each
representation independently, all 161,596 cells from SO1 cores
1 through 14 were concatenated, row-L2-normalized solely for cosine distance,
inserted into an independent sparse FAISS HNSW 30-nearest-neighbor graph using a
deterministic core-independent permutation, and clustered with seeded Leiden at
resolution 1.0. There was no PCA or mean-centering. Sampled exact recall@30 was
required to be at least 0.90 for both graphs.

- Intrinsic clusters: 17; size range [1170, 17345]
- Contextual clusters: 26; size range [317, 16676]
- Intrinsic clusters >90% from one core: None
- Contextual clusters >90% from one core: S1C8, S1C13, S1C15, S1C16, S1C17, S1C22, S1C24, S1C25
- Raw delta-norm mean/median: 14.4851 / 13.4646
- Shared delta plotting limits (global p1/p99): 10.8513 / 23.4481

## Interpretation constraints

Intrinsic clusters represent patterns in the cell's own standardized expression
and permitted metadata embedding. Contextual clusters represent patterns after
graph-based neighborhood processing by this trained model. `delta_h_l2` is only
the descriptive magnitude of representation change after contextual processing.

None of these quantities independently establishes cell type, signaling,
biological influence, or causality. Cluster names are deliberately limited to
the model-derived `S1I` and `S1C` namespaces. Marker-based and pathological
validation will be conducted separately.

This is a fit-only, transductive, single-seed exploratory readout. Core-dominated
clusters are flagged but not removed or integrated.

## Programmatic reproduction

Run only after the registry marks the exact run complete and the immutable last
checkpoint verifies:

```python
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry
from spatial_benchmark.so1_model_embedding_clustering import run_so1_model_embedding_clustering

paths = ProjectPaths.from_environment(anchor=__file__)
registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
run_so1_model_embedding_clustering(registry=registry, paths=paths)
```

Launch Python with `CUDA_VISIBLE_DEVICES=""`, `PYTHONPATH=src`, and the repository
virtual environment. Per-core extraction, joint clustering, and plotting are
separate checksum-verified resumable stages.
