# Six-core intrinsic/contextual embedding clustering

Status: complete exploratory post-hoc analysis of a locked, completed model.
No model retraining, preprocessing refit, batch correction, neighbor sampling,
or per-core clustering was performed.

## Selected model and coverage

- Run ID: `r_20260824T121803Z_16144620_s000_f00_a02_62498796`
- Model seed: `0`
- Checkpoint: `artifacts/runs/2026/08/r_20260824T121803Z_16144620_s000_f00_a02_62498796/checkpoints/last.ckpt`
- Checkpoint SHA-256: `c5b7fdd6e3192c3146d415f1c77474f9e4483bba090c4fc9c2ac7d79ab554c0c`
- Model width/layers: `256` / `4`
- Cells: Core 1: 8,924, Core 9: 17,223, Core 13: 5,345, Core 15: 38,145, Core 21: 4,897, Core 23: 43,462; total 117,996
- Joint intrinsic clusters: 15
- Joint contextual clusters: 21
- Intrinsic clusters >90% from one core: I1, I8
- Contextual clusters >90% from one core: C0, C1, C3, C5, C6, C7, C8, C12, C13, C15, C16, C18, C19
- Global raw delta-norm mean/median: 12.3441 / 12.2684
- Shared plotting limits (global p1, p99): 9.4051, 16.4718

The core counts differ materially from the approximate 15,000-per-core planning
expectation. The immutable preparation manifest contains 117,996 cells, and all
117,996 are retained here without downsampling or silent exclusion.

## Interpretation constraints

Intrinsic clusters represent patterns in the cell's own expression and metadata
embedding. Contextual clusters represent patterns after graph-based neighborhood
processing. Delta-h norm measures the magnitude of representation change after
contextual processing and is only a descriptive contextual
representation-change magnitude.

None of these quantities independently establishes cell type, signaling,
biological influence, or causality. The model-derived clusters are deliberately
not assigned biological names. Marker-based and pathological validation will be
conducted separately.

This analysis is transductive and post-hoc. Core-dominated clusters are flagged,
not removed or integrated. No Harmony, scVI, ComBat, or related integration was
applied.

## Reproduction

From the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-embedding-clusters \
  --run-id r_20260824T121803Z_16144620_s000_f00_a02_62498796 \
  --checkpoint artifacts/runs/2026/08/r_20260824T121803Z_16144620_s000_f00_a02_62498796/checkpoints/last.ckpt \
  --n-neighbors 30 \
  --leiden-resolution 1.0 \
  --pca-components 50 \
  --random-seed 20260825 \
  --device cuda:0 \
  --output-dir reports/analyses/cancer_6core_embedding_clustering/r_20260824T121803Z_16144620_s000_f00_a02_62498796
```

The workflow is resumable: a checksum-valid
`embeddings/extraction_manifest.json` skips model inference, and a checksum-valid
`clustering/clustering_manifest.json` skips clustering when only figures remain.
All final files are checksum-bound by `manifest.json`.
