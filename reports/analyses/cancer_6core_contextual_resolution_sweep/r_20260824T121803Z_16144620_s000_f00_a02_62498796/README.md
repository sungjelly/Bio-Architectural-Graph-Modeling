# Contextual hL Leiden resolution sweep

Status: complete exploratory sensitivity analysis of the locked contextual
embedding. This bundle uses only the final graph embedding `hL`. It creates no
intrinsic `h0` clustering figures and no delta-h figures. The model was not
retrained and model inference was not rerun.

## Source and coverage

- Source run ID: `r_20260824T121803Z_16144620_s000_f00_a02_62498796`
- Model seed: `0`
- Source checkpoint: `/workspace/BAGM/artifacts/runs/2026/08/r_20260824T121803Z_16144620_s000_f00_a02_62498796/checkpoints/last.ckpt`
- Checkpoint SHA-256: `c5b7fdd6e3192c3146d415f1c77474f9e4483bba090c4fc9c2ac7d79ab554c0c`
- Source analysis bundle: `/workspace/BAGM/reports/analyses/cancer_6core_embedding_clustering/r_20260824T121803Z_16144620_s000_f00_a02_62498796`
- Cells: Core 1: 8,924, Core 9: 17,223, Core 13: 5,345, Core 15: 38,145, Core 21: 4,897, Core 23: 43,462; total 117,996
- Contextual embedding width: 256

All cells from cores 1, 9, 13, 15, 21, and 23 were clustered jointly. One
mean-centered 50-component PCA representation and one cosine 30-nearest-neighbor
FAISS-HNSW graph were computed from `hL`, checksum-matched to the completed
analysis, saved, and reused without alteration for every resolution. Only the
Leiden resolution parameter varies. The shared kNN edge checksum is
`180afebe30ee6232c8c39368f7da18d420680ed290e4d56670820f3f383dd037`. Resolution 1.0 labels match
the established contextual labels exactly.

## Resolution results

| Leiden resolution | Clusters | Cluster-size range | Clusters >90% from one core |
|---:|---:|---:|:---|
| 0.25 | 7 | 5,137–29,952 | R0p25_C0, R0p25_C2, R0p25_C3, R0p25_C6 |
| 0.5 | 12 | 4,008–16,090 | R0p5_C0, R0p5_C2, R0p5_C3, R0p5_C5, R0p5_C7, R0p5_C9, R0p5_C11 |
| 0.75 | 17 | 874–13,510 | R0p75_C0, R0p75_C1, R0p75_C3, R0p75_C5, R0p75_C6, R0p75_C7, R0p75_C11, R0p75_C12, R0p75_C14, R0p75_C16 |
| 1 | 21 | 509–10,388 | R1p0_C0, R1p0_C1, R1p0_C3, R1p0_C5, R1p0_C6, R1p0_C7, R1p0_C8, R1p0_C12, R1p0_C13, R1p0_C15, R1p0_C16, R1p0_C18, R1p0_C19 |
| 1.25 | 24 | 516–9,370 | R1p25_C0, R1p25_C2, R1p25_C4, R1p25_C6, R1p25_C8, R1p25_C9, R1p25_C11, R1p25_C12, R1p25_C13, R1p25_C14, R1p25_C16, R1p25_C17, R1p25_C19, R1p25_C20, R1p25_C21, R1p25_C22 |
| 1.5 | 26 | 535–8,602 | R1p5_C0, R1p5_C2, R1p5_C7, R1p5_C8, R1p5_C9, R1p5_C10, R1p5_C11, R1p5_C12, R1p5_C13, R1p5_C14, R1p5_C15, R1p5_C16, R1p5_C17, R1p5_C21, R1p5_C22, R1p5_C23, R1p5_C24 |
| 2 | 32 | 547–6,456 | R2p0_C1, R2p0_C3, R2p0_C5, R2p0_C7, R2p0_C8, R2p0_C9, R2p0_C10, R2p0_C11, R2p0_C12, R2p0_C13, R2p0_C14, R2p0_C15, R2p0_C17, R2p0_C18, R2p0_C19, R2p0_C21, R2p0_C22, R2p0_C23, R2p0_C25, R2p0_C26, R2p0_C28, R2p0_C29 |

Cluster identifiers are resolution-qualified. A similarly numbered cluster at
two resolutions is not asserted to be the same population. Higher resolution
changes clustering granularity; it is not evidence that one setting is more
biologically correct.

## Interpretation constraints

These contextual clusters represent patterns after graph-based neighborhood
processing by the trained model. They are model-derived partitions, not
independently established cell types. Neither a partition nor its spatial
appearance independently establishes cell type, signaling, biological
influence, or causality. Marker-based and pathological validation will be
conducted separately. Core-dominated clusters are reported without removal,
merging, or batch integration.

## Reproduction

From the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-contextual-resolution-sweep \
  --run-id r_20260824T121803Z_16144620_s000_f00_a02_62498796 \
  --resolutions 0.25 0.5 0.75 1 1.25 1.5 2 \
  --random-seed 20260825 \
  --output-dir reports/analyses/cancer_6core_contextual_resolution_sweep/r_20260824T121803Z_16144620_s000_f00_a02_62498796
```

The workflow is resumable. Checksum-valid shared-graph and partition receipts
skip PCA, kNN construction, and Leiden if plotting is interrupted. Every final
table and figure is checksum-bound by `manifest.json`.
