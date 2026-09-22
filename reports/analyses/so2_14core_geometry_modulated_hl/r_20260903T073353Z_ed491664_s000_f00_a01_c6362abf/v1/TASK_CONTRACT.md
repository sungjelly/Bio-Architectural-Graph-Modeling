# Geometry-modulated SO2 hL spatial map

Phase: pilot. Outcome: pending. This is exploratory post-hoc visualization.

## Task contract

Generate one verified PNG of joint final-layer hL clusters for the most recently
completed training run, `r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf`
(seed 0, epoch 200, completed 2026-09-04). Use its registered final checkpoint;
selection is by completion recency, never by loss or visual appeal. The campaign
is already registered; this versioned report is linked to that immutable run.

Question: how are groups in this model's final 256-dimensional representation
distributed over the fitted tissue? The operational hypothesis is that the
checkpoint can produce finite, cell-aligned hL for all 246,063 mapped cells.
Failure to reload exactly, reproduce extraction, or retain every cell falsifies
that operational hypothesis. Apparent spatial groups may instead reflect
same-cell expression, morphology, broad fields, or core/batch effects. The map
alone cannot discriminate these biological alternatives.

The estimand is joint clustering of the fourth graph-block output with fully
observed expression (zero gene mask), in evaluation/inference mode. It is a
descriptive representation map, not masked-prediction evaluation, cell-type
annotation, communication, or causal evidence. Cells are plotted observations;
cores are fitted spatial units, not independent patient replication. There is
no held-out split or patient-generalization claim. No clinical/vendor labels
enter extraction or clustering; source coordinates only place cells on the map.

Use the existing PCA-50, L2-normalization, cosine kNN-30, undirected-union Leiden
pipeline at resolution 1.0 and seed 20260825. These settings match the earlier
SO2 hL workflow and are fixed before viewing this checkpoint's map. Cluster
numbers/colors are local to this partition and do not establish correspondence
to older maps. Do not tune the resolution after viewing the result.

Primary operational metric: 100% finite embeddings and plotted-cell coverage.
Positive controls: strict checkpoint state checksum/reload; repeated pilot
extraction with rtol=1e-6 and atol=1e-5; seeded Leiden replay with identical
labels. Negative checks: reject altered inputs, missing cells, nonzero masks,
nonfinite embeddings, missing palette entries, or a different architecture.
Biological baselines, null calibration, segmentation robustness, cross-seed
stability and independent replication are not evaluated by this visualization;
no corresponding claim can be made or promoted to results/.

Inputs: immutable checkpoint/config/provenance bundle and its checksummed
`data/processed/so2_14core_relative_qkv_v1` and
`data/processed/so2_14core_relative_qkv_graphs_v1` artifacts. SO2 FOV246 remains
explicitly excluded. Recover missing model code and original campaign records
byte-for-byte from the run's provenance patch, checking the recorded hashes.
Keep existing user changes and the successful source bundle untouched.

Resource plan: inspect all four RTX3090 GPUs and framework visibility. Run a
small-core repeated pilot and largest-core memory pilot before distributing the
remaining independent complete cores over available GPUs 0--3. Use FP32 model
inference with exact receiver chunking, no neighbor sampling, and CPU memory
maps for graph/geometry. Require at least 2 GiB GPU headroom; record throughput,
peak allocated/reserved VRAM, host RSS, software versions and failures. CPU
PCA/FAISS/Leiden preserves the established deterministic clustering algorithm.

Active files live under `scratch/active_runs/<source_run_id>/posthoc_reports/`
and the verified report is published once under
`reports/analyses/so2_14core_geometry_modulated_hl/<source_run_id>/v1/`.
Expected artifacts: PNG, 14 aligned embedding shards, label and composition
tables, parameters, source/code/environment provenance, append-only events,
checksummed manifests and completion marker. Stop on failed integrity,
alignment, numerical, resource, or publication checks; retain failure evidence.

## Reproduction and verification

From the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q tests/unit/spatial_benchmark/test_geometry_modulated_relative_qkv_graph_transformer.py tests/unit/spatial_benchmark/test_so2_hl_clustering.py
PYTHONPATH=src /venv/main/bin/python scripts/analysis/create_so2_geometry_hl_map.py
PYTHONPATH=src /venv/main/bin/python scripts/analysis/create_so2_geometry_hl_map.py --verify-only
```

Completion requires all source/output checksums, all 14 expected cell counts,
pilot and Leiden repeat checks, a complete readable PNG with one shared palette,
and an updated status here. No scientific conclusion record is warranted by an
attractive visualization alone.
