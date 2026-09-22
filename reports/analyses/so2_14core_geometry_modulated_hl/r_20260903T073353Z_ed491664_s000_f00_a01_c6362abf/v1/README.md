# SO2 geometry-modulated hL map

Run `r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf`, final epoch 200, model seed 0. All 246,063 cells across 14 fitted cores; 20 joint clusters (sizes [1114, 19503]). Fully observed expression; final graph-block hL, PCA50, cosine kNN30, Leiden resolution1, seed20260825.

Repeated pilot extraction and seeded Leiden replay passed. Spatial groups can reflect expression, morphology, core/batch effects or broad fields. This is an exploratory representation map with no cell-type, communication, patient-generalization or causal claim. No biological null or cross-seed stability was tested.

Clusters with >90% membership from one core: ['C0', 'C5', 'C9', 'C10', 'C16', 'C17', 'C18'].

Reproduce from the project root: `PYTHONPATH=src /venv/main/bin/python scripts/analysis/create_so2_geometry_hl_map.py`. Add `--verify-only` to check the completed report. See TASK_CONTRACT.md and manifests for provenance.
