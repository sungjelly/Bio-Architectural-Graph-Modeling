# Core Assignment Scripts

This directory contains scripts for assigning and plotting CosMx FOVs against tissue-core labels.

- `plot_fov_core_layout.py`: when its clinical policy is repaired, writes only to `scratch/preprocessing/legacy_core_assignment/` until outputs are reviewed and registered.
- `../data/select_adjacent_normal_cores.py`: the separate, fail-closed
  adjacent-normal selection workflow described below.

## Current Blocker

Do not treat the current script output as canonical. The script searches for a
nonexistent `Gastric Study_2.xlsx`. The available newer workbook is a pathology
review/correction sheet and does not supply the donor/stage schema the script
expects; the legacy workbook supplies donor pairing but needs deliberate
reconciliation with seven revised diagnoses.

Read [`docs/legacy_data_guide.md`](../../docs/legacy_data_guide.md) before repairing or running
this utility. A repair
must define and version the clinical-label policy rather than changing only the
workbook path.

## Reviewed Adjacent-Normal Selection

`scripts/data/select_adjacent_normal_cores.py` does not repair or authorize the
legacy plotting utility. It implements the narrower
`adjacent_normal_reviewed_balanced_v1` policy:

- the only accepted exact legacy labels are `주변조직` (8 rows) and
  `주변조직 (N)` (6 rows), both mapped to canonical `AdjacentNormal`;
- all 14 candidates must have distinct restricted donors;
- the current review must say diagnosis `Normal`, result `정상조직확인`, and
  have a blank correction;
- cell counts come from slide-qualified metadata, and eligibility requires at
  least 5,001 cells;
- five candidates per slide are ranked by absolute distance from the
  slide-specific median eligible cell count, with the protected core key used
  only as a deterministic tie-break.

Adjacent-normal tissue is not the same biological category as the single
documented true-Normal core. Keep that distinction in experiment names and
claims.

The CLI requires a caller-owned output path. Its stdout contains only aliases,
slides, and FOV routing. The mode-0600 JSON is a protected local artifact:
it includes source core keys and must not be committed or published, but never
contains raw donor identifiers.

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/data/select_adjacent_normal_cores.py \
  --output scratch/preprocessing/adjacent_normal_10_core_selection/selection.json

PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_adjacent_normal_selection.py
```
