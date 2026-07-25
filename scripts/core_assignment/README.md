# Core Assignment Scripts

This directory contains scripts for assigning and plotting CosMx FOVs against tissue-core labels.

- `plot_fov_core_layout.py`: when its clinical policy is repaired, writes only to `scratch/preprocessing/legacy_core_assignment/` until outputs are reviewed and registered.

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
