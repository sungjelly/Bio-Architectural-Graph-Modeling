# CosMx Data Guide

This is the read-first map of the local CosMx spatial transcriptomics data. It
describes the snapshot profiled on 2026-07-24; it is not a substitute for a
versioned manifest in a conclusion-bearing workflow.

## Non-Negotiable Rules

- Treat `data/raw/`, `data/clinical/`, and the root `data.tar.gz` as immutable,
  untracked inputs.
- Do not publish donor identifiers or row-level clinical metadata in commits,
  logs, prompts, reports, or external services.
- Resolve raw files through the nested layout
  `data/raw/<filename>/<filename>`.
- Add slide identity before every join. FOV and numeric cell identifiers restart
  on each slide.
- Stream or scan the transcript and polygon tables. Do not load either complete
  transcript CSV into memory.
- Define and record a clinical-label policy before using disease labels. The
  current and legacy workbooks serve different purposes and are not directly
  interchangeable.

## Snapshot

| Quantity | SO_1 | SO_2 | Total |
|---|---:|---:|---:|
| Cells in expression/metadata | 161,596 | 246,403 | 407,999 |
| Raw FOVs | 205 | 246 | 451 |
| FOVs assigned to a core | 205 | 245 | 450 |
| Tissue cores | 14 | 14 | 28 |
| Transcript records | 42,498,373 | 74,516,776 | 117,015,149 |
| Cell-assigned transcripts | 32,620,874 | 59,679,722 | 92,300,596 |
| Unassigned transcripts | 9,877,499 | 14,837,054 | 24,714,553 (21.1%) |
| Polygon vertices | 4,198,698 | 5,731,515 | 9,930,213 |
| Expression targets | 1,207 | 1,207 | same ordered panel |
| Biological targets | 1,000 | 1,000 | same ordered panel |
| Technical controls | 207 | 207 | 10 negative + 197 system controls |
| Raw CSV size | 3.01 GB | 5.16 GB | 8.17 GB |

Expression and metadata contain one row per cell and have matching row counts,
but joins must still use keys rather than row position.

## Raw Files

Each basename is prefixed by `26040302SO_1` or `26040302SO_2`.

| Suffix | Grain | Columns and role |
|---|---|---|
| `_exprMat_file.csv` | one row per cell | `fov`, `cell_ID`, then 1,207 integer target-count columns |
| `_metadata_file.csv` | one row per cell | 100 columns containing keys, coordinates, morphology, protein intensity, RNA/technical QC, and vendor-derived annotations |
| `_tx_file.csv` | one row per detected transcript | `fov,cell_ID,cell,x_local_px,y_local_px,x_global_px,y_global_px,z,target,CellComp` |
| `-polygons.csv` | one row per boundary vertex | `fov,cellID,cell,x_local_px,y_local_px,x_global_px,y_global_px` |
| `_fov_positions_file.csv` | one row per FOV | `FOV,x_global_px,y_global_px,x_global_mm,y_global_mm` |

The two slides have identical ordered schemas for every corresponding export,
including the expression panel. Raw file names use inconsistent separators
(`-polygons` versus `_...`) and column spelling/case varies by table.

## Canonical Keys and Joins

Normalize slide names to `SO_1` and `SO_2`.

| Entity | Canonical key | Source spellings |
|---|---|---|
| Cell | `(slide, fov, cell_ID)` | expression/metadata/transcripts use `cell_ID`; polygons use `cellID` |
| FOV | `(slide, fov)` | positions use `FOV`; the core map uses `fov` |
| Core | `core_label` | integer 1-28; currently unique across both slides |
| Donor | restricted donor field from the legacy workbook | never infer it from cell, FOV, core, or slide |

`cell_ID` is only local to an FOV, and `(fov, cell_ID)` is only local to a
slide: 102,528 pair keys collide between the two slides. The metadata columns
`cell` and `cell_id` are identical composite strings such as `c_1_1_5`, meaning
slide 1, FOV 1, numeric cell 5. Construct the tuple key explicitly instead of
trusting row order. Expression, metadata, and polygon cell sets match exactly
when keyed correctly. Include slide in the core-map join even though core
numbers are currently globally unique.

Use global pixel coordinates for cross-FOV geometry. Local pixels are
FOV-relative, with opposite local/global Y direction:

```text
x_global_px = FOV_x_global_px + x_local_px
y_global_px = FOV_y_global_px - y_local_px
```

This transform is exact for metadata centers and polygon vertices. Transcript
exports have rounding residuals of at most one pixel, so recompute from local
coordinates and FOV origins or validate with a ±1 px tolerance. The FOV position
table also supplies global millimetres; the observed scale is approximately
`0.000120281 mm/px` (`0.120281 µm/px`). Preserve source coordinates and record
the canonical frame used to build a graph.

## Expression Panel

- The matrix has 1,209 columns: two keys plus 1,207 complete, nonnegative
  integer-count targets.
- Biological feature sets must exclude names beginning with `Negative` or
  `SystemControl`. Controls are interspersed alphabetically, so never remove a
  positional tail; retain the prefix-filtered controls separately for QC.
- The sum of all assigned transcript rows equals the matrix grand total, with
  zero per-cell or per-target mismatches. Metadata `nCount_RNA`,
  `nCount_negprobes`, and `nCount_falsecode` reconcile respectively to the
  biological, `Negative*`, and `SystemControl*` matrix sums.
- Nineteen probe targets represent combined or ambiguous symbols containing
  `/`; they are probe-level measurements and must not be split into gene-level
  values. Examples include `FCGR3A/B`, `HBA1/2`, `HLA-DQB1/2`, `KRT6A/B/C`,
  `SAA1/2`, `TPSAB1/B2`, and `XCL1/2`.
- Classic gastric lineage markers `MUC1`, `MUC2`, `MUC5AC`, `MUC6`, `TFF1`,
  `TFF2`, and `TFF3` are absent. Do not claim that this targeted panel can
  resolve those lineages as if whole-transcriptome evidence were available.
- Vendor cell type, cluster, neighborhood, niche, and posterior-probability
  fields are post-hoc annotations by default, not permitted model inputs.

## Transcript and Polygon Details

- Unassigned transcript rows use `cell_ID = 0`; all 24,714,553 have blank
  `CellComp` and must not become graph nodes. Assigned rows use only `Nuclear`
  or `Cytoplasm`; `z` spans 0-7 and no membrane compartment label exists.
- Every metadata cell has assigned transcripts and exactly one polygon. Polygon
  vertices form contiguous cell blocks, lack an explicit order column, and use
  implicit closure. Preserve source row order within each cell.
- The 25,806 adjacent identical transcript records are included in the exactly
  reconciled count matrix. With no molecule identifier, do not blindly
  deduplicate them; characterize coordinate collisions for the intended use.
- Polygon shoelace area and metadata `Area` differ by more than 25% for 1.14% of
  SO_1 cells and 1.57% of SO_2 cells. Treat them as different estimators and
  sensitivity-test any area or contact threshold.

## Metadata Orientation

Use the exact header rather than positional column numbers. Important groups
are:

- identity and geometry: `slide_ID`, `fov`, `cell_ID`, `cell`, `cell_id`,
  `CenterX_local_px`, `CenterY_local_px`, `CenterX_global_px`,
  `CenterY_global_px`, `Area`, `Area.um2`, `Width`, `Height`, and nuclear-shape
  fields;
- imaging: mean/max `PanCK`, `G`, `Membrane`, `CD45`, and `DAPI`;
- RNA/technical QC: `nCount_RNA`, `nFeature_RNA`, negative/false-code
  summaries, `propNegative`, `complexity`, `unassignedTranscripts`, cell QC
  flags, and FOV QC flags;
- derived vendor results: the long cell-typing cluster/probability columns,
  Leiden-like cluster, twelve neighborhood-composition columns, and niche
  assignment.

There are no duplicate `(fov, cell_ID)` rows or missing primary cell keys in
either metadata table. SO_1 has 21 blanks (0.013%) in each of `median_negprobes` and the six
negative-probe quantile columns; its other fields and all SO_2 metadata fields
are complete. Vendor cell QC marks 156,290 of 161,596 SO_1 cells and
237,946 of 246,403 SO_2 cells as passed: 394,236 of 407,999 overall (96.6%).
Seven FOVs are vendor-flagged in total. Treat these as documented vendor QC
signals, not an automatic universal filter; every workflow must state its
filter and sensitivity analysis.

SO_1 has mean `nCount_RNA` 198.8 and mean `nFeature_RNA` 100.6; SO_2 has 239.5
and 113.5. This slide difference is an early warning for batch/confounding
checks, not evidence of a biological stage effect.

## Core and Clinical Metadata

`data/clinical/fov_core_map.csv` has 450 unique `(slide, fov)` assignments:
SO_1 FOVs 1-205 map to cores 1-14, while SO_2 FOVs 1-245 map to cores 15-28.
Raw SO_2 FOV 246 is the only unmapped FOV; it contains 340 cells and 57,912
transcript rows (45,549 assigned). Do not infer a core, or silently discard it;
add a verified mapping or record explicit exclusion/unknown handling in the
workflow.

The workbooks have distinct roles:

- `Gastric Study_Old.xlsx` contains the legacy core-to-donor and tissue labels:
  28 cores from 14 donors, with two cores per donor. Its legacy distribution is
  Cancer 6, AdjacentNormal 14, HGD 1, LGD 6, and true Normal 1.
- `Gastric Study.xlsx` is a newer pathology review/correction sheet for the same
  28 cores. Seven rows are marked as incorrect diagnoses and contain revised
  findings, including gastritis and intestinal-metaplasia categories. It does
  not contain the donor mapping expected by the current assignment script.

Do not treat either workbook as a drop-in canonical label table. A workflow
using histology must create a local, versioned reconciliation policy that
preserves the legacy donor pairing, applies the pathology review deliberately,
distinguishes tissue type from donor-level diagnosis, records provenance, and
is reviewed before modeling.

Under the legacy labels, every donor contributes a lesion or true-Normal core
paired with an AdjacentNormal core. Hold out donors, not cells or FOVs. Cancer
and HGD occur only on SO_1; LGD and true Normal occur only on SO_2; only
AdjacentNormal spans both slides. Consequently, slide/batch and lesion stage
are strongly confounded. HGD and true Normal have one core each and are
descriptive groups, not adequate bases for population-level claims or model
selection.

## Safe Access Pattern

1. Read only headers and file sizes first.
2. Resolve the nested raw path and derive slide identity from the explicit file
   prefix, not from FOV number.
3. Select only required columns and set compact dtypes.
4. Stream transcript/polygon CSVs in chunks or use a lazy scanner.
5. Validate key uniqueness and join coverage before writing an intermediate.
6. Write derived columnar data under `data/processed/`, never beside or over the
   source files.
7. Save an input manifest with paths, byte sizes, checksums, row counts, schema,
   exclusions, label policy, and conversion code for any full experiment.

For a filename `name`, use this resolver logic:

```python
from pathlib import Path

direct = Path("data/raw") / name
path = direct / name if (direct / name).is_file() else direct
```

Never load both transcript tables with a normal eager `read_csv`. For a first
pass, stream only `fov`, `cell_ID`, `target`, `CellComp`, and the coordinates
required by the question.

## Preflight Checklist for Every Workflow

- [ ] The input snapshot and checksums are recorded.
- [ ] Slide-qualified cell and FOV keys are used and join coverage is reported.
- [ ] SO_2 FOV 246 has an explicit handling rule.
- [ ] Technical controls and combined probes have explicit feature policies.
- [ ] Allowed inputs are separated from post-hoc/leakage fields.
- [ ] The clinical reconciliation policy and experimental unit are stated.
- [ ] Donor-held-out and spatial-block splits are created before learned
      preprocessing.
- [ ] Slide/stage confounding, vendor QC, panel coverage, and segmentation
      spillover are tested or named as limitations.

## Known Tooling Blocker

Do not currently run `scripts/core_assignment/plot_fov_core_layout.py` as a
source of canonical clinical assignments. It searches for a nonexistent
`Gastric Study_2.xlsx`; the newer workbook also does not match the donor/stage
schema assumed by the script. Repair requires an explicit clinical
reconciliation policy, not merely a path change.
