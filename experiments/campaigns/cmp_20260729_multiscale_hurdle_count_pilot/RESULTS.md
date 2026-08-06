# Multiscale hurdle-count graph result

## Decision

- Outcome: **negative**
- Registered run:
  `r_20260729T172139Z_1833137a_s000_f00_a01_df2d8e14`
- Scientific gate: failed
- Execution status: completed successfully
- Artifact bundle: checksum-valid and registry-consistent
- Decision: stop the real-data graph arms and graph interpretation

A successful process exit is not a scientific pass. This run completed the
locked diagnostic, but three prespecified Stage-0 relations failed.

## Registered observed-geometry result

The run trained four parameter-matched arms for 96 fixed FP32 epochs on
generated expression over the locked 160-node ANC-05 observed geometry. There
was no validation/test selection and no post-outcome tuning.

| Arm or diagnostic | Whole-node hurdle loss |
|---|---:|
| Self + regional | 0.262242 |
| True local | 0.253390 |
| Permuted local | 0.252845 |
| Null injection, true local | 0.328063 |

True-local improved over self+regional by `3.38%`, but was `0.22%` worse than
the topology- and geometry-identical permuted-sender control. The learned
planted contribution had the expected positive sign, but its deletion result
was not faithful:

| Prespecified check | Observed result | Pass |
|---|---:|---:|
| True-local loss below self+regional | advantage `+0.008851` | yes |
| True-local loss below permuted-local | advantage `-0.000545` | no |
| Mean planted contribution positive | `+0.002115` | yes |
| Top deletion exceeds distance-matched deletion | `0.000417` vs `0.002199` | no |
| No analogous null discovery | analogous discovery observed | no |

The sender-state permutation itself passed QC: `100%` of node mappings and
local edge-slot source identities changed, `96.25%` of nodes moved more than
`75 µm`, and local topology, receivers, and edge attributes remained
unchanged. Thus the negative result is not explained by a failed permutation
gate.

## What preceded the registered run

Two graph-null designs failed before outcome-bearing GPU training:

1. The first 2-switch materialization changed only `6.33%` to `8.20%` of final
   relations because later switches could restore original edges.
2. After strengthening the requirement to at least `50%` final replacement,
   a degree-relaxed feasibility bound showed that no tested distance
   stratification could also preserve mean distance within `10%` on all five
   cores. Weakening the control after seeing this was not allowed, so rewiring
   was replaced by the frozen sender-state permutation.

`stage0_prerun_negative_diagnostic_v1.json` was then run on deterministic
generated unit-test coordinates. It was unregistered and did **not** use the
ANC-05 observed geometry. It also failed, but its contract explicitly says
that it cannot substitute for the registered result. The registered run above
used the locked ANC-05 geometry and independently reached a negative decision.

The follow-up signed-program architectures are reported separately in
`../cmp_20260729_signed_program_stage0_pilot/RESULTS.md`; none of their three
prespecified variants passed either.

## Resource and provenance record

- Device: GPU 0, NVIDIA GeForce RTX 3090
- PyTorch/CUDA build: `2.11.0+cu130` / `13.0`
- Model arms: 4, each with 10,360 trainable parameters
- Fixed training budget: 96 epochs per arm
- Measured diagnostic duration: `39.58 s` (`0.0110 GPU-hours`)
- Peak allocated VRAM: `76,796,416` bytes (`0.0715 GiB`)
- Finalized bundle size: `4,028,316` bytes
- Last-checkpoint SHA-256:
  `c489a218a0a1eb25ecde07437fb055aed2169fddf212db7bd518592146f4d98f`
- Checkpoint catalog status: `verified`
- Bundle verification: no bundle or registry issues

The immutable bundle is under
`artifacts/runs/2026/07/r_20260729T172139Z_1833137a_s000_f00_a01_df2d8e14/`.
Its resolved configuration, command, environment, hardware, input
fingerprints, metrics, predictions, diagnostics, checkpoint, and checksums are
preserved.

## Interpretation

Question: did the additive multiscale QKV model recover a known local
sender-state dependency well enough to justify real-expression graph runs?

Observed answer: no. Reconstruction improved over self+regional, but the
spatially displaced sender control did slightly better, the selected
contributions failed the matched deletion test, and the null injection
produced the prohibited analogous discovery.

The strongest remaining alternative is an optimization or identifiability
failure under this architecture, mask mixture, and fixed budget. One fixture
and one seed cannot show that all graph architectures fail. Changing the seed,
effect, epoch budget, or gates after observing the result would be
post-outcome tuning and is not an allowed rescue of this campaign.

Maximum defensible claim: the tested QKV graph architecture was not validated
for biological interpretation under its locked synthetic recovery gate. The
result says nothing affirmative about real-cell predictive gain, independent
biological signal, communication, mechanism, patient generalization, or
causality.

## Reproduction and verification

Run from the repository root. The first command validates the locked
configuration without creating another run:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/train/enqueue_multiscale_synthetic_recovery.py --check-only

PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_hurdle_continuous.py \
  tests/unit/spatial_benchmark/test_local_source_permutation.py \
  tests/unit/spatial_benchmark/test_multiscale_graphs.py \
  tests/unit/spatial_benchmark/test_multiscale_hybrid.py \
  tests/unit/spatial_benchmark/test_multiscale_hurdle_training.py \
  tests/unit/spatial_benchmark/test_multiscale_synthetic.py

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 verify-artifacts \
  --run-id r_20260729T172139Z_1833137a_s000_f00_a01_df2d8e14

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 resolve-checkpoint \
  r_20260729T172139Z_1833137a_s000_f00_a01_df2d8e14 --role last
```
