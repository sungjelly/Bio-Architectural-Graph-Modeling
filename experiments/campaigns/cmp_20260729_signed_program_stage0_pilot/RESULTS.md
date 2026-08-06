# Signed-program Stage-0 result

## Decision

- Outcome: **negative**
- Passing variants: `0/3`
- Decision: stop this architecture branch; do not launch a real-expression or
  production graph run from this campaign
- Receipt:
  `scratch/diagnostics/signed_program_stage0_result.json`
- Canonical receipt-payload checksum:
  `eb1465cd20deaf20be162c9badb0cc418bd17abad67a6ef0936ab9fe1b74fd76`
- Receipt-file SHA-256:
  `9e447ce28bd60c448e828b04184ce7ca3e38edf39c276d90738eeb5fa3d70a25`

The diagnostic used the checksum-locked 160-node ANC-05 geometry fixture
(`f53076d07e81842b4e73a4bb6ec66b795d2899d207a620c35a136377b2aa1cd7`)
and the unchanged seed, effect size, masks, optimizer, 96 epochs, and Stage-0
gate.

## Observed result

| Variant | Self+regional loss | True-local loss | Permuted-local loss | True vs self | True vs permuted | Planted top-deletion delta | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `count_sum` | 0.298026 | 0.293530 | 0.282303 | +1.51% | -3.98% | -0.038285 | fail |
| `count_plus_concentration` | 0.298026 | 0.292023 | 0.281984 | +2.01% | -3.56% | -0.036718 | fail |
| `radial_count` | 0.298026 | 0.275229 | 0.262280 | +7.65% | -4.94% | -0.031801 | fail |

Positive `True vs self` means lower loss for true-local. Negative
`True vs permuted` means the permuted-sender control had lower loss than
true-local.

Every variant:

- was exactly parameter matched across its four arms;
- passed sender-permutation QC;
- learned a positive mean continuous contribution on planted edges;
- beat the self+regional reconstruction loss;
- failed because permuted sender state predicted better than correct sender
  state; and
- failed faithfulness because deleting the top positive planted-edge
  contributions **reduced** target loss instead of increasing it.

The null-injection arm did not produce the analogous positive-contribution plus
deletion discovery. This useful negative control does not rescue the two failed
positive-control relations.

## Interpretation

Question: does preserving unnormalized sender counts repair the failed QKV
Stage-0 recovery?

Observed answer: no. Raw count preservation increased apparent graph gain,
especially for the radial variant, but that gain was neither specific to
correct sender alignment nor faithful under deletion.

Strongest alternative explanation: optimization under the mixed hurdle
objective may distribute the planted relationship across self, regional, and
local coefficients in a way that makes a single local coefficient look
positive while its net predictive use is harmful. The parameter-matched
permuted arm and exact edge-deletion intervention directly expose this failure.
The single fixture and seed cannot distinguish optimization pathology from a
more general limitation of this constrained family.

Remaining uncertainty: this is one generated fixture, one fixed seed, and one
fixed training budget. It does not show that every signed additive architecture
must fail. Changing those quantities after seeing this outcome would be
post-outcome tuning and is prohibited in this campaign.

Maximum defensible claim: none of the three prespecified constrained
signed-program architectures passed the locked synthetic recovery gate. Their
reconstruction improvements alone are not evidence of meaningful biological
signal and do not authorize real-data interpretation.

## Verification and execution record

Focused checks:

```text
27 passed
```

This includes the new eight-test suite plus the existing synthetic fixture and
multiscale training suites. The final diagnostic ran on CPU in 18.16 measured
model seconds, used no GPU VRAM, and wrote only the compact JSON receipt.

An initial identical execution completed model fitting but failed while
serializing the receipt because the diagnostic used the wrong graph-receipt
dictionary key (`checksums` instead of `bundle_checksums`). No outcome was
printed or inspected. The only change before the deterministic rerun corrected
that receipt lookup; no scientific, model, seed, effect, epoch, loss, or gate
setting changed.

Exact commands:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_signed_program_synthetic.py \
  tests/unit/spatial_benchmark/test_multiscale_synthetic.py \
  tests/unit/spatial_benchmark/test_multiscale_hurdle_training.py

PYTHONPATH=src /venv/main/bin/python \
  scripts/diagnostics/run_signed_program_synthetic.py \
  --device cpu \
  --output scratch/diagnostics/signed_program_stage0_result.json
```
