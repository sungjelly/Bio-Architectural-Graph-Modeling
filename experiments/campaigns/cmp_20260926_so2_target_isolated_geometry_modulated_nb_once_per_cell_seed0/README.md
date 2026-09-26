# SO2 target-isolated geometry-modulated NB2, once per cell

Campaign ID:
`cmp_20260926_so2_target_isolated_geometry_modulated_nb_once_per_cell_seed0`

Status: design and implementation locked. Launch is gated on the
checksum-bound implementation passing every gate in the frozen task contract.

## Question

Can the four-block geometry-modulated Relative-QKV NB2 model learn a more
faithful masked-count objective when every cell is predicted exactly once per
epoch from a masked target stream and a fully observed, static neighbor stream?

This is a new experiment. It does not amend the completed 2026-09-07 NB run.
The existing SO2 overlay is reused only for its donor-grouped split,
training-only preprocessing, raw targets, and immutable source references. Its
legacy ten-view validation masks are not consumed by this campaign.

## Data and claims

- Training: SO2-C15 through SO2-C26, 208,696 cells from six donor groups.
- Validation: SO2-C27 and SO2-C28, 37,367 cells from one held-out donor group.
- Test: none.
- Validation controls scheduling, early stopping, and checkpoint selection.
- Results are exploratory and validation-selected. They are not an unbiased
  test estimate and do not support a generalization claim.

## Target-isolated two-stream routing

The model retains four independent 256-dimensional graph blocks, eight
32-dimensional attention heads, a 1,024-dimensional feed-forward sublayer, a
1,024-dimensional decoder, and an NB2 output with one gene-shared inverse
dispersion per gene. The parameter count remains exactly 5,135,088.

Each receiver cell is represented by two views that share the same encoder
parameters:

1. The target-query stream masks the receiver's selected genes, includes the
   explicit mask channel and allowed metadata, and evolves through all four
   graph blocks.
2. The neighbor key/value stream encodes fully observed cells once and remains
   cell-autonomous and static across graph blocks.

Every block forms queries from the evolving masked target state and keys and
values from static fully observed neighbor states. Only incoming non-self
neighbor-to-target edges are evaluated. Target states are never used as
neighbor keys or values, and context states never receive messages. Therefore
a target's hidden expression cannot leave the target stream and return through
a graph path. The receiver is also excluded from its own context, so its fully
observed context encoding cannot reveal its masked values to itself.

## One visit and one mask

In each global epoch, every one of the 208,696 training cells is a target
exactly once. Its masked-gene count is sampled independently from
`Uniform{1,...,1000}`, and positions are selected without replacement. Empty
target masks are impossible. Neighbor context expression is never masked.

Two complete cores form one optimizer update, giving six updates per epoch.
For each core, all four ranks process disjoint contiguous target shards while
each rank sees that core's complete static context. The two cores are staged
sequentially: the first backward executes under DDP `no_sync`, the second is
synchronized, and the optimizer steps once after both cores. Coverage receipts
must prove no missing or duplicate target, one mask per target, and zero masked
neighbor entries.

For masked entries `M`, the update objective is

```text
sum_{(cell,gene) in M} NB2_NLL(cell,gene) / |M|.
```

Before either backward, the four-rank implementation must all-reduce `|M|`
across all target shards of both cores. Each local two-core NLL sum is scaled
by `world_size / |M|` before ordinary DDP gradient averaging. Thus cells,
cores, ranks, and target shards receive no accidental equal-unit reweighting;
each masked entry has exactly equal weight.

## Fixed validation and stopping

Every validation cell is targeted exactly once using one immutable nonempty
mask derived from the campaign validation namespace. The same masks are
regenerated and checksum-verified at every validation. The primary metric is
the pooled full-constant NB2 NLL: the total NLL sum divided by the total number
of masked validation entries across both cores.

Training uses AdamW with learning rate `1e-4`, weight decay `1e-5`, global
gradient clipping at 1.0, mixed precision for the model, and FP32 likelihood
evaluation outside autocast. ReduceLROnPlateau uses factor 0.5, patience 8,
absolute threshold `1e-4`, and minimum learning rate `1e-6`. Best-checkpoint
tracking begins at epoch 1; stopping patience begins after epoch 50 and is 25
validations with absolute minimum improvement `1e-4`. Epoch 300 is the hard
maximum.

The run keeps atomic latest and validation-best checkpoints while active. Once
the best checkpoint reloads and reproduces its fixed-mask validation metric,
latest is deleted and only best remains.

## Hypotheses and alternatives

Primary optimization hypothesis: the leakage-isolated model will reduce its
fixed-mask pooled validation NB NLL by more than `1e-4` relative to epoch 1
while maintaining finite, non-degenerate gradients.

Mechanistic hypothesis: fully observed local expression context can improve a
masked receiver prediction without allowing the receiver's hidden values to
return through message passing.

Alternatives include no material validation improvement, degradation caused
by static rather than evolving context, a mismatch between NB2 and the count
distribution, or an overly difficult `Uniform{1,...,1000}` masking mixture.
This single run cannot attribute any gain to one mechanism or establish
superiority over the earlier NB run because the masking and routing protocols
differ. A superiority claim requires a matched comparator under this exact
target/mask protocol.

## Acceptance and falsification

Implementation acceptance requires all of the following before launch:

- exact 5,135,088-parameter topology and four independent graph blocks;
- exact once-per-cell target coverage with one nonempty mask per target;
- fully observed neighbor rows and incoming non-self routing only;
- prediction invariance when masked target input values or raw likelihood
  targets are perturbed before the loss is applied;
- no target-to-context or context-update path in any of the four blocks;
- equality of four-rank gradients and a single-process masked-entry-pooled
  reference on deliberately unequal mask counts;
- deterministic fixed validation masks and pooled metric replay;
- finite FP32 likelihood, finite pre-clip gradients for every block and theta,
  AMP/FP32 equivalence within declared tolerances, checkpoint reload, VRAM
  headroom, and at least 20 GiB free storage.

Any missing or duplicate target, empty target mask, masked neighbor value,
self edge, return path, unequal weighting, mask drift, non-finite value, or
failed checkpoint replay falsifies the implementation and blocks launch. The
optimization hypothesis is not supported if the best fixed-mask pooled
validation NLL does not improve on epoch 1 by more than `1e-4`.

## Recorded outputs

Every completed epoch records pooled training loss, all validation metrics,
learning rate, inverse-dispersion summaries, target and mask coverage receipts,
duration, throughput, and peak VRAM. Gradient diagnostics include consecutive
optimizer-step and epoch-to-epoch cosine similarity for the full model, each
of four graph blocks, and the gene-dispersion parameter. Gradient vectors are
never persisted.

The authoritative immutable specification is
[`frozen_task_contract.yaml`](frozen_task_contract.yaml).
