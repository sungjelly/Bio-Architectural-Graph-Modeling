# Configuration

## Composition

Configuration groups live under `configs/<group>/<name>.yaml`. A root file can
declare ordered defaults:

```yaml
defaults:
  - model: g1
  - masking: p_n_b
  - dataset: normal_core_cosmx
  - features: cosmx_morphology_topology
  - graph: k16_r75_mutual
  - trainer: spatial_benchmark_locked
  - evaluation: masked_expression_regression_v1
  - launcher: local_one_gpu
```

Each group file contains a mapping named for its group. Defaults are
deep-merged in order, then the root mapping is deep-merged last. `defaults`
does not appear in the resolved configuration. Missing files, cycles, unknown
group values, and invalid cross-field combinations are errors. Save the fully
resolved mapping as `config.resolved.yaml` for every attempt.

The baseline example is `configs/base.yaml`. The concrete topology-only and
edge-conditioned variants are
`configs/experiment/edge_feature_ablation_g1.yaml` and
`configs/experiment/edge_feature_ablation_g2.yaml`.

## Parameter ownership

| Parameter | Owning group |
|---|---|
| model family, hidden/embedding dimensions, graph depth, attention heads | `model` |
| masking type, rates, schedule, warm-up, mask-bundle settings | `masking` |
| dataset ID/version, preprocessing version, split ID/fingerprint | `dataset` |
| node/edge feature definitions and `use_edge_features` | `features` |
| `neighbor_k`, radius, symmetry, edge construction and graph restrictions | `graph` |
| optimizer, learning rate, batch size/unit, epochs, precision, checkpoint monitor | `trainer` |
| metrics, uncertainty unit, evaluation splits, sealed-test policy | `evaluation` |
| seed, fold, attempt, campaign and variant labels | experiment root |
| lifecycle stage, study axis, retention class, classification confidence, source batch | top-level `classification` |
| GPU request, worker concurrency, heartbeat, disk threshold, paths | `launcher` |

Do not duplicate a parameter across groups to make overrides convenient.
`embedding_dim` belongs to `model`; `neighbor_k` belongs to `graph`; edge
feature use and definitions belong to `features`; seed and fold are execution
dimensions rather than variant dimensions.

Classification controls discovery and interpretation, not model behavior. New
experiment definitions should declare it explicitly:

```yaml
classification:
  lifecycle_stage: exploratory_screen
  study_axis: edge_feature_ablation
  retention_class: retain_exploratory_evidence
  classification_confidence: high
  source_batch: cmp_edge_feature_ablation
```

Controlled lifecycle stages are `diagnostic`, `exploratory_screen`,
`validation_confirmation`, `locked_final`, `posthoc_evaluation`, and
`unknown`. Missing classification is not guessed from a filename, timestamp,
or metric: it resolves to an explicit unknown/unclassified category.

`configs/schema/experiment_v1.yaml` defines required fields and cross-field
checks. In particular, G1 requires `use_edge_features: false`; G2 requires
`use_edge_features: true` and a non-empty edge schema. Dataset and split IDs
must resolve in `data/registry/`, and the primary metric must exist in the
versioned metric registry.

## Current benchmark example

The example reuses actual, previously validated spatial-benchmark values:

- G1 topology-only GATv2 or G2 edge-conditioned GATv2;
- P+N+B masking;
- protected normal-core CosMx prepared dataset and split;
- k=16, radius 75 µm, mutual graph;
- 512 hidden/embedding dimensions and two graph layers;
- 64-dimensional G2 edge embedding;
- learning rate 0.0003, one complete split graph per optimization batch,
  up to 200 epochs, early-stopping patience 25;
- `val/masked_huber` as the primary minimized metric;
- seed and fold explicitly at the experiment level.

This example is not authorization to reopen the legacy sealed test set or
enqueue training. Review and register a campaign first.

## Validation and provenance

The queue creates the run record and scratch bundle first so an invalid attempt
remains auditable, then validates before starting any training subprocess.
Validation checks types, required fields, metric references,
dataset/split registry references, feature/model consistency, graph
constraints, output-root safety, and scientific ownership.

Resolved configs are immutable attempt inputs. `scientific_id` removes only
the declared execution fields; `repro_id` additionally binds code, dirty state,
data, split, preprocessing, and environment. See
`docs/experiment_protocol.md` for the exact contract.

For a future worker run, the resolved configuration also drives
`run_semantics_from_configuration`: it registers the date-free semantic alias,
campaign → stage → study-axis → variant → execution category, and explicit
knownness for seed/fold/attempt before training. After successful artifact
publication, the worker automatically indexes and verifies the checkpoint.
For imported history, source-index evidence overrides placeholders; the
normal-core fold and attempt remain unknown.

Path values should be project-relative or obtained from
`spatial_benchmark.paths`. Local overrides use `BAGM_*_ROOT` environment
variables. Never place credentials, direct patient identifiers, or a server
home directory in tracked YAML.
