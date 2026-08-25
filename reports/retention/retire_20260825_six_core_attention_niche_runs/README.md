# Six-core attention-niche run-bundle retirement

Decision ID: `retire_20260825_six_core_attention_niche_runs`

Status: planned; apply receipt pending.

On 2026-08-25, the repository owner explicitly requested deletion of all run
artifacts created by campaign
`cmp_20260825_six_core_attention_routing_niches` to recover local storage. The
owner accepted that the scientific tables, static figures, partial diagnostics,
logs, resolved run configurations, and terminal markers would need to be
recreated by rerunning the retained code.

This decision covers exactly the campaign's seven registered run IDs:

- `r_20260825T051807Z_825125bb_s000_f00_a01_058e2607`
- `r_20260825T053454Z_0da9fbf2_s000_f00_a01_69a07db0`
- `r_20260825T055100Z_0da9fbf2_s000_f00_a01_ead1d2c2`
- `r_20260825T081845Z_0da9fbf2_s000_f00_a01_dc19543d`
- `r_20260825T105008Z_0da9fbf2_s000_f00_a01_21554c05`
- `r_20260825T110043Z_0da9fbf2_s000_f00_a01_5e477ab9`
- `r_20260825T120155Z_0da9fbf2_s000_f00_a01_a41036ce`

The four upstream trained-model runs and their final checkpoints are expressly
outside this decision and must remain unchanged.

The immutable plan contains all 244 original artifact identities and all seven
terminal markers. Of those artifact rows, 240 are currently present and four
were already tombstoned by
`cleanup_20260825_relative_qkv_final_only_v2`. The new action plans to remove
240 registered files plus seven markers, totaling 47,115,175,143 apparent
bytes. The earlier four tombstones and their prior decision lineage remain
unchanged and are not counted as newly deleted.

The current generic bundle verifier already reports the two runs containing
those four earlier diagnostics/metadata tombstones as invalid because its
file-level exception is restricted to checkpoint and prediction paths. This is
a pre-existing verifier limitation, not evidence that the registered
tombstones or remaining files changed. Full retirement replaces each absent
bundle with an external receipt that is accepted only if the exact receipt,
complete original inventory, deletion plan, run status, terminal-marker
identity, and absent canonical root all verify.

The operation preserves run IDs, five failed and two completed run statuses,
queue and failure history, aliases, categories, metrics, scientific and
reproduction IDs, original artifact paths, artifact checksums, and retention
classes. Present artifact rows transition through
`retention_deletion_pending` to `deleted_by_retention`; registry rows are never
removed. An SQLite-consistent pre-deletion snapshot is required before any
payload is unlinked.

Plan SHA-256:

```text
f152f35df6a8f227d3ba70a03f5b1cf7039091d81196910c4bd4c42ad3c20c0d
```

The applied command is the same exact seven-run invocation used for planning,
with `--apply` appended. The generated `application_receipt.json`, per-run
receipts, registry verification, measured free-space delta, and final status
will be recorded here after application.
