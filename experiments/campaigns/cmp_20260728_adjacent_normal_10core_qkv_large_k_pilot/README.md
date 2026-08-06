# Adjacent-normal QKV-GAT k=5,000 resource pilot

This is a one-run implementation and resource diagnostic for the ten-core
production campaign. It uses opaque core alias `ANC-01`, whose verified exact
`k=5,000` graph has the largest directed-edge count in the selected cohort.

The pilot keeps the production dataset, graph identity, 36,749,480-parameter
QKV architecture, optimizer, precision, and exact-attention execution. It
changes only the fixed epoch budget to two and uses one diagnostic evaluation
mask per mode. It has no validation or test partition, is not
conclusion-bearing, and cannot enter the ten-core scientific comparison.

Production may be enqueued only after the pilot completes with two finite
epochs, the expected model and graph identities, an immutable verified bundle,
and acceptable GPU memory and runtime.
