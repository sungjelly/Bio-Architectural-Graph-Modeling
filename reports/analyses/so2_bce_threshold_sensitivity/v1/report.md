# Exploratory threshold sensitivity

This post-hoc diagnostic uses retained selected-best validation histograms for all four confirmation runs. The primary threshold remains 0.5; terminal-epoch model selection is unchanged. No threshold was optimized or selected.

![Precision, recall, F1 and positive prediction frequency at four declared cutoffs](threshold_sensitivity.png)

The table reports exact metrics at the displayed upward-aligned cutoff. Brackets give histogram discretization bounds in percentage units at the requested nominal cutoff; these are not statistical confidence intervals. Within-bin positions were not retained. Requested cutoffs 0.1, 0.2 and 0.3 fall inside bins; the 0.5 row exactly matches saved confusion counts. Full-precision bounds are also in thresholds.csv and summary.json.

| Run | Requested | Actual cutoff | Precision | Recall | F1 | Predicted positive |
|---|---:|---:|---:|---:|---:|---:|
| comparator seed 0 | 0.1 | 0.100097656 | 18.618% [18.581, 18.637] | 64.295% [64.295, 64.373] | 28.875% [28.830, 28.905] | 28.033% [28.033, 28.096] |
| comparator seed 0 | 0.2 | 0.200195312 | 32.116% [32.067, 32.140] | 30.921% [30.921, 30.955] | 31.507% [31.483, 31.536] | 7.816% [7.816, 7.831] |
| comparator seed 0 | 0.3 | 0.300048828 | 47.040% [46.965, 47.074] | 17.668% [17.668, 17.692] | 25.688% [25.676, 25.718] | 3.049% [3.049, 3.056] |
| comparator seed 0 | 0.5 | 0.500000000 | 68.700% | 8.852% | 15.683% | 1.046% |
| comparator seed 1 | 0.1 | 0.100097656 | 18.547% [18.509, 18.565] | 64.591% [64.591, 64.666] | 28.819% [28.773, 28.848] | 28.271% [28.271, 28.335] |
| comparator seed 1 | 0.2 | 0.200195312 | 31.885% [31.838, 31.908] | 31.219% [31.219, 31.252] | 31.548% [31.525, 31.577] | 7.948% [7.948, 7.963] |
| comparator seed 1 | 0.3 | 0.300048828 | 46.912% [46.834, 46.949] | 17.689% [17.689, 17.715] | 25.691% [25.679, 25.723] | 3.061% [3.061, 3.068] |
| comparator seed 1 | 0.5 | 0.500000000 | 69.061% | 8.815% | 15.635% | 1.036% |
| incumbent seed 0 | 0.1 | 0.100097656 | 19.693% [19.651, 19.715] | 60.272% [60.272, 60.353] | 29.687% [29.638, 29.721] | 24.845% [24.845, 24.905] |
| incumbent seed 0 | 0.2 | 0.200195312 | 34.106% [34.057, 34.130] | 28.177% [28.177, 28.208] | 30.860% [30.840, 30.887] | 6.707% [6.707, 6.719] |
| incumbent seed 0 | 0.3 | 0.300048828 | 49.052% [48.976, 49.083] | 16.298% [16.298, 16.318] | 24.467% [24.457, 24.493] | 2.697% [2.697, 2.703] |
| incumbent seed 0 | 0.5 | 0.500000000 | 69.507% | 8.428% | 15.033% | 0.984% |
| incumbent seed 1 | 0.1 | 0.100097656 | 18.859% [18.821, 18.877] | 63.279% [63.279, 63.352] | 29.058% [29.013, 29.087] | 27.238% [27.238, 27.299] |
| incumbent seed 1 | 0.2 | 0.200195312 | 32.632% [32.584, 32.656] | 29.994% [29.994, 30.027] | 31.258% [31.236, 31.287] | 7.462% [7.462, 7.475] |
| incumbent seed 1 | 0.3 | 0.300048828 | 48.252% [48.175, 48.289] | 16.624% [16.624, 16.649] | 24.729% [24.719, 24.761] | 2.797% [2.797, 2.803] |
| incumbent seed 1 | 0.5 | 0.500000000 | 70.323% | 8.226% | 14.729% | 0.950% |

Observed validation prevalence is 0.08117838. Predicting every entry positive gives precision 8.118%, recall 100%, and F1 = 2p/(1+p) = 0.15016649. This can exceed a model's F1 at 0.5 while producing many more false positives. F1 ignores true negatives and depends on the operating point; a 0.5 cutoff is not generally F1-optimal for rare positives. A lower F1 than this trivial rule does not by itself establish poor probability ranking or failed optimization. Read precision, recall and prediction frequency together with AP and probability scores. AP summarizes ranking across cutoffs; the constant-score, tied-score AP reference equals prevalence. These retrospective count references do not choose a threshold or constitute an independent fitted baseline. Always-negative precision is undefined and displayed as zero by convention.

| Count-derived reference | TP | FP | TN | FN | Precision | Recall | F1 | Predicted positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| always_positive | 1518702 | 17189505 | 0 | 0 | 8.118% | 100.000% | 15.017% | 100.000% |
| always_negative | 0 | 0 | 17189505 | 1518702 | 0.000% | 0.000% | 0.000% | 0.000% |

Selected-best probability metrics (AP is 4096-bin average precision; all are threshold-independent):

| Run | Selected epoch | BCE | Brier | AP | AUROC |
|---|---:|---:|---:|---:|---:|
| comparator seed 0 | 25 | 0.23841607 | 0.06551386 | 0.30746046 | 0.77514338 |
| comparator seed 1 | 28 | 0.23840900 | 0.06550816 | 0.30765735 | 0.77507983 |
| incumbent seed 0 | 32 | 0.23877644 | 0.06558402 | 0.30601856 | 0.77453581 |
| incumbent seed 1 | 31 | 0.23870651 | 0.06558243 | 0.30565521 | 0.77454903 |

Lowering a cutoff changes the precision/recall tradeoff; it does not improve the model's probability estimates or prove failed optimization. These same-donor, selected-checkpoint observations provide no independent validation of a deployment threshold, graph-specific gain, count reconstruction, or causality. All four runs are shown; optimization seeds are not independent biological replicates.
