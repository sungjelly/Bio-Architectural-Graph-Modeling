# GeneMAE Jacobian-pair annotation

## Result

A verified 39-by-39 signed gene-to-gene sensitivity matrix was extracted from the seven-seed GeneMAE ensemble. The leading pairs are dominated by collagen/ECM and simple-epithelial keratin genes. Six have direct structural or biochemical evidence, three have shared-program or indirect functional evidence, and ACTA2-PSCA has only a weak unvalidated co-capture. This is not independent validation: the 39 genes were biology-selected, the analysis is held-in, and the upstream graph-gradient structure gate passed in 0/10 cores.

The primary pair score is `0.5*(abs(J[i,j]) + abs(J[j,i]))`. It is an unsigned mutual model-sensitivity magnitude, not a gene-expression correlation coefficient. Rows of `J` are targets and columns are sources.

## Top ten mutual sensitivities and biological annotation

| Rank | Pair | Mutual | Directed J | Evidence | Biological annotation |
|---:|---|---:|---|---|---|
| 1 | `COL1A1–COL3A1` | 0.07119 | COL1A1<-COL3A1 0.09508; COL3A1<-COL1A1 0.04730 | direct ECM crosslink and functional coupling | Type I and III collagen molecules occupy shared fibrils and can form intermolecular covalent crosslinks; Col3a1 loss disrupts collagen-I fibrillogenesis. [Covalent crosslinks between type I and type III collagen molecules](https://pubmed.ncbi.nlm.nih.gov/6120835/); [Type III collagen is crucial for collagen I fibrillogenesis and normal cardiovascular development](https://pubmed.ncbi.nlm.nih.gov/9050868/) |
| 2 | `KRT8–KRT19` | 0.06298 | KRT8<-KRT19 0.07221; KRT19<-KRT8 0.05374 | direct alternative keratin-filament partnership | KRT8 and KRT19 can form type-II/type-I heterodimers and filament arrays; KRT8 expression stabilizes KRT19. [K8 expression stabilizes K19 and supports K8-K19 filament formation](https://pubmed.ncbi.nlm.nih.gov/9857055/); [Cytokeratin expression profile in gastric carcinomas](https://pubmed.ncbi.nlm.nih.gov/15138932/) |
| 3 | `KRT8–KRT18` | 0.03766 | KRT8<-KRT18 0.04068; KRT18<-KRT8 0.03465 | direct heterodimer complex | KRT8 and KRT18 are the canonical type-II/type-I coiled-coil keratin heterodimer and assemble into epithelial intermediate filaments. [Direct structural analysis of the K8-K18 heterodimer](https://pubmed.ncbi.nlm.nih.gov/1691189/); [Reciprocal K8-K18 stabilization and filament rescue](https://pubmed.ncbi.nlm.nih.gov/2471065/); [Cytokeratin expression profile in gastric carcinomas](https://pubmed.ncbi.nlm.nih.gov/15138932/) |
| 4 | `KRT18–KRT19` | 0.03459 | KRT18<-KRT19 0.03301; KRT19<-KRT18 0.03617 | shared epithelial program and functional redundancy; no direct dimer | KRT18 and KRT19 are coexpressed type-I keratins with compensatory epithelial functions, but canonical keratin heterodimers require a type-II partner such as KRT8. [K18 and K19 functional compensation and double-knockout lethality](https://pubmed.ncbi.nlm.nih.gov/11013209/); [Cytokeratin expression profile in gastric carcinomas](https://pubmed.ncbi.nlm.nih.gov/15138932/) |
| 5 | `COL1A1–COL1A2` | 0.03423 | COL1A1<-COL1A2 0.06002; COL1A2<-COL1A1 0.00845 | direct molecular complex | Normal type-I collagen is a heterotrimer containing two alpha-1(I) chains and one alpha-2(I) chain. [Structural basis of type I collagen heterotrimer formation](https://pubmed.ncbi.nlm.nih.gov/28281531/); [Developmental coordination of type I and type III collagen transcripts](https://pubmed.ncbi.nlm.nih.gov/8785585/) |
| 6 | `COL1A1–DCN` | 0.03195 | COL1A1<-DCN 0.04708; DCN<-COL1A1 0.01681 | direct binding and matrix-function evidence | Decorin binds a mapped region of the type-I collagen alpha-1 chain and regulates collagen fibril organization. [Decorin binding site mapped to the type I collagen alpha-1 chain](https://pubmed.ncbi.nlm.nih.gov/10823816/); [Decorin loss disrupts collagen fibril morphology and tissue strength](https://pubmed.ncbi.nlm.nih.gov/9024701/) |
| 7 | `COL3A1–DCN` | 0.02903 | COL3A1<-DCN 0.03288; DCN<-COL3A1 0.02519 | direct in-vitro biochemical interaction | Decorin binds type-III collagen fibrils during fibrillogenesis and changes fibril incorporation and diameter. [Decorin alters type III collagen fibrillogenesis and fibril diameter](https://pubmed.ncbi.nlm.nih.gov/16903686/) |
| 8 | `COL1A2–COL3A1` | 0.02730 | COL1A2<-COL3A1 0.01233; COL3A1<-COL1A2 0.04227 | shared fibril system; chain-specific direct binding not shown | COL1A2 is an obligatory type-I collagen chain, and type-I and type-III collagens form coupled fibrils; direct COL1A2-to-COL3A1 chain binding was not established. [Covalent crosslinks between type I and type III collagen molecules](https://pubmed.ncbi.nlm.nih.gov/6120835/); [Type III collagen is crucial for collagen I fibrillogenesis and normal cardiovascular development](https://pubmed.ncbi.nlm.nih.gov/9050868/); [Developmental coordination of type I and type III collagen transcripts](https://pubmed.ncbi.nlm.nih.gov/8785585/) |
| 9 | `COL1A1–ACTA2` | 0.02269 | COL1A1<-ACTA2 0.03880; ACTA2<-COL1A1 0.00657 | shared myofibroblast program with limited regulatory evidence | COL1A1 and ACTA2 co-mark activated myofibroblast and fibrotic states; no direct protein complex is established. [Collagen and alpha-SMA association and induction in gastric-cancer fibroblasts](https://pubmed.ncbi.nlm.nih.gov/41625254/); [ACTA2 knockdown suppresses TGF-beta-induced COL1A1 and collagen production](https://pubmed.ncbi.nlm.nih.gov/37021273/) |
| 10 | `ACTA2–PSCA` | 0.02143 | ACTA2<-PSCA -0.00830; PSCA<-ACTA2 -0.03456 | weak unvalidated co-capture; compartment association more plausible | One PSCA-FLAG proteomics table listed ACTA2 with two peptides, but the study did not validate or functionally test that hit; gastric epithelial-versus-stromal compartment mixing is better supported. [PSCA-FLAG co-immunoprecipitation mass spectrometry candidate table](https://aacrjournals.org/mcr/article/18/3/501/90114/A-PSCA-PGRN-NF-B-Integrin-4-Axis-Promotes-Prostate); [PSCA gastric-cancer susceptibility and gastric epithelial localization](https://pubmed.ncbi.nlm.nih.gov/18488030/); [ACTA2 signal localizes to gastric stromal and alpha-SMA-positive CAF compartments](https://pmc.ncbi.nlm.nih.gov/articles/PMC10173146/); [IntAct exact ACTA2-PSCA pair query returned no curated interaction](https://www.ebi.ac.uk/Tools/webservices/psicquic/intact/webservices/current/search/query/id%3AP62736%20AND%20id%3AO43653?format=count); [STRING ACTA2-PSCA network query returned no pair edge](https://string-db.org/api/json/network?identifiers=ACTA2%0DPSCA&species=9606&required_score=0) |

`Direct` in the evidence column refers to an independently documented physical/biochemical relationship between the gene products. `Shared program` is weaker: it means a common matrix, epithelial, or cell-state program without a demonstrated direct pairwise interaction. Literature support does not validate the model's sign, direction, graph dependence, or tissue mechanism.

## Literal Jacobian-profile correlations

A separate Spearman analysis asked whether two target rows respond similarly across other source genes, or whether two source columns have similar downstream target profiles. Both pair coordinates were excluded. These are profile similarities, not direct edges.

### Receiver-row profiles

| Rank | Pair | Spearman rho |
|---:|---|---:|
| 1 | `COL1A1–COL3A1` | 0.9329 |
| 2 | `NKG7–CD79A` | 0.9286 |
| 3 | `CD3E–MS4A1` | 0.9260 |
| 4 | `KRT5–LGR5` | 0.9199 |
| 5 | `VWF–LGR5` | 0.9071 |
| 6 | `MKI67–CD163` | 0.9021 |
| 7 | `CD8A–MS4A1` | 0.8988 |
| 8 | `CD163–LGR5` | 0.8940 |
| 9 | `KRT5–LUM` | 0.8938 |
| 10 | `CD3E–CD8A` | 0.8909 |

### Source-column profiles

| Rank | Pair | Spearman rho |
|---:|---|---:|
| 1 | `COL1A1–COL3A1` | 0.8841 |
| 2 | `CDH1–CD3E` | 0.8587 |
| 3 | `COL1A2–COL3A1` | 0.8492 |
| 4 | `COL1A2–LUM` | 0.8428 |
| 5 | `KRT8–KRT19` | 0.8409 |
| 6 | `CD3E–CD8A` | 0.8298 |
| 7 | `KRT8–KRT18` | 0.8286 |
| 8 | `EPCAM–OLFM4` | 0.8231 |
| 9 | `CDH1–CD8A` | 0.8184 |
| 10 | `PCNA–BIRC5` | 0.7966 |

Only `COL1A1–COL3A1` appears in both top-ten profile lists. Several other high profile correlations cross canonical cell lineages, which is evidence that aggregate scale, cell-state mixtures, or low-rank model structure can create high rho without a direct biological relationship. These 741 post-hoc pairwise profile comparisons are descriptive and have no inferential p-values.

## Interpretation limits

- Upstream matrix note: J is the equal-core mean of seven fixed-seed source-style unmasked summed-output gradients on the ten-core adjacent-normal cohort; it is not a replay of MyJJu's unavailable historical slide/checkpoint output.
- The ensemble was selected for held-in partial-gene reconstruction (Huber 0.348210), not patient-held-out graph prediction.
- Its global graph-use improvement was 1.09%, below the frozen 2% gate; only KRT8 passed target-specific graph-use eligibility.
- The upstream graph-gradient structure null failed in every core: the locked-pair statistic was stronger after node-label permutation in 10/10 cores.
- The matrix uses unmasked inputs and mixes the node-wise branch, same-cell graph self-loops, and cross-cell paths up to four hops.
- It covers 39 preselected markers rather than the full 1,000-gene panel; favorable literature overlap is therefore circular.
- No independent cohort, orthogonal measurement, or controlled perturbation tests these model-derived pairs.

## Maximum defensible claim

The listed pairs are literature-annotated, held-in aggregate GeneMAE model-implied sensitivities. They are not biologically verified model relationships, graph-dependent communication edges, mechanisms, or causal effects.

Complete matrices and rankings are provided as CSV files in this directory; `provenance.json` and `verification.json` bind them to the immutable upstream artifact.
