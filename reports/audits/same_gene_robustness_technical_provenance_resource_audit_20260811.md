# Same-gene robustness 기술·provenance·자원 독립 감사

- 감사 시각: 2026-08-11T08:36Z
- 대상 campaign: `cmp_20260810_same_gene_robustness_multiverse_v1`
- immutable 분석 출력: `reports/analyses/same_gene_robustness_20260811`
- 판정: **PASS**
- 범위 제한: 과학적 effect를 재계산하거나 해석하지 않았다. 이 문서는 실행 권위, attempt lineage, bundle 무결성, 기술 controls, 자원, 그래프 불변식, publication/verify-only 증거만 감사한다.

## 1. 결론

140개 production coverage slot은 `7 variants × 5 base model seeds × 4 folds`로 정확히 한 번씩 채워졌고, plan·ledger·registry·job marker·archive bundle·analysis attempt history·`run_verification.csv`가 모두 같은 140개 run을 가리킨다. Production attempt는 전부 attempt 1의 selected success이며 missing, duplicate, failed, superseded production attempt가 없다.

7개 pilot은 각 variant마다 attempt 1 실패와 명시적 recovery attempt 2 성공의 연속 lineage를 가진다. 7개 receipt 모두 attempt 2만 선택하고, attempt 1을 terminal unsuccessful로 보존한다. 14개 pilot bundle 전체가 현재 checksum 검증을 통과했다.

Full coordinator wall time은 1.759419시간이며 contract 예상 1.5–2.0시간 안이다. Registry에 기록된 모델 runtime 합은 5.491576 GPU-hours이고, coordinator가 GPU worker process를 점유한 시간 합은 6.459590 GPU-hours이다. 둘의 정의를 혼동하지 않았다. Peak VRAM은 16.789804GB로 frozen gate 20.5GB보다 3.710196GB 낮다.

분석 publication은 세 번째 분석 invocation에서 성공했고, 별도 `--verify-only` invocation도 exit 0으로 성공했다. Verify-only 전후 9개 output 파일의 SHA-256은 byte-identical하다. Manifest, `_SUCCESS`, 356개 provenance source, 48개 launch source가 모두 현재 bytes와 일치한다.

## 2. 권위 파일과 독립 해시 확인

| 권위 | SHA-256 | 판정 |
|---|---|---:|
| Frozen contract | `8761098cbafa91ae81b53a1c9cd0d8dcd293476d967f74cddde9be7dc5c990e9` | PASS |
| Parent pilot launch | `99bf0a633ecfe8961a584212d50d6b03a12c3ac42552913dd1b1282e36d8a7c1` | PASS |
| Recovery launch | `38fd482c1343f10efc8ec18507b05f5110f226898f236815a97b9671f7269223` | PASS |
| Pilot attempt-1 plan | `0f6d0db7a5bfaf095c0e83759bdac2fd20ecadf8640ce01ee010e267af42055b` | PASS |
| Pilot attempt-1 ledger | `67f177b7e21c41c95fff1a79d95989ac5ae75321cf795ba4f038cb8799efecbd` | PASS |
| Pilot attempt-2 plan | `69515be03ae513e39680e12cd8f00d908b24eb4de7c142b32830c86d748de758` | PASS |
| Pilot attempt-2 ledger | `783c29e5a2ca072961a10bcd916f16cef85a791fd24c55e25a5cf60b1ba703c0` | PASS |
| Full plan | `2c8b08c37ec6997c45417adbb4e74e0f394cb133845129e7a2c3fece66a0d0a1` | PASS |
| Full ledger | `9e2a27d989e3e26e02af1bc667b999e6fced905c8575383aed94aff4ec77ef9e` | PASS |
| Registry DB | `3a5e5b895e74852fee79699345d3df7de4ada3799b806c670a6b641df0db6443` | PASS |
| Canonical analyzer | `0f77667e4e805b2d0b62a6ec75204c366badefb8b62d6bd1f4f1aea3ad11db85` | PASS |
| Pilot recovery amendment | `cfb085ed104e9ca2cf272a06cbded7f0e7804a3ff72f06e573ae3bfe80a3fc75` | PASS |
| Analysis recovery v1 amendment | `7e4053886d402e50d6a417242cfa62edaa6a180f7fc0e3b0b5e56da96311e684` | PASS |
| Analysis recovery v2 amendment | `6f4bdcd1eed3fdd7138e956cada2849dd2ef12f19a4527b968b5314fbabf0b50` | PASS |

Recovery launch의 48개 bound source는 48/48 path·size·SHA가 일치했다. Analysis provenance의 source inventory는 356개 고유 경로, 총 14,153,185 bytes이며 356/356 path·size·SHA가 일치했다. 감사 시 계산한 정렬 source inventory digest는 `d21799f8d4f26df722708a282e2ac56f499bca786fe9353455e92958178b8954`이다.

현재 v2 authority-check 결과는 `verified:true`, `scientific_outcomes_read:false`이다. v1/v2 recovery 전용 테스트는 64/64 PASS했다.

## 3. Production 140-run coverage와 bundle binding

Frozen slot universe는 다음과 같다.

- Variants: `V0`–`V6`, 7개
- Base model seeds: `20260810`, `20261810`, `20262810`, `20263810`, `20264810`, 5개
- Folds: `0`, `1`, `2`, `3`, 4개
- Expected/observed: 140/140

Registry의 resolved trainer seed는 재현 가능한 fold-specific effective seed인 `base model seed + fold`이다. 따라서 registry config를 직접 세면 effective seed가 20개로 보이지만, frozen coverage axis와 materialized job config의 base model seed는 정확히 5개다. 이 둘을 혼동하지 않았다.

| 계층 | 행/객체 수 | 상태 |
|---|---:|---|
| Full plan jobs | 140 | exact slot universe |
| Full ledger jobs | 140 | `completed: 140` |
| Registry full runs | 140 | `completed: 140` |
| Aggregate attempt-history rows | 140 | selected attempt 1 only |
| `run_verification.csv` rows | 140 | all four verification flags true |
| Registry artifact rows | 11,340 | 81 per run, all `present` |

각 slot에서 다음을 직접 대조했다.

1. Plan의 config path와 `expected_config_sha256`.
2. Materialized config의 variant, base seed, fold, attempt.
3. Ledger의 단일 process attempt, return code 0, `marker_and_optional_verifier_passed`.
4. Job marker canonical payload digest, config SHA, run ID, artifact `_SUCCESS` file SHA.
5. Registry status, attempt, fold, artifact path, decoded config, effective seed.
6. Aggregate attempt-history와 `run_verification.csv`의 run/config/plan/scientific ID.
7. Native seven-file manifest와 checkpoint/Jacobian SHA.
8. Registry 81 artifact metadata rows와 archive checksum inventory.

감사 crosswalk digest는 `9b7a2af0cb01da73b0defb260881e349d1c847332562e41a5fa4826c4f003c80`이고, 정렬된 140 run-ID set digest는 `5fa1c3bf5a75b2414e0bf874f86f04c41ae07096e40816b8a8692e94c5fb1f4d`이다.

### Bundle 재검증

Archive 계층의 공식 read-only `verify_run_bundle(require_success_contract=True)`를 140개 전부 다시 실행했다.

- Verified success bundles: 140/140
- Checksum-bound files: 80/run
- Present files: 80/run
- Tombstoned files: 0/run
- Native scientific bundle manifest: 7 files/run
- Registry artifact rows: 81/run; 80 checksum-bound files + checksum manifest 자체
- Current directory files: 82/run; 위 81개 + registry에 등록하지 않는 `_SUCCESS`

Set-level 기술 digest:

- Config SHA set: `7b23cba330a77c0a16d045f9ae5812f1117837d315a11de888336e5b636cdd5a`
- Marker payload SHA set: `6681b28044bdafd0d97b41c1f36dfc8c5ae12d729fbc3321a5d43ebe436c7095`
- `_SUCCESS` file SHA set: `edc6c50964a7ac945164b0216b2fb06804b5198a667dd243a3cab4f09666c6d6`
- Native manifest-payload SHA set: `e8bacfb56a4b34dc57a07dcf9857ea7959002ee2b0e2a2d4254b0f605fb056a0`
- Archive checksum-manifest SHA set: `6f6046059444248cf17b2a4c97680664e494aabde6615a38b9356c836b7db349`
- Registry artifact metadata digest: `612402b41f1fa0469aa416166ce03c16b72acc253928e8ee4408ea377cc6798c`

### Production 기술 controls

140개 `results.json.controls`, registry, `run_verification.csv`를 독립 집계했다.

- `all_outputs_finite`: 140/140 true
- Analytical nonlinear control: 140/140 passed
- Maximum analytical/autograd error: `5.551115123125783e-17`
- Maximum analytical/finite-difference error: `4.968164768470729e-12`
- Checkpoint GPU replay metric/prediction maximum error: 둘 다 `0.0`
- Replay device: 140/140 `cuda`
- Identity oracle actually executed: 140/140; row top-1 fraction `1.0`
- Graph-specific invariants: 140/140 true
- Source/config/data hashes verified: 140/140 true
- Train/validation/test component overlap: 140/140 false
- Outer test untouched: 140/140 true
- Receiver RNA or RNA-derived covariate input: 140/140 false
- Peak-VRAM gate and projected-runtime gate: 140/140 pass
- Maximum Jacobian reconstruction error: `1.0583336741698535e-09`, tolerance `1e-06`
- Fold별 component prediction rows: fold 0/2/3은 7, fold 1은 6; 27개 component axis와 정확히 일치

## 4. Pilot attempt lineage와 receipts

Pilot attempt 1은 7/7 실패했으며 결과 해석 대상이 아니다.

- V0, V1, V3: frozen genes가 해당 fit population의 numerical eligibility를 만족하지 못함
- V2, V5: 더 작은 동일 유형의 numerical eligibility 실패
- V4, V6: 실패 artifact 직렬화 시 non-finite `inf`가 strict JSON에서 거부됨
- Parent ledger 최종 상태: `interrupted`, `retry_required: 7`

Recovery amendment는 각 variant에 대해 immutable attempt 2만 추가했다. 7개 receipt를 canonical digest부터 다시 검증한 결과:

- Receipt: 7/7 valid
- Attempt-history rows: 14; 각 receipt에 정확히 `[1, 2]`
- Attempt 1: 7 failed, 7 superseded, 0 selected
- Attempt 2: 7 completed, 7 selected
- `retry_of`: 7/7 attempt-2가 해당 attempt-1 run을 정확히 지시
- Receipt crosswalk digest: `a81d264f008d1fb933fe9fd2aca8dae571b22aeacef400511639a5dce54018e0`
- Receipt SHA set digest: `4794a8e7e896f6260fdcbcc64ae4c533cc52936fffeca620a11be695117e5bf6`
- Attempt-history SHA set digest: `caabb356ed2c2ca17ce711b08873b9008f7a8fafb6cd3fc40ae4dd232603cd14`

14개 pilot artifact를 공식 archive verifier로 다시 검사했다. 7개 `_FAILED`와 7개 `_SUCCESS`가 모두 checksum-valid이고 tombstone은 없다. 실패 stage에 따라 failed bundle의 checksum-bound file count는 6개 또는 10개이고, selected success bundle은 79개다.

Pilot gate maxima:

- Peak VRAM: `16.789803504943848GB` ≤ `20.5GB`
- Projected full hours/fold: `0.09189721803291466` ≤ `0.25`
- Analytical/autograd error: `5.551115123125783e-17` ≤ `1e-10`
- Analytical/finite-difference error: `4.968164768470729e-12` ≤ `1e-8`
- Checkpoint replay errors: `0.0` ≤ `1e-7`
- All seven: finite, no split overlap, no receiver RNA input, identity oracle executed, graph invariants true, CUDA replay, source hashes verified, outer test untouched, production authorized

## 5. 자원 감사

### Full wall time와 GPU-hours

| 정의 | 값 |
|---|---:|
| Full ledger wall | 6,333.909461 s = 1.759419 h |
| Full supervisor wall, pre/post 포함 | 6,387.378 s = 1.774272 h |
| Registry/model runtime 합 | 19,769.675241 s = 5.491576 GPU-h |
| Coordinator process occupancy 합 | 23,254.522683 s = 6.459590 GPU-h |
| 4-GPU ledger capacity | 7.037677 GPU-h |
| Coordinator occupancy fraction | 91.785819% |

`runs.duration_seconds`와 각 `results.json.duration_seconds`는 140/140 exact match했다. Registry/model runtime은 모델 실행기가 측정한 시간이고, coordinator occupancy는 process 시작부터 verifier 종료까지 GPU slot이 점유된 시간이다. 실제 scheduling 비용을 말할 때는 후자를, 모델 runtime을 말할 때는 전자를 사용해야 한다.

GPU lane별 coordinator occupancy:

- GPU 0: 35 jobs, 1.625548 h, wall 대비 92.391150%
- GPU 1: 34 jobs, 1.571819 h, wall 대비 89.337364%
- GPU 2: 36 jobs, 1.634509 h, wall 대비 92.900480%
- GPU 3: 35 jobs, 1.627714 h, wall 대비 92.514282%

### Variant별 runtime, VRAM, archive bytes

| Variant | Registry GPU-h | Scheduler GPU-h | Median run s | Max run s | Peak VRAM GB | Checksum-declared archive bytes |
|---|---:|---:|---:|---:|---:|---:|
| V0 | 0.772018 | 0.916186 | 137.936 | 149.769 | 7.271592 | 3,604,841,888 |
| V1 | 0.770457 | 0.896031 | 136.791 | 150.303 | 7.271592 | 3,604,838,294 |
| V2 | 0.751979 | 0.882866 | 134.088 | 147.915 | 7.271592 | 3,604,857,554 |
| V3 | 0.789422 | 0.911866 | 141.011 | 166.793 | 7.271592 | 3,604,738,031 |
| V4 | 0.801899 | 0.933137 | 142.177 | 154.988 | 15.250035 | 3,617,334,862 |
| V5 | 0.785996 | 0.926265 | 141.493 | 179.261 | 7.271592 | 3,604,756,124 |
| V6 | 0.819805 | 0.993239 | 143.824 | 170.361 | 16.789804 | 3,695,249,571 |

140개 registry/model runtime의 min/median/mean/max는 각각 `121.210199 / 140.849644 / 141.211966 / 179.261106`초다. Peak VRAM min/median/mean/max는 `7.271592 / 7.271592 / 9.771114 / 16.789804GB`다. 최대값은 V6 runs에서 관측됐고 20.5GB gate의 81.901481%이며 margin은 3.710196GB다.

Pilot registry/model runtime은 failed attempt 1이 0.031610 GPU-h, selected attempt 2가 0.059836 GPU-h다. Coordinator occupancy는 각각 0.050009와 0.120060 GPU-h다.

### Storage bytes

| 범위/정의 | Bytes | GiB |
|---|---:|---:|
| 140 native seven-file manifests 합 | 21,160,963,918 | 19.707683 |
| 140 archive checksum inventories 합 | 25,336,616,324 | 23.596563 |
| 140 current bundle logical bytes | 25,338,962,484 | 23.598748 |
| 140 current bundle allocated bytes | 25,367,375,872 | 23.625210 |
| Full plan projected output | 30,645,627,380 | 28.540965 |
| 7 failed pilot bundles, logical | 292,146,271 | 0.272082 |
| 7 selected pilot bundles, logical | 1,263,845,913 | 1.177048 |
| Prepared robustness data root | 33,356,549,373 | 31.065707 |
| Analysis publication, 7 payload files | 988,477,067 | 0.920591 |
| Analysis publication, 9 total files | 988,478,757 | 0.920593 |

현재 full bundle logical size는 projection보다 5,306,664,896 bytes, 즉 17.316222% 작다. `archive checksum inventories`는 `_SUCCESS`와 checksum manifest 자체를 제외한 immutable archive payload이며, `current bundle logical`은 둘까지 포함한다.

## 6. 실행 chronology

모든 시각은 UTC다.

| 단계 | 시작 | 종료 | 경과 | 기술 결과 |
|---|---|---|---:|---|
| Final graph preparation | 2026-08-10 23:17:00.302 | 23:26:54.248 | 593.946 s | exit 0 |
| Prepared-data verify | 2026-08-10 23:27:41.122 | 23:28:30.270 | 49.148 s | exit 0 |
| Pilot attempt 1 ledger | 2026-08-11 00:20:04.729 | 00:24:59.031 | 294.302 s | 7 retry-required; service was restarted, then stopped |
| Pilot recovery attempt 2 ledger | 2026-08-11 01:28:36.235 | 01:31:10.143 | 153.907 s | 7 completed; supervisor wall 207.067 s |
| Full ledger | 2026-08-11 01:35:49.893 | 03:21:23.802 | 6,333.909 s | 140 completed |
| Analysis attempt 1 | 2026-08-11 03:22:57.614 | 03:48:01.349 | 1,503.735 s | exit 2, registry config API/raw JSON mismatch |
| Analysis attempt 2 | 2026-08-11 04:33:27.327 | 05:29:31.018 | 3,363.691 s | exit 1, int-key component map passed to string-key canonical JSON |
| Analysis attempt 3, v2 | 2026-08-11 05:50:29.514 | 07:01:44.127 | 4,274.613 s | exit 0, publication success |
| Separate verify-only | 2026-08-11 07:03:26.962 | 08:13:24.722 | 4,197.760 s | exit 0, stderr 0 |

Attempt 1은 첫 selected run의 registry configuration reconciliation에서 멈췄다. 자동화된 reader가 첫 run payload를 로드했지만 aggregate를 계산하거나 output을 publish하지 않았고 staging도 남지 않았다.

Attempt 2는 140 run 검증과 `_build_payload` 진입 후 `_verify_component_coverage`에서 멈췄다. 오류는 integer component keys를 string-key-only canonical JSON helper에 전달한 타입 계약 모순이었다. Gates, bootstrap, gene-label null, publication은 시작되지 않았고 output/staging은 남지 않았다.

Attempt 3의 v2 layer는 component dictionaries를 coercion 없이 `tuple(sorted(mapping.items()))`로 비교하며, canonical analyzer 원본 bytes를 변경하지 않는다.

## 7. Prepared graph 감사

현재 bytes에서 preparation verifier의 write-free `--verify-only`를 다시 실행했다.

- Result: `verified:true`
- Root processed fingerprint: `0b28d5c37ef9ae1ae3fbc646065d3d253d14d27b5b616ca0d8e0b80d6353eab5`
- Root manifest SHA: `c01e44e3542c4acc5e5c8c5053711756b3d53ac7536c2242896665b65624c715`
- Prepared variants verified: 8/8; V0–V6가 사용하는 7 variants와 auxiliary `a0_component_cp10k_all`
- Graph audit digest: `59259d23b6a4f56e959cdaf3807ca4f5bc6ebb209cd1bd62b3adb9235eeb51d0`

Verifier가 실제 arrays에서 다시 확인한 항목은 declared content의 path/size/SHA, CSR degree, near/permuted degree 보존, within-FOV permutation bijection, receiver collision 부재, geometry-component/fold isolation, self-edge 부재, within-FOV variant의 cross-FOV edge 부재, native/primary eligibility, QC-induced eligibility, frozen 932-gene identity다.

| Variant | Partition | Active nodes | Near directed edges | Annular directed edges | Near cross-FOV | Annular cross-FOV | Fixed source states |
|---|---|---:|---:|---:|---:|---:|---:|
| V0 | within FOV | 407,999 | 4,296,323 | 4,814,322 | 0 | 0 | 0 |
| V1 | within geometry component | 407,999 | 4,385,108 | 4,845,883 | 165,821 | 371,933 | 0 |
| V2 | within FOV, QC-induced | 394,236 | 4,123,067 | 4,647,370 | 0 | 0 | 1 |
| V3 | within FOV | 407,999 | 4,296,323 | 4,814,322 | 0 | 0 | 0 |
| V4 | within FOV | 407,999 | 4,296,323 | 4,814,322 | 0 | 0 | 0 |
| V5 | within geometry component, QC-induced | 394,236 | 4,207,360 | 4,679,931 | 152,854 | 349,395 | 1 |
| V6 | within FOV | 407,999 | 4,296,323 | 4,814,322 | 0 | 0 | 0 |

모든 variant/slide에서 receiver collision은 0, receiver degree preservation은 true, zero-distance candidate pair는 0이었다. V1/V5의 cross-FOV edge는 오류가 아니라 frozen `within_geometry_component` 정의에서 명시적으로 예상되는 edge이며 component/fold를 넘지 않는다. V2/V5의 fixed source state 1개는 SO_2 FOV 245의 permutation mapping 제약으로 동일 source가 남은 경우다. 두 경우 모두 receiver collision은 0이고 changed-source fraction은 `0.9999957973657889`다.

## 8. Analysis manifest, provenance, verify-only proof

Publication directory에는 정확히 9개 파일만 있다: manifest-declared payload 7개, `analysis_manifest.json`, `_SUCCESS`. Undeclared file은 없다.

| 파일 | Bytes | SHA-256 |
|---|---:|---|
| `aggregate_jacobians.npz` | 987,143,947 | `767c907fc4ee9d7aceec40feeac02b958c5b05d3d36000833b7704022aa31fef` |
| `aggregate_results.json` | 424,428 | `d188f44ac642994e95586cb7ea426eb127284fe2f8325ef1302342bf6fc8e37b` |
| `analysis_provenance.json` | 98,101 | `005fa4318ff28578e88c106c9a47b7cb72d953dbcc8f088a0f68f608b578c01e` |
| `component_metrics.csv` | 688,346 | `02a297a0e40f9ac9d569d8c1b150798421a4a9d92f8338878b5e3d05ea07585e` |
| `eligible_gene_summary.csv` | 55,692 | `0a49cb35bc3c1e6f387c70b193655f28302a31dc1bbb7914726134d4da8422ed` |
| `report.md` | 3,260 | `2ee29bea380a3119d4085c9485036912a9ebe1f84bf919058669461ec308a440` |
| `run_verification.csv` | 63,293 | `bda476437412e6bc63a5115c4c2f6811b410e8fad425aaa54dab4440cca0e069` |

Envelope bindings:

- `analysis_manifest.json` SHA: `73c8f76e7e1ef6b546424217c0f556dce1fc51464ac86fdecc92ed7f928fbbd2`
- Canonical `files_sha256`: `78e7e9781c08e1c53ffb05eda8cff960d0c07f9d9abb6d2f443706201eb5a202`
- `_SUCCESS.payload` canonical digest: `27556c0603f343561ec0a9d8df8255c87b4354acdbcbadecdc7d84c0d2173a30`
- `_SUCCESS` file SHA: `11ef11980bfed909d0991c18477ce63a8c55cb14cfeedea5aea0280a96caf5a6`

각 payload file의 bytes/SHA, manifest의 canonical file-list digest, `_SUCCESS`의 manifest/files binding을 독립 재계산해 모두 PASS했다.

별도 verify-only supervisor는 exit status 0(expected), stderr 0 bytes였다. Stdout payload는 다음과 같다.

```json
{"file_count":7,"manifest_sha256":"73c8f76e7e1ef6b546424217c0f556dce1fc51464ac86fdecc92ed7f928fbbd2","output":"/workspace/Bio-Architectural-Graph-Modeling/reports/analyses/same_gene_robustness_20260811","payload_sha256":"d188f44ac642994e95586cb7ea426eb127284fe2f8325ef1302342bf6fc8e37b","verified":true}
```

Publication 직후 기록한 9개 baseline SHA와 verify-only 종료 후 재계산한 9개 SHA가 모두 동일했다. 따라서 verify-only가 output bytes를 변경하지 않았다는 byte-level 증거가 있다.

## 9. 독립 재검증한 것과 하지 않은 것

독립 재검증:

- 48/48 launch sources와 356/356 analysis provenance sources의 path/size/SHA
- Pilot 7 receipts의 canonical digest, contiguous lineage, registry/plan/launch/config/bundle bindings
- Pilot 14 bundles와 production 140 bundles의 공식 archive checksum/marker/success-contract 검증
- Production 140-slot plan/ledger/registry/marker/attempt-history/CSV/native-manifest crosswalk
- 11,340 registry artifact metadata rows
- Registry/results duration과 VRAM exact equality 및 자원 합계
- Prepared-data write-free verification과 array-level graph invariants
- Analysis manifest의 exact inventory, 모든 file SHA, canonical envelope digests
- v2 authority-check와 v1/v2 focused tests 64개
- Verify-only supervisor exit/stderr/stdout와 output hash 불변성

하지 않은 것:

- Prediction/Jacobian effect의 통계적 재계산 또는 방향·크기 해석
- Gate, bootstrap, gene-label null, gene ranking의 과학적 재평가
- 새로운 학습 또는 artifact 수정

이 감사 보고서는 analyzer가 검증하는 immutable publication inventory 밖 `reports/audits`에 의도적으로 저장했다. Publication directory 안에 이 파일을 복사하거나 추가하면 exact-inventory 검증을 깨뜨린다.
