# Same-gene RNA robustness multiverse v1 — 상세 한국어 결과

작성일: 2026-08-11 UTC

캠페인: `cmp_20260810_same_gene_robustness_multiverse_v1`

상태: **140/140 전체 학습 완료, 불변 집계 출판 완료, verify-only 및 2단계 독립 수치 감사 통과**

## 한 문장 결론

V0 secondary gene-label null에서 공간적으로 가까운 세포의 **같은 이름 RNA가 무작위 이름보다 훨씬 강하게 정렬**됐고, same-name absolute enrichment gate는 V0–V5 primary 전처리에서 일관되게 통과했다(V6 secondary도 같은 방향). 그러나 같은 이름이 각 RNA 행에서 충분히 독점적으로 가장 높은 값만 차지한다는 **strict selectivity 주장은 지지되지 않았다**.

좀 더 정확히 말하면 다음과 같다.

| 질문 | 사전 고정 기준 | 결과 | 판정 |
|---|---|---|---|
| 같은 이름 대각의 절댓값이 일반 비대각보다 큰가? | median \|diagonal\| / median \|off-diagonal\| ≥ 2 | V0–V6에서 2.958–3.474 | **강하게 지지** |
| 같은 이름이 그 행의 1위인가? | 932개 중 ≥25% | V0 24.79%; 다른 변형 25.64–28.00% | V0 실패, V1–V6 개별 기준 통과 |
| 같은 이름이 그 행의 상위 1%(상위 10개)인가? | 932개 중 ≥50% | 모든 변형 39.48–44.96% | **모든 변형 실패** |
| 두 row 기준을 동시에 만족하는가? | top-1 ≥25% **그리고** top-10 ≥50% | V0–V6에서 5개 base-seed aggregate 모두 실패 | **robust gate failure** |
| 가까운 RNA가 morphology-only보다 예측을 ≥2% 개선하는가? | component-equal gain ≥2%, ≥3/4 folds | V0–V6에서 0.305–1.543% | **모든 변형 실패** |
| 12 epoch 예산 부족이 위 실패를 설명했는가? | saturation + prediction/row attribution 동시 만족 | V0 40/40 saturated, 오히려 gain −0.0616 percentage point | **설명하지 못함** |

따라서 사용자의 질문을 가장 직접적으로 답하면 다음과 같다.

> **“같은 이름끼리 절댓값이 상대적으로 큰가?”에는 예. “같은 이름의 signed 값만 높거나, 그 행에서 같은 이름만 독점적으로 높은가?”에는 아니오.**

## 1. 여기서 `same-name`과 “round RNA”가 뜻하는 것

현재 데이터에는 `round RNA_type 1`/`round RNA_type 2`라는 별도 assay-round annotation이 없다. 이번 검증은 이를 다음과 같이 조작적으로 정의했다.

- 행: receiver 세포에서 예측할 target probe/RNA 이름
- 열: 주변 source 세포들의 평균 RNA feature 이름
- 같은 이름: target 행 이름과 source 열 이름이 같은 대각 원소
- near graph: `(0, 25]` µm, 최대 `k=12`, self edge 제외
- annular graph: `(25, 50]` µm, 최대 `k=12`
- Jacobian 방향: `target row × source column`

즉, 이번 결과는 **서로 다른 assay round의 동일 RNA 측정치 재현성**을 직접 검정한 것이 아니라, “주변 세포의 같은 이름 RNA feature를 바꿀 때 receiver의 같은 이름 RNA 예측이 얼마나 민감하게 변하는가”를 검정한 것이다. 실제 round별 데이터가 따로 존재한다면 round ID를 보존한 직접 paired 분석이 추가로 필요하다.

Jacobians는 주변 이웃의 **평균 standardized expression**에 대한 민감도다. 개별 sender 세포 하나의 효과가 아니며, 개별 sender 단위로 옮기려면 graph degree와 source/target scaling을 함께 고려해야 한다.

## 2. 데이터와 공간 분할

| 항목 | 값 |
|---|---:|
| SO_1 cells | 161,596 |
| SO_2 cells | 246,403 |
| 총 cells | 407,999 |
| FOV | 451 |
| V0 matched receiver cells | 396,622 |
| QC-pass active/source cells (V2/V5) | 394,236 |
| V2/V5 matched degree-eligible evaluation receivers | 384,521 |
| frozen geometry components | 27 |
| outer folds | 4 |
| prediction probe 수 | 1,000 |
| frozen Jacobian probe 수 | 932 |

FOV 중심 거리 0.75 mm 연결 성분을 slide 내부에서만 만들고, component 전체를 한 fold에 배정했다. 모든 train/validation/test graph는 split 경계를 넘지 않게 별도로 검증됐다.

V0 test component의 eligible cell 수는 306–49,503으로 매우 불균형하다. median은 12,227이다. 이 때문에 component-equal과 cell-weighted estimand가 달라질 수 있으며, 사전 지정 primary는 component-equal이다.

중요한 한계는 27개 component가 환자 27명이 아니라 **두 장의 slide 안 공간 섬 27개**라는 점이다. 네 fold 모두 같은 두 slide에서 나온 component를 나누므로 patient-held-out 검증이 아니다.

## 3. 전체 실험 설계

각 variant마다 5개 base model seed와 4개 outer fold를 사용했다. 실제 trainer/optimizer seed는 `base seed + fold`이므로 variant마다 서로 다른 execution seed는 20개다. 각 process는 아래 네 arm을 모두 학습했다.

1. `morphology_only`
2. `observed_near`
3. `observed_annular`
4. `within_fov_permuted_near`

총 scientific slots는 다음과 같다.

- 7 variants × 5 base model seeds × 4 folds = **140 GPU processes**
- 140 processes × 4 arms = **560 selected fresh-final-refit arm outputs**
- tuning trajectory와 fresh final-refit을 각각 세면 optimizer training trajectories는 **1,120개**
- candidate epochs: 12, 24, 48, 96, 192
- validation-selected epoch로 fresh final refit
- 같은 final-refit trajectory의 epoch 12를 anchor로 저장
- AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 4096
- additive neighbor MLP: full linear neighbor skip + rank-64 GELU residual
- process 4-arm parameter-count 합: 3,476,192 (`morphology_only` 23,000; 각 neighbor arm 1,151,064)

### 3.1 Variant 정의

| ID | 역할 | 변경점 | 직접 비교 시 주의 |
|---|---|---|---|
| V0 | primary | within-FOV, 407,999 source nodes와 antecedent-matched degree-eligible receiver cohort 396,622, log1p; antecedent-exact observed arms + corrected permutation | 기본 해석 |
| V1 | primary | 같은 geometry component 안에서 FOV 경계를 넘는 edge 허용 | target/cohort는 V0와 동일 |
| V2 | primary | 394,236 QC-pass active receiver/source cell로 graph 재구축 | matched degree-eligible evaluation cohort는 384,521 |
| V3 | primary | log1p count 대신 panel CP10k 후 log1p | response scale/estimand 변경 |
| V4 | primary | train-only panel library-size residual | conditional-expression estimand |
| V5 | primary | cross-FOV component graph + 394,236 QC-pass active receiver/source cell + CP10k | matched degree-eligible evaluation cohort는 384,521; 여러 변경을 결합 |
| V6 | mechanistic secondary | train-only vendor cell type + library-size residual | RNA-derived type 사용; V0–V5 결론을 rescue할 수 없음 |

V3–V6의 절대 MSE는 response 정의가 달라 V0와 직접 비교하면 안 된다. 각 variant 안의 arm contrast가 해석 단위다.

## 4. 실행·복구·검증 계보

### 4.1 Pilot

첫 pilot attempt 7개는 모두 기술적으로 실패했다.

- V0/V1/V3: capped pilot fit population에서 frozen genes 16개가 fold-local numerical eligibility check를 통과하지 못함
- V2/V5: 같은 유형의 numerical eligibility check에서 7개 gene이 통과하지 못함
- V4/V6: identity oracle의 이론적 `+∞` ratio를 strict JSON으로 직렬화하지 못함

이는 scientific gate 결과가 아니라 technical implementation failure였다. 실패 bundle은 보존했고 해석에서 제외했다. 직접 failure trigger는 위 두 failure 유형(numerical eligibility와 nonfinite JSON)이었다. recovery amendment는 frozen 932 gene axis를 pilot cap에서 재선택하지 않도록 하고 무한 oracle ratio를 `null + positive_infinity flag`로 표현했으며, checkpoint normalization/replay, pre-transform eligibility provenance, uncapped pilot `final_train`, publication·lineage guard도 함께 고정했다. 해당 attempt 2는 7/7 통과했다.

### 4.2 Production

- 140/140 completed
- full production 실패/재시도: 0
- return code 0: 140/140
- selected attempt: 모두 attempt 1
- GPU 배정: GPU0 35, GPU1 34, GPU2 36, GPU3 35 jobs
- 시작: 2026-08-11 01:35:49 UTC
- 종료: 2026-08-11 03:21:23 UTC
- 4-GPU wall time: 1시간 45분 34초

### 4.3 Aggregate analysis recovery

과학 결과를 바꾸지 않는 두 개의 analysis-only API adapter가 필요했다.

1. `Registry.get_run()`이 `config_json`을 decoded `config` mapping으로 반환하는데 analyzer가 raw 문자열을 기대한 문제
2. component coverage 비교에서 integer-key mapping을 canonical JSON으로 직접 집합화한 문제

두 실패 모두 final output/staging을 출판하지 않았고, 원 analyzer와 registry source는 byte-identical로 보존했다. 별도 immutable amendments와 wrappers로 decoded registry-config adapter, component-coverage equality view, 그리고 각 recovery provenance hook만 고쳤다.

최종 v2 publication과 동일 경로 `verify-only`가 모두 성공했다.

| 검증 항목 | 값 |
|---|---|
| verify-only | `verified=true` |
| verify stderr | 0 bytes |
| `_SUCCESS` SHA-256 | `11ef11980bfed909d0991c18477ce63a8c55cb14cfeedea5aea0280a96caf5a6` |
| manifest SHA-256 | `73c8f76e7e1ef6b546424217c0f556dce1fc51464ac86fdecc92ed7f928fbbd2` |
| aggregate JSON SHA-256 | `d188f44ac642994e95586cb7ea426eb127284fe2f8325ef1302342bf6fc8e37b` |
| publication inventory | manifest-declared payload 7개 + manifest/`_SUCCESS` = 총 9개 |
| verify-only 시간 | 약 1시간 10분 |

## 5. Prediction 결과

### 5.1 Component-equal MSE

| Variant | Morphology | Near | Annular | Permuted near | Cells |
|---|---:|---:|---:|---:|---:|
| V0 | 0.914319 | 0.900246 | 0.905930 | 0.914142 | 396,622 |
| V1 | 0.914319 | 0.900213 | 0.905684 | 0.913980 | 396,622 |
| V2 | 0.919777 | 0.906808 | 0.912115 | 0.919963 | 384,521 |
| V3 | 0.960825 | 0.954567 | 0.957991 | 0.963165 | 396,622 |
| V4 | 0.948271 | 0.935712 | 0.941601 | 0.949369 | 396,622 |
| V5 | 0.966769 | 0.961689 | 0.964798 | 0.970052 | 384,521 |
| V6 | 0.955734 | 0.952822 | 0.956219 | 0.961381 | 396,622 |

Near는 모든 variant에서 annular보다 descriptive MSE가 낮았다. 하지만 primary gate는 near-vs-morphology와 near-vs-matched-permutation이다.

### 5.2 Relative gain과 bootstrap

양수 gain은 near가 해당 baseline보다 낮은 MSE라는 뜻이다. CI는 두 slide 안 27개 geometry component를 slide-stratified resampling한 조건부 interval이며 환자-level CI가 아니다.

| Variant | Near vs morph gain | 95% component bootstrap | Cell-weighted gain | Near vs permutation gain | 95% component bootstrap |
|---|---:|---:|---:|---:|---:|
| V0 | 1.5392% | [0.7111%, 2.4572%] | 2.1315% | 1.5201% | [1.0579%, 2.0278%] |
| V1 | 1.5429% | [0.7126%, 2.4612%] | 2.1372% | 1.5063% | [1.0504%, 1.9986%] |
| V2 | 1.4100% | [0.5834%, 2.3227%] | 2.0512% | 1.4300% | [1.0157%, 1.8711%] |
| V3 | 0.6513% | [0.1937%, 1.2192%] | 0.9363% | 0.8926% | [0.6747%, 1.1470%] |
| V4 | 1.3244% | [0.6421%, 2.0475%] | 1.8708% | 1.4385% | [1.0093%, 1.8873%] |
| V5 | 0.5255% | [0.0667%, 1.0855%] | 0.8438% | 0.8621% | [0.6585%, 1.0957%] |
| V6 | 0.3048% | [−0.0863%, 0.7101%] | 0.6739% | 0.8903% | [0.5995%, 1.1962%] |

핵심 판정:

- near-vs-morphology 2% gate: V0–V6 모두 consensus 실패, seed 0/5 pass
- near-vs-permutation 1% gate: V0/V1/V2/V4 pass, V3/V5/V6 fail
- V0–V5 near-vs-permutation 결론: `preprocessing_sensitive`
- near는 seed-평균 component 비교에서 permutation보다 모든 variant의 27/27 components에서 낮았다. V3/V5/V6 실패는 방향이 반대여서가 아니라 1% magnitude 기준에 못 미쳐서다.

### 5.3 Weighting 민감도

| Variant | Component-equal | Cell-weighted | Fold-equal | Slide-equal |
|---|---:|---:|---:|---:|
| V0 | 1.539% | 2.131% | 1.537% | 1.540% |
| V1 | 1.543% | 2.137% | 1.540% | 1.543% |
| V2 | 1.410% | 2.051% | 1.410% | 1.411% |
| V3 | 0.651% | 0.936% | 0.654% | 0.651% |
| V4 | 1.324% | 1.871% | 1.327% | 1.322% |
| V5 | 0.525% | 0.844% | 0.530% | 0.525% |
| V6 | 0.305% | 0.674% | 0.312% | 0.303% |

V0/V1/V2는 cell-weighted 값만 2%를 넘는다. 이는 component 크기가 306–49,503 cells로 매우 다르고, 큰 component 213과 204의 V0 gain이 각각 5.27%, 3.34%이기 때문이다. 반면 component-equal, fold-equal, slide-equal은 약 1.54%로 일치한다.

따라서 “세포 한 개를 동일 가중한 평균 성능”에서는 2%를 넘지만, 사전 지정된 “공간 component 하나를 동일 가중한 일반화 성능”에서는 넘지 못했다. 이것은 코드 오류가 아니라 estimand 차이다.

V0에서 SO_1/SO_2 component-equal gain은 각각 1.551%, 1.527%로 방향은 두 slide에서 일치했다.

## 6. Same-name Jacobian 결과

### 6.1 Variant별 absolute enrichment와 row selectivity

top-1 최소 통과 count는 233/932, top-1%는 `ceil(0.01×932)=10`이므로 최소 466/932가 필요하다.

| Variant | Diag/off ratio | Top-1 | Top-10 | Top-1 margin | Top-10 deficit | Positive / Negative diag | Median signed diag |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 | 3.4125 | 231/932 (24.79%) | 401/932 (43.03%) | −2 genes | −65 genes | 389 / 543 | −0.00412 |
| V1 | 3.4676 | 242/932 (25.97%) | 406/932 (43.56%) | +9 genes | −60 genes | 391 / 541 | −0.00423 |
| V2 | 3.4743 | 239/932 (25.64%) | 408/932 (43.78%) | +6 genes | −58 genes | 384 / 548 | −0.00534 |
| V3 | 2.9581 | 241/932 (25.86%) | 374/932 (40.13%) | +8 genes | −92 genes | 487 / 445 | +0.00126 |
| V4 | 3.4133 | 247/932 (26.50%) | 406/932 (43.56%) | +14 genes | −60 genes | 399 / 533 | −0.00364 |
| V5 | 3.1345 | 249/932 (26.72%) | 368/932 (39.48%) | +16 genes | −98 genes | 463 / 469 | −0.00022 |
| V6 | 3.4696 | 261/932 (28.00%) | 419/932 (44.96%) | +28 genes | −47 genes | 400 / 532 | −0.00376 |

모든 variant에서 diag/off ≥2는 base-seed aggregate 5/5와 consensus가 모두 통과했다. 그러나 모든 variant에서 top-10 <50%였고, strict conjunction을 통과한 base-seed aggregate는 0/5였다.

따라서 정확한 해석은 다음과 같다.

- 같은 이름 대각은 전체적으로 비대각보다 훨씬 큰 절댓값을 가진다.
- 그러나 932 target 중 55.0–60.5%에서는 같은 이름이 그 행의 상위 10개 안에도 들지 않는다.
- V0에서는 75.21%의 target에서 같은 이름보다 절댓값이 큰 다른 source가 적어도 하나 있다.
- V0 median absolute row rank는 23.5다. 전형적인 same-name 값은 상위 약 2.5% 수준이지, 항상 1위나 상위 1%는 아니다.

### 6.2 부호

V0 대각 932개 중 389개만 양수이고 543개는 음수다. median signed diagonal도 −0.00412다. V3를 제외하면 대부분 variant에서 음수 대각이 더 많다.

이번 primary gate는 부호가 아니라 **절댓값 magnitude**를 썼다. 따라서 3.4배 enrichment를 “같은 RNA가 주변에서 높으면 receiver에서도 양의 방향으로 올라간다” 또는 “positive regulation”으로 해석하면 안 된다. 음수도 모델의 conditional sensitivity이지 생물학적 inhibition 증거가 아니다.

### 6.3 V0 weighting/slide 민감도

| Weighting | Diag/off | Top-1 | Top-10 | Positive fraction |
|---|---:|---:|---:|---:|
| Fold-equal, component-equal primary | 3.4125 | 24.79% | 43.03% | 41.74% |
| Fold-equal, cell-weighted | 3.3943 | 24.79% | 43.24% | 41.74% |
| Slide-equal | 3.4048 | 24.79% | 43.03% | 41.74% |
| SO_1 only | 3.3827 | 24.89% | 42.27% | 42.81% |
| SO_2 only | 3.4041 | 25.00% | 43.24% | 41.31% |

Same-name magnitude enrichment와 strict-row 실패는 weighting과 두 slide에서 모두 같은 방향이다.

### 6.4 Linear/nonlinear decomposition

| V0 selected component | Diag/off | Top-1 | Top-10 | Positive fraction | Median signed |
|---|---:|---:|---:|---:|---:|
| Total | 3.4125 | 24.79% | 43.03% | 41.74% | −0.00412 |
| Full linear skip | 3.8582 | 28.76% | 46.67% | 36.37% | −0.01058 |
| GELU nonlinear residual | 2.8680 | 9.44% | 26.18% | 90.24% | +0.00819 |

`total = linear + nonlinear`은 원소별로 성립하지만 ratio나 rank 같은 요약치는 더할 수 없다. 독립 감사에서 42개 decomposition matrix의 최대 원소 오차는 `1.67e-16`이었다.

## 7. Frozen 7-gate 결과

`P`는 consensus와 5/5 base-seed aggregates가 gate를 통과한 `robust_pass`, `F`는 consensus가 실패하고 base-seed aggregate도 0/5가 통과한 `robust_gate_failure`다.

| Variant | Near vs morph ≥2% | Near vs perm ≥1% | Diag/off ≥2 | Strict row | Near/perm diag ≥1.25 | Fold stability | Technical |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 | F | P | P | F | P | P | P |
| V1 | F | P | P | F | P | P | P |
| V2 | F | P | P | F | P | P | P |
| V3 | F | F | P | F | P | P | P |
| V4 | F | P | P | F | P | P | P |
| V5 | F | F | P | F | P | P | P |
| V6 secondary | F | F | P | F | P | P | P |

V0–V5 cross-preprocessing 판정:

- near-vs-morphology failure: `robust_across_preprocessing`, 공통 분류 `robust_gate_failure`
- same-name diagonal enrichment: `robust_across_preprocessing`, 공통 분류 `robust_pass`
- strict row selectivity failure: `robust_across_preprocessing`, 공통 분류 `robust_gate_failure`
- near/permuted diagonal: `robust_across_preprocessing`, 공통 분류 `robust_pass`
- fold stability: `robust_across_preprocessing`, 공통 분류 `robust_pass`
- technical validity: `robust_across_preprocessing`, 공통 분류 `robust_pass`
- near-vs-permutation prediction: `preprocessing_sensitive`

여기서 `robust_across_preprocessing`은 반드시 공통 분류와 함께 읽어야 한다. “모든 변형이 성공”을 뜻하지 않고, 모든 변형이 같은 pass/fail 분류를 가졌다는 뜻이다.

## 8. Observed vs permuted diagonal과 fold 안정성

| Variant | Observed/permuted median diagonal ratio | Median fold-pair Spearman | ≥3/4 fold sign-consistent genes |
|---|---:|---:|---:|
| V0 | 2.4907 | 0.8744 | 94.53% |
| V1 | 2.5101 | 0.8745 | 94.42% |
| V2 | 2.3979 | 0.8729 | 94.53% |
| V3 | 2.1282 | 0.8862 | 91.95% |
| V4 | 2.4328 | 0.8757 | 93.99% |
| V5 | 2.2371 | 0.8834 | 91.74% |
| V6 | 2.5477 | 0.8755 | 93.99% |

모든 variant에서 4/4 fold ratio가 1.25 이상이었고, stability 기준 Spearman ≥0.70 및 sign fraction ≥0.75를 넉넉히 넘었다.

## 9. Deterministic gene-label null

V0 selected observed-near consensus matrix에서 source-column gene 이름만 derangement했다. 두 family를 각각 10,000회 사용했다.

| Null family | Statistic | Observed | Null mean | Null 95% reference range | Upper-tail p |
|---|---|---:|---:|---:|---:|
| Full label | Median \|diag\| | 0.016731 | 0.004903 | [0.004545, 0.005278] | 1/10001 |
| Full label | Top-1 fraction | 0.247854 | 0.000815 | [0, 0.003219] | 1/10001 |
| Full label | Top-10 fraction | 0.430258 | 0.010285 | [0.004292, 0.017167] | 1/10001 |
| Prevalence/SD matched | Median \|diag\| | 0.016731 | 0.005205 | [0.004838, 0.005581] | 1/10001 |
| Prevalence/SD matched | Top-1 fraction | 0.247854 | 0.000908 | [0, 0.003219] | 1/10001 |
| Prevalence/SD matched | Top-10 fraction | 0.430258 | 0.017402 | [0.009657, 0.025751] | 1/10001 |

`null 95%`는 관측 효과의 confidence interval이 아니라 null draw의 reference range다. 가능한 최소 add-one p-value가 나왔으므로 이름 alignment는 random relabeling으로 설명되기 어렵다. prevalence와 target SD를 맞춘 29 strata에서도 동일했다.

그러나 이 secondary null은 frozen strict threshold를 바꾸거나 rescue하지 않는다. “random보다 훨씬 강함”과 “절대적 exclusivity 50% 통과”는 서로 다른 질문이다.

## 10. 12→192 epoch 최적화 horizon 및 saturation 감사

V0 morphology와 observed-near의 5개 base model seeds × 4 folds × 2 arms = 40 trajectories를 검사했다.

- saturated: 40/40
- observed-near selected epoch: 24에서 20/20
- morphology selected epoch: 48에서 3, 96에서 10, 192에서 7
- 96→192 relative validation gain 최대: 0.0321%, frozen 0.1% saturation 기준 미만

| 항목 | 값 |
|---|---:|
| Anchor-12 near-vs-morph gain | 1.6008% |
| Selected near-vs-morph gain | 1.5392% |
| Selected − anchor gain | −0.0616 percentage point |
| Paired component bootstrap | [−0.0946, −0.0328] percentage point |
| Positive bootstrap draws | 0/20,000 |

Row 지표는 top-1이 231/932로 그대로였고, top-10은 392→401로 늘었지만 466 threshold에 크게 못 미쳤다. seed별 row attribution 조건도 통과하지 않았다.

Frozen verdict는 `budget_explanation_not_supported`다. 즉 12 epoch가 충분했다고 일반화하는 것이 아니라, **이번 고정 optimizer/model/schedule에서 192까지 연장해도 두 원래 실패를 설명하지 못했다**는 뜻이다.

## 11. Permutation graph audit

| Variant | Fixed source states | Near cross-FOV directed edges | Annular cross-FOV directed edges | Receiver collisions | Degree preserved |
|---|---:|---:|---:|---:|---:|
| V0 | 0 | 0 | 0 | 0 | Yes |
| V1 | 0 | 165,821 | 371,933 | 0 | Yes |
| V2 | 1 | 0 | 0 | 0 | Yes |
| V3 | 0 | 0 | 0 | 0 | Yes |
| V4 | 0 | 0 | 0 | 0 | Yes |
| V5 | 1 | 152,854 | 349,395 | 0 | Yes |
| V6 | 0 | 0 | 0 | 0 | Yes |

V2/V5의 한 fixed source는 SO_2 FOV 245에서 발생했다. QC-pass active 11개 중 receiver collision 없이 완전 derangement가 수학적으로 불가능한 Hall violation이 있어 237,946개 source state 중 정확히 1개만 고정됐다. receiver collision은 여전히 0이다.

## 12. 기술 controls와 자원

### 12.1 140-run 기술 검증

| 항목 | 관측 최대/결과 | Gate |
|---|---:|---:|
| Peak VRAM | 16.789804 GiB | ≤20.5 GiB |
| Analytical vs autograd error | 5.55e−17 | ≤1e−10 |
| Analytical vs finite-difference error | 4.97e−12 | ≤1e−8 |
| Checkpoint replay metric error | 0 | ≤1e−7 |
| Checkpoint replay prediction error | 0 | ≤1e−7 |
| Identity oracle row top-1 | 1.0 | 1.0 |
| Split component overlaps | 0 | 0 |
| Receiver RNA/derived covariate as model input | False 140/140 | False |
| Bundle/native manifest/registry/marker verification | 140/140 | 140/140 |
| Max published Jacobian reconstruction error | 1.06e−9 | ≤1e−6 |

### 12.2 Resource usage

| Variant | Mean registry/model duration/run | Min–max | Max VRAM |
|---|---:|---:|---:|
| V0 | 139.0 s | 130.4–149.8 s | 7.27 GiB |
| V1 | 138.7 s | 129.6–150.3 s | 7.27 GiB |
| V2 | 135.4 s | 121.2–147.9 s | 7.27 GiB |
| V3 | 142.1 s | 130.9–166.8 s | 7.27 GiB |
| V4 | 144.3 s | 135.5–155.0 s | 15.25 GiB |
| V5 | 141.5 s | 123.7–179.3 s | 7.27 GiB |
| V6 | 147.6 s | 136.2–170.4 s | 16.79 GiB |

- 합산 registry/model runtime: 5.491576 GPU-hours
- coordinator GPU-slot 점유: 6.460 GPU-hours
- 4-GPU scheduler 점유율: 91.79%
- 4-GPU production wall time: 1.759 hours
- prepared robustness data: 33,356,549,373 bytes = 31.066 GiB
- selected 140 run artifact bytes: 25,338,962,484 bytes = 23.599 GiB
- immutable aggregate output: 988,478,757 bytes = 0.921 GiB
- verify 완료 후 2026-08-11 08:14 UTC 확인 시 네 GPU 모두 idle

## 13. Gene-level descriptive 예시

아래는 V0 consensus에서 absolute diagonal이 큰 probe의 예다. 이는 descriptive ranking이며 gene-level p-value나 biological mechanism 주장이 아니다.

상위 양수 예: `LYZ`, `REG1A`, `PSCA`, `IGHA1`, `OLFM4`, `IGHM`, `PIGR`, `MALAT1`, `LTF`, `DMBT1`, `MHC I`, `MT2A`, `IGHG1`, `IGKC`, `IFI6`.

절댓값이 큰 음수 예: `NOTCH1`, `EFNA5`, `PTGES`, `PTGES2`, `KRT23`, `CEACAM1`, `TIE1`, `CXCR6`, `RARG`, `ADIPOQ`, `MYL7`, `IL17RB`, `ST6GAL1`, `HCK`, `PDGFD`.

큰 양수 diagonal probe들과 음수 same-name sensitivity가 함께 존재한다는 점이 absolute enrichment와 signed biological effect를 구분해야 하는 이유다.

## 14. 선행 실험과 비교

| 결과 세대 | Near vs morph | Near vs perm | Diag/off | Top-1 | Top-10 | 결론 |
|---|---:|---:|---:|---:|---:|---|
| Linear antecedent | 2.23% | 1.30% | 2.852 | 24.5% | 41.1% | row fail |
| Nonlinear 12-epoch | 1.59% | 1.41% | 3.265 | 24.4% | 42.5% | prediction/row fail |
| Single-seed convergence | 1.523% | 1.454% | 3.387 | 24.8% | 42.8% | budget explanation 반대 |
| 이번 V0 5-seed | 1.539% | 1.520% | 3.413 | 24.8% | 43.0% | 선행 핵심 결론과 일관 |

이번 V0 permutation은 antecedent의 degree/collision 문제를 수정한 control이라 near-vs-permutation 숫자를 완전 동일-input replication으로 부르면 안 된다. observed near/annular와 primary cohort는 antecedent-exact다.

## 15. 독립 감사

### 15.1 Aggregate four-file audit

Campaign analyzer나 `spatial_benchmark`를 import하지 않은 standalone auditor가 다음을 직접 재계산했다.

- 42,140/42,140 checks PASS
- 7,560 component rows
- 147 NPZ keys, 140 float64 1,000×1,000 matrices
- prediction weighting, bootstrap, Jacobian rank/sign/ratio, budget, full-label null
- 20,000 budget draws elementwise maximum error 0
- full-label null 30,000 values elementwise maximum error 0
- 42 `total = linear + nonlinear` matrices maximum error `1.67e−16`

### 15.2 Run-authority audit

140 immutable run bundles와 V0 prepared arrays를 원자료로 사용한 두 번째 auditor가 다음을 재구성했다.

- 67,768/67,768 checks PASS
- selected/anchor component MSE, MAE, cell counts
- 모든 seed/fold Jacobian gates와 fold stability
- 40 validation trajectories
- matched prevalence/SD null 29 strata, 30,000 values, maximum error 0
- published near/permuted 932×932 aggregate maximum error 0
- complete 1,000-gene order SHA 일치

감사 산출물:

- `reports/audits/same_gene_robustness_20260811_standalone_audit.md`
- `reports/audits/same_gene_robustness_20260811_standalone_audit.json`
- `reports/audits/same_gene_robustness_20260811_run_authority_audit.md`
- `reports/audits/same_gene_robustness_20260811_run_authority_audit.json`

### 15.3 Technical/provenance/resource audit

세 번째 독립 감사는 effect 값을 재계산하지 않고 plan·ledger·registry·bundle·pilot lineage·graph·source hash·publication envelope와 자원을 검사했다.

- production 140/140 exact slots, pilot 14/14 lineage bundles PASS
- production bundle 140개와 pilot bundle 14개 공식 archive verifier PASS
- registry artifact rows 11,340개 exact
- launch sources 48/48 및 analysis provenance sources 356/356 hash PASS
- prepared variants 8/8 write-free verify PASS
- publication 9개 파일의 manifest/`_SUCCESS` binding PASS
- 별도 verify-only 전후 9개 파일 모두 byte-identical

산출물: `reports/audits/same_gene_robustness_technical_provenance_resource_audit_20260811.md`

## 16. 허용되는 주장과 금지되는 주장

### 데이터가 지지하는 주장

1. V0 selected matrix에서 same-name Jacobian absolute magnitude는 full 및 prevalence/SD-matched gene-label null보다 강하다.
2. V0–V5에서 diag/off와 observed/permuted diagonal enrichment gate는 graph boundary, QC, CP10k, library residual, 5개 base model seeds(variant마다 20개 distinct execution seeds)에 일관되게 통과했고 V6 secondary도 같은 방향이었다.
3. strict row exclusivity는 V0–V5 전처리 전반에서 안정적으로 실패했고 V6 secondary도 실패했다.
4. Near는 morphology보다 평균 MSE가 낮았지만 V0–V5 모두 component-equal 2% practical gate를 통과하지 못했다. Near-vs-permutation 1% gate는 V0/V1/V2/V4만 통과하고 V3/V5는 실패해 preprocessing-sensitive였으며, V6 secondary도 실패했다.
5. V0 40-trajectory audit에서 12-epoch budget은 고정 모델/optimizer의 prediction 및 row 실패를 설명하지 못했다.

### 데이터가 지지하지 않는 주장

- 세포 간 RNA communication이 증명됐다.
- 같은 RNA가 이웃 세포의 같은 RNA를 인과적으로 조절한다.
- 대각이 주로 양의 regulation이다.
- 환자 일반화가 검증됐다.
- 개별 sender cell의 효과가 측정됐다.
- 실제 assay round type 1/type 2의 기술적 일치성이 검증됐다.
- 상위 gene이 통계적으로 유의한 biological mechanism이다.

## 17. 주요 한계

1. **두 slide뿐이다.** 27 geometry components는 patient replicates가 아니다.
2. **Post-hoc robustness study다.** antecedent test outcomes를 본 뒤 설계했으며 독립 replication이 아니다.
3. **Transductive masked reconstruction이다.** receiver self RNA는 제외하지만, test cell의 RNA가 다른 test receiver의 neighbor feature로 쓰일 수 있다. unseen-patient inductive prediction이 아니다.
4. **공간/상태 confounding이 남는다.** cell type, tissue compartment, spatial autocorrelation, technical field, segmentation spillover가 same-name alignment를 만들 수 있다.
5. **절댓값 gate다.** 932 대각 중 V0에서 543개가 음수이므로 positive signaling 주장이 아니다.
6. **표준화·aggregation 의존적이다.** 저장된 Jacobian은 standardized aggregate neighbor mean에 대한 값이라 source/target standardization에 의존한다. 이를 individual-sender 민감도로 환산할 때는 sender degree와 scaling을 추가로 고려해야 한다.
7. **Prediction과 Jacobian gene axis가 다르다.** MSE는 1,000 probes, Jacobian gate는 frozen 932 probes다.
8. **Variant 간 response가 다르다.** CP10k와 residual variants의 절대 MSE를 V0와 직접 비교할 수 없다.
9. **V6 cell type은 RNA-derived다.** 독립 임상 covariate adjustment가 아니다.
10. **Bootstrap은 fitted model을 고정한다.** training uncertainty, seed population, 환자 sampling uncertainty를 모두 포함하지 않는다.

## 18. 다음 단계

이번 outcome을 본 뒤 선택한 follow-up은 기존 confirmatory family의 일부로 간주할 수 없다. 아래 실험은 별도 frozen exploratory 또는 새로운 confirmatory contract와 pilot을 만든 뒤 수행해야 한다.

1. 두 방향 leave-one-slide-out로 slide generalization stress test
2. 새 환자/새 slide의 진짜 독립 replication
3. native graph eligibility sensitivity
4. prespecified distance-band sensitivity
5. cell type/compartment 안에서만 하는 spatial permutation/null bank
6. sender-level graph model 또는 contact/barrier-aware graph
7. 실제 assay round annotation 확보 후 동일 RNA의 type-1/type-2 paired measurement 분석

현재 결과만으로 가장 우선순위가 높은 것은 **새 slide/환자 독립 replication**과 **cell-state/compartment-conditioned null**이다.

## 19. 불변 결과와 재현 경로

- Immutable aggregate: `reports/analyses/same_gene_robustness_20260811/`
- Canonical report: `reports/analyses/same_gene_robustness_20260811/report.md`
- Aggregate JSON: `reports/analyses/same_gene_robustness_20260811/aggregate_results.json`
- Aggregate NPZ: `reports/analyses/same_gene_robustness_20260811/aggregate_jacobians.npz`
- Component rows: `reports/analyses/same_gene_robustness_20260811/component_metrics.csv`
- Eligible genes: `reports/analyses/same_gene_robustness_20260811/eligible_gene_summary.csv`
- Run verification: `reports/analyses/same_gene_robustness_20260811/run_verification.csv`
- Provenance: `reports/analyses/same_gene_robustness_20260811/analysis_provenance.json`

Frozen authority:

- Contract SHA-256: `8761098cbafa91ae81b53a1c9cd0d8dcd293476d967f74cddde9be7dc5c990e9`
- Child launch SHA-256: `38fd482c1343f10efc8ec18507b05f5110f226898f236815a97b9671f7269223`
- Full plan SHA-256: `2c8b08c37ec6997c45417adbb4e74e0f394cb133845129e7a2c3fece66a0d0a1`
- Pilot recovery amendment SHA-256: `cfb085ed104e9ca2cf272a06cbded7f0e7804a3ff72f06e573ae3bfe80a3fc75`
- Analysis recovery v1 SHA-256: `7e4053886d402e50d6a417242cfa62edaa6a180f7fc0e3b0b5e56da96311e684`
- Analysis recovery v2 SHA-256: `6f4bdcd1eed3fdd7138e956cada2849dd2ef12f19a4527b968b5314fbabf0b50`

이 상세 보고서는 immutable publication inventory 바깥에 두었다. 따라서 원 결과 manifest와 `_SUCCESS` 해시는 변하지 않는다.
