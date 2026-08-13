# 그래프 대 비그래프 실험: 중간 결과 요약

> **중요:** 이 문서는 2026-08-13 UTC 기준의 중간 진행 보고서입니다. 최종 graph 효과의 확정 결과가 아닙니다.

## 한 문장 요약

이번 실험은 **형태학 정보만 사용하는 모델(no-graph)** 과 **주변 세포의 RNA를 그래프로 전달받는 모델(graph)** 을 같은 조건에서 비교하도록 설계되었습니다. 현재까지는 실험 파이프라인과 1차 하이퍼파라미터 탐색이 정상적으로 진행되었다는 것만 확인되었고, **graph가 no-graph보다 실제로 좋은지는 아직 결론 내릴 단계가 아닙니다.**

## 무엇을 비교했나?

| 조건 | 모델에 추가되는 정보 | 목적 |
|---|---|---|
| `no_graph` | 형태학/이미징 22개 변수 + broad spatial field | 그래프가 없을 때의 공정한 기준선 |
| `observed_near` | 기준선 + 실제 0–25 µm 이웃의 source-state RNA context | 우리가 검증하려는 실제 그래프 |
| `permuted_near` | 기준선 + 거리/차수 조건을 보존한 source-state permutation | 단순한 이웃 분포나 차수 효과 통제 |
| `observed_annular` | 기준선 + 25–50 µm annular context | 가까운 이웃 특이성 검증 |

데이터는 두 CosMx slide의 약 407,999개 cell, 1,000개 biological probe, 22개 morphology/imaging feature, 27개 geometry component를 사용합니다. Receiver의 RNA와 library size는 입력에서 제외하고, graph arm에서만 사전에 정의한 one-hop source-state context를 허용했습니다.

## 실험을 어떻게 공정하게 했나?

- 4개의 geometry outer fold를 사용했습니다.
- 각 fold에서 graph와 no-graph를 **독립적으로** 튜닝했습니다.
- 모든 arm에 같은 모델 용량, epoch 후보, batch size, seed 예산을 적용했습니다.
- Stage A에서 16개 하이퍼파라미터 후보를 먼저 비교하고, Stage B에서 선택된 후보를 추가 seed로 재검증합니다.
- 최종 confirmation에서는 4개 arm × 4개 outer fold × 5개 seed = **80개 실행**을 고정된 선택 receipt로 평가합니다.
- test 결과는 선택 과정에서 읽지 않도록 분리했습니다.
- synthetic positive/null gate를 먼저 통과해야 실제 결과를 해석할 수 있도록 했습니다.

## 현재까지 실제로 끝난 것

### Stage A: 완료

- 256/256 실행 완료
- 실패 0개
- 모든 job의 exit code 정상
- result file과 success marker 누락 없음
- plan과 ledger의 job ID가 정확히 일치
- frozen contract, input manifest, integrity manifest, 실행 소스 hash, synthetic gate hash가 일치
- fold별로 각 arm에서 다음 단계로 넘길 후보 3개가 선택됨

이 단계는 “어떤 hyperparameter 후보를 다음 단계에서 검증할지”를 정하는 단계입니다. 따라서 Stage A 완료만으로 graph의 성능 우위를 의미하지 않습니다.

### Stage B: 진행 중

최신 스냅샷:

- 96/96 완료
- 실패 0개
- 모든 결과와 success marker 확인
- confirmation 단계는 아직 시작하지 않음

Stage B는 Stage A에서 선택된 후보를 추가 seed로 다시 비교해, 특정 seed나 우연한 hyperparameter 선택에 의한 결과인지 확인하는 단계입니다. 이제 Stage B 전체가 끝났지만, 이 수치는 여전히 outer-test가 아닌 validation 결과입니다.

### 완료된 Stage B의 초기 preview

Stage B 완료 후에는 사전에 정한 nested selection 규칙으로 각 outer fold/arm의 configuration을 고정하고, 해당 configuration의 3개 seed validation 값을 평균했습니다. 이 값은 **validation preview**이며 최종 test 성능이 아닙니다.

| arm | 완료 수 | 평균 validation MSE | 평균 validation MAE |
|---|---:|---:|---:|
| `no_graph` | 4 folds × 3 seeds | 0.9452 | 0.5182 |
| `observed_near` | 4 folds × 3 seeds | 0.9070 | 0.5091 |
| `permuted_near` | 4 folds × 3 seeds | 0.9237 | 0.5144 |
| `observed_annular` | 4 folds × 3 seeds | 0.9141 | 0.5131 |

선택된 validation 값만 비교하면 `observed_near`는 `no_graph`보다 MSE가 약 **4.05% 낮고**, `permuted_near`보다 약 **1.81% 낮습니다**. MAE도 각각 약 **1.76%**, **1.04%** 낮습니다. 네 outer fold 모두 `observed_near`가 `no_graph`와 `permuted_near`보다 낮았습니다.

fold별 MSE gain은 다음과 같습니다.

| outer fold | near vs no-graph | near vs permutation |
|---:|---:|---:|
| 0 | 4.12% | 2.17% |
| 1 | 3.46% | 1.89% |
| 2 | 3.80% | 1.38% |
| 3 | 4.69% | 1.72% |

이것은 graph 방향의 꽤 일관된 초기 신호입니다. 하지만 다음 이유로 최종 결과가 아닙니다.

- 이는 outer-test가 아닌 validation metric이며, confidence interval과 component/slide별 재현성 검사가 없습니다.
- fold별 configuration과 epoch를 validation에서 선택했기 때문에 이 수치 자체가 선택 편향을 포함할 수 있습니다.
- MAE 개선폭은 MSE보다 작고, 아직 gene-level/Jacobian 결과가 없습니다.
- 전체 Stage B가 끝난 뒤 선택 receipt를 고정하고, 별도의 80개 confirmation에서 다시 검증해야 합니다.

따라서 현재 가장 정확한 표현은 **“초기 validation preview에서는 observed near graph가 no-graph보다 좋아 보이는 신호가 있지만, 아직 통계적으로 확인된 graph 효과는 아니다”**입니다.

## RNA 이름이 비슷한 유전자끼리의 gradient/Jacobian 분석

추가로 계획한 분석은 학습 gradient norm이 아니라 **gene×gene Jacobian**입니다. 즉,

> source RNA gene A의 입력을 조금 변화시켰을 때 target RNA gene B의 예측값이 얼마나, 어떤 방향으로 변하는가?

를 계산합니다. 구현된 Jacobian은 1,000×1,000 행렬이며, RNA program 단위로 요약합니다.

- 행(row): target RNA program
- 열(column): source RNA program
- `signed_mean_sensitivity`: 방향을 포함한 평균 민감도
- `mean_absolute_sensitivity`: 영향 크기만 본 평균 민감도
- 같은 이름/같은 program 내부 영향과 다른 program 영향의 비율: `same_name_absolute_enrichment`

다만 이 값은 **최종 confirmation의 observed-near 20개 run**에서만 생성하도록 했습니다. 현재 Stage B tune 결과에는 이 Jacobian 파일이 없고, 현재 confirmation도 시작되지 않았으므로 지금 단계에서 “비슷한 RNA 이름끼리 몇 % 더 높다”고 말할 수 있는 숫자는 아직 없습니다. 최종적으로 값이 나오더라도 이는 모델 sensitivity이지 RNA 간 상관, 세포 간 communication, 생물학적 기전 또는 인과효과를 의미하지 않습니다.

### 아직 남은 단계

1. Stage B 96개 전체 완료
2. outer fold별 최종 configuration receipt 고정
3. confirmation 80개 실행
4. component-equal MSE, MAE, slide-stratified bootstrap CI, seed/component heterogeneity 계산
5. permutation, annular, 10–25 µm edge-removal sensitivity 분석
6. 최종 report와 verdict 생성

## 지금 말할 수 있는 결과와 말할 수 없는 결과

### 지금 말할 수 있는 것

- graph/no-graph를 같은 용량과 같은 튜닝 예산으로 비교하는 실험 코드가 실행되고 있습니다.
- Stage A의 256개 실행은 모두 정상 완료되었습니다.
- 현재까지 실패한 Stage B job은 없습니다.
- 데이터 split, 입력 hash, 실행 소스 hash가 사전에 고정된 값과 일치합니다.

### 아직 말하면 안 되는 것

- “graph가 no-graph보다 몇 % 좋다”
- “graph가 생물학적 상호작용이나 인과 효과를 증명한다”
- “다른 환자/slide에도 일반화된다”

최종적으로는 `observed_near`가 independently tuned `no_graph`보다 최소 2% 개선되고, component-bootstrap CI가 0을 넘으며, 두 slide와 대부분의 component/seed에서 일관되게 우세해야 graph-support 판정을 내립니다. 그 조건을 충족하지 않으면 결과는 negative 또는 inconclusive로 보고합니다.

## 이전 결과와의 관계

과거 탐색 결과에는 near context의 개선이 약 1.5–2.2%로 관찰된 적이 있지만, preprocessing과 estimand가 달라 이번 nested-CV 결과와 직접 합칠 수 없습니다. 특히 cell-type/library residual을 제거하면 개선폭이 약 0.3%까지 줄어든 분석도 있어, 이번 실험은 **graph 신호와 shared state/library confounding을 분리하는 것**을 중요한 목적으로 삼았습니다.

## 결론

현재 결론은 다음과 같습니다.

> **실험 시스템과 1차 튜닝은 정상적으로 진행 중이다. 그러나 최종 graph 대 no-graph 성능 차이는 confirmation과 통계 요약이 끝난 뒤에만 판단할 수 있다.**

최종 수치가 나오면 이 문서의 중간 상태를 최종 report 링크와 함께 갱신하겠습니다.
