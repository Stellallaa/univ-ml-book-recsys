# Hybrid 추천과 실험 결과 집계 (D3, D4)

> **핵심 요약:** 검증에서 Hybrid 후보 중 switch-at-3을 선택했지만, Cold Tune 평균 NDCG@10은 MF 0.2744, Hybrid 0.2491이었습니다. 최종 Holdout에서도 Hybrid가 MF를 앞서지 않았습니다.

## 과정

### D3 Hybrid

B4에서 선택한 CB(C17)와 C2/C3에서 선택한 순위 학습 MF 및 Fold-in을 공통 Recommender 인터페이스로 연결했습니다. 두 모델의 원시 점수 범위가 달라 그대로 더하지 않고, 각 사용자에 대해 10,000권 책의 점수를 백분위 순위로 변환했습니다. 사용자 이력 길이에 따라 고정 MF 가중치, 임계값 전환, 점진 전환을 적용하고 총 9개 후보를 비교했습니다.

모든 후보는 `split.train`으로 학습했습니다. 선택 점수는 Warm Validation 1,000명(시드 42)의 NDCG@10과 Cold Tune 2,500명의 이력 1·3·5·10개 평균 NDCG@10을 같은 비중으로 평균한 값입니다. Test와 Cold Final은 설정 선택에 사용하지 않았습니다. 선택 후 최종 평가 단계에서는 train+validation으로 학습한 뒤 test/cold_final에서 점수를 기록했습니다.

### D4 결과 집계

평가 결과 CSV를 Validation과 Holdout 단계로 나누어 수집했습니다. 공통 열에 모델, 평가 목표, 이력 크기, 사용자 수, Precision/Recall/NDCG, to-read 적중 및 해당 시 RMSE를 정리했습니다. 누락된 원본 파일은 건너뛰고, 모델·평가 목표·이력 크기 조합이 중복되면 오류를 내어 중복 행을 숨기지 않습니다. 순위 지표는 인기도, RMSE는 Bias 기준선과의 차이를 계산합니다. 각 단계의 CSV와 Warm/Cold NDCG 비교 그림을 저장합니다.

## 결과

### Hybrid 후보 선택 및 성능

| 검증 결과 | Warm NDCG@10 | Cold Tune 평균 NDCG@10 | 선택 점수 |
|---|---:|---:|---:|
| Hybrid switch-at-3 | 0.075703 | 0.249146 | 0.162425 |
| 단독 MF | 0.075703 | 0.274434 | - |

switch-at-3은 비교한 Hybrid 후보 중 선택됐지만, Cold Tune 평균은 단독 MF보다 낮았습니다. Warm 결과는 동일합니다. 선택 설정은 이력 1개에서 CB, 3개 이상에서 MF를 사용합니다.

| 최종 조건 | 인기도 | CB | MF | Hybrid |
|---|---:|---:|---:|---:|
| Warm Test | 0.032039 | 0.042463 | 0.057218 | 0.057218 |
| Cold Final, 이력 1개 | 0.233752 | 0.159614 | 0.256665 | 0.159614 |
| Cold Final, 이력 3개 | 0.233752 | 0.169103 | 0.273512 | 0.273512 |
| Cold Final, 이력 5개 | 0.233752 | 0.179504 | 0.283455 | 0.283455 |
| Cold Final, 이력 10개 | 0.233752 | 0.221658 | 0.304823 | 0.304823 |

### D4 집계 산출물

- `results/final/validation_comparison.csv`: 27행의 Validation 비교 결과.
- `results/final/holdout_comparison.csv`: 27행의 최종 Holdout 비교 결과.
- `docs/figures/D4_warm_val_comparison.png`, `D4_cold_tune_comparison.png`, `D4_warm_test_comparison.png`, `D4_cold_final_comparison.png`: 단계별 NDCG 비교 그림.
- 순위 성능은 NDCG@10 기준이며, RMSE는 평점 예측 결과로 별도 취급합니다. 최종 조건에서 MF가 제시된 Warm/Cold NDCG 비교를 이끌고, Hybrid는 임계값 전환 뒤 MF와 같은 추천 점수를 냅니다.
## 결론

Hybrid 구현은 기존 CB와 MF를 재사용해 프로젝트 구조 안에서 결합했습니다. 그러나 결합 자체가 성능 개선을 보장하지 않았습니다. switch-at-3은 Hybrid 후보 중 선택됐으나 Cold Tune 평균에서 단독 MF보다 낮았고, 최종 평가에서도 이력 3개 이상에서는 MF와 같으며 이력 1개에서는 CB 점수를 사용해 인기도와 MF보다 낮았습니다. 이 결과를 숨기지 않고 제한된 실험 범위의 결론으로 보고합니다.

D4는 서로 다른 모델의 결과를 동일한 열과 비교 키로 정리해 코호트와 평가 단계를 혼동하지 않게 했습니다. 누락 파일과 중복 결과를 명시적으로 처리해 재실행 시에도 결과 형식을 유지합니다.

## 이미지

- D3 후보 시각 자료: 별도 그래프 파일은 확인되지 않아 candidate_comparison.csv를 표로 사용하고 전환 규칙은 직접 구조도로 표현합니다.
- Validation 후보 수치: `results/hybrid/candidate_comparison.csv`
- 공통 검증 결과: `results/hybrid/validation_models.csv`
- D4 Warm Validation: `docs/figures/D4_warm_val_comparison.png`
- D4 Cold Tune: `docs/figures/D4_cold_tune_comparison.png`
- D4 Warm Test: `docs/figures/D4_warm_test_comparison.png`
- D4 Cold Final: `docs/figures/D4_cold_final_comparison.png`
- 통합 표: `results/final/validation_comparison.csv`, `results/final/holdout_comparison.csv`

## 2분 발표 대본

“저는 콘텐츠 기반 모델과 협업 필터링 모델을 결합하고, 그 결과를 기존 모델과 같은 기준으로 비교했습니다. 먼저 CB의 코사인 점수와 MF 점수는 범위가 달라 그대로 더하면 한쪽이 과도한 영향을 줄 수 있어 사용자별 백분위 순위로 변환했습니다. 고정 가중치, 이력 길이 임계값 전환, 점진 전환 방식으로 9개 후보를 만들고, 학습 데이터만으로 학습한 뒤 Warm Validation과 Cold Tune을 같은 비중으로 평가했습니다. 그 결과 switch-at-3이 Hybrid 후보 중 선택됐습니다. 평점 1개일 때 CB를 사용하고, 3개 이상부터 MF로 바뀌는 방식입니다. 하지만 Cold Tune 평균 NDCG는 Hybrid 0.2491, MF 0.2744여서 Hybrid가 MF보다 좋아졌다고 말할 수 없습니다. 최종 Test에서도 Warm은 MF와 Hybrid가 0.0572로 같았고, Cold 이력 3개 이상에서는 같은 MF 결과를 냈습니다. 이력 1개에서는 CB를 사용해 Hybrid 점수가 0.1596으로 인기도 0.2338보다 낮았습니다. 마지막으로 D4 집계 스크립트가 모델별 결과를 Validation과 Holdout CSV로 통일하고 네 개 비교 그림을 저장합니다. 결합 모델의 선택과 성능 우위는 별개의 주장이라는 점을 확인했습니다.”

## 어려웠던 점과 해결

- **모델 점수 척도 차이:** CB 코사인 점수와 MF 점수를 직접 합산하지 않고 사용자별 백분위 순위로 정규화했습니다.
- **실험 조건 공정성:** 후보마다 같은 학습 분할과 검증 코호트를 사용하고 Test/Cold Final은 선택 과정에서 제외했습니다.
- **Hybrid 후보 선택과 MF 우위의 혼동:** “Hybrid 후보 중 가장 높은 점수”와 “단독 MF보다 나은가”를 따로 보고했습니다.
- **이질적인 결과 파일:** D4에서 모델·goal·history_size를 공통 비교 키로 사용하고 중복 키를 오류로 처리했습니다. 파일이 없는 경우 해당 결과를 건너뜁니다.
- **그래프 해석:** Warm/Cold 및 Validation/Holdout을 별도 그래프로 저장해 서로 다른 코호트의 점수를 한 축에서 섞지 않았습니다.

## 개인별 배운 점

모델 결합은 시스템 통합과 성능 개선을 구별해 평가해야 합니다. 또한 Validation은 설정 선택, Holdout은 최종 보고라는 역할을 지키고, 실험 CSV를 공통 구조로 집계해야 수치와 그래프가 같은 결론을 전달할 수 있다는 점을 배웠습니다.

## 참고한 외부 코드·자료 (출처)

- Hybrid 점수 결합 구현은 저장소의 `src/bookrec/models/hybrid/recommender.py`이며 외부 구현을 복사하지 않았습니다.
- D4 집계 구현과 실행 방법: `scripts/aggregate_results.py`, `results/README.md`.
- D3 결과와 선택 기준: `results/hybrid/D3_hybrid_tuning.md`, `results/hybrid/selected_config.json`.
- 순위 변환 API: SciPy `scipy.stats.rankdata` 문서, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html.
- CB 및 MF 상세한 외부 참고문헌은 각 팀 문서 `docs/parts/content_based.md`, `docs/parts/mf.md`의 출처 항목을 참조합니다.
