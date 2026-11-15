# 발표 슬라이드 구성안 - Goodbooks-10k 도서 추천 시스템

> 머신러닝 전공 수업 발표용입니다. 총 11장, 약 7분 40초를 목표로 합니다. 슬라이드별 시간·담당자·발표 목표는 표시하지 않습니다. 다른 팀원의 `docs/parts` 문서에 정리된 과정·결과·결론·이미지·발표 대본·어려움·배운 점·출처를 통합해 시각 자료 중심으로 구성합니다.

## 전체 흐름과 팀 작업 반영

데이터·평가(A) → 콘텐츠 기반 추천·군집(B) 및 MF·Fold-in(C) → EDA·Hybrid·결과 집계(D) → 공통 평가 결과와 한계. 개별 파트를 나열하지 않고 하나의 실험 흐름으로 연결합니다. D의 작업은 D1 EDA, D3 CB/MF Hybrid, D4 결과 집계로 반영합니다.

## 슬라이드 구성

### 1. Goodbooks-10k 도서 추천 시스템
- **핵심 내용:** 희소한 평점으로 기존 독자와 신규 독자에게 책을 추천하고 실제 결과를 비교합니다.
- **시각 자료:** 책 표지와 사용자 → 평점 이력 → 추천 목록 아이콘 흐름.
- **추천 레이아웃:** 제목과 팀명만 중앙에 배치하고 설명은 최소화합니다.
- **자료:** `README.md`, `docs/timeline.html`

### 2. 데이터, 전처리와 Warm/Cold 분할
- **핵심 내용:** Goodbooks-10k는 평점 5,976,479개, 사용자 53,424명, 책 10,000권이며 관측 밀도는 1.12%입니다. Goodreads ID 연결 및 태그 정제 후, 시드 42로 사용자 5,000명을 Cold 평가용으로 분리해 Cold Tune/Cold Final에 각 2,500명씩 배정합니다. 이 사용자는 기본 모델 학습에서 제외됩니다. 나머지 Warm 사용자 48,424명은 사용자별 평점 행 순서로 train/validation/test를 70/15/15로 나눕니다. 타임스탬프가 없어 사용자별 원본 행 순서를 시간의 근사치로 사용합니다.
- **시각 자료:** 데이터 규모 숫자 4개 + 원본 → ID 연결·태그 정제 → Warm/Cold 분할도. Warm에는 사용자별 70/15/15, Cold에는 Tune/Final 각 2,500명을 표시합니다.
- **추천 레이아웃:** 상단에 데이터 규모, 중앙에 전처리 흐름, 하단에 Warm/Cold 분기. Cold 사용자의 평점은 학습에서 제외되고 신규 사용자 평가에 쓰인다는 설명은 한 줄만 둡니다.
- **자료:** `docs/data.md`, `docs/pipeline.md`, `src/bookrec/data/loaders.py`, `src/bookrec/data/tag_filter.py`, `src/bookrec/data/split.py`

### 3. 전처리 후 EDA: 희소성, 긴 꼬리, 태그 편중
- **핵심 내용:** 관측 밀도는 1.12%이고 사용자·책별 평점 수가 크게 다릅니다. `to-read` 태그는 10,000권 중 9,983권에 나타납니다. 이 특성은 인기도 기준선과 콘텐츠 모델, Cold 평가를 함께 살펴야 하는 배경입니다.
- **시각 자료:** 평점 분포, 사용자/책 평점 수 로그 분포, 상위 태그 그래프 중 3개.
- **추천 레이아웃:** 그래프 세 개를 동등하게 배치하고 “희소성 / 긴 꼬리 / 태그 편중”만 강조합니다.
- **자료:** `notebooks/eda/D1_eda_graphs.ipynb`, `docs/figures/d1_rating_distribution.png`, `d1_user_rating_distribution_log.png`, `d1_book_rating_distribution_log.png`, `d1_top_20_tags.png`
### 4. 기준선과 공통 평가
- **핵심 내용:** 인기도는 학습 평점 수, Goodreads는 외부 평균 평점으로 정렬하고 Bias는 전역 평균과 사용자·책 편향으로 평점을 예측합니다. 평점 4 이상을 정답으로 Top-10의 Precision@10, Recall@10, NDCG@10, to_read_hit@10을 평가합니다. RMSE는 평점 예측에만 적용합니다.
- **시각 자료:** 기준선 카드와 정답 → Top-10 → 순위 지표 흐름도. RMSE는 별도 경로.
- **추천 레이아웃:** 왼쪽은 세 기준선, 오른쪽은 추천 평가 과정.
- **자료:** `src/bookrec/models/baselines/`, `src/bookrec/evaluation/`

### 5. Content-Based Filtering과 K-means
- **핵심 내용:** 태그로 희소 책 벡터를 만들고 선호 책 벡터를 합쳐 독자 프로필을 만듭니다. 선택 CB는 binary, min_df=1, positive 가중치와 코사인 유사도를 사용하며 프로필이 비면 인기도로 대체합니다. K-means K=10의 표본 실루엣은 0.0614입니다. 이는 군집 내부 지표이지 추천 품질 점수가 아닙니다. SVD 후보는 확인 표본 개선이 없어 채택하지 않았습니다.
- **시각 자료:** 태그 → 책 벡터 → 독자 프로필 → 유사 책 추천. 옆에 군집 크기/대표 책과 실루엣을 별도로 표시합니다.
- **추천 레이아웃:** CB 흐름을 크게, 군집 탐색은 보조 패널로 배치합니다.
- **자료:** `docs/parts/content_based.md`, `src/bookrec/models/content_based/`, `results/content_based/B4_selected_config.json`, `B5_selected_clustering.json`, `B5_improvement_selected_config.json`

### 6. MF 별점 예측: 학습 곡선과 과적합
- **핵심 내용:** Matrix Factorization은 독자와 책을 잠재 벡터로 표현하고, 전체 평균·독자 편향·책 편향·두 벡터의 내적으로 평점을 예측합니다. 팀 C는 NumPy로 gradient descent와 L2 정규화를 구현했습니다. 학습 오차는 계속 감소하지만 검증 오차가 다시 오르는 구간이 나타나며, 검증 오차가 가장 낮은 시점에서 early stopping을 적용했습니다.
- **시각 자료:** 실제 학습·검증 곡선 `C1_mf_loss_curve.png`. 학습 곡선 옆에 간단한 예측식 `r_hat = 평균 + 독자 편향 + 책 편향 + p_u · q_i`.
- **추천 레이아웃:** 왼쪽 2/3에 학습·검증 곡선, 오른쪽 1/3에 예측식과 “검증 오차가 오르기 전 멈춤”을 표시합니다.
- **자료:** `docs/parts/mf.md`, `docs/figures/C1_mf_loss_curve.png`, `results/collaborative_filtering/C1_rating_history.csv`, `src/bookrec/models/collaborative_filtering/mf.py`

### 7. 평점 예측에서 순위 추천으로
- **핵심 내용:** 평점이 관측된 책만 학습한 초기 MF는 검증 NDCG@10이 0.0031로 인기도 기준선 0.0382보다 낮았습니다. 순위 학습에서는 관측되지 않은 책 일부를 음성 예시로 뽑아 목표값 0으로 학습합니다. 탐색 결과에서 음성 샘플 비율이 커진 후보는 추천 NDCG가 높고 평점 RMSE가 나빠지는 trade-off가 관찰됐습니다. 단, 그래프의 후보는 다른 파라미터도 달라지므로 비율 하나의 인과 효과로 단정하지 않습니다. 이에 평점 예측용과 순위 추천용을 분리했습니다. 평점 예측은 ALS 검증 RMSE 0.8392를 선택했고 최종 Test RMSE는 0.8337입니다. 최종 추천 MF는 gradient descent, 요인 128, 학습률 0.01, 정규화 0.05, 음성 비율 8, 4 epoch입니다.
- **시각 자료:** `C2_mf_rmse_vs_ndcg.png`로 RMSE와 NDCG trade-off를 보이고, 하단에 “평점 예측: ALS / 순위 추천: MF + 음성 샘플링” 두 카드를 둡니다. 초기 rating-only 결과 0.0031 대 0.0382는 작은 callout으로 표시합니다.
- **추천 레이아웃:** 왼쪽 그래프, 오른쪽 두 학습 목적 비교. 최종 하이퍼파라미터는 작은 카드 하나로 제한합니다.
- **자료:** `docs/parts/mf.md`, `results/collaborative_filtering/C2_mf_tuning.md`, `results/collaborative_filtering/C2_selected_config.json`, `docs/figures/C2_mf_rmse_vs_ndcg.png`, `results/evaluation/mf_rmse_test.csv`

### 8. 신규 독자 개인화: Fold-in
- **핵심 내용:** 학습에 없던 신규 독자는 기존 MF 사용자 벡터가 없습니다. 책 벡터를 고정하고 신규 독자의 짧은 평점 이력으로 사용자 벡터와 편향을 Ridge 방식으로 계산합니다. Fold-in 정규화 계수는 Cold Tune에서 비교해 10을 선택했습니다. 최종 Test 성능 수치는 10번 슬라이드에서 비교합니다.
- **시각 자료:** `docs/figures/C3_fold_in_cold.png`. 두 패널은 Cold Tune 검증 결과입니다. 왼쪽은 이력 1·3·5·10개에 따른 MF Fold-in NDCG와 인기도 비교, 오른쪽은 정규화 계수 비교 및 선택값 10을 보여줍니다.
- **추천 레이아웃:** 상단에 `책 벡터 고정 → 신규 평점 이력 → Ridge로 사용자 벡터 계산 → 추천` 흐름도를 짧게 두고, 아래에 C3 검증 그래프를 배치합니다. 그래프에는 “Cold Tune 검증”이라고 표시합니다.
- **자료:** `docs/parts/mf.md`, `results/collaborative_filtering/C3_fold_in.md`, `results/collaborative_filtering/C3_fold_in_reg.csv`, `docs/figures/C3_fold_in_cold.png`

### 9. Hybrid: CB와 MF 결합을 검증하다
- **핵심 내용:** CB 코사인 점수와 MF 점수는 사용자별 백분위 순위로 바꾼 뒤 결합했습니다. 고정 가중치·이력 길이 전환·점진 전환 9개 후보를 `train`으로 학습하고 Validation/Cold Tune에서 비교했습니다. 선택 기준은 Warm Validation 1,000명과 Cold Tune 이력 1·3·5·10개 평균 NDCG@10의 같은 비중 평균입니다. 선택된 `switch-at-3`은 이력 1개에서 CB, 3개 이상에서 MF를 사용합니다. Hybrid 후보 중 가장 높았지만 단독 MF보다 성능이 개선되지는 않았습니다.
- **시각 자료:** 위쪽에는 `CB 점수 + MF 점수 → 사용자별 백분위 순위 → 이력 길이별 전환` 구조도. 아래쪽에는 Validation 비교표를 둡니다.

  | 검증 모델 | Warm NDCG@10 | Cold Tune 평균 NDCG@10 |
  |---|---:|---:|
  | Hybrid switch-at-3 | 0.075703 | 0.249146 |
  | 단독 MF | 0.075703 | 0.274434 |

  표 아래에 “Hybrid 후보 중 선택됐지만 MF 대비 개선은 확인되지 않음”을 강조합니다. D3 후보 비교 전용 그림은 없으므로 `candidate_comparison.csv`와 `validation_models.csv`에서 표를 구성합니다.
- **추천 레이아웃:** 상단 절반에 점수 결합과 전환 규칙, 하단 절반에 두 모델 비교표와 결론 한 줄을 둡니다. 최종 Test/Cold Final 그래프는 다음 슬라이드에서 보여 중복을 피합니다.
- **자료:** `src/bookrec/models/hybrid/recommender.py`, `results/hybrid/D3_hybrid_tuning.md`, `results/hybrid/candidate_comparison.csv`, `results/hybrid/validation_models.csv`, `results/hybrid/selected_config.json`

### 10. 실험 설계와 최종 성능
- **핵심 내용:** Validation/Cold Tune은 설정 선택에, Test/Cold Final은 최종 비교에만 사용합니다. Warm Test는 정답 사용자 48,266명, Cold Final은 2,498명입니다. 최종 NDCG@10:
  
  | 모델 | Warm Test | Cold Final 1개 | Cold Final 10개 |
  |---|---:|---:|---:|
  | 인기도 | 0.0320 | 0.2338 | 0.2338 |
  | Goodreads | 0.0022 | 0.0101 | 0.0101 |
  | CB | 0.0425 | 0.1596 | 0.2217 |
  | MF | **0.0572** | **0.2567** | **0.3048** |
  | Hybrid | 0.0572 | 0.1596 | 0.3048 |

  Warm Test에서 순위 MF는 NDCG 0.0572로 인기도 0.0320의 약 1.79배입니다. Cold 이력 3·5개 MF는 0.2735·0.2835이며 Hybrid는 같은 결과입니다. 별도 평점 예측 Test에서 ALS RMSE는 0.8337, Bias 기준선은 0.8555입니다. RMSE와 NDCG는 다른 과제의 지표입니다.
- **시각 자료:** Warm Test 막대그래프와 Cold Final 이력별 선 그래프를 분리합니다. Hybrid의 1개 이력 결과가 인기도 및 MF보다 낮다는 점을 표시하고, RMSE는 별도 패널에 둡니다.
- **추천 레이아웃:** Warm/Cold를 좌우로 나누고, 검증과 테스트 수치를 같은 축에 섞지 않습니다.
- **자료:** `results/final/holdout_comparison.csv`, `results/final/validation_comparison.csv`, `docs/figures/D4_warm_test_comparison.png`, `docs/figures/D4_cold_final_comparison.png`, `results/evaluation/mf_rmse_test.csv`, `results/evaluation/baselines_rmse_test.csv`


### 11. 결론과 한계
- **핵심 내용:** 이번 분할에서 MF가 Warm 및 Cold Final 비교 조건의 NDCG@10 최고였습니다. Hybrid는 3개 이상 이력에서 MF와 같고 MF를 넘지 못했으며, 1개 이력에서는 인기도보다 낮았습니다. D 작업에서 EDA, 공정한 결합 검증, 일관된 결과 집계의 중요성을 확인했습니다. 시간 정보 부재와 태그 품질 편차는 한계입니다.
- **시각 자료:** “MF가 가장 높음 / 한 개 이력 Cold는 과제 / Hybrid 개선은 확인되지 않음” 세 카드.
- **추천 레이아웃:** 결론 한 문장 아래에 확인 결과와 후속 실험을 나눕니다.
- **자료:** `results/final/holdout_comparison.csv`, `results/hybrid/D3_hybrid_tuning.md`, `docs/data.md`

## 어려웠던 점과 배운 점

- CB·MF 점수 범위 차이는 사용자별 백분위 순위 변환 후 후보 비교로 해결했습니다.
- Validation/Test, Warm/Cold의 평가 역할을 분리해 결과 과장을 막았습니다.
- Hybrid 성능 개선이 확인되지 않은 결과도 그대로 보고합니다.
- D4 집계에서 알고리즘·코호트·이력 크기·지표를 같은 형식으로 맞춰 비교 가능하게 했습니다.
- 배운 점: 모델을 결합했다고 자동으로 좋아지지 않으며, EDA와 일관된 평가·집계가 결론의 신뢰성을 좌우합니다.

## 참고 자료와 발표 전 확인

Goodbooks-10k 출처·라이선스는 `docs/data.md`의 기록을 사용합니다. 팀 내부 참고 문서: `docs/parts/content_based.md`, `docs/parts/mf.md`, `docs/pipeline.md`, `docs/data.md`. 실제 슬라이드에 넣는 외부 자료는 각 팀 문서의 출처를 확인해 각주로 기재하고, 확인되지 않은 코드는 인용하지 않습니다.

- 군집 실루엣을 추천 NDCG와 같은 지표처럼 비교하지 않습니다.
- Warm/Cold 및 Validation/Test 범례를 분명히 표시합니다.
- Hybrid가 MF보다 우수하다고 표현하지 않습니다.
- 본 문서는 구성안이며 PPTX를 생성하지 않습니다.
- MF theory: Y. Koren, R. Bell, C. Volinsky, Matrix Factorization Techniques for Recommender Systems, IEEE Computer 42(8), 2009.
- Implicit-feedback model reference: Y. Hu, Y. Koren, C. Volinsky, Collaborative Filtering for Implicit Feedback Datasets, ICDM, 2008.
- Content vector and clustering API references from Team B: scikit-learn TfidfTransformer, normalize, KMeans, TruncatedSVD, silhouette_samples documentation (links are listed in docs/parts/content_based.md).
- Dataset source and license: zygmuntz/goodbooks-10k, CC BY-SA 4.0 (https://github.com/zygmuntz/goodbooks-10k).

## 팀 파트 문서와 통합 위치

- D1 EDA 과정·결과·이미지·2분 대본·배운 점: [eda.md](eda.md)
- D3 Hybrid 및 D4 결과 집계 과정·결과·이미지·2분 대본·배운 점: [hybrid.md](hybrid.md)
- B의 콘텐츠 기반 추천·군집 내용: [content_based.md](content_based.md)
- C의 MF·Fold-in 내용: [mf.md](mf.md)
- 이 구성안은 위 파트 문서와 A의 공통 데이터·평가 문서 및 실제 결과 CSV를 한 발표 흐름으로 통합합니다.
