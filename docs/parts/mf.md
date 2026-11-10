# MF·fold-in (C1, C2, C3)

담당: 팀원 C (이다영) · 코드 `src/bookrec/models/collaborative_filtering/` · 노트북 `notebooks/collaborative_filtering/` · 기록 `results/collaborative_filtering/`

## 과정

1. **C1 MF 학습**
   - 예측값 = 전체 평균 + 독자 bias + 책 bias + (독자 벡터 · 책 벡터)
   - loss = (정답 - 예측값)^2 + L2 정규화. 수업에서 배운 mini-batch gradient descent로 줄였고 NumPy로 직접 구현했습니다.
   - epoch마다 검증 점수를 재고, 2번 연속 나아지지 않으면 멈춘 뒤 가장 좋았던 epoch로 되돌립니다(early stopping).
   - 평점만 학습한 MF는 별점 예측은 잘하지만 추천 적중률이 인기순보다 낮았습니다. 그래서 평점 1개마다 무작위 책 몇 권을 "정답 0점"으로 함께 학습하는 방식(negative sampling)을 추가했습니다.
2. **C2 MF 튜닝**
   - 별점 예측용(기준: 검증 RMSE)과 추천용(0점을 함께 학습, 기준: val NDCG@10)을 따로 골랐습니다.
   - 1차로 학습률, 잠재 차원, 정규화 세기, neg_ratio를 한 번에 하나씩 바꿔 15개 설정을 비교했습니다.
   - 추가 실험으로 더 큰 값(neg_ratio 8, 차원 128), neg_ratio x 차원 네 조합, 추천용 학습률, epoch 상한을 올린 재비교를 했습니다. gradient descent 설정은 모두 26개입니다.
   - 제안서에 적은 대로 다른 최적화 방법인 ALS도 구현해 정규화 세기 6개를 비교했습니다.
   - 모든 후보는 `train`으로 학습하고 `val`로만 비교했습니다.
3. **C3 fold-in**
   - 학습된 책 벡터는 고정하고, 독자의 평점만으로 독자 벡터를 Ridge regression의 닫힌 해로 구합니다. 다시 학습하지 않습니다.
   - 학습에 없던 신규 독자도 평점 1개만 있으면 MF 점수를 낼 수 있습니다. fold-in의 정규화 세기는 `cold_tune`으로 골랐습니다.

## 결과

### 최종 설정

| 용도 | 방법 | 잠재 차원 | 학습률 | 정규화 | neg_ratio | epoch | fold_in_reg | 검증 점수 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 별점 예측용 | ALS | 64 | - | 0.1 | - | 25 | - | 검증 RMSE 0.8392 |
| 추천용 (hybrid에 들어가는 모델) | gradient descent | 128 | 0.01 | 0.05 | 8 | 4 | 10 | val NDCG@10 0.0723 (인기순 0.0382) |

### 최종 점수 (`test`, `cold_final`)

| model | goal | history_size | precision@10 | recall@10 | ndcg@10 | to_read_hit@10 | test_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popularity | warm_test | all | 0.0268 | 0.0230 | 0.0292 | 0.0303 |  |
| mf_ranking | warm_test | all | 0.0493 | 0.0435 | 0.0549 | 0.0521 |  |
| mf_rating | warm_test | all |  |  |  |  | 0.8515 |
| popularity | cold_final | 1 | 0.2101 | 0.0308 | 0.2237 | 0.0230 |  |
| popularity | cold_final | 3 | 0.2101 | 0.0308 | 0.2237 | 0.0230 |  |
| popularity | cold_final | 5 | 0.2101 | 0.0308 | 0.2237 | 0.0230 |  |
| popularity | cold_final | 10 | 0.2101 | 0.0308 | 0.2237 | 0.0230 |  |
| mf_ranking | cold_final | 1 | 0.2466 | 0.0362 | 0.2574 | 0.0197 |  |
| mf_ranking | cold_final | 3 | 0.2611 | 0.0387 | 0.2736 | 0.0228 |  |
| mf_ranking | cold_final | 5 | 0.2691 | 0.0401 | 0.2791 | 0.0242 |  |
| mf_ranking | cold_final | 10 | 0.2866 | 0.0430 | 0.2980 | 0.0276 |  |

- 기존 독자(Goal 2): 추천용 MF의 NDCG@10은 0.0549로 인기순(0.0292)의 1.88배입니다. 별점 예측용 MF(ALS)의 test RMSE는 0.8515입니다.
- 신규 독자(Goal 1): MF(fold-in)의 NDCG@10은 평점 1개일 때 0.257, 10개일 때 0.298이고 인기순은 0.224입니다.
- 읽고 싶은 책(Goal 3): to_read 적중은 기존 독자에서 MF 0.0521, 인기순 0.0303입니다.
- 신규 독자의 to_read 적중은 평점 1개일 때 MF 0.0197, 인기순 0.0230, 평점 10개일 때 MF 0.0276입니다. 평점이 적을 때는 MF가 인기순보다 낮습니다.
- `test`·`cold_final`은 1차 튜닝 설정으로 한 번 측정한 적이 있고, 위 표는 `val`만 보고 다시 고른 최종 설정으로 다시 측정한 값입니다(`S1_mf_final_scores.md`).

### 기준선·CB와의 비교 (val, cold_tune, NDCG@10)

| 대상 | 인기순 | Goodreads 평점순 | bias | CB (팀원 B 최종 설정) | MF |
| --- | --- | --- | --- | --- | --- |
| 기존 독자 (val 5,000명) | 0.0382 | 0.0018 | 0.0014 | 0.0516 | 0.0718 |
| 신규 독자, 평점 1개 | 0.2213 | 0.0089 | 0.0074 | 0.1518 | 0.2529 |
| 신규 독자, 평점 3개 | 0.2213 | 0.0089 | 0.0074 | 0.1682 | 0.2709 |
| 신규 독자, 평점 5개 | 0.2213 | 0.0089 | 0.0074 | 0.1868 | 0.2784 |
| 신규 독자, 평점 10개 | 0.2213 | 0.0089 | 0.0074 | 0.2277 | 0.2956 |

- 독자 단위 bootstrap(2,000번)으로 MF와의 차이의 95% 구간을 냈습니다. 모든 비교에서 구간이 0을 포함하지 않았습니다.
- 검증 RMSE: 전체 평균 0.9865, bias 기준선 0.8659, MF 별점 예측용 (ALSMatrixFactorization) 0.8392
- 자세한 표는 `results/collaborative_filtering/C2c_mf_baselines_uncertainty.md`에 있습니다.

### 실험에서 확인한 것 (val 기준)

| 실험 | 결과 |
| --- | --- |
| 평점만 학습한 MF의 학습 곡선 (차원 32) | 검증 RMSE가 epoch 21에서 0.8477로 최저, 그 뒤 다시 증가. 학습 RMSE는 계속 감소 (과적합) |
| 평점만 학습한 MF로 추천 | NDCG@10 0.0031 (인기순 0.0382). 추천된 책의 평점 수 중앙값 122개 (인기순 11,421개) |
| neg_ratio 0 → 1 → 2 → 4 (차원 32) | NDCG@10 0.003 → 0.054 → 0.060 → 0.063, 검증 RMSE 0.85 → 1.74 → 1.94 → 2.26 |
| neg_ratio x 차원 조합 (학습률 0.02) | (4, 64) 0.0672 / (4, 128) 0.0684 / (8, 64) 0.0689 / (8, 128) 0.0706 |
| 추천용 학습률 0.01 / 0.02 / 0.04 (차원 64, neg_ratio 4) | NDCG@10 0.0689 / 0.0672 / 0.0610 |
| 정규화 세기 (gradient descent, 별점 예측용) 0.02 / 0.05 / 0.1 | 검증 RMSE 0.8574 / 0.8463 / 0.8626 |
| ALS와 gradient descent (별점 예측) | 검증 RMSE: ALS 0.8392, gradient descent 최저 0.8453 |
| ALS 정규화 세기 0.05 / 0.1 / 0.2 | 검증 RMSE 0.8700 / 0.8392 / 0.8615 |
| fold_in_reg 1 / 5 / 10 / 20 / 50 | cold_tune 평균 NDCG@10 0.2555 / 0.2726 / 0.2744 / 0.2703 / 0.2589 |

## 결론

- RMSE가 좋은 모델이 추천을 잘하는 것은 아닙니다. 평점만 학습한 MF는 RMSE는 좋지만 추천 적중률은 인기순의 1/10 이하였습니다. 평점이 있는 칸만 보면 "어떤 책을 읽을지"를 배우지 못하기 때문입니다.
- 평점 없는 칸을 0점으로 함께 학습하면 추천 적중률이 인기순의 약 1.9배(val)가 됩니다. 대신 별점 예측은 나빠져서 용도별로 모델을 나눴습니다.
- 별점 예측은 같은 모델이라도 최적화 방법에 따라 달랐습니다. ALS가 gradient descent보다 검증 RMSE가 낮아 별점 예측용은 ALS를 골랐습니다.
- fold-in으로 신규 독자에게도 MF 점수를 낼 수 있고, 평점 1개만 있어도 인기순보다 높았습니다. 평점이 늘수록 점수가 올라갑니다.
- 다만 평점이 적을 때 MF의 추천은 많이 읽힌 책 쪽으로 몰립니다. 예를 들어 Pride and Prejudice, Jane Eyre, Wuthering Heights를 고른 가상의 독자에게 1위로 To Kill a Mockingbird이(가) 나왔습니다(`C3_fold_in.md`의 추천 예시).

## 이미지

| 파일 | 내용 |
| --- | --- |
| `docs/figures/C1_mf_loss_curve.png` | 평점만 학습한 MF의 epoch별 학습·검증 RMSE (과적합이 시작되는 지점 표시) |
| `docs/figures/C1_mf_ranking_curve.png` | 0점을 함께 학습한 MF의 epoch별 학습 RMSE와 val NDCG@10 |
| `docs/figures/C2_mf_tuning_rating.png` | 별점 예측용 1차 설정 비교 (학습률, 잠재 차원, 정규화) |
| `docs/figures/C2_mf_tuning_ranking.png` | 추천용 1차 설정 비교 (neg_ratio, 잠재 차원, 정규화) |
| `docs/figures/C2_mf_rmse_vs_ndcg.png` | neg_ratio에 따른 검증 RMSE와 val NDCG@10 |
| `docs/figures/C2b_mf_extended.png` | 추가 실험 (추천용 학습률, neg_ratio x 차원 조합, 차원별 RMSE와 ALS) |
| `docs/figures/C2c_mf_vs_baselines.png` | 기준선·CB와의 비교 (기존 독자, 신규 독자) |
| `docs/figures/C3_fold_in_cold.png` | 신규 독자의 평점 수별 NDCG@10, fold-in 정규화 세기 비교 |
| `docs/figures/S1_mf_final_scores.png` | 최종 점수 (기존 독자 test, 신규 독자 cold_final) |

PPT에는 `C1_mf_loss_curve`, `C2_mf_rmse_vs_ndcg`, `S1_mf_final_scores` 3장을 우선 추천합니다.

## 2분 발표 대본

Collaborative filtering은 Matrix Factorization으로 구현했습니다. 독자와 책을 각각 숫자 벡터로 표현하고, 두 벡터의 내적에 독자와 책의 bias를 더해 별점을 예측합니다. 학습은 수업에서 배운 gradient descent와 L2 정규화를 NumPy로 직접 구현했습니다.

왼쪽 그래프는 학습 곡선입니다. 학습 오차는 계속 내려가지만 검증 오차는 어느 시점부터 다시 올라갑니다. 과적합이 시작되는 지점이고, 여기서 학습을 멈췄습니다.

그런데 이 모델로 추천을 하면 적중률이 인기순의 10분의 1도 되지 않았습니다. 평점이 있는 칸만 학습해서, 읽은 책에 몇 점을 줄지는 배우지만 어떤 책을 읽을지는 배우지 못했기 때문입니다. 그래서 평점 없는 책을 0점으로 함께 학습시켰습니다. 가운데 그래프처럼 0점을 많이 섞을수록 별점 예측은 나빠지지만 추천은 좋아집니다. 그래서 별점 예측용과 추천용 모델을 따로 두었습니다. 별점 예측용은 ALS라는 다른 최적화 방법도 비교했고, 오차가 더 낮아 ALS를 선택했습니다. test RMSE는 0.85입니다.

최종 결과입니다. 기존 독자에게 추천용 MF의 NDCG는 인기순의 약 1.9배입니다. 신규 독자는 학습된 책 벡터를 고정하고 그 독자의 평점 몇 개로 독자 벡터만 Ridge regression으로 계산했습니다. 평점이 1개뿐이어도 인기순보다 높았고, 평점이 늘수록 점수가 올라갔습니다. 다만 평점이 적을 때는 많이 읽힌 책 위주로 추천하는 경향이 있어서, 이 부분을 content-based와 합친 hybrid로 보완합니다.

## 어려웠던 점과 해결

| 어려웠던 점 | 해결 |
| --- | --- |
| RMSE는 좋은데 추천 적중률이 인기순보다 낮았음 (NDCG@10 0.003 vs 0.038) | 추천된 책을 확인해 평점 수가 적은 책에 몰린다는 것을 찾았고, 평점 없는 칸을 0점으로 함께 학습(negative sampling)해 0.072까지 올림 |
| 한 모델로 별점 예측과 추천을 모두 잘할 수 없었음 | 용도별로 설정을 따로 고르고 `load_selected_mf("rating" / "ranking")`으로 나눠 쓰게 함 |
| 평점 377만 건을 한 건씩 갱신하면 Python 반복문으로는 너무 느림 | 2,048건씩 묶어 NumPy로 한 번에 계산. 한 묶음에 같은 독자·책이 여러 번 나와 `np.add.at`으로 기울기를 누적 |
| 학습에 없던 신규 독자는 독자 벡터가 없음 | 책 벡터를 고정하고 독자 벡터만 Ridge regression의 닫힌 해로 계산(fold-in) |
| 1차 튜닝에서 가장 좋은 값이 비교 범위의 끝에 있었고, epoch 상한에 걸린 후보가 있었음 | 범위를 넓히고 상한을 올려 다시 비교. 조합(neg_ratio x 차원)도 함께 비교 |
| 최종 추천용 설정은 학습에 10분 넘게 걸림 | 학습된 모델을 파일로 저장해 두고 다른 노트북과 hybrid에서 다시 쓰게 함 (`fit_selected_mf`) |
| 튜닝 실행이 오래 걸려 중간에 끊김 | 실험이 끝날 때마다 CSV에 저장하고, 다시 실행하면 끝난 실험은 건너뛰게 함 |
| ALS에서 평점 전체의 벡터를 한 번에 꺼내 메모리가 부족했음 | 독자·책 구간마다 필요한 행만 꺼내 계산 |

### 다시 한다면

- 추천용의 neg_ratio 8, 차원 128은 여전히 비교한 범위의 가장 큰 값입니다. 키울수록 조금씩 올랐지만 학습 시간이 2배씩 늘어 여기서 멈췄습니다. 더 빠른 구현으로 더 큰 값을 비교하겠습니다.
- 추천용에도 ALS를 적용해 비교하겠습니다.
- 0점으로 넣을 책을 무작위가 아니라 다른 기준(예: 많이 읽혔는데 그 독자는 읽지 않은 책)으로 뽑는 방식을 비교하겠습니다.

## 개인별 배운 점

(이다영) 평가 지표가 목적과 맞아야 한다는 것을 배웠습니다. RMSE를 낮추는 데만 집중했다면 추천에는 쓸 수 없는 모델을 최종 모델로 냈을 것입니다. 숫자가 이상할 때 추천된 책을 직접 확인한 것이 원인을 찾는 데 가장 도움이 됐습니다. 또 수업에서 배운 gradient descent, L2 정규화, Ridge regression, early stopping이 실제 600만 건 데이터에서 그대로 동작하는 것을 직접 구현하며 확인했습니다.

## 참고한 외부 코드·자료 (출처)

- Y. Koren, R. Bell, C. Volinsky, "Matrix Factorization Techniques for Recommender Systems", IEEE Computer, 42(8), 2009. (bias를 포함한 예측식과 gradient descent 갱신식)
- Y. Hu, Y. Koren, C. Volinsky, "Collaborative Filtering for Implicit Feedback Datasets", IEEE ICDM, 2008. (평점 없는 칸을 낮은 가중치의 0으로 보는 방식, 책 벡터를 고정하고 독자 벡터를 닫힌 해로 구하는 식)
- Y. Zhou, D. Wilkinson, R. Schreiber, R. Pan, "Large-Scale Parallel Collaborative Filtering for the Netflix Prize", AAIM, 2008. (ALS: 한쪽을 고정하고 번갈아 푸는 방식, 정규화 세기에 평점 수를 곱하는 방식)
- 외부 코드는 가져오지 않았습니다. NumPy, pandas만 사용했고, 수업에서 다루지 않은 함수(`np.add.at`, `np.linalg.solve`, `np.savez` 등)는 코드 주석에 설명했습니다.
