# C1 MF 학습

노트북: `notebooks/collaborative_filtering/C1_mf_training.ipynb` · 코드: `src/bookrec/models/collaborative_filtering/mf.py`

## 과정
- 예측값 = 전체 평균 + 독자 bias + 책 bias + (독자 벡터 · 책 벡터)
- loss = (정답 - 예측값)^2 + reg * (파라미터 크기^2). mini-batch gradient descent(batch 2,048)로 줄입니다. NumPy로 직접 구현했습니다.
- `train`(3,769,078건, 독자 48,424명)만 학습하고 `val`로만 점수를 쟀습니다. `test`·`cold_final`은 쓰지 않았습니다.
- 추천 지표는 val 독자 5,000명(시드 42, 정답이 있는 독자 4,974명)으로 쟀습니다.

## 결과
| model | precision@10 | recall@10 | ndcg@10 | val_rmse |
| --- | --- | --- | --- | --- |
| popularity (인기순) | 0.0342 | 0.0308 | 0.0382 | - |
| mf_rating (평점만 학습) | 0.0029 | 0.0029 | 0.0031 | 0.8477 |
| mf_ranking (0점을 함께 학습, neg_ratio 2) | 0.0535 | 0.0500 | 0.0601 | 1.9427 |

설정: 차원 32, 정규화 0.05. mf_rating은 학습률 0.01, mf_ranking은 학습률 0.02입니다.

- 평점만 학습한 MF: 검증 RMSE는 epoch 21에서 0.8477로 가장 낮고 그 뒤로 다시 커집니다(과적합 시작). 학습 RMSE는 계속 내려갑니다. 모든 평점을 전체 평균으로 예측하면 0.9865라서 14.1% 줄었습니다.
- 같은 모델의 예측 별점 순으로 추천하면 NDCG@10이 0.0031로 인기순(0.0382)보다 훨씬 낮습니다. 독자 1,000명에게 추천된 책의 `train` 평점 수 중앙값이 122개(인기순은 11,421개, 전체 책은 145개)였습니다. 평점이 적고 평균 별점이 높은 책을 추천하는데, 독자가 그 책을 실제로 읽는 경우가 드물어 적중하지 않습니다.
- 원인: 평점이 있는 칸만 학습하면 "읽은 책에 몇 점을 줄지"만 배우고 "어떤 책을 읽을지"는 배우지 못합니다.
- 해결: 평점 1개마다 무작위 책 neg_ratio권을 "정답 0점"으로 함께 학습했습니다(negative sampling). NDCG@10이 0.0601로 인기순보다 높아졌습니다. 대신 예측값이 별점보다 낮아져 RMSE는 1.94로 나빠집니다.

## 결론
MF를 용도별로 둘로 나눕니다. 별점 예측(RMSE)에는 평점만 학습한 MF를, 추천(top-10)과 hybrid에는 0점을 함께 학습한 MF를 씁니다.

## 이미지와 기록
- `docs/figures/C1_mf_loss_curve.png`: 평점만 학습한 MF의 epoch별 학습·검증 RMSE
- `docs/figures/C1_mf_ranking_curve.png`: 0점을 함께 학습한 MF의 epoch별 학습 RMSE와 val NDCG@10
- `C1_rating_history.csv`, `C1_ranking_history.csv`: epoch별 기록 / `C1_validation_scores.csv`: 위 표
