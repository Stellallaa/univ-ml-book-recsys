# collaborative_filtering

model-based collaborative filtering 코드를 두는 곳입니다. 담당은 팀원 C입니다.

| 작업 | 하는 일 |
| --- | --- |
| C1 학습 | A4에서 나눈 학습 데이터로 모델을 학습하고 학습 곡선(epoch별 오차)을 그림 |
| C2 튜닝 | 하이퍼파라미터를 바꿔 비교하고 최종 설정을 고름 |
| C3 fold-in | 학습 때 없던 신규 독자에게도 점수를 낼 수 있게 함 |

- 어떤 기법으로 구현할지는 담당자가 정합니다. 제안서에서는 Matrix Factorization(MF)을 계획했습니다.
- 모델은 `Recommender`를 상속합니다(`../README.md` 참고). 별점을 예측하는 모델이면 `RatingPredictor`도 상속해 RMSE 평가를 받습니다.
- 라이브러리를 써도 됩니다. 수업에서 다루지 않은 라이브러리·함수는 파라미터까지 주석으로 설명합니다(과제 필수사항).
- 튜닝 실험은 `notebooks/collaborative_filtering/`에서 하고, 재사용할 코드만 여기에 둡니다.
