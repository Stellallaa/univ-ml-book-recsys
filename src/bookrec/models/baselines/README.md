# baselines

다른 모델의 결과를 비교할 기준선 모델을 두는 곳입니다(A6). 담당은 팀원 A입니다.

| 파일 | 기준선 | 상속 | 채점 |
| --- | --- | --- | --- |
| `popularity.py` | train에서 평점이 많이 달린 순(A2 예시 모델 겸용) | `Recommender` | 순위 지표 |
| `goodreads_rating.py` | `books.csv`의 `average_rating`이 높은 순 | `Recommender` | 순위 지표 |
| `bias.py` | 전체 평균 + 독자 성향 + 책 성향으로 별점 예측 | `RatingPredictor` | RMSE |

- 채점 방법과 결과는 `../../evaluation/README.md`의 「기준선 (A6)」을 봅니다.
