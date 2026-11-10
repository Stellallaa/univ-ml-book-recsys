# models

추천 모델을 두는 곳입니다. 모든 모델은 `base.py`의 공통 인터페이스를 따릅니다.

| 위치 | 작업 | 담당 |
| --- | --- | --- |
| `base.py` | A2 공통 인터페이스 (`Recommender`, `RatingPredictor`) | 팀원 A |
| `popularity.py` | A2 예시 모델, A6 인기순 기준선 | 팀원 A |
| `goodreads_rating.py` | A6 Goodreads 평점순 기준선 | 팀원 A |
| `bias.py` | A6 bias 기준선 (RMSE용) | 팀원 A |
| `content_based/` | B2~B5 content-based filtering | 팀원 B |
| `collaborative_filtering/` | C1~C3 model-based collaborative filtering | 팀원 C |
| `hybrid/` | D3 hybrid filtering | 팀원 D |

## 모델을 만들 때 지킬 것

- `Recommender`를 상속하고 `fit(train)`과 `score(history)`를 구현합니다. `recommend(history, k)`는 물려받으므로 따로 만들지 않습니다.
- `score`는 행이 `user_id`, 열이 `BOOK_IDS`(책 1~10000)인 점수표를 돌려줍니다. 점수가 높을수록 추천합니다.
- `score`는 독자 ID가 아니라 평점 이력(`user_id`, `book_id`, `rating`)을 받습니다. 이력이 1개뿐인 신규 독자에게도 동작해야 합니다.
- 별점 자체를 예측하는 모델(MF, bias 기준선)은 `RatingPredictor`도 상속하고 `predict(pairs)`를 구현합니다. RMSE 평가에 쓰입니다.
- 만드는 방법은 `popularity.py`를 참고합니다.
