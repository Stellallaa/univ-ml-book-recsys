# B3 콘텐츠 기반 추천기

## 과정

`ContentBasedRecommender`는 기존 `Recommender`를 상속합니다. B1·B2의 태그 특징, A의 인기순 기준선과 공통 평가 함수를 그대로 사용했습니다. 다른 팀원의 파일은 수정하지 않았습니다.

```python
from bookrec.data import load_book_tags, load_split
from bookrec.data.tag_filter import filter_book_tags
from bookrec.models.content_based.features import build_book_features
from bookrec.models.content_based.recommender import ContentBasedRecommender

features = build_book_features(filter_book_tags(load_book_tags()))
split = load_split()
model = ContentBasedRecommender(features).fit(split.train)
scores = model.score(split.cold_tune.groupby("user_id", sort=False).head(1))
# score 전체 독자 호출은 메모리를 많이 사용한다. 공통 평가 함수는 500명씩 나눠 호출한다.
```

### 점수 계산

- B2의 책 벡터를 사용합니다. 기본 설정은 확장 태그 정제, TF-IDF, min_df=2, sublinear_tf=True입니다.
- `profile_weighting="positive"`가 기본입니다. 가중치는 `max(rating - 3, 0)`이므로 4점 책은 1, 5점 책은 2, 1~3점 책은 0으로 반영합니다.
- 독자 벡터는 책 벡터의 가중합을 L2 정규화한 값입니다. 점수는 정규화된 독자·책 벡터의 내적(코사인 유사도)입니다.
- `profile_weighting="rating"`도 선택할 수 있으며, 별점 자체를 가중치로 씁니다. 이 방식의 비교는 B4에서 진행합니다.
- 취향 벡터가 비면 기존 `PopularityRecommender`의 train 책별 평점 개수를 최대 개수로 나눈 점수를 사용합니다. 학습 데이터가 비면 대체 점수도 전부 0입니다.
- 모든 점수는 0~1이고, 행은 이력의 독자 ID 오름차순, 열은 공통 `BOOK_IDS`(1~10,000)입니다. hybrid에 이 점수표를 전달할 수 있습니다.
- 이미 읽은 책 제외와 동점 처리는 공통 `recommend`가 담당합니다. 별도 `read` 인수도 그대로 사용합니다.
- `fit`은 train 인기순 통계를 준비하고, 독자 취향은 매번 전달된 history에서 계산합니다. 학습에 없는 신규 독자도 처리합니다.
- `profile_summary(history)`로 이력 수와 대체 추천 여부를 확인할 수 있습니다.

## 첫 검증 결과

이 결과는 최종 점수가 아닙니다. 기존 독자는 val에서 seed=42로 추출한 동일한 1,000명, 신규 독자는 cold_tune 전원 2,500명입니다. 두 모델 모두 train으로만 학습했으며, test·cold_final 채점은 수행하지 않았습니다.

| 목표 | 이력 | 모델 | Precision@10 | Recall@10 | NDCG@10 | to_read_hit@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 기존 독자 val | train | 인기순 | 0.036446 | 0.032146 | 0.041433 | 0.027956 |
| 기존 독자 val | train | CB | 0.041365 | 0.039683 | 0.048862 | 0.042320 |
| 신규 독자 tune | 1 | 인기순 | 0.207240 | 0.030281 | 0.221323 | 0.020148 |
| 신규 독자 tune | 1 | CB | 0.125360 | 0.019320 | 0.139881 | 0.022367 |
| 신규 독자 tune | 3 | 인기순 | 0.207240 | 0.030281 | 0.221323 | 0.020148 |
| 신규 독자 tune | 3 | CB | 0.138840 | 0.021037 | 0.153870 | 0.025805 |
| 신규 독자 tune | 5 | 인기순 | 0.207240 | 0.030281 | 0.221323 | 0.020148 |
| 신규 독자 tune | 5 | CB | 0.158200 | 0.023835 | 0.172518 | 0.026414 |
| 신규 독자 tune | 10 | 인기순 | 0.207240 | 0.030281 | 0.221323 | 0.020148 |
| 신규 독자 tune | 10 | CB | 0.198040 | 0.030241 | 0.216142 | 0.027459 |

기존 독자의 좋아한 책 지표는 정답이 있는 996명, to_read 지표는 해당 정답이 있는 905명으로 평균을 냈습니다. 신규 독자는 각각 2,500명과 2,298명입니다. 공통 평가 함수가 정답 없는 독자를 지표별 평균에서 제외합니다.

인기순은 이력을 취향 계산에 사용하지 않으며, 신규 독자 평가의 제외 책은 네 조건 모두 앞 10권으로 같으므로 네 결과가 같습니다. 기존 독자와 신규 독자는 정답 구간이 다르므로 목표 간 수치 크기를 직접 비교하지 않습니다.

### 인기순 대체 비율

| 목표 | 이력 | 전체 독자 | 대체 독자 | 대체 비율 |
| --- | --- | ---: | ---: | ---: |
| 기존 독자 val | train | 1,000 | 0 | 0.00% |
| 신규 독자 tune | 1 | 2,500 | 616 | 24.64% |
| 신규 독자 tune | 3 | 2,500 | 158 | 6.32% |
| 신규 독자 tune | 5 | 2,500 | 67 | 2.68% |
| 신규 독자 tune | 10 | 2,500 | 15 | 0.60% |

이 비율은 정답 유무에 관계없이 추천 대상 전원을 기준으로 계산합니다.

## 검증

기존 테스트 42개와 B3 테스트 사례 14개가 통과해 전체 56개가 통과했습니다.

- 학습에 없는 독자, 단일 평점 이력, 이력에 따른 점수 차이와 입력 불변성
- 수작업 가중 코사인 값, positive/rating 가중 방식, 유한한 0~1 점수
- 저평점 이력·특징 없는 책의 대체 추천과 train 통계만 사용
- 상속한 recommend의 이미 읽은 책 제외와 별도 read 인수
- 빈 학습·빈 이력, fit 전 호출, 잘못된 입력과 반복 학습
- 공통 기존·신규 독자 평가 함수와 연결한 작은 데이터의 정확한 지표

실제 데이터에서는 학습에 없는 독자에게 4점 책 한 권만 제공한 추천을 추가 확인했습니다. 기존 책 제외, 추천 10권의 중복 없음, 점수 범위와 유한성 검증을 통과했습니다.

```bash
uv run pytest
```

실행 노트북: `notebooks/content_based/B3_content_based.ipynb`
원 수치: `B3_validation_scores.csv`
대체 추천 통계: `B3_profile_summary.csv`

## 결론과 다음 작업

기존 독자 검증 표본에서는 CB가 인기순보다 높은 순위 지표를 보였습니다. 신규 독자는 이력이 늘수록 CB 지표가 높아졌지만, 네 조건 모두 인기순보다 NDCG@10이 낮았습니다. 현재 결과로 신규 독자 추천에서 우수하다고 주장하지 않습니다.

첫 평점이 낮으면 취향을 추정하지 못하고, 높은 평점도 매우 적으면 일부 시리즈와 태그에 추천이 집중될 수 있습니다. 1~3점 책은 positive 취향에 반영되지 않지만 이미 읽은 책으로는 제외됩니다. 추천 가능한 책이 모두 0점이면 공통 규칙대로 작은 책 ID가 먼저 나옵니다.

B4에서 태그 정제 범위·특징 가중치·독자 가중치를 같은 검증 데이터로 비교합니다. 현재 CB에는 별점 예측 기능이 없으므로 RMSE는 계산하지 않습니다.

## 참고

- `src/bookrec/models/base.py`, `popularity.py`: 공통 인터페이스·대체 추천
- `src/bookrec/evaluation/README.md`: 지표, 정답 구간, 배치 평가 규칙
- `src/bookrec/models/content_based/features.py`: B2 태그 특징
- SciPy `csr_matrix`: 독자 x 책 가중치와 희소 행렬 곱
- scikit-learn `normalize(norm="l2")`: 책·독자 벡터 정규화
