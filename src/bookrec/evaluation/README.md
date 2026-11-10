# evaluation

모델을 평가하고 결과를 모으는 공용 코드를 두는 곳입니다.

| 작업 | 하는 일 | 담당 |
| --- | --- | --- |
| A5 공통 평가 함수 | 어떤 모델을 넣어도 같은 방식으로 채점 | 팀원 A |
| D4 결과 집계 스크립트 | 모든 모델의 결과를 한 표와 비교 그래프로 모으고, 기준선 대비 향상 정도를 함께 보여 줌 | 팀원 D |

## 무엇을 채점하나

**누구에게** 추천하는지(목표)와 **무엇을 정답으로** 볼지(평가 기준)를 나눠서 채점합니다.

| 평가 기준 → <br> 목표 ↓ | 좋아한 책 (4점 이상) | 읽고 싶은 책 (to_read) | 실제 별점 |
| --- | --- | --- | --- |
| **Goal 1 신규 독자** (평점 1·3·5·10개) | Precision@10, Recall@10, NDCG@10 | to_read 적중 | 해당 없음 |
| **Goal 2 기존 독자** (기록 충분) | Precision@10, Recall@10, NDCG@10 | to_read 적중 | RMSE (별점을 예측하는 모델만) |

- 신규 독자의 RMSE는 재지 않습니다. 학습에 없던 독자의 별점을 예측하려면 이력을 함께 넘겨야 하는데, 지금 공통 형태의 `predict`는 (독자, 책) 쌍만 받습니다.

## 공통 평가 함수 (A5)

`evaluate.py`의 함수 세 개로 모든 모델을 같은 방식으로 채점합니다. 모델은 이력만 받고 정답은 보지 않습니다. 평가 함수가 `model.recommend(이력, 10)`을 불러 받은 추천 10권을 정답과 비교합니다.

```python
from bookrec.data import load_split, load_to_read
from bookrec.evaluation import evaluate_cold, evaluate_rmse, evaluate_warm

split = load_split()
to_read = load_to_read()
model.fit(split.train)

# Goal 2 기존 독자: 튜닝은 val, 최종은 test (마지막에 한 번)
evaluate_warm(model, split, on="val", to_read=to_read)
evaluate_warm(model, split, on="test", to_read=to_read)

# Goal 1 신규 독자: 튜닝은 cold_tune, 최종은 cold_final (마지막에 한 번)
evaluate_cold(model, split, on="tune", to_read=to_read)
evaluate_cold(model, split, on="final", to_read=to_read)

# 실제 별점 (기존 독자, 별점을 예측하는 모델만)
evaluate_rmse(model, split, on="val")
evaluate_rmse(model, split, on="test")
```

| 함수 | 돌려주는 것 |
| --- | --- |
| `evaluate_warm(model, split, on="val"/"test", to_read=None, n_users=None)` | `precision@10`, `recall@10`, `ndcg@10`, `users`, (`to_read`를 넘기면) `to_read_hit@10`, `to_read_users`를 담은 dict. 여러 모델은 `pd.DataFrame([결과1, 결과2])`로 모읍니다 |
| `evaluate_cold(model, split, on="tune"/"final", to_read=None)` | 행 = 남긴 평점 수(1, 3, 5, 10), 열 = 위와 같은 지표인 표 |
| `evaluate_rmse(model, split, on="val"/"test")` | RMSE 숫자 하나 |

- 최종 채점 전에 고른 설정으로 `train` + `val`을 다시 학습할지는 모델 담당자가 정합니다.
- `evaluate_warm`은 기본으로 채점 구간의 기존 독자 **전원**을 채점합니다. 설정 후보를 빠르게 비교하고 싶으면 `n_users=5000`처럼 넘겨 무작위로 줄일 수 있습니다(시드 고정이라 모든 모델이 같은 독자). 줄일지는 쓰는 사람이 정합니다.

### 숫자의 뜻

| 지표 | 뜻 | 예시 (정답 8권, 추천 10권 중 2·5·10위가 적중) |
| --- | --- | --- |
| Precision@10 | 추천 10권 중 정답 비율 | 3 / 10 = 0.3 |
| Recall@10 | 정답 중 추천 10권에 들어간 비율 | 3 / 8 = 0.375 |
| NDCG@10 | 적중한 책이 위쪽 순위일수록 높은 점수(0~1). 정답이 맨 위부터 꽉 찼을 때를 1로 맞춥니다 | 약 0.33 |
| to_read_hit@10 | 추천 10권 중 to_read 목록에 있는 비율 | - |
| RMSE | 예측 별점과 실제 별점 차이의 크기. 작을수록 좋습니다 | 오차 1, -1, 0, -2 → 약 1.22 |
| `users`, `to_read_users` | 평균을 낸 독자 수 | - |

### 채점 규칙

| 규칙 | 내용 |
| --- | --- |
| 좋아한 책 | 4점 이상 준 책 |
| 읽고 싶은 책 | 그 독자의 to_read 목록에서 이미 읽은 책(Goal 2는 넣은 이력, Goal 1은 앞 10개)을 뺀 것 |
| 평균 | 독자마다 계산한 뒤 독자들의 평균을 냅니다. 정답이 하나도 없는 독자는 그 지표의 평균에서만 빠집니다 |
| Goal 2 이력 | `on="val"`이면 train, `on="test"`면 train + val을 이력으로 넣습니다. test 시점의 독자는 val의 책도 이미 읽었기 때문입니다 |
| Goal 2 정답 | 채점 구간(val 또는 test)에서 4점 이상 준 책 |
| Goal 1 이력 | 신규 독자의 평점을 파일 순서(시간순)대로 세어 앞 1·3·5·10개를 넣습니다. 모델 하나로 네 번 추천합니다 |
| Goal 1 정답 | 11번째 평점부터 4점 이상 준 책. 1·3·5·10개 모두 같은 정답으로 채점합니다 |
| Goal 1 이미 읽은 책 | 넣은 이력이 1·3·5개여도 앞 10개 책은 모두 추천에서 뺍니다. 독자가 이미 읽은 책인데 정답 구간(11번째부터)에도 없어서, 추천되면 틀림으로 세지기 때문입니다 |
| RMSE | 채점 구간(val 또는 test)의 (독자, 책) 쌍마다 `predict`로 예측해 실제 별점과 비교합니다 |
| 메모리 | 독자를 500명씩 나눠 추천받습니다. 한 번에 넣으면 점수표(독자 수 × 1만 권)가 수 GB가 됩니다 |

- **Goal 1과 Goal 2의 숫자를 서로 비교하지 않습니다.** Goal 1은 11번째 이후 평점 전체(독자당 약 100권)가 정답 후보이고, Goal 2는 마지막 15%(약 17권)만 정답 후보라 값의 크기가 다릅니다. 비교는 같은 목표 안에서 모델끼리만 합니다.

## 기준선 (A6)

모든 모델 결과를 기준선 대비로도 보여 주기 위해, 아주 단순한 추천 방법 세 가지를 같은 평가 함수로 채점합니다.

| 기준선 | 위치 | 하는 일 | 채점 |
| --- | --- | --- | --- |
| 인기순 | `models/baselines/popularity.py` | train에서 평점이 많이 달린 책을 모든 독자에게 똑같이 추천 | 순위 지표 |
| Goodreads 평점순 | `models/baselines/goodreads_rating.py` | `books.csv`의 `average_rating`이 높은 책을 모든 독자에게 똑같이 추천 | 순위 지표 |
| bias | `models/baselines/bias.py` | 전체 평균 + 독자 성향(후하게 주는지) + 책 성향(평이 좋은지)으로 별점 예측 | RMSE |

```bash
uv run python scripts/run_baselines.py          # val, cold_tune으로 채점
uv run python scripts/run_baselines.py --final  # test, cold_final으로 채점 (마지막에 한 번)
```

결과는 `results/evaluation/`에 `baselines_warm_<구간>.csv`(Goal 2), `baselines_cold_<구간>.csv`(Goal 1), `baselines_rmse_<구간>.csv`(RMSE)로 저장됩니다. `--final`은 val까지 학습에 넣어 다시 학습합니다.

### 튜닝 구간 결과 (val, cold_tune)

| 목표 | 기준선 | P@10 | R@10 | NDCG@10 | to_read 적중 |
| --- | --- | --- | --- | --- | --- |
| Goal 2 기존 독자 (48,150명) | 인기순 | 0.0329 | 0.0293 | 0.0364 | 0.0286 |
| Goal 2 기존 독자 (48,150명) | Goodreads 평점순 | 0.0017 | 0.0015 | 0.0020 | 0.0006 |
| Goal 1 신규 독자 (2,500명) | 인기순 | 0.2072 | 0.0303 | 0.2213 | 0.0201 |
| Goal 1 신규 독자 (2,500명) | Goodreads 평점순 | 0.0084 | 0.0012 | 0.0089 | 0.0006 |

| 기준선 | val RMSE |
| --- | --- |
| bias | 0.8616 |

- 두 순위 기준선은 독자의 이력을 보지 않아 Goal 1에서 남긴 평점 수(1·3·5·10개)와 상관없이 점수가 같습니다.
- Goodreads 평점순은 인기순보다 훨씬 낮습니다. 평균 별점 상위가 『The Complete Calvin and Hobbes』, 『Words of Radiance』(시리즈 2권), 해리 포터 박스 세트처럼 시리즈 후속권·모음집이라, 이미 팬인 독자만 읽어서 평이 높고 대부분의 독자는 읽지 않은 책입니다.
- `average_rating`은 우리 `ratings.csv`가 아니라 Goodreads 사이트 전체 독자의 집계입니다. 외부 정보를 쓴 기준선이라는 점을 결과와 함께 밝힙니다.
- bias는 MF 예측식(전체 평균 + 독자 성향 + 책 성향 + 독자·책 벡터 곱)에서 벡터 곱만 뺀 모델이라, MF와 RMSE를 나란히 놓으면 벡터를 더해 얼마나 좋아졌는지 볼 수 있습니다. 다만 규제 방식과 학습 방법이 MF와 달라서 RMSE 차이 전부를 벡터 덕분이라고 하지는 않습니다. 방법 출처는 Koren & Bell (2011), "Advances in Collaborative Filtering", Recommender Systems Handbook 2.1절입니다.
- bias의 규제 세기(`USER_REG`·`BOOK_REG` = 10)와 반복 횟수(`N_ITERS` = 10)는 val RMSE로 골랐습니다. 0~25 사이 조합에서 val RMSE는 0.8616~0.8643으로 차이가 작았습니다. 예측은 MF와 같이 1~5로 자릅니다.
