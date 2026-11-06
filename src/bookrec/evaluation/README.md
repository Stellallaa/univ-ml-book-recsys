# evaluation

모델을 평가하고 결과를 모으는 공용 코드를 두는 곳입니다.

| 작업 | 하는 일 | 담당 |
| --- | --- | --- |
| A5 공통 평가 함수 | 어떤 모델을 넣어도 Goal 1~3의 지표를 같은 방식으로 계산 | 팀원 A |
| D4 결과 집계 스크립트 | 모든 모델의 결과를 한 표와 비교 그래프로 모으고, 기준선 대비 향상 정도를 함께 보여 줌 | 팀원 D |

| 목표 | 지표 |
| --- | --- |
| Goal 1 신규 독자 | 남긴 평점 수(1, 3, 5, 10개)별 Precision@10, Recall@10 |
| Goal 2 기존 독자 | Precision@10, Recall@10, NDCG@10, RMSE(별점을 예측하는 모델만) |
| Goal 3 읽고 싶은 책 | 추천 10권 중 to_read 목록에 있는 비율 |

- 평가 함수는 모델의 `recommend` 결과로 순위 지표를, `predict` 결과로 RMSE를 계산합니다.
- 그래프는 `bookrec.plot_style`로 저장합니다.

## 공통 평가 함수 (A5)

`evaluate.py`의 함수 세 개로 모든 모델을 같은 방식으로 채점합니다. 모델은 이력만 받고 정답은 보지 않습니다. 평가 함수가 `model.recommend(이력, 10)`을 불러 받은 추천 10권을 정답과 비교합니다.

```python
from bookrec.data import load_split, load_to_read
from bookrec.evaluation import evaluate_cold, evaluate_rmse, evaluate_warm

split = load_split()
to_read = load_to_read()
model.fit(split.train)

# Goal 2·3 튜닝: train을 이력으로 넣고 val로 채점
evaluate_warm(model, split, on="val", to_read=to_read)
# Goal 2·3 최종: train + val을 이력으로 넣고 test로 채점 (마지막에 한 번)
evaluate_warm(model, split, on="test", to_read=to_read)

# Goal 1 튜닝 / 최종 (마지막에 한 번)
evaluate_cold(model, split, on="tune")
evaluate_cold(model, split, on="final")

# RMSE (별점을 예측하는 모델만): 튜닝은 val, 최종은 test
evaluate_rmse(model, split, on="val")
evaluate_rmse(model, split, on="test")
```

| 함수 | 돌려주는 것 |
| --- | --- |
| `evaluate_warm(model, split, on="val"/"test", to_read=None)` | `precision@10`, `recall@10`, `ndcg@10`, `users`, (`to_read`를 넘기면) `to_read_hit@10`, `to_read_users`를 담은 Series |
| `evaluate_cold(model, split, on="tune"/"final")` | 행 = 남긴 평점 수(1, 3, 5, 10), 열 = `precision@10`, `recall@10`, `users`인 표 |
| `evaluate_rmse(model, split, on="val"/"test")` | RMSE 숫자 하나 |

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
| 정답 | 4점 이상 준 책 |
| 평균 | 독자마다 계산한 뒤 독자들의 평균을 냅니다. 정답이 하나도 없는 독자는 그 지표의 평균에서만 빠집니다 |
| Goal 2 이력 | `on="val"`이면 train, `on="test"`면 train + val을 이력으로 넣습니다. test 시점의 독자는 val의 책도 이미 읽었기 때문입니다 |
| Goal 2 독자 | 채점 구간(val 또는 test)에 나오는 독자 중 5,000명을 무작위로 뽑습니다(시드 42, 모든 모델이 같은 독자). 500명씩 나눠 추천해 메모리를 아낍니다 |
| Goal 3 정답 | 그 독자의 to_read 목록에서 이력에 있는 책을 뺀 것. 추천 목록은 Goal 2와 같습니다 |
| Goal 1 이력 | 신규 독자의 평점을 파일 순서(시간순)대로 세어 앞 1·3·5·10개를 넣습니다. 모델 하나로 네 번 추천합니다 |
| Goal 1 정답 | 11번째 평점부터 4점 이상 준 책. 1·3·5·10개 모두 같은 정답으로 채점합니다 |
| RMSE | 채점 구간(val 또는 test)의 (독자, 책) 쌍마다 `predict`로 예측해 실제 별점과 비교합니다 |

- **Goal 1과 Goal 2의 숫자를 서로 비교하지 않습니다.** Goal 1은 11번째 이후 평점 전체(독자당 약 100권)가 정답 후보이고, Goal 2는 마지막 15%(약 17권)만 정답 후보라 값의 크기가 다릅니다. 비교는 같은 Goal 안에서 모델끼리만 합니다.
