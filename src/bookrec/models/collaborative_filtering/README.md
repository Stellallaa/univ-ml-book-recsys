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

## 파일

| 파일 | 내용 |
| --- | --- |
| `mf.py` | `MatrixFactorization`(gradient descent 학습 `fit`, 별점 예측 `predict`, fold-in `fold_in`, 추천 점수 `score`, 저장·불러오기 `save`·`load`), 설정 `MFConfig` |
| `als.py` | `ALSMatrixFactorization`(ALS 학습 `fit`, 별점 예측 `predict`). 별점 예측용 최종 모델 |
| `tuning.py` | 설정 하나를 학습·채점하는 `run_experiment`, 최종 설정으로 모델을 만드는 `load_selected_mf`, 학습까지 해서 돌려주는 `fit_selected_mf` |

## 사용법 (평가·hybrid에서 쓸 때)

MF는 용도에 따라 최종 모델이 둘입니다. 이유와 설정은 `results/collaborative_filtering/C1_mf_training.md`, `C2b_mf_tuning_extended.md`에 있습니다.

| 용도 | 모델 | 쓰는 곳 |
| --- | --- | --- |
| `"ranking"` 추천용 | gradient descent로 평점과 0점을 함께 학습한 MF | `evaluate_warm`, `evaluate_cold`, hybrid |
| `"rating"` 별점 예측용 | ALS로 평점만 학습한 MF | `evaluate_rmse` |

```python
from bookrec.data import load_split
from bookrec.evaluation import evaluate_cold, evaluate_rmse, evaluate_warm
from bookrec.models.collaborative_filtering import fit_selected_mf

split = load_split()

# 추천(top-10)용
mf = fit_selected_mf("ranking", split.train)
mf.score(history)             # 점수표 (행 user_id, 열 책 1~10000). 학습에 없던 독자도 이력만 있으면 됩니다
mf.recommend(history, k=10)   # 독자별 추천 10권
evaluate_warm(mf, split, on="val")
evaluate_cold(mf, split, on="tune")

# 별점 예측용 (predict만 있고 score·recommend는 없습니다)
mf_rating = fit_selected_mf("rating", split.train)
evaluate_rmse(mf_rating, split, on="val")
```

- `fit_selected_mf(용도, train)`은 `results/collaborative_filtering/C2_selected_config.json`의 설정으로 학습한 모델을 돌려줍니다.
- 추천용은 학습에 10분 넘게 걸립니다. 처음 한 번만 학습하고 `data/models/`에 저장한 뒤, 다음부터는 저장된 파일을 불러옵니다(`data/`는 커밋하지 않습니다). 파일 이름에 설정과 학습 평점 수가 들어 있어, 설정이나 학습 데이터가 바뀌면 새로 학습합니다.
- 별점 예측용(ALS)은 매번 학습하고 약 3분 걸립니다.
- 학습 전 모델만 필요하면 `load_selected_mf(용도)`를 씁니다.
- `score`의 값은 0~1이 아닙니다. hybrid에서 합치기 전에 독자별로 정규화해야 합니다.
- `score`는 기존 독자와 신규 독자를 구분하지 않고, 넘어온 이력으로 독자 벡터를 다시 계산합니다(fold-in). 이력이 1개여도 동작합니다.
- 시드가 고정되어 있어 같은 `train`으로 학습하면 항상 같은 모델이 나옵니다.

## 최종 채점 (test, cold_final)

최종 채점은 기준선(`scripts/run_baselines.py --final`)과 같은 방식으로 합니다. val로 고른 설정을 그대로 두고, `train`과 `val`을 합쳐 다시 학습한 뒤 잽니다.

```python
import pandas as pd

final_train = pd.concat([split.train, split.val], ignore_index=True)
mf = fit_selected_mf("ranking", final_train)        # hybrid의 최종 채점에도 이 모델을 씁니다
mf_rating = fit_selected_mf("rating", final_train)
```

MF의 채점 스크립트는 `scripts/run_mf.py`입니다. 기준선과 같은 형식으로 `results/evaluation/`에 저장합니다.

```bash
uv run python scripts/run_mf.py          # val, cold_tune → mf_warm_val.csv, mf_cold_tune.csv, mf_rmse_val.csv
uv run python scripts/run_mf.py --final  # test, cold_final → mf_warm_test.csv, mf_cold_final.csv, mf_rmse_test.csv
```

- `mf_warm_*`, `mf_cold_*`는 추천용 MF, `mf_rmse_*`는 별점 예측용 MF(ALS)의 점수입니다.
- `notebooks/collaborative_filtering/S1_mf_final_scores.ipynb`의 값은 `train`만 학습해 잰 사전 측정입니다. 공식 최종 점수는 `results/evaluation/mf_*_test.csv`, `mf_cold_final.csv`입니다.
