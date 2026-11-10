"""기준선(인기순, Goodreads 평점순, bias)을 공통 평가 함수로 채점해 results/evaluation/에 CSV로 저장한다(A6).

기본은 튜닝용 구간(val, cold_tune)이다. --final을 주면 최종 구간(test, cold_final)을 채점한다.
"""

import argparse
from pathlib import Path

import pandas as pd

from bookrec.data import load_books, load_split, load_to_read
from bookrec.evaluation import evaluate_cold, evaluate_rmse, evaluate_warm
from bookrec.models.baselines.bias import BiasPredictor
from bookrec.models.baselines.goodreads_rating import GoodreadsRatingRecommender
from bookrec.models.baselines.popularity import PopularityRecommender

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "evaluation"


def main():
    # NOTE: argparse는 명령행 옵션을 읽는다. action="store_true"는 --final을 주면 True, 안 주면 False다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="test·cold_final로 채점한다")
    warm_on, cold_on = ("test", "final") if parser.parse_args().final else ("val", "tune")

    split = load_split()
    to_read = load_to_read()
    # NOTE: 최종 채점 때는 val까지 학습에 넣는다.
    train = split.train if warm_on == "val" else pd.concat([split.train, split.val])
    rankers = {"popularity": PopularityRecommender(), "goodreads_rating": GoodreadsRatingRecommender(load_books())}

    warm, cold = [], {}
    for name, model in rankers.items():
        model.fit(train)
        # NOTE: dict | dict는 두 dict를 합친다(Python 3.9+). 모델 이름 열을 지표 dict 앞에 붙인다.
        warm.append({"model": name} | evaluate_warm(model, split, on=warm_on, to_read=to_read))
        cold[name] = evaluate_cold(model, split, on=cold_on, to_read=to_read)
        print(f"done {name}")
    rmse = pd.DataFrame([{"model": "bias", "rmse": evaluate_rmse(BiasPredictor().fit(train), split, on=warm_on)}])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warm = pd.DataFrame(warm)
    # NOTE: concat(dict, names=["model"])은 표들을 위아래로 잇고 dict의 키를 model 인덱스로 붙인다.
    cold = pd.concat(cold, names=["model"]).reset_index()
    warm.to_csv(OUT_DIR / f"baselines_warm_{warm_on}.csv", index=False)
    cold.to_csv(OUT_DIR / f"baselines_cold_{cold_on}.csv", index=False)
    rmse.to_csv(OUT_DIR / f"baselines_rmse_{warm_on}.csv", index=False)
    print(*(table.to_string(index=False) for table in (warm, cold, rmse)), sep="\n\n")


if __name__ == "__main__":
    main()
