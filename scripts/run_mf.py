"""MF(추천용, 별점 예측용)를 공통 평가 함수로 채점해 results/evaluation/에 CSV로 저장한다(C 파트).

기본은 튜닝용 구간(val, cold_tune)이다. --final을 주면 최종 구간(test, cold_final)을 채점한다.
기준선(scripts/run_baselines.py)과 채점 방식·파일 형식이 같다.
"""

import argparse
from pathlib import Path

import pandas as pd

from bookrec.data import load_split, load_to_read
from bookrec.evaluation import evaluate_cold, evaluate_rmse, evaluate_warm
from bookrec.models.collaborative_filtering import fit_selected_mf

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "evaluation"


def main():
    # NOTE: argparse는 명령행 옵션을 읽는다. action="store_true"는 --final을 주면 True, 안 주면 False다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="test·cold_final로 채점한다")
    warm_on, cold_on = ("test", "final") if parser.parse_args().final else ("val", "tune")

    split = load_split()
    to_read = load_to_read()
    # NOTE: 최종 채점 때는 val까지 학습에 넣는다(기준선과 같은 방식). 설정은 val로 고른 값을 그대로 쓴다.
    #       ignore_index=True는 두 표를 이으면서 행 번호를 0부터 새로 매긴다.
    train = split.train if warm_on == "val" else pd.concat([split.train, split.val], ignore_index=True)

    # 추천 지표: 추천용 MF (gradient descent, 0점을 함께 학습, 신규 독자는 fold-in)
    ranker = fit_selected_mf("ranking", train)
    warm = pd.DataFrame([{"model": "mf"} | evaluate_warm(ranker, split, on=warm_on, to_read=to_read)])
    cold = evaluate_cold(ranker, split, on=cold_on, to_read=to_read).reset_index()
    # NOTE: insert(0, 이름, 값)은 맨 앞에 열을 추가한다. 기준선 CSV와 열 순서를 맞춘다.
    cold.insert(0, "model", "mf")
    print("done ranking")

    # 별점 예측: 별점 예측용 MF (ALS)
    rmse = pd.DataFrame([{"model": "mf", "rmse": evaluate_rmse(fit_selected_mf("rating", train), split, on=warm_on)}])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warm.to_csv(OUT_DIR / f"mf_warm_{warm_on}.csv", index=False)
    cold.to_csv(OUT_DIR / f"mf_cold_{cold_on}.csv", index=False)
    rmse.to_csv(OUT_DIR / f"mf_rmse_{warm_on}.csv", index=False)
    print(*(table.to_string(index=False) for table in (warm, cold, rmse)), sep="\n\n")


if __name__ == "__main__":
    main()
