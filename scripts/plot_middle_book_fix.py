"""신규 독자 채점에서 앞 10권을 추천에서 빼기 전후의 인기순 점수를 비교한다(A5 발표용).

인기순은 이력을 보지 않으므로 넣은 평점 수와 상관없이 점수가 같아야 한다. 빼기 전에는 2~10번째 책이 추천되면
틀림으로 세져 평점 수에 따라 점수가 오른다. 결과는 results/evaluation/, 그래프는 docs/figures/에 저장한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bookrec import plot_style
from bookrec.data import load_split
from bookrec.evaluation import evaluate_cold
from bookrec.models.baselines.popularity import PopularityRecommender

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "evaluation"


class HistoryOnlyExclusion(PopularityRecommender):
    """고치기 전 규칙: 넣은 이력에 있는 책만 추천에서 뺀다."""

    def recommend(self, history, k=10, read=None):
        return super().recommend(history, k)


def main():
    split = load_split()
    scores = {}
    for name, model in {"before": HistoryOnlyExclusion(), "after": PopularityRecommender()}.items():
        scores[name] = evaluate_cold(model.fit(split.train), split, on="tune")[["precision@10", "ndcg@10"]]
    table = pd.concat(scores, names=["rule"]).reset_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "A5_middle_book_fix.csv", index=False)
    print(table.round(4).to_string(index=False))

    plot_style.setup()
    fig, ax = plt.subplots()
    labels = {"before": "고치기 전 (넣은 이력만 제외)", "after": "고친 후 (앞 10권 모두 제외)"}
    styles = {"before": dict(color="#898781", linestyle="--"), "after": dict(color="#2a78d6")}
    # NOTE: 두 선이 10에서 만나므로 고친 후는 점 위에, 고치기 전은 점 아래에 값을 적는다.
    offsets = {"before": -16, "after": 8}
    for name, group in table.groupby("rule"):
        ax.plot(group["history_size"], group["ndcg@10"], marker="o", label=labels[name], **styles[name])
        for size, value in zip(group["history_size"], group["ndcg@10"]):
            ax.annotate(f"{value:.3f}", (size, value), textcoords="offset points", xytext=(0, offsets[name]), ha="center", fontsize=10)
    ax.set_ylim(0.195, 0.228)
    ax.yaxis.set_major_formatter("{x:.3f}")
    ax.set_xticks([1, 3, 5, 10])
    ax.set_xlabel("신규 독자에게 넣은 평점 수")
    ax.set_ylabel("인기순 NDCG@10 (cold_tune 2,500명)")
    ax.set_title("신규 독자 채점 규칙 수정 전후")
    ax.legend(loc="lower right")
    plot_style.save(fig, "A5_middle_book_fix")


if __name__ == "__main__":
    main()
