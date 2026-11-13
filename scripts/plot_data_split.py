"""A4 데이터 분할 구조를 발표용 그림으로 그린다. 숫자는 load_split()으로 실제로 센다."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from bookrec import plot_style
from bookrec.data import load_split

COLORS = {"train": "#2a78d6", "val": "#1baf7a", "test": "#eb6834", "tune": "#8e6ccf", "final": "#b15f9e", "box": "#f1f0ec"}


def bar(ax, x, y, width, parts, height=0.13):
    """parts = [(비율, 색, 위 글자, 아래 글자)]를 가로로 이어 그린다."""
    for share, color, label, note in parts:
        w = width * share
        ax.add_patch(Rectangle((x, y), w, height, facecolor=color, edgecolor="white", linewidth=2))
        ax.text(x + w / 2, y + height / 2, label, ha="center", va="center", color="white", fontsize=11 if share > 0.3 else 9.5, linespacing=1.2)
        ax.text(x + w / 2, y - 0.035, note, ha="center", va="top", fontsize=8, color="#3a3a38", linespacing=1.4)
        x += w


def main():
    split = load_split()
    count = {name: (len(table), table["user_id"].nunique()) for name, table in vars(split).items()}
    total = sum(rows for rows, _ in count.values())
    warm_users, cold_users = count["train"][1], count["cold_tune"][1] + count["cold_final"][1]

    plot_style.setup()
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    # NOTE: FancyBboxPatch(좌하단, 너비, 높이, boxstyle)는 모서리가 둥근 상자다. pad는 글자와 테두리 사이 여백이다.
    ax.add_patch(FancyBboxPatch((0.3, 0.84), 0.4, 0.1, boxstyle="round,pad=0.01", facecolor=COLORS["box"], edgecolor="#898781"))
    ax.text(0.5, 0.89, f"원본 평점 {total:,}건 · 독자 {warm_users + cold_users:,}명", ha="center", va="center", fontsize=12)
    for x in (0.27, 0.77):
        ax.annotate("", xy=(x, 0.7), xytext=(0.5, 0.835), arrowprops=dict(arrowstyle="->", color="#898781", lw=1.5))

    ax.text(0.03, 0.66, f"일반 독자 {warm_users:,}명", fontsize=12)
    ax.text(0.03, 0.615, "독자마다 자기 평점을 시간순으로 나눔", fontsize=9.5, color="#5a5a57")
    bar(ax, 0.03, 0.42, 0.57, [
        (0.70, COLORS["train"], "train 70%", f"{count['train'][0]:,}건\n모델 학습"),
        (0.15, COLORS["val"], "val\n15%", f"{count['val'][0]:,}건\n튜닝"),
        (0.15, COLORS["test"], "test\n15%", f"{count['test'][0]:,}건\n최종(1회)"),
    ])

    ax.text(0.64, 0.66, f"신규 독자 {cold_users:,}명", fontsize=12)
    ax.text(0.64, 0.615, "무작위로 먼저 떼어 학습에 전혀 쓰지 않음", fontsize=9.5, color="#5a5a57")
    bar(ax, 0.64, 0.42, 0.33, [
        (0.5, COLORS["tune"], "cold_tune", f"{count['cold_tune'][1]:,}명 · {count['cold_tune'][0]:,}건\n튜닝"),
        (0.5, COLORS["final"], "cold_final", f"{count['cold_final'][1]:,}명 · {count['cold_final'][0]:,}건\n최종(1회)"),
    ])

    # NOTE: 신규 독자 한 명의 평점을 시간순으로 펼쳐 이력·이미 읽은 책·정답 구간을 보여 준다.
    ax.text(0.03, 0.2, "신규 독자 한 명의 채점", fontsize=10.5)
    cells = 16
    for i in range(cells):
        color = COLORS["tune"] if i < 3 else ("#d9d7d0" if i < 10 else "#f7c9a8")
        ax.add_patch(Rectangle((0.3 + i * 0.04, 0.17), 0.036, 0.07, facecolor=color, edgecolor="white"))
    ax.text(0.3 + 1.5 * 0.04, 0.12, "넣는 이력 (1·3·5·10개)", ha="center", va="top", fontsize=8.5)
    ax.text(0.3 + 6.5 * 0.04, 0.12, "이미 읽은 나머지 책", ha="center", va="top", fontsize=8.5)
    # NOTE: 앞 10칸 아래에 괄호 선을 그어, 넣는 이력까지 포함한 10권 전체가 추천에서 빠진다는 것을 보여 준다.
    ax.plot([0.3, 0.3 + 10 * 0.04 - 0.004], [0.06, 0.06], color="#5a5a57", lw=1)
    ax.text(0.3 + 5 * 0.04, 0.045, "앞 10권은 모두 추천에서 제외", ha="center", va="top", fontsize=8.5, color="#3a3a38")
    ax.text(0.3 + 13 * 0.04, 0.12, "11번째부터 4점 이상 = 정답", ha="center", va="top", fontsize=8.5)

    ax.set_title("데이터 분할", fontsize=14, pad=4)
    plot_style.save(fig, "A4_data_split")


if __name__ == "__main__":
    main()
