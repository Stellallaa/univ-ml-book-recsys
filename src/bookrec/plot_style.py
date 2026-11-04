"""팀 공통 그래프 스타일과 저장 함수."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_PATH = Path(__file__).parent / "assets" / "NanumGothic-Regular.ttf"
FIG_DIR = Path(__file__).resolve().parents[2] / "docs" / "figures"
FIG_SIZE = (8, 4.5)  # NOTE: PPT 16:9, dpi 200 기준 1600x900px

MODEL_COLORS = {"baseline": "#898781", "cb": "#2a78d6", "mf": "#eb6834", "hybrid": "#1baf7a"}


def setup():
    """팀 공통 그래프 스타일을 적용한다."""
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=FONT_PATH).get_name(),
        "axes.unicode_minus": False,
        "figure.figsize": FIG_SIZE,
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig, name):
    """그래프를 docs/figures/{name}.png로 저장한다."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(*FIG_SIZE)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
