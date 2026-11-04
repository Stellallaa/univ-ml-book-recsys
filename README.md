# univ-ml-book-recsys

2026-2 머신러닝 텀프로젝트 9조 · Goodbooks-10k 기반 도서 추천 시스템 (content-based + MF + hybrid)

## 빠른 시작

1. [uv](https://docs.astral.sh/uv/)를 설치합니다.
    - macOS·Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. 레포 루트에서 가상환경을 만듭니다. Python 3.11과 패키지가 `uv.lock`에 고정된 버전으로 설치됩니다.
    ```bash
    uv sync
    ```
3. 원본 데이터를 내려받습니다. `data/raw/`에 CSV 5개(약 100MB)가 생깁니다.
    ```bash
    uv run python scripts/download_data.py
    ```
4. 노트북은 VS Code·Jupyter에서 커널을 `.venv`로 선택해 실행합니다.

패키지를 추가할 때는 `uv add <패키지>`를 쓰고, 바뀐 `pyproject.toml`과 `uv.lock`을 함께 커밋합니다.

## 폴더 구조

| 경로 | 내용 |
| --- | --- |
| `scripts/` | 데이터 다운로드 등 실행 스크립트 |
| `src/bookrec/` | 팀 공용 코드 (모델, 평가 함수, 그래프 스타일 등) |
| `notebooks/` | 개인 실험 노트북 |
| `results/` | 실험 결과 |
| `docs/parts/` | 발표 준비용 파트별 정리(md) |
| `docs/figures/` | 발표용 그래프 이미지 |
| `data/` | 원본·가공 데이터 (커밋하지 않음) |

## 그래프 저장

PPT에 들어갈 그래프는 공용 함수로 저장합니다. 폰트, 크기(1600×900px), 모델별 색이 자동으로 맞춰집니다.

```python
import matplotlib.pyplot as plt
from bookrec.plot_style import setup, save, MODEL_COLORS

setup()
fig, ax = plt.subplots()
ax.plot(epochs, val_rmse, color=MODEL_COLORS["mf"], label="MF")
save(fig, "c1_mf_loss_curve")  # docs/figures/c1_mf_loss_curve.png
```

- 모델 색은 `MODEL_COLORS`(`baseline`, `cb`, `mf`, `hybrid`)에서 가져옵니다.
- 그래프 글자에는 특수문자 마이너스(−) 대신 키보드의 하이픈(-)을 씁니다. 폰트에 특수문자 마이너스가 없어 깨집니다.

## 데이터 출처

- Zajac, Z. (2017). [Goodbooks-10k: a new dataset for book recommendations](http://fastml.com/goodbooks-10k-a-new-dataset-for-book-recommendations/). FastML.
- 원본 저장소: [zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) (CC BY-SA 4.0)
- 책 정보와 ID는 Goodreads에서 왔고, 평점의 출처는 수집자가 "goodreads.com과 비슷한 사이트"라고만 밝혔습니다.
- 원본 데이터는 레포에 올리지 않고 다운로드 스크립트만 둡니다.

## 사용한 외부 자료

| 자료 | 라이선스 | 위치 |
| --- | --- | --- |
| 나눔고딕 폰트 (NHN) | SIL Open Font License 1.1 | `src/bookrec/assets/` |
