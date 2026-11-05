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

## 작업 방식

- `main`에 직접 push하지 않습니다. 브랜치를 만들어 작업하고 PR로 병합합니다. 브랜치 이름은 `타입/작업ID-설명`으로 짓습니다(예: `feat/b3-content-based`).
- PR은 다 같이 리뷰할 수 있도록 합니다.
- 작업 카드는 [9조 작업 보드](https://github.com/users/Stellallaa/projects/1)에서 관리합니다. 카드 하나가 Issue 하나이고, 제목은 `[작업ID] 이름`입니다. 작업을 시작하면 카드를 In Progress로 옮깁니다.
- PR 본문에 `Closes #이슈번호`를 적어 카드와 연결합니다. PR이 머지되면 Issue가 닫히고 카드가 Done으로 옮겨집니다.
- 의사결정이나 구현에 관한 질문은 해당 카드의 Issue에서 주고받습니다.
- PR 제목은 `타입(작업 ID): 소문자로 시작하는 짧은 영어 요약`으로 씁니다. 작업 ID는 [타임라인](docs/timeline.html)의 카드 번호입니다.

| 타입 | 언제 | 예시 |
| --- | --- | --- |
| `feat` | 모델, 평가 함수, 스크립트 등 기능 추가 | `feat(B3): add content-based recommender` |
| `fix` | 버그 수정 | `fix(A4): exclude cold-start users from training split` |
| `exp` | 실험·튜닝 결과(노트북, 결과표) | `exp(C2): tune MF latent factors` |
| `docs` | README, `docs/parts/` 정리 자료 | `docs(S2): add MF part summary and script` |
| `chore` | 환경, 설정, 패키지 추가 | `chore(A1): set up repo with uv, data download, and shared plot style` |

## 테스트

```bash
uv run pytest
```

| 규칙 | 내용 |
| --- | --- |
| 위치·이름 | `tests/test_<모듈>.py`, 함수는 `test_<무엇을_확인>` |
| 구조 | 모든 테스트에 `# Given` / `# When` / `# Then` 주석을 답니다 |
| 대상 | 공개 메서드(`fit`, `score`, `recommend` 등)의 동작을 확인하고, 내부 helper는 직접 테스트하지 않습니다 |
| 데이터 | 실제 CSV 대신 테스트 안에서 작은 표를 직접 만듭니다. 원본 데이터가 없는 환경에서도 돌아가야 합니다 |
| 범위 | 공용 코드(`src/bookrec/`)는 테스트를 필수로 하고, 노트북 실험은 제외합니다 |
| 실행 시점 | PR을 올리기 전에 `uv run pytest`가 통과해야 합니다 |

- `tests/`는 지금은 한 폴더에 둡니다. 테스트가 많아지면 `src/bookrec/`과 같은 폴더 구조로 나눕니다.

## 폴더 구조

각 폴더의 README에 그 폴더에서 하는 작업이 적혀 있습니다.

| 경로 | 내용 | 작업 |
| --- | --- | --- |
| `scripts/` | 데이터 다운로드 등 실행 스크립트 | A1 |
| `src/bookrec/data/` | 전처리, 태그 정제, 데이터 분할, 신규 독자 실험 세트 | A3, B1, A4, A7 |
| `src/bookrec/models/` | 공통 인터페이스, 인기순 기준선 | A2, A6 |
| `src/bookrec/models/content_based/` | content-based filtering | B2~B5 |
| `src/bookrec/models/collaborative_filtering/` | model-based collaborative filtering | C1~C3 |
| `src/bookrec/models/hybrid/` | hybrid filtering | D3 |
| `src/bookrec/evaluation/` | 공통 평가 함수, 결과 집계 | A5, D4 |
| `src/bookrec/plot_style.py` | 공용 그래프 함수 | A1 |
| `tests/` | 공용 코드 테스트 | 전원 |
| `notebooks/` | 담당별 실험 노트북 (`data/`, `content_based/`, `collaborative_filtering/`, `hybrid/`, `eda/`) | 전원 |
| `results/` | 담당별 실험 결과·기록 (`data_evaluation/`, `content_based/`, `collaborative_filtering/`, `hybrid/`), 최종 결과표(`final/`) | 전원 |
| `docs/data.md` | 원본 데이터 파일·컬럼 설명 | A1 |
| `docs/timeline.html` | 역할 분담·작업 타임라인 (내려받아 브라우저로 열기) | - |
| `docs/parts/` | 발표 준비용 파트별 정리(md) | S2 |
| `docs/figures/` | 발표용 그래프 이미지 | 전원 |
| `data/` | 원본·가공 데이터 (커밋하지 않음) | - |

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
