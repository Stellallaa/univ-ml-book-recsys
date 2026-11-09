# B5 책 군집과 대표 책

## 과정

B4_selected_config.json의 C01 설정을 읽어 기존 B1~B4 함수로 책 특징을 재생성했습니다. basic 정제, binary 책 벡터, min_df=2, L2 정규화된 10,000권 x 17,042개 태그입니다. profile_weighting은 독자 취향 설정이므로 책 군집에는 영향을 주지 않습니다.

다른 팀원의 파일, B4 결과, .vscode는 수정하지 않았습니다. 독자 평점과 최종 평가 데이터는 사용하지 않았습니다. 이번 구현은 책 군집과 대표 책 선정까지이며 가입 화면은 포함하지 않습니다.

- k 후보: 5, 10, 15, 20.
- KMeans: k-means++ 초기화, n_init=10, max_iter=300, tol=1e-4, algorithm=lloyd, random_state=42.
- 영벡터 책은 제외하고 결과에서 군집 -1로 표시합니다. 실제 데이터에는 영벡터가 없습니다.
- inertia는 전체 학습 책의 군집 중심까지 제곱거리 합입니다.
- silhouette는 후보마다 같은 표본 2,000권(seed=42)의 Euclidean 거리로 계산했습니다.
- 미리 정한 기준은 최소 군집 100권 이상인 후보 중 silhouette 최대, 동점이면 작은 k입니다. 조건을 만족하는 후보가 없으면 중단하도록 구현했습니다.

## 군집 수 선택

| k | inertia | silhouette | sample_books | min_books | max_books | unassigned_books | n_iter | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 7425.748010 | 0.052822 | 2000 | 1100 | 3920 | 0 | 12 | True |
| 10 | 6885.840286 | 0.062676 | 2000 | 393 | 1546 | 0 | 17 | True |
| 15 | 6630.583214 | 0.056178 | 2000 | 385 | 990 | 0 | 28 | True |
| 20 | 6458.223473 | 0.053485 | 2000 | 164 | 981 | 0 | 40 | True |

모든 후보가 최소 군집 크기 조건을 만족했습니다. silhouette가 가장 높은 **k=10**을 선택했습니다. 이는 네 후보와 고정된 초기화 설정 내의 선택 결과입니다.

## 군집별 대표 책

대표 책은 해당 군집 내에서 중심까지 Euclidean 거리가 가까운 5권씩, 총 50권입니다. 동점이면 작은 book_id를 먼저 둡니다. 아래는 군집별 첫 번째 대표 책이며, 전체 50권은 B5_representative_books.csv에 있습니다.

| cluster | books | distinctive_tags | title |
| --- | --- | --- | --- |
| 0 | 1301 | literature, classics, literary-fiction, classic | Cannery Row |
| 1 | 393 | paranormal-romance, paranormal-fantasy, urban-fantasy, paranormal | Hunting Ground (Alpha & Omega, #2) |
| 2 | 1385 | mystery-thriller, mystery-suspense, suspense, thriller | Killing Floor (Jack Reacher, #1) |
| 3 | 1546 | nonfiction, non-fiction, to-read-non-fiction, biography | The Art of Asking; or, How I Learned to Stop Worrying and Let People Help |
| 4 | 1061 | scifi-fantasy, sci-fi-fantasy, science-fiction-fantasy, fantasy-sci-fi | The Eye of the World (Wheel of Time, #1) |
| 5 | 1264 | general-fiction, contemporary-fiction, bookclub, adult-fiction | The Light Between Oceans |
| 6 | 516 | contemporary-romance, new-adult, alpha-male, 4-stars | Fallen Too Far (Rosemary Beach, #1; Too Far, #1) |
| 7 | 1203 | teen, ya-fiction, ya, ya-books | The Raven Boys (The Raven Cycle, #1) |
| 8 | 938 | children-s-books, kids-books, kids, children-s | James and the Giant Peach |
| 9 | 393 | graphic-novel, graphic-novels, comic, comics | Fables, Vol. 1: Legends in Exile |

- top_tags: 군집 중심의 가중치가 높은 태그.
- distinctive_tags: 군집 중심 가중치에서 전체 비영벡터 책의 평균 가중치를 뺀 차이가 높은 태그.
- 차이가 양수인 태그만 distinctive_tags에 표시합니다. 이 값은 장르 정답이나 통계적 유의성을 뜻하지 않습니다.
- 군집 크기는 393~1,546권이며, 10,000권 모두 배정됐습니다.

## 해석과 한계

태그에서 문학·고전, 초자연 로맨스, 미스터리, 논픽션, 판타지·SF, 현대 소설, 로맨스, 청소년, 어린이, 그래픽노블 등의 경향을 읽을 수 있습니다. 이는 대표 태그와 제목을 보고 한 해석이며, 모델이 해당 장르 이름을 학습하거나 정답 장르로 평가된 결과는 아닙니다. 군집 번호 자체에 의미나 순서는 없습니다.

선택 silhouette는 약 0.062676으로 0에 가깝습니다. 여러 태그가 겹치는 책들을 뚜렷하게 분리했다고 주장할 수 없습니다. 고차원 특징과 장르 중첩의 영향이 있을 수 있지만 이 실험만으로 원인을 입증하지는 않습니다. 표본 내부 지표이며 추천 정확도 개선이나 최종 평가 점수가 아닙니다.

B4의 basic 정제 설정을 유지했으므로 favorites, 별점, 독서 연도와 일부 to-read 변형 태그가 남아 있습니다. 군집은 순수한 내용 정보만으로 나뉜 것이 아닐 수 있습니다. 이 결과를 보고 B1이나 B4 선택 설정을 임의로 바꾸지는 않았습니다.

중심과 가까운 책을 고르는 방식이라 한 시리즈나 한 저자가 여러 대표 책을 차지할 수 있습니다. 예를 들어 그래픽노블 군집에는 Scott Pilgrim 시리즈 여러 권이 포함됩니다. 실제 가입 화면에서 다양성을 높이려면 별도의 시리즈·저자 중복 제한 정책이 필요합니다. 현재 50권은 중심 거리 기준의 결과입니다.

## 검증

기존 66개와 B5 테스트 사례 10개를 포함해 전체 76개 테스트가 통과했습니다.

- 작은 인공 데이터의 알려진 두 군집을 정확히 분리하고, 영벡터 책을 -1로 유지.
- 같은 시드로 학습한 군집 재현, 입력 행렬 불변성.
- 고정 표본 silhouette와 군집 크기, 최소 군집 크기·동점 선택 규칙.
- 대표 책의 소속, 거리와 동점 순서, 태그 설명.
- 잘못된 k·표본 크기·중복 후보·책 ID 순서 불일치 거부.
- 실제 선택 k=10을 같은 시드로 재학습하여 10,000권의 군집 번호가 저장 결과와 정확히 일치함을 확인.
- 실제 대표 책 50권의 중복 없음·소속·제목 결측 없음·거리 순서 확인.
- 노트북 코드 셀 전체 실행 및 오류 출력 없음.
- 공용 save() 두 번 호출과 PNG 2개의 1600 x 900 크기를 확인하고 그림 배치도 점검.

현재 제한된 실행 환경에서 joblib이 물리 코어 수 조회 경고를 내고 논리 코어 수로 대체했습니다. 학습·테스트는 정상 완료됐고, 같은 시드의 군집 결과도 재현됐습니다.

```bash
uv run pytest
```

## 재현

```python
import json
from pathlib import Path
from bookrec.data import load_book_tags
from bookrec.models.content_based.tuning import ContentBasedConfig, make_content_based
from bookrec.models.content_based.clustering import fit_book_clusters, representative_books

# 레포 루트에서 실행합니다.
saved = json.loads(Path("results/content_based/B5_selected_clustering.json").read_text())
features = make_content_based(load_book_tags(), ContentBasedConfig(**saved["feature_config"])).features
clusters = fit_book_clusters(features, saved["k"], seed=saved["seed"])
representatives = representative_books(features, clusters, n_books=5)
```

모든 후보와 그래프까지 재실행하려면 notebooks/content_based/B5_book_clustering.ipynb를 위에서부터 실행합니다. 노트북의 그림 저장은 bookrec.plot_style.save()를 사용하며 공용 파일은 수정하지 않았습니다.

## 산출물

- src/bookrec/models/content_based/clustering.py: 군집 학습·비교·선택·태그 요약·대표 책 함수.
- tests/test_book_clustering.py: 작은 인공 데이터 기반 공개 함수 테스트.
- notebooks/content_based/B5_book_clustering.ipynb: 실제 실행 코드와 결과.
- B5_cluster_comparison.csv: k별 inertia·silhouette·군집 크기·반복 횟수.
- B5_assignments.csv: 10,000권의 책 ID와 군집 번호. 전처리 입력이 아닌 모델 실험 결과입니다.
- B5_cluster_overview.csv: 군집별 책 수와 태그.
- B5_representative_books.csv: 군집별 대표 5권의 ID·거리·제목·대표 저자.
- B5_selected_clustering.json: 선택한 k·특징 설정·학습 및 선택 조건.
- docs/figures/B5_cluster_count_comparison.png: k 결정 그래프.
- docs/figures/B5_cluster_sizes.png: 선택한 군집의 책 수.

## 참고

- docs/timeline.html: B5의 k-means, 군집 수 결정 그래프와 대표 책 산출물.
- src/bookrec/models/content_based/README.md: B2 특징을 B5에 재사용하는 역할.
- B4_selected_config.json: 선택한 책 특징 설정.
- scikit-learn KMeans, silhouette_score, euclidean_distances: 사용한 설정과 거리 기준은 위 과정 및 코드 주석에 명시했습니다.
