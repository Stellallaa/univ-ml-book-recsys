# B5 silhouette 진단과 SVD 개선 실험

## 결과

기존 basic/binary/min_df=2 특징을 유지하면서 **TruncatedSVD 50차원 → L2 재정규화 → k-means(k=10)**를 추가 결과로 채택했습니다. B4의 추천 설정은 변경하지 않았습니다. 기존 B5 파일도 덮어쓰지 않고 기준선으로 보존했습니다.

| 평가 | 기존 원본 k=10 | SVD 50차원 k=10 |
| --- | ---: | ---: |
| 같은 원본 공간·표본 silhouette (학습 시드 42) | 0.062676 | 0.063772 |
| 실제 군집 공간 silhouette (학습 시드 42) | 0.062676 | 0.178420 |
| 같은 원본 공간·표본, silhouette 음수 책 비율 | 6.95% | 3.75% |
| 같은 원본 공간·표본, 학습 시드 3개 평균 | 0.062416 | 0.063222 |
| 별도 표본, 같은 원본 공간 silhouette | 0.062817 | 0.065548 |

**직접 비교 가능한 개선은 같은 원본 공간의 0.062676 → 0.063772입니다.** 0.178420은 차원 축소 후 공간에서 계산한 값이라 기존 원본 점수와 같은 기준의 개선으로 해석하지 않습니다. 시드 42의 절대 개선은 약 0.001096, 상대 개선은 약 1.75%입니다. 여전히 원본 공간에서는 군집 중첩이 큽니다.

## 보존과 변경 범위

- 수정: B5에서 작성한 clustering.py, test_book_clustering.py 두 파일.
- 추가: 개선 노트북, B5_improvement_* 결과, 같은 접두어 그래프 3개.
- 기존 B1~B4 파일과 다른 팀원 파일, 의존성 설정, .vscode는 수정하지 않았습니다.
- 기존 B5 노트북·결과·그래프와 B4_selected_config.json의 SHA-256을 전후 비교했습니다. 해시는 노트북 실행 중 메모리에서 비교하고, 별도 파일로 저장하지 않습니다.
- 기존 fit_book_clusters(features, k) 호출은 SVD 없이 동작합니다. 새 결과를 쓰려면 개선 JSON과 prepare_cluster_space를 명시적으로 사용해야 합니다.
- 독자 평점·test·cold_final은 이번 실험에서 사용하지 않았습니다.

## 구현

### 원본 특징과 군집 좌표 분리

- BookFeatures는 기존 책 x 태그 행렬과 ID·태그 이름을 그대로 보존합니다.
- ClusterSpace는 원본 또는 SVD 군집 좌표, 책 ID, 축소 차원과 설명 분산 비율을 담습니다.
- prepare_cluster_space(features, n_components=None)는 원본 공간을 반환합니다.
- n_components를 지정하면 TruncatedSVD(n_iter=7, random_state=42)를 비영벡터 책에 학습하고 L2 재정규화합니다. 원본 영벡터는 0으로 유지합니다.
- SVD의 음수 좌표는 허용합니다. 태그의 비음수 가중치와 다른 개념입니다.
- fit_book_clusters의 space 인자를 통해 같은 SVD 좌표를 여러 k와 학습 시드에서 재사용합니다.
- 군집 설명 태그는 원본 특징에서 군집별 평균을 계산합니다. SVD 축을 태그 이름과 직접 대응시키지 않습니다.
- 대표 책은 실제 군집 좌표에서 중심과의 Euclidean 거리 순서로 고릅니다. 동점은 작은 book_id 순서입니다.

### 평가 공간 구분

cluster_diagnostics는 같은 표본의 책별 점수와 군집별 평균·음수 비율을 반환합니다.

- original_silhouette: 해당 후보의 축소 전 특징 공간.
- clustering_silhouette: 실제 군집에 사용한 공간. SVD가 없으면 original과 같습니다.
- reference_silhouette: 모든 후보를 공정하게 비교하는 고정 C01 원본 특징 공간.

현재 채택된 후보는 태그 설정을 바꾸지 않아 original과 reference가 같습니다. 추후 다른 태그 설정을 시험해도 reference는 그대로 고정할 수 있습니다. 표본에 영벡터나 미배정 책이 있으면 자동으로 제외하지 않고 오류를 내어 평가 대상 변경을 막습니다.

## 실험 기준

평가 전 노트북에 후보와 기준을 고정했습니다.

1. 차원 비교: 원본·50·100·200차원, k=10, 학습 시드=42.
2. 표현 선택: 최소 군집 100권·미배정 0권을 만족하는 후보 중 고정 원본 공간 silhouette 우선. 동점은 낮은 차원·후보 ID 순서.
3. 원본 기준에서 SVD가 0.0001 이상 개선되지 않으면 basic/extended 및 binary/TF-IDF의 다른 조합 3개를 원본·유망 SVD 차원에서 비교합니다. 이번에는 SVD가 개선되어 이 조건부 탐색을 실행하지 않았습니다.
4. 안정성: 선택 표현을 고정하고 k=5·8·10·12·15·20과 학습 시드 42·7·2026을 비교합니다. SVD 시드는 42로 고정해 k-means 초기화 영향과 분리했습니다.
5. 모든 학습 시드에서 군집 크기·배정 조건을 만족하는 k 중 공통 원본 점수 평균 최대를 선택합니다. 동점은 작은 k입니다.
6. 원본 k=10도 같은 세 학습 시드로 비교합니다.
7. 채택 조건: 세 시드 평균 개선, 시드 42 선택 표본 개선, 별도 표본 개선이 각각 0.0001을 초과해야 합니다. 배포 결과는 최고 시드를 고르지 않고 시드 42로 고정합니다.

0.0001은 실용적인 최소 차이 기준이며 통계적 유의성 임계값이 아닙니다. 선택 표본은 seed=42의 고정 2,000권, 별도 표본은 seed=2026의 2,000권입니다. 일부 책이 중복될 수 있으며 같은 카탈로그에서의 감도 확인입니다. 독립적인 일반화 평가가 아닙니다. 최종 후보를 고정한 뒤 별도 표본을 확인했고, 이를 보고 다른 후보로 다시 탐색하지 않았습니다.

## 기존 결과 진단

기존 k=10 군집 배정 10,000권과 silhouette를 먼저 재현했습니다. 선택 표본에서 군집 0(문학·고전 경향)은 음수 비율 약 23.02%, 군집 3(논픽션 경향)은 약 12.65%였습니다. 서로 다른 내용의 경계나 넓은 주제 묶음일 가능성이 있지만, 이 진단만으로 원인을 확정하지 않습니다.

군집별 수치와 점수가 낮은 책 사례는 노트북의 기존 결과 진단 셀에 출력했습니다. 최종 군집 진단은 대표 책 CSV에 군집 정보와 함께 합쳤습니다. 음수 책은 삭제하지 않았습니다.

## 차원 비교

| candidate_id | reference_silhouette | clustering_silhouette | reference_negative_rate | explained_variance_ratio |
| --- | --- | --- | --- | --- |
| baseline | 0.062676 | 0.062676 | 0.069500 | nan |
| svd_50 | 0.063772 | 0.178420 | 0.037500 | 0.427776 |
| svd_100 | 0.063395 | 0.138926 | 0.057500 | 0.517602 |
| svd_200 | 0.063290 | 0.111406 | 0.064500 | 0.620680 |

50차원이 같은 원본 공간에서 가장 높은 점수를 보였습니다. 설명 분산 비율 합은 약 0.427776입니다. 이 수치는 군집 정확도나 정답 의미를 보존한 비율을 뜻하지 않습니다.

## k와 시드 비교

| k | reference_mean | reference_std | clustering_mean | minimum_books | eligible | mean_ari_vs_seed42 |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.055223 | 0.000000 | 0.144674 | 1084 | True | 0.999891 |
| 8 | 0.057607 | 0.002152 | 0.159085 | 394 | True | 0.944972 |
| 10 | 0.063222 | 0.000476 | 0.177196 | 388 | True | 0.899860 |
| 12 | 0.060484 | 0.002476 | 0.176143 | 356 | True | 0.885820 |
| 15 | 0.059607 | 0.002587 | 0.177566 | 228 | True | 0.924142 |
| 20 | 0.057392 | 0.001126 | 0.179281 | 109 | True | 0.870001 |

k=10이 공통 원본 공간 점수 평균으로 가장 높았습니다. k=12가 시드 42에서 축소 공간 silhouette는 더 높지만, 같은 원본 공간·세 시드 평균 선택 기준으로는 선택되지 않았습니다.

표의 mean_ari_vs_seed42는 시드 42 자신(ARI=1)을 포함한 세 행의 평균입니다. 같은 번호의 군집끼리 직접 맞추지 않고 ARI로 초기화에 따른 배정 변화를 확인했습니다. 시드가 달라져도 군집이 완전히 같다는 뜻은 아닙니다. 그래프의 오차 막대는 세 시드 표준편차이며 신뢰구간이 아닙니다.

## 별도 표본 확인

| candidate_id | sample | sample_seed | reference_silhouette |
| --- | --- | --- | --- |
| baseline | confirmation | 2026 | 0.062817 |
| proposed | confirmation | 2026 | 0.065548 |

별도 표본의 같은 원본 공간 개선은 약 0.002731이었습니다. 세 조건을 충족해 개선 설정을 별도 산출물로 채택했습니다. 추가적인 태그 정제·가중치 탐색은 수행하지 않았습니다.

## 최종 군집과 해석

10,000권 모두 배정됐고, 군집 크기는 389~1,377권입니다. 군집별 대표 책 5권씩 총 50권을 선정했습니다.

| cluster | book_id | title | main_author |
| --- | --- | --- | --- |
| 0 | 5047 | The Scar (Bas-Lag, #2) | China Miéville |
| 1 | 4237 | The Magician's Assistant | Ann Patchett |
| 2 | 1585 | Forever (The Wolves of Mercy Falls, #3) | Maggie Stiefvater |
| 3 | 2055 | The Woods | Harlan Coben |
| 4 | 6384 | The Happiness Hypothesis: Finding Modern Truth in Ancient Wisdom | Jonathan Haidt |
| 5 | 4419 | My Father's Dragon (My Father's Dragon, #1) | Ruth Stiles Gannett |
| 6 | 264 | The Sun Also Rises | Ernest Hemingway |
| 7 | 6461 | The Umbrella Academy, Vol. 1: The Apocalypse Suite | Gerard Way |
| 8 | 1595 | Long Walk to Freedom | Nelson Mandela |
| 9 | 9067 | The Golden Dynasty (Fantasyland, #2) | Kristen Ashley |

대표 태그를 보면 판타지·SF, 현대 소설, 청소년, 미스터리, 철학·자기계발, 어린이, 문학·고전, 그래픽노블, 전기·회고록, 로맨스 등의 경향을 해석할 수 있습니다. 군집 번호는 기존 결과와 대응하지 않으며, 장르 정답이 아닙니다. 기존 초자연 로맨스 묶음과 논픽션 묶음의 배정도 달라졌습니다.

basic 정제에서 남은 favorites·별점·독서 기록 태그가 여전히 존재합니다. 내용 중심 군집인지, 실제 가입 화면에서 대표 책이 충분히 다양하고 알아보기 쉬운지는 별도 사용자 검토가 필요합니다. 원본 점수의 소폭 개선이 이러한 제품 품질이나 추천 성능 향상을 입증하지는 않습니다.

## 검증

- 기존 테스트 포함 88개 통과. 원본 경로 동작, SVD 좌표 재현·정규화·영벡터, 입력 불변성, 실제 군집 공간의 대표 책 거리, 원본 태그 해석을 확인했습니다.
- 원본·축소·고정 기준 공간 silhouette를 작은 인공 데이터의 독립 sklearn 계산과 대조했습니다.
- 노트북의 모든 코드 셀을 실제 데이터로 실행했고 오류 출력이 없습니다.
- 기존 k=10 배정과 선택 표본의 점수가 저장된 B5와 재현됐습니다.
- 개선 설정의 같은 시드 재학습, 저장 결과와 진단 일치, 10,000권 대응·대표 책 50권의 소속·거리 순서를 확인했습니다.
- 기존 B5·B4 설정 파일 해시가 유지됐습니다.
- 공용 plot_style.save를 통해 차원 비교와 k·시드 비교 그림 2개를 1600 x 900 PNG로 생성했습니다.
- 기존 환경과 같은 joblib 물리 코어 조회 경고가 발생했으나 논리 코어 대체로 학습이 완료됐습니다.

```bash
uv run pytest
```

## 재현 및 사용

```python
import json
from pathlib import Path
from bookrec.data import load_book_tags
from bookrec.models.content_based.tuning import ContentBasedConfig, make_content_based
from bookrec.models.content_based.clustering import (
    prepare_cluster_space, fit_book_clusters, representative_books,
)

saved = json.loads(Path("results/content_based/B5_improvement_selected_config.json").read_text())
features = make_content_based(load_book_tags(), ContentBasedConfig(**saved["feature_config"])).features
space = prepare_cluster_space(features, saved["n_components"], seed=saved["svd_seed"])
clusters = fit_book_clusters(features, saved["k"], seed=saved["fit_seed"], space=space)
representatives = representative_books(features, clusters, n_books=5)
```

전체 실험은 노트북을 처음부터 실행합니다. 실행 시간은 환경에 따라 달라지며 패키지 버전은 선택 JSON에 기록했습니다.

## 파일별 역할

개선 실험의 새 파일은 아래 8개만 남겼습니다. 수정한 공용 B5 코드·테스트 2개는 이 개수에 포함하지 않습니다.

| 파일 | 내용 |
| --- | --- |
| notebooks/content_based/B5_clustering_improvement.ipynb | 전체 재현 코드와 실행 출력·진단·k 요약 |
| results/content_based/B5_improvement_comparison.csv | 모든 설정의 지표·시간, 시드별 ARI, 별도 표본 결과를 합친 비교표 |
| results/content_based/B5_improvement_selected_config.json | 채택 여부·최종 설정·선택 기준·표본 seed·환경 |
| results/content_based/B5_improvement_assignments.csv | 개선 군집의 책 ID 대응 |
| results/content_based/B5_improvement_representative_books.csv | 대표 책 50권과 군집별 크기·태그·진단을 합친 표 |
| results/content_based/B5_improvement.md | 실험 설명과 재현 방법(이 문서) |
| docs/figures/B5_improvement_dimension_comparison.png | 원본·축소 공간 점수를 구분한 차원 비교 |
| docs/figures/B5_improvement_k_stability.png | k와 초기화 시드의 변동 |

비교표의 stage는 dimensions(차원 비교), features(필요 시 태그 비교), stability(k·시드 비교), baseline_stability(기준선 시드 비교), confirmation(별도 표본 확인)입니다. 이번에는 features 단계가 실행되지 않았습니다. sample_seed는 선택 표본 42 또는 확인 표본 2026을 표시합니다. ari_vs_seed42는 stability 행에서만 계산하며, 확인 단계는 학습 시간 재측정이 아니므로 seconds_fit_diagnostics를 비워 둡니다. 비교표는 총 27행입니다.

대표 책 표의 cluster_books는 해당 군집 전체 책 수입니다. top_tags와 distinctive_tags는 원래 태그 공간의 설명입니다. sample_books와 silhouette·negative_rate는 선택 표본의 군집별 진단이며 같은 군집 대표 책 5행에 반복됩니다. 책별 중심 거리와 군집 단위 평균 silhouette는 서로 다른 값입니다.

별도 표본 ID·해시·책별 전체 점수 CSV, 중복 군집 요약·진단 CSV와 군집별 진단 PNG는 제거했습니다. 표본은 JSON의 seed로 재생성하고, 상세 진단과 요약은 노트북 실행 출력으로 확인합니다. 노트북을 다시 실행해도 이 보조 파일들은 생성하지 않습니다.

## 참고

- scikit-learn 공식 문서: [텍스트 군집과 LSA](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html).
- scikit-learn 공식 문서: [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).
- 기존 B5_book_clustering.md와 B4_selected_config.json: 기준선과 원본 책 표현.
