# B5 차원·군집 수·초기화 재검토

## 최종 판단
이번 재실험의 개선 후보 채택 여부는 **False**입니다.
최종 군집은 차원=원본, k=10입니다.
선택 표본에서 제안된 SVD 50차원·k=12은
3시드 원본 공간 평균=0.062086로 기준선 평균=0.060802보다 높았지만,
겹치지 않는 확인 표본의 차이는 -0.000493여서 채택하지 않았습니다.
이전에 채택했던 SVD50/k10 결과는 커밋6e859f5에 있으며, 개정된 태그·특징의 이번 결과와 섞지 않습니다.

## 사전 고정한 절차
차원 원본·50·100·200 x k=5·8·10·12·15·20 x k-means 시드42·7·2026의72조합을 모두 비교합니다.
SVD는 seed42/n_iter7, k-means는 n_init10/max_iter300/tol1e-4/Lloyd입니다.
모든 시드의 군집 크기100권 이상·미배정0권을 만족하는 후보 중 원본 공간 silhouette 평균 최대를 고릅니다.
동점은 낮은 차원·작은 k입니다. 배포 시드는42로 고정합니다.
채택에는 시드 평균·시드42·확인 표본의 개선이 각각0.0001을 초과해야 합니다.
SVD시드7·2026 확인에서도 크기·배정 조건과 평균 개선 기준을 충족해야 합니다.
이 임계값은 통계적 유의성 기준이 아닙니다.

선택 표본2000권(seed42), 확인 표본2000권(seed2026)은 서로 겹치지 않습니다.
두 표본 모두 군집 학습에 포함된 같은 카탈로그의 책이므로 독립 테스트셋은 아닙니다.
확인 결과를 보고 다른 후보로 재탐색하지 않았습니다. 이번에는 다른 태그 설정의 군집 탐색을 추가하지 않았습니다.

## 조합별 원본 공간 점수
| n_components | k | reference_mean | reference_std | minimum_books | eligible |
| --- | --- | --- | --- | --- | --- |
| 0 | 5 | 0.051668 | 0.000324 | 1103 | True |
| 0 | 8 | 0.060360 | 0.000712 | 392 | True |
| 0 | 10 | 0.060802 | 0.000507 | 387 | True |
| 0 | 12 | 0.059662 | 0.003529 | 379 | True |
| 0 | 15 | 0.055692 | 0.001140 | 291 | True |
| 0 | 20 | 0.053013 | 0.000416 | 197 | True |
| 50 | 5 | 0.053582 | 0.000007 | 1078 | True |
| 50 | 8 | 0.054483 | 0.000042 | 535 | True |
| 50 | 10 | 0.060103 | 0.003320 | 386 | True |
| 50 | 12 | 0.062086 | 0.000103 | 328 | True |
| 50 | 15 | 0.058264 | 0.000947 | 299 | True |
| 50 | 20 | 0.056189 | 0.000402 | 106 | True |
| 100 | 5 | 0.053240 | 0.000589 | 1075 | True |
| 100 | 8 | 0.054366 | 0.000045 | 528 | True |
| 100 | 10 | 0.059774 | 0.003512 | 384 | True |
| 100 | 12 | 0.061860 | 0.000020 | 323 | True |
| 100 | 15 | 0.058705 | 0.001881 | 295 | True |
| 100 | 20 | 0.054367 | 0.000781 | 230 | True |
| 200 | 5 | 0.053589 | 0.000017 | 1057 | True |
| 200 | 8 | 0.056581 | 0.003840 | 395 | True |
| 200 | 10 | 0.061179 | 0.000312 | 386 | True |
| 200 | 12 | 0.060379 | 0.002730 | 384 | True |
| 200 | 15 | 0.056321 | 0.000829 | 304 | True |
| 200 | 20 | 0.055043 | 0.001255 | 108 | True |

## 확인과 SVD 초기화
| candidate_id | stage | reference_silhouette | clustering_silhouette | min_books | eligible |
| --- | --- | --- | --- | --- | --- |
| confirmation_d0_k10_fit42_svd42 | confirmation | 0.063066 | 0.063066 | 387 | True |
| confirmation_d50_k12_fit42_svd42 | confirmation | 0.062573 | 0.182268 | 385 | True |
| svd_stability_d50_k12_fit42_svd7 | svd_stability | 0.062050 | 0.178548 | 386 | True |
| svd_stability_d50_k12_fit42_svd2026 | svd_stability | 0.062052 | 0.183540 | 384 | True |

확인 표본의 책별 silhouette 차이를2000회 bootstrap한 조건부95% 구간은
[-0.0017909503906640269, 0.0008383950016397745]입니다. 표본 감도 설명용이며 일반화나 선택 편향 보정 구간은 아닙니다.
SVD시드3개 원본 점수 평균은 0.062029입니다.
전체 비교표에는 원본·축소 점수를 분리하고 ARI는 같은 표현/k의 시드42 배정 기준으로 기록합니다.
시드42 자기 자신(ARI=1)을 다른 시드의 안정성인 것처럼 해석하지 않습니다.

## 최종 대표 책
| cluster | book_id | title | main_author |
| --- | --- | --- | --- |
| 0 | 1161 | Cannery Row | John Steinbeck |
| 1 | 330 | The Eye of the World (Wheel of Time, #1) | Robert Jordan |
| 2 | 1143 | Beautiful Bastard (Beautiful Bastard, #1) | Christina Lauren |
| 3 | 689 | Beautiful Ruins | Jess Walter |
| 4 | 537 | The Power of Habit: Why We Do What We Do in Life and Business | Charles Duhigg |
| 5 | 1077 | Fables, Vol. 1: Legends in Exile | Bill Willingham |
| 6 | 335 | James and the Giant Peach | Roald Dahl |
| 7 | 477 | Killing Floor (Jack Reacher, #1) | Lee Child |
| 8 | 290 | Is Everyone Hanging Out Without Me? (And Other Concerns) | Mindy Kaling |
| 9 | 928 | The Raven Boys (The Raven Cycle, #1) | Maggie Stiefvater |

군집별5권, 총50권입니다.
원래 중심거리순 목록 대비 다양성 제약 후 평균 거리 증가는 0.001600입니다.
군집 배정은 바꾸지 않고 대표 목록만 바꿉니다.
저자·명시적 시리즈를 먼저 제한하고 부족할 때 시리즈·저자·제목 제한을 순서대로 완화합니다.
각 선택의 candidate_rank/selection_rule을 CSV에 남겼습니다.
전체 태그·규칙·진단은 대표 책 표에 합쳐 별도 보조 CSV를 늘리지 않았습니다.

## 공식 사용
clustering.load_selected_clusters()는 최종 JSON을 읽어 BookFeatures와 BookClusters를 반환합니다.
adopted=False는 오류가 아니라 개선 후보를 기각하고 명시된 최종 원본 설정을 사용한다는 뜻입니다.
파일 누락·필드 누락·지원하지 않는 학습 파라미터에는 오류를 냅니다.
대표 책은 representative_books(features, clusters, books=load_books(), diversify=True)로 재현합니다.

## 검증과 파일
실제 노트북 모든 셀 실행, 최종 로더 재학습 후10000권 배정·대표 책의 ID·거리 재현을 확인했습니다.
작은 인공 데이터로 원본/SVD·태그 해석·영벡터·다양성·설정 검증을 테스트합니다.
새 원본 공간과 예전 원본 공간은 태그가 달라졌으므로 그 점수끼리는 직접적인 군집 개선 비교가 아닙니다.

- B5_clustering_improvement.ipynb: 전체 재실행 코드·출력.
- B5_improvement_comparison.csv: 72개 조합과 확인/SVD 민감도 총76행.
- B5_improvement_selected_config.json: 채택 여부·최종 설정·기준·불확실성.
- B5_improvement_assignments.csv: 최종10000권 배정.
- B5_improvement_representative_books.csv: 대표 책과 설명·진단.
- 이 문서.
- B5_improvement_dimension_comparison.png, B5_improvement_k_stability.png: 공용 save로 저장한 그림 **2개**.

수정 전 결과는 커밋 6e859f5와 그 이전 이력에 보존되어 있습니다. 현재 결과는 수정된 코드·태그 규칙으로 재실행한 값이며 이전 수치와 혼합하지 않습니다.

기존 조원 파일과 .vscode는 수정하지 않았습니다. 노트북은 실제 데이터로 다시 실행했고, 공용 코드 테스트는 작은 인공 데이터로 검사합니다.

## 참고
docs/timeline.html, B4_selected_config.json, sklearn KMeans·TruncatedSVD·silhouette_samples.
