# B4 CB 설정 비교

## 과정과 사전 기준
태그 정제 2종(`basic`, `extended`), 책 가중치 3종(`binary`, `count`, `tfidf`), 독자 가중치 2종(`positive`, `rating`)을 조합한 12개 후보를 먼저 비교합니다. TF-IDF의 로그 TF를 끈 4개 후보도 이 단계에 포함합니다.
그중 평균 점수 상위 3개는 `min_df`를 1·5·10으로 바꿔 비교합니다. 실행 전에 정한 규칙으로 중복 설정을 제외하고 총 25개를 평가했습니다.

`cold_tune`의 이력 1·3·5·10개 조건에서 NDCG@10을 같은 비중으로 평균해 후보를 고릅니다. 동점이면 `warm_val` NDCG@10, 설정 ID 순으로 정합니다. 인기순은 비교 기준선입니다.
모든 후보는 `train`으로 준비하고 기존 독자 `val` 표본 1,000명(시드 42)과 신규 독자 `cold_tune` 2,500명을 평가합니다.
test·cold_final은 평가하지 않았습니다.

## 후보 순위
| candidate_id | cold_mean_ndcg@10 | warm_ndcg@10 | extended | weighting | min_df | sublinear_tf | profile_weighting |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C17 | 0.183624 | 0.050634 | False | binary | 1 | True | positive |
| C01 | 0.183285 | 0.050609 | False | binary | 2 | True | positive |
| C18 | 0.181950 | 0.049593 | False | binary | 5 | True | positive |
| C20 | 0.180905 | 0.049564 | True | binary | 1 | True | positive |
| C07 | 0.180773 | 0.050599 | True | binary | 2 | True | positive |
| C19 | 0.180570 | 0.049415 | False | binary | 10 | True | positive |
| C21 | 0.178908 | 0.049894 | True | binary | 5 | True | positive |
| C22 | 0.176942 | 0.048054 | True | binary | 10 | True | positive |
| C23 | 0.175369 | 0.052847 | False | tfidf | 1 | True | positive |
| C05 | 0.173558 | 0.050446 | False | tfidf | 2 | True | positive |
| C24 | 0.171030 | 0.044791 | False | tfidf | 5 | True | positive |
| C11 | 0.170137 | 0.049679 | True | tfidf | 2 | True | positive |
| C25 | 0.168332 | 0.045941 | False | tfidf | 10 | True | positive |
| C02 | 0.166816 | 0.046136 | False | binary | 2 | True | rating |
| C08 | 0.164229 | 0.046389 | True | binary | 2 | True | rating |
| C06 | 0.158316 | 0.047511 | False | tfidf | 2 | True | rating |
| C12 | 0.154536 | 0.045061 | True | tfidf | 2 | True | rating |
| C13 | 0.126633 | 0.028788 | False | tfidf | 2 | False | positive |
| C15 | 0.117069 | 0.023624 | True | tfidf | 2 | False | positive |
| C03 | 0.111791 | 0.023184 | False | count | 2 | True | positive |
| C14 | 0.108992 | 0.024600 | False | tfidf | 2 | False | rating |
| C16 | 0.099147 | 0.020994 | True | tfidf | 2 | False | rating |
| C09 | 0.098878 | 0.019820 | True | count | 2 | True | positive |
| C04 | 0.094948 | 0.022822 | False | count | 2 | True | rating |
| C10 | 0.082637 | 0.017018 | True | count | 2 | True | rating |

## 선택
후보 C17: {'extended': False, 'weighting': 'binary', 'min_df': 1, 'sublinear_tf': True, 'profile_weighting': 'positive'}

검증 NDCG@10은 신규 독자 이력 조건 평균 0.183624, 기존 독자 표본 0.050634입니다.
이는 탐색한 후보 내 선택이며 전체 가능한 설정의 최적해를 뜻하지 않습니다.
후속 min_df 실험도 같은 검증 데이터로 선택했으므로 선택 편향이 남습니다.

binary에서는 sublinear_tf 값이 점수에 영향을 주지 않습니다.

## 같은 독자의 점수 차이
독자 단위로 2,000회 재표집한 백분위수 부트스트랩 구간입니다(시드 42).
cold_mean은 독자마다 네 이력의 NDCG 평균을 먼저 구합니다.
| goal | against | users | mean_difference | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- |
| warm_val | popularity | 996 | 0.009202 | 0.000518 | 0.018144 |
| cold_mean | popularity | 2500 | -0.037699 | -0.047276 | -0.028284 |
| warm_val | C11 | 996 | 0.000955 | -0.004286 | 0.006299 |
| cold_mean | C11 | 2500 | 0.013487 | 0.010841 | 0.016170 |
| warm_val | C01 | 996 | 0.000026 | -0.001248 | 0.001174 |
| cold_mean | C01 | 2500 | 0.000340 | -0.000041 | 0.000722 |

이 구간은 검증 데이터에 조건부인 비교이며 후보 선택 편향을 보정하지 않습니다.
최종 평가나 일반화 성능의 확정 근거가 아닙니다.
1·2위 점수 차이의 구간은 0을 포함합니다.
신규 독자 평균에서는 선택 CB가 인기순보다 낮습니다.

## 실행
최종 추천기는 `tuning.load_selected_content_based()`로 불러옵니다. 선택 JSON이나 필수 필드가 누락되면 오류가 발생합니다. `ContentBasedConfig()`의 기본값은 초기 실험 재현에 사용합니다.
전체 비교는 B4_configuration_compare.ipynb를 순서대로 실행합니다.
CSV는 후보·검증 지표·대체 비율·시간·순위 기록, JSON은 최종 선택과 bootstrap 조건 기록입니다.
그래프 2개는 공용 `bookrec.plot_style.save()`로 저장합니다.
