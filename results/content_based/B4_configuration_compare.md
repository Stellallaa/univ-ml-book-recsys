# B4 CB 설정 비교

## 과정과 선택 기준

B1 태그 필터, B2 책 특징, B3 추천기, A의 분할·인기순 기준선·평가 함수를 그대로 사용했습니다. 기존 조원 파일과 .vscode는 수정하지 않았습니다.

후보와 선택 기준은 평가 전에 고정했습니다. 신규 독자 추천이 CB의 주된 목적이므로 cold_tune의 이력 1·3·5·10에 대한 NDCG@10을 동일 가중치로 평균해 선택합니다. 동점이면 warm_val NDCG@10, 설정 ID 순서입니다. 인기순은 비교용이며 CB 후보 순위에서는 제외합니다.

### 후보 16개

- 기본 비교 12개: 태그 정제 basic/extended x 책 가중치 binary/count/tfidf x 독자 가중치 positive/rating.
- 민감도 비교 4개: extended TF-IDF에서 sublinear_tf=False 2개, min_df=5 2개(각각 positive/rating).
- 나머지 기본값은 min_df=2, sublinear_tf=True입니다. binary/count에서는 sublinear_tf가 적용되지 않습니다.
- 기본 12개의 C01~C12는 정제·가중치·독자 가중치 순서로 생성합니다. C11은 B3의 초기 설정입니다.
- 전체 후보 목록은 B4_candidates.csv에 있습니다. 저자 등 새 특징은 이번 비교에 포함하지 않았습니다.

### 공통 평가 조건

- 학습: split.train만 사용, 시드 42.
- 기존 독자: val의 같은 표본 1,000명(n_users=1000, seed=42).
- 신규 독자: cold_tune 전원 2,500명, 이력 1·3·5·10개.
- 정답과 이미 읽은 책 제외 규칙은 공통 평가 함수를 따릅니다.
- test·cold_final의 채점은 수행하지 않았습니다. 이 결과로 최종 일반화 성능을 주장하지 않습니다.
- 좋아한 책 지표 평균의 기존 독자 수는 996명, 신규 독자는 2,500명입니다. to_read 지표는 각각 905명과 2,298명입니다.
- 목표별 정답 구간이 다르므로 기존 독자와 신규 독자의 점수 크기를 직접 비교하지 않습니다.

## 후보 순위

| candidate_id | cold_mean_ndcg@10 | warm_ndcg@10 | extended | weighting | min_df | sublinear_tf | profile_weighting |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | 0.183821 | 0.051571 | False | binary | 2 | True | positive |
| C07 | 0.180887 | 0.051281 | True | binary | 2 | True | positive |
| C05 | 0.174507 | 0.051094 | False | tfidf | 2 | True | positive |
| C11 | 0.170602 | 0.048862 | True | tfidf | 2 | True | positive |
| C15 | 0.167980 | 0.044470 | True | tfidf | 5 | True | positive |
| C02 | 0.167219 | 0.047142 | False | binary | 2 | True | rating |
| C08 | 0.164322 | 0.046030 | True | binary | 2 | True | rating |
| C06 | 0.159034 | 0.047254 | False | tfidf | 2 | True | rating |
| C12 | 0.154860 | 0.045743 | True | tfidf | 2 | True | rating |
| C16 | 0.151766 | 0.041355 | True | tfidf | 5 | True | rating |
| C13 | 0.117791 | 0.023933 | True | tfidf | 2 | False | positive |
| C03 | 0.112029 | 0.023210 | False | count | 2 | True | positive |
| C14 | 0.099911 | 0.021043 | True | tfidf | 2 | False | rating |
| C09 | 0.099118 | 0.019859 | True | count | 2 | True | positive |
| C04 | 0.095163 | 0.022948 | False | count | 2 | True | rating |
| C10 | 0.083092 | 0.016978 | True | count | 2 | True | rating |

## 선택 결과

선택한 설정은 **C01: basic 태그 정제 + binary 책 벡터 + min_df=2 + positive 독자 가중치**입니다. binary는 태그 count 크기 대신 태그의 존재 여부를 사용하고 L2 정규화합니다. positive는 max(rating-3, 0)입니다. sublinear_tf=True는 설정 구조상 저장되지만 binary에는 효과가 없습니다.

basic는 책장·매체 태그를 제거하되 favorites나 명시적 독서 연도 태그는 남깁니다. extended는 이 태그도 제거합니다. 선택 결과를 보고 B1 기본 동작 자체를 바꾸지는 않았습니다. 사용 시 반드시 선택 JSON의 config로 재현합니다.

선택 모델의 특징 행렬은 10000권 x 17042개 태그입니다.

| goal | history_size | precision@10 | recall@10 | ndcg@10 | to_read_hit@10 |
| --- | --- | --- | --- | --- | --- |
| warm_val | train | 0.045582 | 0.041668 | 0.051571 | 0.042210 |
| cold_tune | 1 | 0.139080 | 0.021214 | 0.151586 | 0.024369 |
| cold_tune | 3 | 0.154800 | 0.023303 | 0.168967 | 0.027937 |
| cold_tune | 5 | 0.174200 | 0.026405 | 0.186475 | 0.028285 |
| cold_tune | 10 | 0.213040 | 0.032178 | 0.228256 | 0.028460 |

초기 C11의 신규 독자 평균 NDCG@10은 0.170602, 선택 C01은 0.183821로 약 7.75% 높았습니다. 기존 독자 검증 표본 NDCG@10은 0.048862에서 0.051571로 높아졌습니다.

인기순의 신규 독자 평균 NDCG@10은 0.221323으로, 선택한 CB보다 여전히 높습니다. CB의 개선은 시험한 CB 후보와 초기 설정에 대한 개선이며, 신규 독자에서 인기순보다 우수하다는 뜻이 아닙니다. 이력 크기별 수치는 B4_validation_scores.csv에 모두 남겼습니다.

## 해석과 한계

- 시험한 조건에서는 binary가 count보다 좋았습니다. 정확한 원인은 이 비교만으로 입증할 수 없지만, 큰 태그 count가 특정 책장·인기 신호에 치우치는 영향과 관련이 있을 수 있습니다.
- 기본 비교의 같은 정제·벡터 설정에서는 positive가 rating보다 높은 신규 독자 평균 점수를 보였습니다. 낮은 별점을 좋아한 정도로 그대로 더하는 것은 이 실험에서 도움이 되지 않았습니다.
- TF-IDF에서는 TF 로그 완화를 해제한 후보의 점수가 낮았습니다.
- C01과 C07의 차이만으로 모든 개인 태그가 유익하다고 일반화할 수 없습니다. 두 규칙 집합의 묶음 비교이며 태그별 효과를 분리한 실험은 아닙니다.
- 최고 점수는 16개 후보 중 검증 점수가 높은 설정입니다. 후보 수가 제한되어 있고 검증 데이터로 선택했으므로 최종 성능과는 구분합니다.
- 첫 평점이 낮은 독자는 positive 설정에서 인기순 대체 추천을 받습니다. 설정별 비율은 B4_profile_summary.csv에 기록했습니다.
- B3와 B2의 초기 설정·과거 결과는 유지했습니다. 이번 결과로 기존 실험 기록을 덮어쓰지 않았습니다.

## 검증

기존 56개와 B4 테스트 사례 10개를 포함해 전체 66개 테스트가 통과했습니다. 테스트는 공용 코드와 동일하게 작은 인공 데이터, Given/When/Then 주석을 사용합니다.

- 설정 생성 시 B1·B2·B3 연결과 입력 불변성 확인.
- 신규 독자 네 조건 평균이 기존 독자 점수보다 우선하는지 확인.
- 동점 순서와 누락·중복·최종 평가·잘못된 NDCG 입력 거부 확인.
- 실제 데이터에서 기준선 1개와 CB 16개 x 목표별 5행 = 85행 확인.
- C11 및 인기순의 여섯 지표가 B3 결과와 허용 오차 1e-12 이내로 재현됨을 확인.
- 선택 JSON을 다시 읽어 모델을 만들고 신규 독자 5명의 점수 범위·유한성·추천 10권·이미 읽은 책 제외를 확인.
- 노트북 코드 셀 전부 실행, 오류 출력 없음. PNG 두 개의 1600 x 900 크기 확인.

```bash
uv run pytest
```

## 재현 및 hybrid 연결

```python
import json
from pathlib import Path
from bookrec.data import load_book_tags, load_split
from bookrec.models.content_based.tuning import ContentBasedConfig, make_content_based

# 레포 루트에서 실행합니다.
saved = json.loads(Path("results/content_based/B4_selected_config.json").read_text())
config = ContentBasedConfig(**saved["config"])
model = make_content_based(load_book_tags(), config)
split = load_split()
model.fit(split.train)
# model.score(history): 행 user_id, 열 BOOK_IDS, 점수 0~1
# model.recommend(history, k=10): 공통 이미 읽은 책 제외 규칙 사용
```

## 파일

- notebooks/content_based/B4_configuration_compare.ipynb: 처음부터 재실행할 실험과 결과.
- B4_candidates.csv: 전체 후보 설정.
- B4_validation_scores.csv: 모든 설정과 인기순의 검증 지표.
- B4_ranking.csv: 후보 순위와 설정.
- B4_selected_config.json: 선택 설정·기준·분할 정보.
- B4_profile_summary.csv: 대체 추천 비율.
- B4_runtimes.csv: 생성·학습과 평가 시간. 실행 환경에 따라 달라집니다.
- docs/figures/B4_configuration_comparison.png, B4_cold_validation.png: 발표용 그래프.

그래프는 공통 setup·폰트·모델 색과 `bookrec.plot_style.save(fig, name)`를 사용합니다. 저장 경로·크기·dpi·레이아웃은 공용 save 함수에 맡깁니다. plot_style.py는 수정하지 않았습니다. 기존의 직접 fig.savefig 호출을 공용 함수로 교체하고, 저장된 비교 결과로 두 그림을 다시 생성했습니다. 공용 save가 두 번 호출되며 docs/figures에 1600 x 900 PNG가 저장되는 것을 확인했습니다.

이전에 parents[2]가 src를 가리킨다는 설명은 잘못됐습니다. src/bookrec/plot_style.py의 parents[0]은 bookrec, parents[1]은 src, parents[2]는 레포 루트이므로 공용 저장 경로가 맞습니다.

## 참고

- docs/timeline.html, src/bookrec/models/content_based/README.md: B4 비교와 최종 설정 산출물.
- src/bookrec/evaluation/README.md: 지표·검증 구간·표본·이미 읽은 책 제외 규칙.
- notebooks/content_based/B3_content_based.ipynb: 초기 CB와 기준선 결과.
