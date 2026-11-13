# 8분 발표 구성안: Goodbooks-10k 도서 추천 시스템

> 머신러닝 전공 과목 발표용 구성안입니다. 총 11장, 목표 발표 시간은 약 7분 40초입니다. 슬라이드별 시간·담당자·발표 목표는 표시하지 않습니다. 내용은 실제 저장소의 코드와 결과를 확인해 작성했으며, 발표 자료에서는 긴 설명 대신 아래 시각 자료를 중심으로 구성합니다.

## 슬라이드 구성

### 1. Goodbooks-10k 도서 추천 시스템
- **핵심 내용:** 희소한 독서 평점으로 기존 독자와 신규 독자에게 책을 추천하는 방법을 비교합니다. 팀 9의 A·B·C·D 작업을 데이터·평가, 콘텐츠 기반, 협업 필터링, EDA·하이브리드 흐름으로 연결합니다.
- **시각 자료:** 책 표지 몇 개와 사용자-책-추천 결과를 잇는 간단한 아이콘 흐름.
- **추천 레이아웃:** 중앙에 제목, 아래쪽에 프로젝트명과 팀 표기. 설명 문장은 넣지 않습니다.
- **근거:** `README.md`, `docs/timeline.html`

### 2. 추천 문제: 기존 독자와 신규 독자
- **핵심 내용:** 기존 독자(Warm)는 학습에 사용하지 않은 책을 과거 평점으로 개인화합니다. 신규 독자(Cold)는 평점 이력 1·3·5·10개만으로 추천합니다. 두 조건은 관측 정보와 정답 집합이 달라 점수를 직접 비교하지 않습니다.
- **시각 자료:** 같은 책 목록에서 Warm 경로와 Cold 경로로 나뉘는 사용자 여정도.
- **추천 레이아웃:** 좌우 2열 비교. Warm에는 긴 평점 이력, Cold에는 1·3·5·10개 이력 아이콘을 배치합니다.
- **근거:** `docs/pipeline.md`, `src/bookrec/data/split.py`, `src/bookrec/evaluation/README.md`

### 3. 데이터 전처리와 평가용 분할
- **핵심 내용:** Goodbooks-10k는 평점 5,976,479개, 사용자 53,424명, 책 10,000권이며 평점은 1~5, 평균은 3.92, 관측 밀도는 1.12%입니다. Goodreads 책 ID를 공통 ID로 연결하고 책 정보와 태그를 결합합니다. 잘못된 태그, 음수 태그 수, 중복 책-태그 쌍을 정리합니다. 시드 42로 Cold 사용자 5,000명을 분리해 2,500명은 조정, 2,500명은 최종 평가에 사용하고, 나머지 Warm 사용자는 사용자별 행 순서로 70/15/15 분할합니다. 원본에 타임스탬프가 없어 행 순서는 시간의 근사치입니다.
- **시각 자료:** 원본 파일 → ID 매핑·태그 정제 → Train / Validation / Test 및 Cold Tune / Cold Final 분할도. 비율과 사용자 수를 함께 표기합니다.
- **추천 레이아웃:** 가로형 데이터 파이프라인. 최종 Test와 Cold Final은 별도 색으로 표시합니다.
- **근거:** `docs/data.md`, `docs/pipeline.md`, `src/bookrec/data/loaders.py`, `src/bookrec/data/tag_filter.py`, `src/bookrec/data/split.py`

### 4. 전처리 데이터의 EDA
- **핵심 내용:** 관측 밀도가 낮고 사용자별·책별 평점 수의 편차가 큽니다. 태그 분포도 불균형하며 `to-read` 태그는 10,000권 중 9,983권에 나타납니다. 이런 데이터 특성은 인기도 기준선, 희소 콘텐츠 벡터, 신규 독자 평가를 함께 살펴야 하는 이유를 보여줍니다.
- **시각 자료:** 평점 분포, 사용자별·책별 평점 수 로그 그래프, 상위 태그 그래프. 기존 D1 그림을 우선 사용합니다.
- **추천 레이아웃:** 그래프 3개를 같은 크기의 패널로 배치하고, 아래에 “희소성·긴 꼬리·태그 편중” 키워드만 둡니다.
- **근거:** `notebooks/eda/D1_eda_graphs.ipynb`, `docs/figures/d1_rating_distribution.png`, `docs/figures/d1_user_rating_count_log.png`, `docs/figures/d1_book_rating_count_log.png`, `docs/figures/d1_top20_tags.png`, `docs/data.md`

### 5. 기준선과 평가 지표
- **핵심 내용:** 인기도 기준선은 학습 데이터의 책별 평점 수, Goodreads 기준선은 외부 평균 평점으로 책을 정렬합니다. 평점 예측 기준선은 전역 평균과 사용자·책 편향을 사용합니다. 추천 평가는 상위 10권에서 평점 4 이상을 정답으로 보고 Precision@10, Recall@10, NDCG@10 및 `to_read_hit@10`을 계산합니다. RMSE는 평점 예측에만 적용합니다.
- **시각 자료:** 기준선 3종 카드와 “정답 → Top-10 → Precision/Recall/NDCG” 평가 흐름. RMSE는 별도 경로로 표시합니다.
- **추천 레이아웃:** 왼쪽 기준선 카드, 오른쪽 평가 파이프라인. NDCG는 순위가 높을수록 적중 점수가 커지는 작은 예시로 설명합니다.
- **근거:** `src/bookrec/models/baselines/`, `src/bookrec/evaluation/evaluate.py`, `src/bookrec/evaluation/README.md`

### 6. 콘텐츠 기반 추천과 책 군집
- **핵심 내용:** CB는 정제 태그로 책별 희소 벡터를 만들고, 독자가 높게 평가한 책의 태그 벡터를 합쳐 취향 프로필을 구성합니다. 선택 설정은 이진 태그 벡터, `min_df=1`, positive 평점 가중치이며 코사인 유사도로 책을 정렬합니다. 프로필이 비면 인기도를 사용합니다. 별도 K-means 실험은 원본 특성 공간에서 K=10, 표본 실루엣 0.0614를 얻었습니다. 이는 약한 내부 군집 분리 지표이지 추천 품질 점수가 아닙니다. SVD 군집 후보는 확인 표본에서 개선을 보이지 않아 채택하지 않았습니다.
- **시각 자료:** 태그 벡터 → 독자 프로필 → 코사인 유사 책 추천 구조도. 옆에는 군집별 대표 책 또는 군집 시각화를 두고 “탐색용”으로 구분합니다.
- **추천 레이아웃:** 좌측 2/3는 CB 추천 원리, 우측 1/3은 K-means 결과와 실루엣 주의 문구.
- **근거:** `src/bookrec/models/content_based/features.py`, `src/bookrec/models/content_based/recommender.py`, `src/bookrec/models/content_based/clustering.py`, `results/content_based/B4_selected_config.json`, `results/content_based/B5_selected_clustering.json`, `results/content_based/B5_improvement_selected_config.json`

### 7. 협업 필터링: Matrix Factorization
- **핵심 내용:** MF는 사용자와 책을 잠재 벡터로 나타내고, 전역 평균·사용자 편향·책 편향·벡터 내적으로 평점을 추정합니다. 추천용 MF는 관측하지 않은 책 일부를 음성 샘플로 추가해 순위를 학습합니다. 평점 예측용 MF와 추천 순위용 MF는 학습 목적이 다릅니다. 선택된 추천 설정은 잠재 요인 128개, 학습률 0.01, 정규화 0.05, 음성 비율 8, 4 epoch입니다.
- **시각 자료:** 희소 사용자-책 행렬 → 사용자/책 잠재 벡터 → 점수 행렬 → 상위 추천. 옆에 `r_hat = 평균 + 사용자 편향 + 책 편향 + 사용자 벡터·책 벡터` 식을 작게 둡니다.
- **추천 레이아웃:** 행렬 분해를 중심에 두고 학습용 음성 샘플링을 하단에 보조 흐름으로 표시합니다.
- **근거:** `src/bookrec/models/collaborative_filtering/mf.py`, `src/bookrec/models/collaborative_filtering/tuning.py`, `results/collaborative_filtering/C2_selected_config.json`

### 8. 신규 독자 대응: Fold-in
- **핵심 내용:** 학습 때 보지 못한 사용자는 책 벡터를 고정하고, 신규 사용자의 짧은 평점 이력으로 사용자 벡터와 편향을 정규화 최소제곱으로 추정합니다. 정규화 계수는 Cold Tune에서 1·5·10·20·50을 비교해 10을 선택했습니다. 최종 Cold 사용자는 이 설정을 고르는 데 사용하지 않습니다.
- **시각 자료:** 고정된 책 벡터 + 신규 사용자 평점 1/3/5/10개 → 사용자 벡터 추정 → 추천. Fold-in 정규화 계수별 Cold Tune NDCG 그래프.
- **추천 레이아웃:** 위쪽에 Fold-in 계산 흐름, 아래쪽에 정규화 계수 실험 그래프를 배치합니다.
- **근거:** `src/bookrec/models/collaborative_filtering/mf.py`, `results/collaborative_filtering/C3_fold_in.md`, `results/collaborative_filtering/C3_fold_in_reg.csv`

### 9. 하이브리드 추천: CB와 MF 결합
- **핵심 내용:** 서로 다른 CB·MF 점수를 사용자별 백분위 순위로 변환한 뒤 결합했습니다. 고정 가중치, 이력 길이 전환, 점진 전환 등 9개 후보를 검증했습니다. 선택된 `switch-at-3`은 평점 1개에서는 CB, 3개 이상에서는 MF를 사용합니다. 선택 기준에서 Hybrid의 Cold Tune 평균 NDCG@10은 0.2491, MF는 0.2744였습니다. 따라서 Hybrid가 MF보다 좋았다고 결론 내리지 않습니다.
- **시각 자료:** 이력 길이 1 → CB, 3/5/10 → MF로 이어지는 전환도. 아래에 검증 점수 막대 2개를 둡니다.
- **추천 레이아웃:** 전환 규칙을 크게, 점수 비교는 작은 보조 그래프로 배치합니다.
- **근거:** `src/bookrec/models/hybrid/recommender.py`, `results/hybrid/selected_config.json`, `results/hybrid/candidate_comparison.csv`, `results/hybrid/D3_hybrid_tuning.md`

### 10. 실험 설계와 최종 성능 비교
- **핵심 내용:** 설정 선택에는 Validation 및 Cold Tune을 사용하고, 최종 결론은 별도 Test 및 Cold Final 결과로 냅니다. Warm Test는 답이 있는 사용자 48,266명, Cold Final은 2,498명이며 Cold 결과는 이력 길이별로 봅니다. NDCG@10 최종 결과:
  
  | 모델 | Warm Test | Cold Final 1개 | Cold Final 10개 |
  |---|---:|---:|---:|
  | 인기도 | 0.0320 | 0.2338 | 0.2338 |
  | Goodreads 평점 | 0.0022 | 0.0101 | 0.0101 |
  | CB | 0.0425 | 0.1596 | 0.2217 |
  | MF | **0.0572** | **0.2567** | **0.3048** |
  | Hybrid | 0.0572 | 0.1596 | 0.3048 |
  
  Cold Final 이력 3개·5개의 MF 점수는 각각 0.2735, 0.2835이며 Hybrid는 이 두 조건에서 MF와 같습니다. 별도 평점 예측 Test에서는 ALS RMSE 0.8337, 편향 기준선 RMSE 0.8555입니다. RMSE와 NDCG는 다른 과제의 지표입니다.
- **시각 자료:** 위 최종 결과 표 또는 Warm/Cold를 분리한 막대그래프. 검증 수치와 테스트 수치를 한 축에 섞지 않습니다.
- **추천 레이아웃:** 좌측은 Warm Test 표/막대, 우측은 Cold Final 이력 길이별 그래프. RMSE는 하단의 작은 별도 상자로 둡니다.
- **근거:** `results/final/holdout_comparison.csv`, `results/final/validation_comparison.csv`, `results/evaluation/baselines_rmse_test.csv`, `results/evaluation/mf_rmse_test.csv`

### 11. 결론과 한계
- **핵심 내용:** 이 데이터 분할과 평가 조건에서는 MF가 Warm 및 Cold Final의 비교 조건에서 가장 높은 NDCG@10을 보였습니다. 선택된 Hybrid는 이력 3개 이상에서 MF와 같은 결과를 내지만 MF를 능가하지 않았고, 이력 1개에서는 인기도보다 낮았습니다. 타임스탬프 부재, 샘플링 음성 예시, 제한된 Cold 사용자 평가가 한계입니다.
- **시각 자료:** “검증된 결과: MF”, “추가 과제: 1개 이력 Cold”, “다음 실험: 시간 기반 분할” 세 가지 요약 카드.
- **추천 레이아웃:** 결론 한 문장을 상단에 두고, 한계와 다음 실험을 하단 2열로 정리합니다.
- **근거:** `results/final/holdout_comparison.csv`, `results/hybrid/D3_hybrid_tuning.md`, `docs/data.md`, `docs/pipeline.md`

## 전체 흐름 요약

데이터 ID 정리와 태그 전처리(A·B) → EDA(D) → 공통 분할·평가 및 기준선(A) → CB와 K-means(B) 및 MF·Fold-in(C) → Hybrid 통합과 결과 집계(D) → 별도 Test/Cold Final 성능 비교와 한계 정리(전원).

## 발표 자료 점검

- Warm/Cold와 Validation/Test를 슬라이드 및 그래프 범례에서 명확히 분리합니다.
- 실루엣 점수는 추천 지표와 같은 성능 표에 넣지 않습니다.
- Hybrid가 MF보다 우수하다는 표현을 사용하지 않습니다.
- 그래프와 표는 텍스트를 줄이고 핵심 수치만 읽을 수 있게 배치합니다.
- 이 문서는 구성안이며 PPTX 파일을 생성하지 않습니다.
