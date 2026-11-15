# hybrid

hybrid filtering 코드를 두는 곳입니다. 담당은 팀원 D입니다.

| 작업 | 하는 일 |
| --- | --- |
| D3 hybrid 구현·튜닝 | CB와 MF 점수를 합치는 방식 3가지(고정 비율, 전환, 점진 전환)를 만들고, 합치는 비율과 전환 기준을 검증 데이터로 정함 |

- hybrid 모델도 `Recommender`를 상속합니다(`../README.md` 참고). 완성된 두 모델(CB, MF)을 받아 각자의 `score` 점수표를 섞습니다.
- 모든 모델의 점수표는 행(독자)과 열(책 1~10000)이 같으므로 칸끼리 바로 합칠 수 있습니다.
- CB·MF가 나오기 전에는 `PopularityRecommender` 같은 기준선 모델 두 개를 넣어 뼈대를 먼저 만들 수 있습니다.

## 구현 및 실험

`HybridRecommender`는 CB와 추천용 MF의 독자별 점수를 백분위 순위로 바꿔 같은 0~1 척도에서 결합합니다. 지원 방식은 고정 MF 비율(`fixed`), 이력 수 경계에서 CB에서 MF로 바꾸는 `switch`, 두 경계 사이에서 MF 비율을 선형으로 올리는 `gradual`입니다. 평가 시 이미 읽은 책 제외는 공통 `Recommender.recommend()`가 처리합니다.

```powershell
.venv\Scripts\python.exe scripts\run_hybrid.py
.venv\Scripts\python.exe scripts\run_hybrid.py --final
```

첫 명령은 `train`으로 적합한 CB/MF를 사용해 검증 후보를 비교하고 최종 설정을 `results/hybrid/selected_config.json`에 씁니다. 후보별 결과와 비교 표는 같은 폴더의 `validation_scores.csv`, `candidate_comparison.csv`입니다. `--export-validation`은 baseline·CB·MF·선택 Hybrid를 같은 검증 코호트에서 비교해 `validation_models.csv`에 저장합니다. 최종 명령은 선택된 설정을 `train + val`로 적합해 `test`와 `cold_final`을 채점합니다. 최종 평가를 하려면 검증 튜닝을 먼저 완료해야 합니다.

선택 기준은 기존 독자 val 1,000명(시드 42)의 NDCG@10과 신규 독자 `cold_tune` 이력 1·3·5·10 평균 NDCG@10의 단순 평균입니다. 데이터 분할은 A4, 지표는 A5를 그대로 사용하며 test/final은 설정 선택에 사용하지 않습니다. 실측상 switch-at-3이 하이브리드 후보 중 가장 좋았지만, 단독 MF 대비 개선은 없었습니다. 과정·수치·한계는 `results/hybrid/D3_hybrid_tuning.md`에 기록합니다.
