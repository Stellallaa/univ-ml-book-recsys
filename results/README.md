# results

실험 결과와 실험 기록을 두는 곳입니다. 담당별 폴더에 둡니다.

| 폴더 | 내용 | 담당 |
| --- | --- | --- |
| `evaluation/` | 기준선 점수, 평가 함수 확인 결과 | 팀원 A |
| `content_based/` | CB 설정 비교 기록 | 팀원 B |
| `collaborative_filtering/` | CF 학습·튜닝 기록 | 팀원 C |
| `hybrid/` | hybrid 결합 방식·비율 비교 기록 | 팀원 D |
| `final/` | 모든 모델의 최종 결과표 (D4 결과 집계 스크립트가 만듦) | 팀원 D |

- 실험할 때마다 설정과 점수를 기록합니다. 결과가 나빠도 남깁니다.
- 기록 형식은 D4 결과 집계 스크립트에 맞춥니다.
- 발표용 그래프는 여기가 아니라 `docs/figures/`에 저장합니다.

## D3·D4 실행

```powershell
.venv\Scripts\python.exe scripts\run_hybrid.py
.venv\Scripts\python.exe scripts\run_hybrid.py --export-validation
.venv\Scripts\python.exe scripts\run_hybrid.py --final
.venv\Scripts\python.exe scripts\run_baselines.py --final
.venv\Scripts\python.exe scripts\aggregate_results.py
```

하이브리드 첫 실행은 추천용 MF를 학습하므로 시간이 걸립니다. 기본 실행은 `val`/`cold_tune`으로 후보를 선택하고, `--final`은 그 설정을 `train + val`로 적합해 `test`/`cold_final`을 채점합니다. 기준선 홀드아웃 점수도 필요하면 `run_baselines.py --final`을 실행합니다. 집계 스크립트는 존재하는 파일을 읽어 `results/final/validation_comparison.csv`와 `holdout_comparison.csv`를 갱신하고, 결측 소스는 생략하며 같은 알고리즘·평가 목표·이력 크기의 중복은 오류로 알립니다. 그래프는 `docs/figures/D4_*_comparison.png`에 저장합니다.

검증 점수로 선택한 뒤에는 holdout 결과를 다시 선택에 쓰지 않습니다. 검증 비교에서 CB는 B4 선택 후보 C17을 사용합니다. RMSE 개선량은 bias 기준선 RMSE에서 모델 RMSE를 뺀 값이며, 추천 지표 개선량은 인기순을 뺀 값입니다.
