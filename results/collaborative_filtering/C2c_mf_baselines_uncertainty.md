# MF와 기준선·CB 비교, 점수 차이의 불확실성

노트북: `notebooks/collaborative_filtering/C2c_mf_baselines_uncertainty.ipynb`

추천용 MF 최종 설정을 같은 독자, 같은 정답으로 다른 방법과 비교했습니다. 모두 `train`으로 학습하고 `val`(독자 5,000명, 시드 42)과 `cold_tune`(2,500명)으로만 쟀습니다.

- 인기순, Goodreads 평점순, bias: A6 기준선(`bookrec.models.baselines`)을 그대로 썼습니다.
- CB: 팀원 B의 최종 content-based 설정(`load_selected_content_based`)
- bias 기준선은 별점만 예측하고 추천 점수가 없어 RMSE만 비교했습니다.
- A6의 기준선 표(`src/bookrec/evaluation/README.md`)는 val 독자 전원(48,150명)으로 잰 값이라, 5,000명 표본인 이 표의 인기순·Goodreads 평점순 숫자와 조금 다릅니다.

## 별점 예측 (검증 RMSE)
| model | val_rmse |
| --- | --- |
| 전체 평균 | 0.9865 |
| bias 기준선 | 0.8616 |
| MF 별점 예측용 (ALSMatrixFactorization) | 0.8392 |

## 추천 (NDCG@10)
| goal | popularity | goodreads_rating | cb | mf_ranking |
| --- | --- | --- | --- | --- |
| warm_val | 0.0382 | 0.0018 | 0.0516 | 0.0718 |
| cold_tune_1 | 0.2213 | 0.0089 | 0.1518 | 0.2529 |
| cold_tune_3 | 0.2213 | 0.0089 | 0.1682 | 0.2709 |
| cold_tune_5 | 0.2213 | 0.0089 | 0.1868 | 0.2784 |
| cold_tune_10 | 0.2213 | 0.0089 | 0.2277 | 0.2956 |

## MF와의 차이 (MF - 비교 대상, 독자 단위 bootstrap 2,000번, 시드 42)
| goal | against | users | mean_difference | ci95_low | ci95_high | significant |
| --- | --- | --- | --- | --- | --- | --- |
| warm_val | popularity | 4974 | 0.0337 | 0.0299 | 0.0377 | True |
| cold_tune_1 | popularity | 2500 | 0.0316 | 0.0272 | 0.0366 | True |
| cold_tune_3 | popularity | 2500 | 0.0496 | 0.0436 | 0.0555 | True |
| cold_tune_5 | popularity | 2500 | 0.0570 | 0.0502 | 0.0641 | True |
| cold_tune_10 | popularity | 2500 | 0.0742 | 0.0660 | 0.0827 | True |
| cold_tune_mean | popularity | 2500 | 0.0531 | 0.0476 | 0.0586 | True |
| warm_val | goodreads_rating | 4974 | 0.0701 | 0.0664 | 0.0738 | True |
| cold_tune_1 | goodreads_rating | 2500 | 0.2440 | 0.2344 | 0.2536 | True |
| cold_tune_3 | goodreads_rating | 2500 | 0.2620 | 0.2522 | 0.2717 | True |
| cold_tune_5 | goodreads_rating | 2500 | 0.2695 | 0.2594 | 0.2790 | True |
| cold_tune_10 | goodreads_rating | 2500 | 0.2867 | 0.2769 | 0.2969 | True |
| cold_tune_mean | goodreads_rating | 2500 | 0.2655 | 0.2566 | 0.2748 | True |
| warm_val | cb | 4974 | 0.0202 | 0.0168 | 0.0237 | True |
| cold_tune_1 | cb | 2500 | 0.1012 | 0.0909 | 0.1110 | True |
| cold_tune_3 | cb | 2500 | 0.1027 | 0.0924 | 0.1131 | True |
| cold_tune_5 | cb | 2500 | 0.0915 | 0.0812 | 0.1020 | True |
| cold_tune_10 | cb | 2500 | 0.0679 | 0.0588 | 0.0776 | True |
| cold_tune_mean | cb | 2500 | 0.0908 | 0.0825 | 0.0998 | True |

- `cold_tune_mean`은 독자마다 평점 1·3·5·10개의 NDCG@10을 먼저 평균 낸 값입니다.
- 모든 비교에서 95% 구간이 0을 포함하지 않았습니다.
- 이 구간은 검증 데이터에서의 비교이고, 설정을 같은 검증 데이터로 골랐기 때문에 MF 쪽에 유리한 선택 편향이 남아 있습니다. 최종 점수는 `S1_mf_final_scores.md`를 봅니다.

## 이미지와 기록
- `docs/figures/C2c_mf_vs_baselines.png`
- `C2c_comparison_scores.csv`, `C2c_bootstrap.csv`, `C2c_val_rmse.csv`
