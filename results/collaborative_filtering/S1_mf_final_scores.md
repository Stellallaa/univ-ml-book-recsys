# S1 MF 최종 점수

노트북: `notebooks/collaborative_filtering/S1_mf_final_scores.ipynb`

C2 추가 실험과 C3에서 고른 최종 설정으로 `train`만 학습하고 `test`(기존 독자 전원)와 `cold_final`(신규 독자 2,500명)로 한 번 잰 값입니다. 이 점수로 설정을 바꾸지 않았습니다.

| model | goal | history_size | precision@10 | recall@10 | ndcg@10 | to_read_hit@10 | test_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popularity | warm_test | all | 0.0268 | 0.0230 | 0.0292 | 0.0303 |  |
| mf_ranking | warm_test | all | 0.0493 | 0.0435 | 0.0549 | 0.0521 |  |
| mf_rating | warm_test | all |  |  |  |  | 0.8515 |
| popularity | cold_final | 1 | 0.2101 | 0.0308 | 0.2237 | 0.0230 |  |
| popularity | cold_final | 3 | 0.2101 | 0.0308 | 0.2237 | 0.0230 |  |
| popularity | cold_final | 5 | 0.2101 | 0.0308 | 0.2237 | 0.0230 |  |
| popularity | cold_final | 10 | 0.2101 | 0.0308 | 0.2237 | 0.0230 |  |
| mf_ranking | cold_final | 1 | 0.2466 | 0.0362 | 0.2574 | 0.0197 |  |
| mf_ranking | cold_final | 3 | 0.2611 | 0.0387 | 0.2736 | 0.0228 |  |
| mf_ranking | cold_final | 5 | 0.2691 | 0.0401 | 0.2791 | 0.0242 |  |
| mf_ranking | cold_final | 10 | 0.2866 | 0.0430 | 0.2980 | 0.0276 |  |

- `mf_ranking`은 추천용 MF(gradient descent, 0점을 함께 학습), `mf_rating`은 별점 예측용 MF(ALS)입니다. `mf_rating`은 RMSE만 쟀습니다.
- `test_rmse`는 별점 예측용 MF만 쟀습니다. 추천용 MF는 0점을 함께 학습해 예측값이 별점보다 낮으므로 별점 예측에 쓰지 않습니다.
- Goal 1(cold_final)과 Goal 2(warm_test)의 숫자는 정답 후보 수가 달라 서로 비교하지 않습니다.
- `test`·`cold_final`은 1차 튜닝의 설정으로 한 번 측정한 적이 있습니다(추천용 차원 64·neg_ratio 4: warm_test NDCG@10 0.0527, cold_final 0.250~0.289 / 별점 예측용 gradient descent: test RMSE 0.8591). 그 뒤 추가 실험에서 설정을 `val`만 보고 다시 골랐고, 이 표는 새 설정으로 다시 측정한 값입니다. 설정을 고르는 데 `test`·`cold_final` 점수는 쓰지 않았습니다.
- 이미지: `docs/figures/S1_mf_final_scores.png`
