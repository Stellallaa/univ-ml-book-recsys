# C3 fold-in (신규 독자용 MF 점수)

노트북: `notebooks/collaborative_filtering/C3_fold_in.ipynb` · 코드: `mf.py`의 `fold_in`, `score`

## 과정
- MF는 학습한 독자만 독자 벡터가 있습니다. fold-in은 학습된 책 벡터·책 bias를 고정하고, 그 독자의 평점만으로 독자 벡터·독자 bias를 구합니다.
- 독자 한 명마다 Ridge regression을 닫힌 해로 풉니다. 입력은 책마다 [책 벡터, 1], 정답은 (정답 - 전체 평균 - 책 bias)입니다. 다시 학습하지 않습니다.
- 이력의 책은 정답 = 별점, 가중치 1입니다. 나머지 책은 학습 때와 같게 정답 0, 가중치 neg_ratio * 이력 수 / 책 수로 둡니다.
- `score(이력)`은 기존 독자와 신규 독자를 구분하지 않고 모두 fold-in으로 처리합니다.
- 모델은 추천용 최종 설정(`C2b_mf_tuning_extended.md`)으로 `train`만 학습했고, `val`과 `cold_tune`으로만 점수를 쟀습니다.

## fold-in 정규화 세기 고르기
기준: cold_tune에서 평점 1·3·5·10개일 때 NDCG@10의 평균. 책 벡터는 그대로 두고 값만 바꿨습니다.

| fold_in_reg | cold_ndcg@10_1 | cold_ndcg@10_3 | cold_ndcg@10_5 | cold_ndcg@10_10 | cold_mean_ndcg@10 | warm_val_ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.2449 | 0.2459 | 0.2513 | 0.2800 | 0.2555 | 0.0739 |
| 5.0000 | 0.2573 | 0.2698 | 0.2730 | 0.2904 | 0.2726 | 0.0723 |
| 10.0000 | 0.2529 | 0.2709 | 0.2784 | 0.2956 | 0.2744 | 0.0718 |
| 20.0000 | 0.2476 | 0.2635 | 0.2745 | 0.2957 | 0.2703 | 0.0703 |
| 50.0000 | 0.2413 | 0.2529 | 0.2621 | 0.2792 | 0.2589 | 0.0675 |

- 선택: fold_in_reg = 10.0
- 기존 독자(val)만 보면 fold_in_reg가 작을수록 높았습니다. fold-in은 신규 독자를 위한 기능이라 cold_tune 기준으로 골랐고, 기존 독자 쪽은 가장 높은 값보다 조금 낮습니다(표의 `warm_val_ndcg@10`).

## 평점 수별 점수 (cold_tune 2,500명)
| model | history_size | precision@10 | recall@10 | ndcg@10 | users |
| --- | --- | --- | --- | --- | --- |
| popularity | 1 | 0.2072 | 0.0303 | 0.2213 | 2500 |
| popularity | 3 | 0.2072 | 0.0303 | 0.2213 | 2500 |
| popularity | 5 | 0.2072 | 0.0303 | 0.2213 | 2500 |
| popularity | 10 | 0.2072 | 0.0303 | 0.2213 | 2500 |
| mf_ranking | 1 | 0.2414 | 0.0354 | 0.2529 | 2500 |
| mf_ranking | 3 | 0.2596 | 0.0383 | 0.2709 | 2500 |
| mf_ranking | 5 | 0.2664 | 0.0395 | 0.2784 | 2500 |
| mf_ranking | 10 | 0.2844 | 0.0426 | 0.2956 | 2500 |

- 인기순은 이력을 보지 않아 네 경우의 점수가 같습니다.
- MF(fold-in)는 평점이 1개여도 점수를 낼 수 있고, 평점이 늘수록 점수가 올라갑니다.

## 추천 예시 (학습에 없는 가상의 독자)

고른 책(모두 5점): Harry Potter and the Sorcerer's Stone / The Hobbit / The Lightning Thief

| rank | title | author |
| --- | --- | --- |
| 1 | Harry Potter and the Order of the Phoenix (Harry Potter, #5) | J.K. Rowling |
| 2 | The Hunger Games (The Hunger Games, #1) | Suzanne Collins |
| 3 | Harry Potter and the Prisoner of Azkaban (Harry Potter, #3) | J.K. Rowling |
| 4 | The Fellowship of the Ring (The Lord of the Rings, #1) | J.R.R. Tolkien |
| 5 | Harry Potter and the Chamber of Secrets (Harry Potter, #2) | J.K. Rowling |
| 6 | Harry Potter and the Deathly Hallows (Harry Potter, #7) | J.K. Rowling |
| 7 | Harry Potter and the Goblet of Fire (Harry Potter, #4) | J.K. Rowling |
| 8 | To Kill a Mockingbird | Harper Lee |
| 9 | Harry Potter and the Half-Blood Prince (Harry Potter, #6) | J.K. Rowling |
| 10 | Catching Fire (The Hunger Games, #2) | Suzanne Collins |

고른 책(모두 5점): Pride and Prejudice / Jane Eyre / Wuthering Heights

| rank | title | author |
| --- | --- | --- |
| 1 | To Kill a Mockingbird | Harper Lee |
| 2 | Harry Potter and the Sorcerer's Stone (Harry Potter, #1) | J.K. Rowling |
| 3 | Sense and Sensibility | Jane Austen |
| 4 | Harry Potter and the Order of the Phoenix (Harry Potter, #5) | J.K. Rowling |
| 5 | The Hobbit | J.R.R. Tolkien |
| 6 | The Great Gatsby | F. Scott Fitzgerald |
| 7 | The Hunger Games (The Hunger Games, #1) | Suzanne Collins |
| 8 | The Diary of a Young Girl | Anne Frank |
| 9 | Harry Potter and the Prisoner of Azkaban (Harry Potter, #3) | J.K. Rowling |
| 10 | Of Mice and Men | John Steinbeck |

## 이미지와 기록
- `docs/figures/C3_fold_in_cold.png`
- `C3_fold_in_reg.csv`, `C3_cold_tune_scores.csv`, `C3_example_recommendations.csv`
