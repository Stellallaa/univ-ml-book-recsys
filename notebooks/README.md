# notebooks

각자 실험하는 노트북을 두는 곳입니다. 담당별 폴더에 둡니다.

| 폴더 | 작업 | 담당 |
| --- | --- | --- |
| `data/` | 전처리(`preprocessing/`), 분할, 평가 함수, 기준선 확인 | 팀원 A |
| `content_based/` | 태그 정제, 책 벡터, CB 설정 비교, 책 군집 | 팀원 B |
| `collaborative_filtering/` | CF 모델 학습, 튜닝, fold-in | 팀원 C |
| `hybrid/` | hybrid 구현·튜닝 | 팀원 D |
| `eda/` | 데이터 탐색 그래프 | 팀원 D |

- 파일 이름은 `작업ID_주제.ipynb`로 짓습니다(예: `content_based/B4_similarity_compare.ipynb`).
- 각자 자기 폴더의 노트북만 수정합니다.
- 여러 곳에서 다시 쓸 코드는 노트북에 두지 말고 `src/bookrec/`으로 옮깁니다.
- 실험 결과와 기록은 `results/`에 남깁니다.
- 커널은 레포의 `.venv`를 선택합니다.
