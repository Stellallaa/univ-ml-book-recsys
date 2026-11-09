# B2 책 특징 벡터

## 과정

기존 `load_book_tags()`와 B1의 `filter_book_tags()`를 사용합니다. 다른 팀원의 파일은 수정하지 않았으며, 새 함수는 `bookrec.models.content_based.features`에서 직접 불러옵니다.

```python
from bookrec.data import load_book_tags
from bookrec.data.tag_filter import filter_book_tags
from bookrec.models.content_based.features import build_book_features

features = build_book_features(filter_book_tags(load_book_tags()))
features.matrix       # CSR 희소 행렬
features.book_ids     # 행의 책 ID (기본은 BOOK_IDS)
features.tag_names    # 열의 태그 이름
features.similar_books(1, k=5)
```

- 태그 하나를 특징 하나로 유지합니다. `science-fiction`, `young-adult`를 단어로 나누지 않습니다.
- 양수 count만 사용하고, 동일한 (책, 태그)가 있으면 count를 합산합니다. 기존 A3 출력에는 이미 중복이 없습니다.
- `min_df=2`: 서로 다른 책 두 권 이상에 붙은 태그를 사용합니다. 한 책에 여러 행이 있어도 책 한 권으로 셉니다.
- `weighting="tfidf"`: TF-IDF를 적용합니다. 기본 `sublinear_tf=True`는 TF를 `1 + log(count)`로 완화합니다.
- IDF는 `log((1 + N) / (1 + df)) + 1`입니다. N은 태그가 없는 책도 포함한 전체 책 수입니다.
- `weighting="count"`는 count 자체, `weighting="binary"`는 태그 존재 여부를 사용합니다. 이 두 방식에서는 sublinear_tf를 적용하지 않습니다.
- 모든 방식에서 책 벡터를 L2 정규화합니다. 영벡터는 그대로 남깁니다.
- 기본 행 순서는 공통 `BOOK_IDS`(1~10,000), 열은 태그 이름 오름차순입니다.
- 반환한 특징 행렬과 책 ID·태그 이름은 함께 사용합니다.

책 정보·태그는 프로젝트의 파이프라인 규칙대로 전체 카탈로그를 사용합니다. 독자 평점이나 val/test 정답은 이 단계에 사용하지 않습니다. min_df와 가중 방식은 초기 후보이며 B4에서 검증 평가로 선택합니다.

## 결과

| 항목 | 값 |
| --- | ---: |
| B1 확장 정제 후 입력 행 | 700,957 |
| 책 수 | 10,000 |
| 특징 수 | 16,958 |
| 0이 아닌 값 수 | 683,823 |
| 밀도 | 약 0.4032% |
| 영벡터인 책 | 0 |
| CSR 배열 메모리 합계 | 8,245,880 bytes |

메모리는 CSR의 data, indices, indptr 배열 합계이며, Python 객체 전체나 실행 중 최대 메모리는 아닙니다. 큰 특징 행렬이나 전 책 간 유사도 행렬은 파일로 저장하지 않습니다. 입력과 설정으로 재생성합니다.

- 행렬 통계: `B2_feature_summary.csv`
- 대표 책 5권의 유사 도서 상위 5권: `B2_similar_books.csv`
- 실행 결과: `notebooks/content_based/B2_book_features.ipynb`

『The Hunger Games』의 유사 도서 1위는 『Catching Fire』(약 0.9342), 2위는 『Mockingjay』(약 0.8729)입니다. 제목을 특징으로 사용하지 않고 태그만으로 계산한 결과입니다. 이 사례는 정성적 점검이며 추천 정확도를 입증하는 평가 결과는 아닙니다. 같은 시리즈나 합본이 높은 유사도를 보일 수 있습니다.

## 검증

기존 테스트 27개와 B2 테스트 사례 15개가 통과해 전체 42개가 통과했습니다.

- 책 ID 순서, 복합 태그 보존, 영벡터 책 유지, 입력 불변성
- 작은 데이터의 TF-IDF를 수작업 계산값과 비교
- 등장 책 수 기반 min_df와 0회 태그 처리
- count/binary 가중치와 정규화 결과
- 유사 도서 조회의 자기 자신 제외, 동점 순서, 영벡터·없는 ID 처리
- 잘못된 설정, 음수·무한대 count, 빈 태그·빈 입력 거부

실제 데이터 노트북에서는 책 ID 대응, 유한한 값, 비음수 가중치, 비어 있지 않은 벡터의 L2 길이 1을 추가 확인했습니다. B1의 태그 정제와 연결해 원본 CSV에서 끝까지 실행했습니다.

```bash
uv run pytest
```

## 다음 작업

B3에서는 `features.matrix`를 사용해 독자별 평점 이력의 가중합으로 취향 벡터를 만들고, 전체 책 점수표를 반환합니다. B5에서도 같은 특징 공간을 사용합니다. 특징 벡터가 없는 독자 이력과 저평점 이력의 대체 추천은 B3의 책임입니다.

## 참고

- `src/bookrec/models/README.md`: 공통 책 ID와 모델 인터페이스
- `docs/pipeline.md`: 책 메타데이터와 독자 평점 분할의 구분
- `notebooks/data/preprocessing/A3_loader_preview.ipynb`: 실제 데이터 출력과 조건 확인 방식
- scikit-learn의 `TfidfTransformer`, `normalize`: 사용 파라미터는 새 코드의 주석과 위 과정에 명시
