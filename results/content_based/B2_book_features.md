# B2 책 특징 벡터

## 과정
B1 extended 결과를 초기 실험 입력으로 사용합니다. 최종 설정은 B4_selected_config.json을 따릅니다.
build_book_features()는 책 x 태그 CSR 행렬과 행 book_ids·열 tag_names를 묶어 반환합니다.
기본 행은 BOOK_IDS(1~10,000)이며 특징 없는 책도 영벡터로 유지합니다.
태그는 분리하거나 이름을 바꾸지 않으며 min_df는 서로 다른 책 수로 셉니다.

count는 숫자 또는 숫자로 변환 가능한 문자열인 0 이상 2**53-1 이하의 정수여야 합니다.
소수·bool·결측·무한대·음수·범위 초과는 거부합니다. 숫자 변환 뒤 중복을 합산하고 합계도 검사합니다.
0회 태그는 제외합니다. 이 계약으로 문자열 연결 오류와 로그 TF의 음수 발생을 막습니다.
tfidf는 TF=1+log(count), IDF=log((1+N)/(1+df))+1을 기본 사용합니다.
count·binary도 지원하며 모든 방식에 L2 정규화를 적용합니다.

## 초기 실험 결과
| weighting | min_df | sublinear_tf | books | features | nonzero_values | density | zero_books | csr_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tfidf | 2 | True | 10000 | 16523 | 662281 | 0.004008 | 0 | 7987376 |

유사 도서는 자기 자신과 유사도0 책을 제외하고 코사인 점수순, 동점은 작은 책 ID순입니다.
B2_similar_books.csv에 책5권의 상위5권을 기록했습니다.
CSR 메모리는 data·indices·indptr 합계이며 프로세스 최대 메모리가 아닙니다.

## 검증과 다음 작업
수작업 TF-IDF·정규화, ID 대응, min_df·영벡터·유사도 순서, 문자열 count 합산과 잘못된 입력 거부를 검사합니다.
큰 행렬은 저장하지 않고 입력·설정으로 재생성합니다. B4가 선택한 모델과 B2 기본 모델을 혼동하지 않습니다.

수정 전 결과는 커밋 6e859f5와 그 이전 이력에 보존되어 있습니다. 현재 결과는 수정된 코드·태그 규칙으로 재실행한 값이며 이전 수치와 혼합하지 않습니다.

기존 조원 파일과 .vscode는 수정하지 않았습니다. 노트북은 실제 데이터로 다시 실행했고, 공용 코드 테스트는 작은 인공 데이터로 검사합니다.

## 참고
src/bookrec/models/README.md, sklearn TfidfTransformer·normalize, A3 로더.
