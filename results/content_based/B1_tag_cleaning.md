# B1 태그 정제

## 과정
A의 load_book_tags() 결과에 filter_book_tags()를 적용합니다. 원본 ID 연결·음수·중복 처리는 A 로더에 맡깁니다.
basic는 읽기 상태·소유·구매·매체 태그, extended는 여기에 개인 평가와 명시적 독서 연도 태그를 제거합니다.

정규화는 규칙 판정에만 적용하며 결과의 원래 태그·행 순서는 보존합니다.
to-read-fiction, own-to-read, on-kindle, books-owned 등의 명확한 혼합 태그도 전체 제거합니다.
일부를 잘라 새 장르 태그로 만들지 않습니다. 동일 의미 count의 중복이나 임의 의미 변환을 피하기 위해서입니다.
1001-books-to-read-before-you-die처럼 숫자-books로 시작하는 명명된 목록은 일괄 제거하지 않습니다.
1984·19th-century 등은 보존합니다. 모든 비내용 태그를 제거했다고 가정하지 않습니다.

## 결과
| mode | rows_before | rows_after | tags_before | tags_after | books_without_tags |
| --- | --- | --- | --- | --- | --- |
| basic | 999898 | 783564 | 34250 | 33203 | 0 |
| extended | 999898 | 678861 | 34250 | 33103 | 0 |

규칙별로 제거된 태그 수입니다.
| reason | tags |
| --- | --- |
| format | 198 |
| personal | 25 |
| reading_year | 75 |
| shelf | 849 |

제거된 태그와 이유는 `B1_removed_tags.csv`, 책별 정제 전후 사례는 `B1_tag_cleaning.ipynb`에 있습니다.
태그가 남는지 여부와 양수 count 특징이 남는지 여부는 다릅니다. 후자는 B2에서 확인합니다.

## 검증과 한계
기본·확장 규칙, 혼합 태그, 명명된 목록 보존, 입력 불변성·빈 입력·통계 집계를 테스트합니다.
basic에서 favorites·독서 연도 등이 남는 것은 선택적 규칙의 의도이며 B4에서 extended와 비교합니다.
명시적 규칙이므로 알려지지 않은 개인 책장 변형은 남을 수 있습니다.

## 참고
docs/timeline.html, src/bookrec/data/README.md, A3_raw_data_check.ipynb, Goodbooks-10k 데이터(루트 README 출처).
