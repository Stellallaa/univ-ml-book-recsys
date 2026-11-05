# 원본 데이터 설명 (Goodbooks-10k)

`uv run python scripts/download_data.py`로 받은 `data/raw/`의 CSV 5개를 설명합니다. 가공하지 않은 원본 그대로의 내용입니다. 수치는 2026-11-04에 받은 파일로 직접 잰 값입니다.

- 출처: [zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) (CC BY-SA 4.0)
- 평점은 Goodreads와 비슷한 사이트에서 모았고, 책 정보와 태그는 Goodreads에서 가져왔습니다.

## 파일 한눈에 보기

| 파일 | 행 수 | 크기 | 한 행의 뜻 |
| --- | --- | --- | --- |
| `ratings.csv` | 5,976,479 | 72 MB | 독자 한 명이 책 한 권에 준 별점 |
| `to_read.csv` | 912,705 | 9 MB | 독자 한 명이 "읽고 싶은 책"에 담은 책 한 권 |
| `books.csv` | 10,000 | 3 MB | 책 한 권의 정보 |
| `book_tags.csv` | 999,912 | 17 MB | 책 한 권에 붙은 태그 하나와 붙은 횟수 |
| `tags.csv` | 34,252 | 0.7 MB | 태그 ID 하나와 태그 이름 |

## 파일 사이의 연결

```
ratings.csv ─┐
             ├─ book_id ── books.csv ── goodreads_book_id ── book_tags.csv ── tag_id ── tags.csv
to_read.csv ─┘
```

- `ratings.csv`, `to_read.csv`는 책을 `book_id`(1~10000)로 가리킵니다.
- `book_tags.csv`는 책을 `goodreads_book_id`로 가리킵니다. 두 ID는 `books.csv`에서 1:1로 대응합니다.
- `book_tags.csv`의 태그는 `tag_id` 숫자로만 있고, 이름은 `tags.csv`에 있습니다.

## ratings.csv

독자가 책에 준 별점입니다. 원본 README에 따르면 시간순으로 정렬되어 있습니다. 시각 컬럼은 없습니다.

| 컬럼 | 타입 | 뜻 | 값 범위 | 결측 |
| --- | --- | --- | --- | --- |
| `user_id` | int | 독자 ID | 1~53,424 (빈 번호 없음) | 0 |
| `book_id` | int | 책 ID | 1~10,000 (빈 번호 없음) | 0 |
| `rating` | int | 별점 | 1~5 정수 | 0 |

| 항목 | 값 |
| --- | --- |
| 독자 / 책 | 53,424명 / 10,000권 |
| 같은 (독자, 책) 쌍 중복 | 0 |
| 별점 비율 | 1점 2.1%, 2점 6.0%, 3점 22.9%, 4점 35.8%, 5점 33.2% (평균 3.92) |
| 독자 한 명의 평점 수 | 최소 19, 중앙값 111, 최대 200 |
| 책 한 권의 평점 수 | 최소 8, 중앙값 248, 최대 22,806 |
| 밀도 | 1.12% (독자×책 5.3억 칸 중 평점이 있는 칸) |

예시:

```
user_id,book_id,rating
1,258,5
2,4081,4
```

## to_read.csv

독자가 "읽고 싶은 책"(to-read)으로 담아 둔 책입니다. 원본 README에 따르면 시간순으로 정렬되어 있습니다.

| 컬럼 | 타입 | 뜻 | 값 범위 | 결측 |
| --- | --- | --- | --- | --- |
| `user_id` | int | 독자 ID (`ratings.csv`와 같은 ID) | 1~53,424 | 0 |
| `book_id` | int | 책 ID (`ratings.csv`와 같은 ID) | 1~10,000 | 0 |

| 항목 | 값 |
| --- | --- |
| 독자 / 책 | 48,871명 / 9,986권 |
| 같은 (독자, 책) 쌍 중복 | 0 |
| 독자 한 명이 담은 책 수 | 최소 1, 중앙값 14, 최대 117 |

## books.csv

책 10,000권의 정보입니다. Goodreads 책 정보(XML)에서 뽑은 값이라, 평점 관련 컬럼은 이 데이터셋의 `ratings.csv`가 아니라 **Goodreads 사이트 전체**의 집계입니다.

### ID

| 컬럼 | 타입 | 뜻 | 결측 |
| --- | --- | --- | --- |
| `book_id` | int | 이 데이터셋의 책 ID. 1~10,000, 빈 번호 없음 | 0 |
| `goodreads_book_id` | int | Goodreads의 책 ID(대표 판본). `book_tags.csv`가 쓰는 ID | 0 |
| `best_book_id` | int | Goodreads가 고른 대표 판본 ID. 97.6%는 `goodreads_book_id`와 같음 | 0 |
| `work_id` | int | Goodreads의 작품 ID(판본을 묶은 단위) | 0 |
| `isbn` | str | ISBN-10 | 700 |
| `isbn13` | float | ISBN-13. 실수로 저장되어 끝자리가 틀린 값이 있음(예: `9780439023480.0`) | 585 |

- Goodreads 주소로 바꿀 수 있습니다: `https://www.goodreads.com/book/show/{goodreads_book_id}`, `https://www.goodreads.com/work/editions/{work_id}`
- 원본 README에 따르면 `ratings.csv`의 평점은 판본별이 아니라 작품(`work_id`) 단위로 합쳐져 있습니다.

### 책 정보

| 컬럼 | 타입 | 뜻 | 결측 | 비고 |
| --- | --- | --- | --- | --- |
| `title` | str | 판본 제목. 시리즈 정보가 괄호로 붙음 | 0 | 예: `The Hunger Games (The Hunger Games, #1)` |
| `original_title` | str | 원제 | 585 | 공백뿐인 값 5개 |
| `authors` | str | 저자. 여러 명이면 쉼표로 이어 한 칸에 씀 | 0 | 2,079권이 여러 명. 예: `J.K. Rowling, Mary GrandPré`(번역가·삽화가 포함) |
| `original_publication_year` | float | 처음 출판된 해 | 21 | -1750~2017. 0 이하 31권은 기원전 책(예: 『오디세이아』 -720) |
| `language_code` | str | 언어 코드 | 1,084 | 25종. `eng` 6,341, `en-US` 2,070, `en-GB` 257 등 |
| `books_count` | int | Goodreads에 있는 판본 수 | 0 | 1~3,455, 중앙값 40 |
| `image_url` | str | 표지 이미지 주소 | 0 | 3,332권은 "사진 없음" 기본 이미지 |
| `small_image_url` | str | 작은 표지 이미지 주소 | 0 | `image_url`과 같은 표지 |

### Goodreads 평점 집계

| 컬럼 | 타입 | 뜻 | 값 범위 |
| --- | --- | --- | --- |
| `average_rating` | float | Goodreads 평균 별점 | 2.47~4.82 |
| `ratings_count` | int | 대표 판본의 Goodreads 평점 수 | 2,716~4,780,653 |
| `work_ratings_count` | int | 모든 판본을 합친 Goodreads 평점 수 | 5,510~4,942,365 |
| `work_text_reviews_count` | int | 모든 판본을 합친 Goodreads 글 리뷰 수 | 3~155,254 |
| `ratings_1`~`ratings_5` | int | 별점별 Goodreads 평점 수 | 다섯 개의 합 = `work_ratings_count` |

- 결측은 없습니다.
- 모든 책에서 `ratings_count`가 `ratings.csv`에 있는 그 책의 평점 수보다 큽니다. `ratings.csv`는 Goodreads 평점 중 일부만 담고 있습니다.

## book_tags.csv

독자들이 책에 붙인 태그(Goodreads 책장 이름)와 그 태그를 붙인 횟수입니다. 원본 README에 따르면 `goodreads_book_id` 오름차순, `count` 내림차순으로 정렬되어 있습니다.

| 컬럼 | 타입 | 뜻 | 값 범위 | 결측 |
| --- | --- | --- | --- | --- |
| `goodreads_book_id` | int | Goodreads 책 ID (`books.csv`의 같은 이름 컬럼) | 10,000권 모두 있음 | 0 |
| `tag_id` | int | 태그 ID (`tags.csv`) | 0~34,251 | 0 |
| `count` | int | 그 책에 그 태그를 붙인 독자 수 | -1~596,234 | 0 |

| 항목 | 값 |
| --- | --- |
| 책 한 권의 태그 수 | 최소 56, 중앙값 100, 최대 100 (책마다 많이 붙은 순서로 최대 100개) |
| `count`가 음수인 행 | 6 (모두 -1) |
| 같은 (책, 태그) 쌍이 두 번 나오는 행 | 8 |

- 태그에는 장르(`fantasy`, `classics`)뿐 아니라 개인 책장 이름(`to-read`, `owned`, `favorites`, `2015-reads` 등)도 섞여 있습니다. `to-read`는 10,000권 중 9,983권에 붙어 있습니다.

## tags.csv

태그 ID와 이름의 대응표입니다.

| 컬럼 | 타입 | 뜻 | 결측 |
| --- | --- | --- | --- |
| `tag_id` | int | 태그 ID. 0~34,251, 빈 번호 없음 | 0 |
| `tag_name` | str | 태그 이름. 34,252개 모두 다름 | 0 |

- 이름은 Goodreads 책장 이름 그대로라 `-`, `--1-` 같은 기호, 일본어·중국어·전각 문자(`ｆａｖｏｕｒｉｔｅｓ`) 등도 있습니다.
