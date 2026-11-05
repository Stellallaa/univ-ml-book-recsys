# data

데이터를 읽고 가공하는 공용 코드를 두는 곳입니다.

| 작업 | 하는 일 | 담당 |
| --- | --- | --- |
| A3 공통 전처리 | B1 책장 태그 거르기를 제외한 책 정보·태그 데이터 정리 | 팀원 A |
| B1 태그 정제 | 책 내용과 상관없는 태그(to-read, owned 같은 책장 태그)를 코드로 걸러 내는 규칙 | 팀원 B |
| A4 데이터 분할 | 평점을 학습·검증·평가용으로 나누고, 신규 독자 실험용 독자를 학습 데이터에서 떼어 둠 | 팀원 A |
| A7 신규 독자 실험 세트 | 떼어 둔 독자들의 평점을 몇 개만 남겨 신규 독자 상황을 흉내 내는 평가 세트 | 팀원 A |

- 원본 CSV는 `data/raw/`에서 읽습니다(`scripts/download_data.py`로 받음).
- 다른 팀원은 데이터를 직접 나누지 않고, 여기의 함수로 분할된 데이터를 불러 씁니다.

## 원본 데이터 불러오기 (A3 공통 전처리)

`loaders.py`의 함수로 원본 CSV를 읽습니다. 정리한 파일을 따로 저장하지 않고, 부를 때마다 아래 처리를 적용합니다(전체 1초 안팎).

```python
from bookrec.data import load_ratings, load_to_read, load_books, load_book_tags

ratings = load_ratings()      # user_id, book_id, rating
books = load_books()          # book_id, title, authors(목록), main_author, ...
book_tags = load_book_tags()  # book_id, tag_name, count
```

| 함수 | 돌려주는 것 | 처리 |
| --- | --- | --- |
| `load_ratings()` | `user_id`, `book_id`, `rating` | 없음. **원본 행 순서(시간순)를 그대로 둡니다** |
| `load_to_read()` | `user_id`, `book_id` | 없음 |
| `load_books()` | 원본 컬럼 전부 + `main_author` (`goodreads_book_id` 제외) | `title` 앞뒤 공백 제거, `authors`를 이름 목록으로 나눔(이름 안의 연속 공백은 한 칸으로), `main_author` = 첫 저자 |
| `load_book_tags()` | `book_id`, `tag_name`, `count` | `goodreads_book_id`를 `book_id`로 바꿈, 태그 이름 붙임, `count`가 음수인 6행 삭제, 같은 (책, 태그) 8쌍은 큰 `count`만 남김. 책별 `count` 내림차순 |

일부러 하지 않는 것:

| 대상 | 이유 |
| --- | --- |
| 출판연도 결측(21권)·기원전(음수, 31권) | 실제 값이라 그대로 둡니다. 연대 묶기는 B2에서 정합니다 |
| `language_code` 결측(1,084권) | 책 내용이 없어 채울 근거가 없습니다 |
| 책장 태그(`to-read`, `owned` 등) 거르기 | B1에서 합니다 |
| 컬럼 고르기 | 어떤 컬럼을 쓸지는 각 담당자가 정합니다 |

- 원본 파일·컬럼 설명은 `docs/data.md`, 판단 근거가 된 확인 과정은 `notebooks/data/preprocessing/A3_raw_data_check.ipynb`에 있습니다.
- 경로를 바꾸려면 `load_books(raw_dir=...)`처럼 `raw_dir`을 넘깁니다.
