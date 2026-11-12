"""Goodbooks-10k 원본 CSV를 읽어 공통 전처리(A3)를 적용한다. 처리 내용은 같은 폴더의 README에 있다."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def load_ratings(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """평점(user_id, book_id, rating)을 원본 행 순서 그대로 읽는다."""
    # NOTE: 시각 컬럼이 없어 행 순서가 유일한 시간 정보다. 정렬하지 않는다.
    return pd.read_csv(raw_dir / "ratings.csv")


def load_to_read(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """읽고 싶은 책(user_id, book_id)을 원본 행 순서 그대로 읽는다."""
    return pd.read_csv(raw_dir / "to_read.csv")


def load_books(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """책 정보를 읽는다. authors는 이름 목록, main_author는 첫 저자이고 goodreads_book_id는 뺀다."""
    books = pd.read_csv(raw_dir / "books.csv").drop(columns="goodreads_book_id")
    books["title"] = books["title"].str.strip()
    # NOTE: "Betty  Smith"처럼 이름 안에 공백이 여러 칸인 경우가 있어 한 칸으로 줄인다.
    books["authors"] = books["authors"].str.split(",").map(lambda names: [" ".join(name.split()) for name in names])
    books["main_author"] = books["authors"].str[0]
    return books


def load_book_tags(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """책별 태그(book_id, tag_name, count)를 읽는다. count가 음수인 행은 빼고, 중복된 (책, 태그)는 큰 count만 남긴다."""
    ids = pd.read_csv(raw_dir / "books.csv", usecols=["book_id", "goodreads_book_id"])
    # NOTE: groupby(as_index=False)는 묶은 열(book_id, tag_name)을 인덱스가 아닌 열로 남긴다.
    #       sort_values(ascending=[True, False])는 book_id 오름차순, 그 안에서 count 내림차순으로 정렬한다.
    return (
        pd.read_csv(raw_dir / "book_tags.csv")
        .query("count >= 0")
        .merge(ids, on="goodreads_book_id")
        .merge(pd.read_csv(raw_dir / "tags.csv"), on="tag_id")
        .groupby(["book_id", "tag_name"], as_index=False)["count"].max()
        .sort_values(["book_id", "count"], ascending=[True, False], ignore_index=True)
    )
