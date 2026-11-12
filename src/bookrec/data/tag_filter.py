"""책 내용과 관계없는 책장 태그를 걸러 낸다(B1)."""

from __future__ import annotations

import re

import pandas as pd

# NOTE: 장르처럼 널리 쓰이는 태그도 있어, 등장 빈도만으로는 제거하지 않는다.
SHELF_TAGS = {
    "to-read", "to-read-next", "want-to-read", "want-to-read-soon", "tbr", "unread",
    "currently-reading", "reading", "read", "read-books", "already-read", "have-read",
    "owned", "owned-books", "books-i-own", "i-own", "own", "own-it", "my-books",
    "my-library", "library", "personal-library", "in-my-library", "on-my-shelf",
    "on-my-bookshelf", "bookshelf", "to-buy", "want-to-buy", "wish-list", "wishlist",
    "borrowed", "borrowed-books", "lent", "default", "all", "all-books",
}
FORMAT_TAGS = {
    "kindle", "kindle-books", "ebook", "ebooks", "e-book", "e-books", "audio",
    "audiobook", "audiobooks", "audio-books", "audible", "paperback", "hardcover",
    "hardback", "nook", "epub", "pdf", "digital", "print", "physical-books",
}
PERSONAL_TAGS = {
    "favorites", "favourites", "favorite", "favourite", "my-favorites", "my-favourites",
    "all-time-favorites", "all-time-favourites", "5-stars", "4-stars", "3-stars",
    "2-stars", "1-star", "five-stars", "four-stars", "did-not-finish", "dnf",
    "abandoned", "unfinished", "re-read", "reread", "re-reads", "rereads",
}
# NOTE: 시대·제목(1984 등)을 지우지 않고, 독서 기록이라고 명시된 연도만 대상으로 한다.
READING_YEAR = re.compile(r"(?:(?:read|reads|reading|books-read|read-in)-(?:19|20)\d{2}|(?:19|20)\d{2}-(?:read|reads|reading))")


# NOTE: 혼합 태그도 전체를 제거한다. 일부를 잘라 장르 태그로 바꾸면 count가 중복될 수 있다.
SHELF_PATTERN = re.compile(r"(?:^|-)(?:to-read|want-to-read|owned|own-to-read|books-owned|books-i-own)(?:-|$)")
FORMAT_PATTERN = re.compile(r"(?:^|-)(?:kindle|audiobooks?|audio-books|ebooks?|e-books|paperback|hardcover|hardback|nook|epub)(?:-|$)")
BOOK_LIST = re.compile(r"^\d+-books(?:-|$)")


def _reason(name: object, extended: bool) -> str:
    if not isinstance(name, str) or not name.strip():
        return "empty"
    tag = re.sub(r"[\s_]+", "-", name.strip().lower())
    if tag in SHELF_TAGS:
        return "shelf"
    if tag in FORMAT_TAGS:
        return "format"
    # NOTE: 1001-books-to-read-before-you-die 같은 명명된 목록은 일괄 삭제하지 않는다.
    if not BOOK_LIST.search(tag):
        if SHELF_PATTERN.search(tag):
            return "shelf"
        if FORMAT_PATTERN.search(tag):
            return "format"
    if extended and tag in PERSONAL_TAGS:
        return "personal"
    if extended and READING_YEAR.fullmatch(tag):
        return "reading_year"
    return "keep"


def tag_filter_report(book_tags: pd.DataFrame, extended: bool = True) -> pd.DataFrame:
    """태그별 책 수·count 합계와 제거 이유를 돌려준다. extended=False면 개인 평가·독서 연도를 남긴다."""
    report = book_tags.groupby("tag_name", dropna=False).agg(
        books=("book_id", "nunique"), count=("count", "sum")
    ).reset_index()
    report["reason"] = report["tag_name"].map(lambda name: _reason(name, extended))
    return report.sort_values(["books", "count"], ascending=False, ignore_index=True)


def filter_book_tags(book_tags: pd.DataFrame, extended: bool = True) -> pd.DataFrame:
    """책장·매체 태그를 제거한다. extended이면 개인 평가·독서 연도 태그도 제거한다.

    태그 이름 정규화는 규칙 판정에만 쓴다. 입력의 값·컬럼·남은 행 순서는 바꾸지 않는다.
    """
    reasons = {name: _reason(name, extended) for name in book_tags["tag_name"].unique()}
    keep = book_tags["tag_name"].map(reasons).eq("keep")
    return book_tags.loc[keep].copy()
