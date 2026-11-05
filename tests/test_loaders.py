import pandas as pd

from bookrec.data import load_book_tags, load_books, load_ratings, load_to_read


def write_csvs(raw_dir, **tables):
    for name, df in tables.items():
        df.to_csv(raw_dir / f"{name}.csv", index=False)


def test_load_ratings_keeps_file_order(tmp_path):
    # Given: 독자가 섞여 있고 정렬되지 않은 평점 파일
    ratings = pd.DataFrame({"user_id": [2, 1, 2], "book_id": [30, 10, 20], "rating": [5, 3, 4]})
    write_csvs(tmp_path, ratings=ratings)

    # When
    loaded = load_ratings(tmp_path)

    # Then: 행 순서가 파일 그대로다
    pd.testing.assert_frame_equal(loaded, ratings)


def test_load_to_read_keeps_file_order(tmp_path):
    # Given
    to_read = pd.DataFrame({"user_id": [3, 1], "book_id": [7, 9]})
    write_csvs(tmp_path, to_read=to_read)

    # When
    loaded = load_to_read(tmp_path)

    # Then
    pd.testing.assert_frame_equal(loaded, to_read)


def test_load_books_splits_authors_and_drops_goodreads_id(tmp_path):
    # Given: 공저자와 연속 공백이 있는 저자, 앞뒤 공백이 있는 제목
    books = pd.DataFrame({
        "book_id": [1, 2],
        "goodreads_book_id": [100, 200],
        "title": [" Angels ", "Odyssey"],
        "authors": ["Betty  Smith", "Homer, Robert Fagles"],
        "original_publication_year": [2001.0, -720.0],
    })
    write_csvs(tmp_path, books=books)

    # When
    loaded = load_books(tmp_path)

    # Then
    assert "goodreads_book_id" not in loaded.columns
    assert loaded["title"].tolist() == ["Angels", "Odyssey"]
    assert loaded["authors"].tolist() == [["Betty Smith"], ["Homer", "Robert Fagles"]]
    assert loaded["main_author"].tolist() == ["Betty Smith", "Homer"]
    assert loaded["original_publication_year"].tolist() == [2001.0, -720.0]


def test_load_book_tags_maps_ids_and_cleans_counts(tmp_path):
    # Given: goodreads_book_id로 된 태그, 음수 count 1행, 같은 (책, 태그) 중복 1쌍
    write_csvs(
        tmp_path,
        books=pd.DataFrame({"book_id": [1, 2], "goodreads_book_id": [100, 200]}),
        tags=pd.DataFrame({"tag_id": [0, 1, 2], "tag_name": ["fantasy", "owned", "default"]}),
        book_tags=pd.DataFrame({
            "goodreads_book_id": [100, 100, 100, 200],
            "tag_id": [1, 1, 0, 2],
            "count": [6066, 6065, 50, -1],
        }),
    )

    # When
    loaded = load_book_tags(tmp_path)

    # Then: book_id로 바뀌고, 음수 행은 빠지고, 중복은 큰 count 하나만 남는다
    expected = pd.DataFrame({"book_id": [1, 1], "tag_name": ["owned", "fantasy"], "count": [6066, 50]})
    pd.testing.assert_frame_equal(loaded, expected)
