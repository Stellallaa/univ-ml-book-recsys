import pandas as pd

from bookrec.data.tag_filter import filter_book_tags, tag_filter_report


def _tags(names):
    return pd.DataFrame({"book_id": range(1, len(names) + 1), "tag_name": names, "count": 10})


def test_filter_removes_shelf_and_format_but_preserves_content():
    # Given
    tags = _tags(["to-read", "owned", "kindle", "fiction", "young-adult", "1984", "19th-century"])
    original = tags.copy(deep=True)

    # When
    cleaned = filter_book_tags(tags)

    # Then
    pd.testing.assert_frame_equal(cleaned, original.iloc[3:])
    pd.testing.assert_frame_equal(tags, original)


def test_filter_matches_case_spaces_and_underscores_without_changing_values():
    # Given
    tags = _tags([" TO_READ ", "Books I Own", "Science Fiction"])

    # When
    cleaned = filter_book_tags(tags)

    # Then
    assert cleaned["tag_name"].tolist() == ["Science Fiction"]


def test_filter_extended_rules_are_explicit_and_optional():
    # Given
    tags = _tags(["favorites", "2015-reads", "read-in-2016", "historical-fiction", "2015", "2015-history"])

    # When
    basic = filter_book_tags(tags, extended=False)
    extended = filter_book_tags(tags)

    # Then
    assert len(basic) == 6
    assert extended["tag_name"].tolist() == ["historical-fiction", "2015", "2015-history"]


def test_filter_handles_empty_input_and_empty_names():
    # Given
    tags = _tags([None, "", "  ", "fantasy"])

    # When
    cleaned = filter_book_tags(tags)
    empty = filter_book_tags(tags.iloc[:0])

    # Then
    assert cleaned["tag_name"].tolist() == ["fantasy"]
    pd.testing.assert_frame_equal(empty, tags.iloc[:0])


def test_report_counts_books_and_explains_removal():
    # Given
    tags = pd.DataFrame({"book_id": [1, 2, 1], "tag_name": ["owned", "owned", "fantasy"], "count": [10, 20, 5]})

    # When
    report = tag_filter_report(tags).set_index("tag_name")

    # Then
    assert report.loc["owned", "books"] == 2
    assert report.loc["owned", "count"] == 30
    assert report.loc["owned", "reason"] == "shelf"
    assert report.loc["fantasy", "reason"] == "keep"

def test_mixed_status_and_format_tags_are_removed_but_named_lists_remain():
    # Given
    tags = _tags(["to-read-fiction", "own-to-read", "on-kindle", "owned-to-read", "books-owned",
                  "1001-books-to-read-before-you-die", "fiction", "kindlewood"])

    # When
    basic = filter_book_tags(tags, extended=False)
    extended = filter_book_tags(tags)

    # Then
    expected = ["1001-books-to-read-before-you-die", "fiction", "kindlewood"]
    assert basic.tag_name.tolist() == extended.tag_name.tolist() == expected
