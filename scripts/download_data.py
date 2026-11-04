"""Goodbooks-10k 원본 CSV를 data/raw/에 내려받는다."""

from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master"
FILES = ["books.csv", "book_tags.csv", "tags.csv", "ratings.csv", "to_read.csv"]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = RAW_DIR / name
        if path.exists():
            print(f"skip {name} (이미 있음)")
            continue
        # NOTE: 받다가 끊기면 불완전한 파일이 남지 않도록 .part로 받은 뒤 이름을 바꾼다.
        part = path.with_suffix(".part")
        urlretrieve(f"{BASE_URL}/{name}", part)
        part.rename(path)
        print(f"done {name} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
