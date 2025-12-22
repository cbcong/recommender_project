import os
import re
import pandas as pd

# 修改这里，让它指向你的工程根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML1M_MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "movielens", "ml-1m", "movies.dat")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "movielens", "ml1m_movies_for_tmdb.csv")

def parse_title_and_year(raw_title: str):
    """
    raw_title 例子: 'Toy Story (1995)'
    有些片名本身带括号，这里用正则取最后一对括号里的年份
    """
    year_match = re.search(r"\((\d{4})\)\s*$", raw_title)
    if year_match:
        year = int(year_match.group(1))
        title = raw_title[:year_match.start()].strip()
    else:
        year = None
        title = raw_title.strip()
    return title, year

def main():
    if not os.path.exists(ML1M_MOVIES_PATH):
        raise FileNotFoundError(f"movies.dat not found at {ML1M_MOVIES_PATH}")

    records = []
    with open(ML1M_MOVIES_PATH, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                movie_id_str, raw_title, genres = line.split("::")
            except ValueError:
                # 极个别行不标准，跳过或打印
                print("Skip malformed line:", line)
                continue
            movie_id = int(movie_id_str)
            title, year = parse_title_and_year(raw_title)
            records.append(
                {
                    "movieId": movie_id,
                    "raw_title": raw_title,
                    "title": title,
                    "year": year,
                    "genres": genres,
                }
            )

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved parsed movies to {OUTPUT_CSV}, total {len(df)} rows.")

if __name__ == "__main__":
    main()
