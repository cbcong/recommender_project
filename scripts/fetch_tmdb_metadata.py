import os
import time
import requests
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "movielens", "ml1m_movies_for_tmdb.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "tmdb", "ml1m_tmdb_metadata.csv")

# ========= 在这里填你的 TMDB 密钥 =========
TMDB_API_KEY = "7ae8e2743b3936ad25e25e2227434804"  # 这里填 v3 API KEY（短的那串）
# 可选：如果想用 Bearer token，也可以用：
TMDB_BEARER_TOKEN = "YOUR_TMDB_BEARER_TOKEN"  # 可留空不用
# =========================================

BASE_URL = "https://api.themoviedb.org/3/search/movie"

def search_movie(title, year=None):
    """
    用 TMDB search/movie 查找电影，优先按 title + year 匹配。
    返回最匹配的一条（或 None）。
    """
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": False,
        "language": "en-US",
    }
    if year is not None:
        params["year"] = int(year)

    resp = requests.get(BASE_URL, params=params, timeout=100)
    if resp.status_code != 200:
        print("TMDB error:", resp.status_code, resp.text)
        return None

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return None

    # 简单策略：取第一条结果作为最匹配
    return results[0]

def main():
    if TMDB_API_KEY == "YOUR_TMDB_API_KEY":
        raise ValueError("请先在脚本中填写 TMDB_API_KEY")

    df = pd.read_csv(INPUT_CSV)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    meta_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching TMDB"):
        movie_id = row["movieId"]
        title = row["title"]
        year = row["year"]

        try:
            result = search_movie(title, year)
        except Exception as e:
            print(f"[Error] movieId={movie_id}, title={title}, error={e}")
            result = None

        if result is None:
            meta_records.append(
                {
                    "movieId": movie_id,
                    "title": title,
                    "year": year,
                    "tmdb_id": None,
                    "imdb_id": None,
                    "overview": None,
                    "poster_path": None,
                    "original_title": None,
                    "release_date": None,
                    "vote_average": None,
                }
            )
        else:
            tmdb_id = result.get("id")
            overview = result.get("overview")
            poster_path = result.get("poster_path")
            original_title = result.get("original_title")
            release_date = result.get("release_date")
            vote_average = result.get("vote_average")

            # 再调一次 /movie/{id} 拿 imdb_id（可选）
            imdb_id = None
            if tmdb_id is not None:
                detail_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
                detail_params = {"api_key": TMDB_API_KEY}
                try:
                    detail_resp = requests.get(detail_url, params=detail_params, timeout=100)
                    if detail_resp.status_code == 200:
                        imdb_id = detail_resp.json().get("imdb_id")
                except Exception as e:
                    print(f"[DetailError] tmdb_id={tmdb_id}, error={e}")

            meta_records.append(
                {
                    "movieId": movie_id,
                    "title": title,
                    "year": year,
                    "tmdb_id": tmdb_id,
                    "imdb_id": imdb_id,
                    "overview": overview,
                    "poster_path": poster_path,
                    "original_title": original_title,
                    "release_date": release_date,
                    "vote_average": vote_average,
                }
            )

        # 很关键：别打太快，避免被 TMDB 限流
        time.sleep(0.2)

    meta_df = pd.DataFrame(meta_records)
    meta_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved TMDB metadata to {OUTPUT_CSV}, total {len(meta_df)} rows.")

if __name__ == "__main__":
    main()
