import os
import requests
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_CSV = os.path.join(PROJECT_ROOT, "data", "tmdb", "ml1m_tmdb_metadata.csv")
POSTER_DIR = os.path.join(PROJECT_ROOT, "data", "tmdb", "posters")

BASE_IMG_URL = "https://image.tmdb.org/t/p/w500"

def download_poster(movie_id, poster_path):
    url = BASE_IMG_URL + poster_path
    ext = os.path.splitext(poster_path)[-1] or ".jpg"
    filename = f"{movie_id}{ext}"
    save_path = os.path.join(POSTER_DIR, filename)

    if os.path.exists(save_path):
        return

    resp = requests.get(url, stream=True, timeout=10)
    if resp.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(1024):
                if chunk:
                    f.write(chunk)
    else:
        print(f"[PosterError] movieId={movie_id}, status={resp.status_code}")

def main():
    df = pd.read_csv(META_CSV)
    os.makedirs(POSTER_DIR, exist_ok=True)

    df_valid = df.dropna(subset=["poster_path"])
    print(f"Total with poster_path: {len(df_valid)}/{len(df)}")

    for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Downloading posters"):
        movie_id = row["movieId"]
        poster_path = row["poster_path"]
        if not isinstance(poster_path, str) or not poster_path.strip():
            continue
        try:
            download_poster(movie_id, poster_path)
        except Exception as e:
            print(f"[Error] movieId={movie_id}, error={e}")

    print(f"Posters saved to {POSTER_DIR}")

if __name__ == "__main__":
    main()
