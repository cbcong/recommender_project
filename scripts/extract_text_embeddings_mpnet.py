import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch  # 放这里就行

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_CSV = os.path.join(PROJECT_ROOT, "data", "tmdb", "ml1m_tmdb_metadata.csv")
FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "features")
OUTPUT_NPY = os.path.join(FEATURE_DIR, "ml1m_text_embeddings_64.npy")
OUTPUT_INDEX = os.path.join(FEATURE_DIR, "ml1m_text_index.csv")

# TODO: 改成你本地 MPNet 模型的路径
LOCAL_MODEL_PATH = r"D:\WorkSpace\pycharm\Python学习路线\基于图书借阅数据的用户潜在借阅预测推荐\_all-mpnet-base-v2-safe"
PCA_DIM = 64


def load_metadata():
    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"Metadata CSV not found: {META_CSV}")
    df = pd.read_csv(META_CSV)
    df["overview"] = df["overview"].fillna("").astype(str)
    return df


def extract_mpnet_embeddings(df):
    os.makedirs(FEATURE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(LOCAL_MODEL_PATH, device=device)

    texts = df["overview"].tolist()
    all_embeddings = []

    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="MPNet encoding"):
        batch_texts = texts[i:i + batch_size]
        emb = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        all_embeddings.append(emb)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print("Raw MPNet embedding shape:", all_embeddings.shape)
    return all_embeddings


def reduce_dim_with_pca(embeddings, dim=PCA_DIM):
    print(f"Running PCA to {dim} dimensions...")
    pca = PCA(n_components=dim)
    reduced = pca.fit_transform(embeddings)
    print("Reduced embedding shape:", reduced.shape)
    return reduced


def main():
    df = load_metadata()
    movie_ids = df["movieId"].tolist()

    embeddings = extract_mpnet_embeddings(df)
    reduced = reduce_dim_with_pca(embeddings, dim=PCA_DIM)

    np.save(OUTPUT_NPY, reduced)
    pd.DataFrame({"movieId": movie_ids}).to_csv(OUTPUT_INDEX, index=False)

    print(f"Saved text embeddings to: {OUTPUT_NPY}")
    print(f"Saved movieId index to:   {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()
