import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_CSV = os.path.join(PROJECT_ROOT, "data", "tmdb", "ml1m_tmdb_metadata.csv")
POSTER_DIR = os.path.join(PROJECT_ROOT, "data", "tmdb", "posters")
FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "features")
OUTPUT_NPY = os.path.join(FEATURE_DIR, "ml1m_image_embeddings_64.npy")
OUTPUT_INDEX = os.path.join(FEATURE_DIR, "ml1m_image_index.csv")

PCA_DIM = 64


def load_movie_poster_paths():
    """
    根据 metadata 和 posters 目录，建立 movieId -> poster_file 的映射。
    我们假设 download_posters.py 保存成 {movieId}.jpg 这种形式。
    """
    df = pd.read_csv(META_CSV)

    # 建立 posters 目录下以 movieId 为文件名的映射
    if not os.path.exists(POSTER_DIR):
        raise FileNotFoundError(f"Poster dir not found: {POSTER_DIR}")

    files = os.listdir(POSTER_DIR)
    id_to_file = {}
    for f in files:
        stem, ext = os.path.splitext(f)
        if not stem.isdigit():
            continue
        movie_id = int(stem)
        id_to_file[movie_id] = os.path.join(POSTER_DIR, f)

    records = []
    for _, row in df.iterrows():
        movie_id = row["movieId"]
        poster_path = id_to_file.get(int(movie_id))
        if poster_path is not None and os.path.exists(poster_path):
            records.append({"movieId": movie_id, "poster_file": poster_path})

    poster_df = pd.DataFrame(records)
    print(f"Found posters for {len(poster_df)} movies.")
    return poster_df


def build_resnet_feature_extractor(device):
    """
    使用预训练 ResNet-50，去掉最后一层 FC，输出 2048 维特征
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # 去掉全连接层，保留到 avgpool
    modules = list(model.children())[:-1]  # 去掉最后的 fc
    backbone = nn.Sequential(*modules)
    backbone.to(device)
    backbone.eval()
    return backbone


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # ImageNet 预训练模型标准化参数
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def extract_image_embeddings(poster_df):
    os.makedirs(FEATURE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_resnet_feature_extractor(device)
    transform = get_transform()

    all_embeddings = []
    movie_ids = []

    with torch.no_grad():
        for _, row in tqdm(poster_df.iterrows(), total=len(poster_df), desc="ResNet encoding"):
            movie_id = int(row["movieId"])
            img_path = row["poster_file"]

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[ImageError] movieId={movie_id}, file={img_path}, err={e}")
                continue

            img_tensor = transform(img).unsqueeze(0).to(device)  # [1,3,224,224]
            feat = model(img_tensor)  # [1,2048,1,1]
            feat = feat.view(1, -1).cpu().numpy()  # [1,2048]

            all_embeddings.append(feat)
            movie_ids.append(movie_id)

    if not all_embeddings:
        raise RuntimeError("No image embeddings extracted.")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print("Raw image embedding shape:", all_embeddings.shape)
    return movie_ids, all_embeddings


def reduce_dim_with_pca(embeddings, dim=PCA_DIM):
    print(f"Running PCA to {dim} dimensions for image features...")
    pca = PCA(n_components=dim)
    reduced = pca.fit_transform(embeddings)
    print("Reduced image embedding shape:", reduced.shape)
    return reduced


def main():
    poster_df = load_movie_poster_paths()
    movie_ids, embeddings = extract_image_embeddings(poster_df)
    reduced = reduce_dim_with_pca(embeddings, dim=PCA_DIM)

    np.save(OUTPUT_NPY, reduced)
    pd.DataFrame({"movieId": movie_ids}).to_csv(OUTPUT_INDEX, index=False)

    print(f"Saved image embeddings to: {OUTPUT_NPY}")
    print(f"Saved movieId index to:    {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()
