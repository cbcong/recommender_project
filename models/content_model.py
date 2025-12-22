import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
if os.path.isdir(UTILS_DIR) and UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

try:
    from preprocess import load_ml1m_ratings, build_id_mappings
except ImportError:
    from utils.preprocess import load_ml1m_ratings, build_id_mappings


class ItemContentEncoder(nn.Module):
    """
    输入 item_idx -> 输出对齐后的内容向量（text + image 拼接）
    支持外部传入 idx2item/n_items，确保与 preprocess mapping 完全一致。
    """

    def __init__(
        self,
        ratings_path: str = None,
        text_feat_path: str = None,
        text_index_path: str = None,
        image_feat_path: str = None,
        image_index_path: str = None,
        use_text: bool = True,
        use_image: bool = True,
        idx2item=None,
        n_items: int = None,
    ):
        super().__init__()

        if ratings_path is None:
            ratings_path = os.path.join(PROJECT_ROOT, "data", "movielens", "ml-1m", "ratings.dat")

        self.use_text = bool(use_text)
        self.use_image = bool(use_image)

        if idx2item is not None:
            self.idx2item = np.array(idx2item, dtype=np.int64)
            self.n_items = int(n_items) if n_items is not None else int(len(self.idx2item))
        else:
            print(f"[ItemContentEncoder] Loading ratings from: {ratings_path}")
            ratings_df = load_ml1m_ratings(ratings_path)
            _, item2idx, _, idx2item_local = build_id_mappings(ratings_df)
            self.idx2item = np.array(idx2item_local, dtype=np.int64)
            self.n_items = int(len(item2idx))

        text_matrix = None
        if self.use_text and text_feat_path and os.path.exists(text_feat_path):
            print(f"[ItemContentEncoder] Loading text features: {text_feat_path}")
            text_emb = np.load(text_feat_path).astype(np.float32)

            if not text_index_path or not os.path.exists(text_index_path):
                raise FileNotFoundError(f"text_index_path not found: {text_index_path}")

            df = pd.read_csv(text_index_path)
            if "movieId" in df.columns:
                key = "movieId"
            elif "movie_id" in df.columns:
                key = "movie_id"
            else:
                raise ValueError(f"Cannot find movieId/movie_id in {text_index_path}")

            movie_ids = df[key].values
            movie_to_row = {int(m): i for i, m in enumerate(movie_ids)}

            n_items = self.n_items
            d = int(text_emb.shape[1])
            text_matrix = np.zeros((n_items, d), dtype=np.float32)

            miss = 0
            for item_idx2, mid in enumerate(self.idx2item):
                ridx = movie_to_row.get(int(mid), None)
                if ridx is None:
                    miss += 1
                    continue
                text_matrix[item_idx2] = text_emb[ridx]
            print(f"[ItemContentEncoder] Text aligned: {n_items - miss}/{n_items}")

        image_matrix = None
        if self.use_image and image_feat_path and os.path.exists(image_feat_path):
            print(f"[ItemContentEncoder] Loading image features: {image_feat_path}")
            img_emb = np.load(image_feat_path).astype(np.float32)

            if not image_index_path or not os.path.exists(image_index_path):
                raise FileNotFoundError(f"image_index_path not found: {image_index_path}")

            df = pd.read_csv(image_index_path)
            if "movieId" in df.columns:
                key = "movieId"
            elif "movie_id" in df.columns:
                key = "movie_id"
            else:
                raise ValueError(f"Cannot find movieId/movie_id in {image_index_path}")

            movie_ids = df[key].values
            movie_to_row = {int(m): i for i, m in enumerate(movie_ids)}

            n_items = self.n_items
            d = int(img_emb.shape[1])
            image_matrix = np.zeros((n_items, d), dtype=np.float32)

            miss = 0
            for item_idx2, mid in enumerate(self.idx2item):
                ridx = movie_to_row.get(int(mid), None)
                if ridx is None:
                    miss += 1
                    continue
                image_matrix[item_idx2] = img_emb[ridx]
            print(f"[ItemContentEncoder] Image aligned: {n_items - miss}/{n_items}")

        if text_matrix is not None:
            self.register_buffer("text_features", torch.from_numpy(text_matrix))
            self.text_dim = int(text_matrix.shape[1])
        else:
            self.text_dim = 0
            self.register_buffer("text_features", torch.zeros(self.n_items, 0, dtype=torch.float32))

        if image_matrix is not None:
            self.register_buffer("image_features", torch.from_numpy(image_matrix))
            self.image_dim = int(image_matrix.shape[1])
        else:
            self.image_dim = 0
            self.register_buffer("image_features", torch.zeros(self.n_items, 0, dtype=torch.float32))

        self.content_dim = int(self.text_dim + self.image_dim)
        print(f"[ItemContentEncoder] content_dim={self.content_dim} (text={self.text_dim}, image={self.image_dim})")

    def forward(self, item_idx: torch.LongTensor) -> torch.Tensor:
        feats = []
        if self.use_text and self.text_dim > 0:
            feats.append(self.text_features[item_idx])
        if self.use_image and self.image_dim > 0:
            feats.append(self.image_features[item_idx])
        if not feats:
            return torch.zeros(item_idx.size(0), 0, device=item_idx.device)
        return torch.cat(feats, dim=-1)
