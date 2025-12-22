import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# (A) 旧接口：显式评分 Dataset
# =========================
class RatingsDataset(Dataset):
    """
    显式评分数据集：返回 (user_idx, item_idx, rating)
    """
    def __init__(self, user_indices, item_indices, ratings):
        assert len(user_indices) == len(item_indices) == len(ratings)
        self.user_indices = torch.LongTensor(user_indices)
        self.item_indices = torch.LongTensor(item_indices)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.user_indices[idx],
            self.item_indices[idx],
            self.ratings[idx],
        )


def load_ml1m_ratings(ml1m_ratings_path: str) -> pd.DataFrame:
    """
    MovieLens-1M ratings.dat:
    UserID::MovieID::Rating::Timestamp
    """
    if not os.path.exists(ml1m_ratings_path):
        raise FileNotFoundError(f"ratings.dat not found at {ml1m_ratings_path}")

    df = pd.read_csv(
        ml1m_ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )
    return df


def build_id_mappings(df: pd.DataFrame):
    """
    将原始 userId, movieId 映射到 [0, n_user-1], [0, n_item-1]
    返回:
        user2idx, item2idx: dict
        idx2user, idx2item: list
    """
    unique_users = sorted(df["userId"].unique())
    unique_items = sorted(df["movieId"].unique())

    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {m: i for i, m in enumerate(unique_items)}

    idx2user = unique_users
    idx2item = unique_items

    return user2idx, item2idx, idx2user, idx2item


def split_by_user_time(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.1):
    """
    旧逻辑保留：按每个用户时间顺序划分 train/val/test：
    - >=3: 最后1条 test，倒数2条 val，其余 train
    - ==2: 1条 train 1条 test
    - ==1: 全 train
    """
    df = df.sort_values(["userId", "timestamp"])
    train_rows, val_rows, test_rows = [], [], []

    for user_id, group in df.groupby("userId"):
        rows = group.to_dict("records")
        n = len(rows)
        if n == 1:
            train_rows.extend(rows)
        elif n == 2:
            train_rows.append(rows[0])
            test_rows.append(rows[1])
        else:
            train_rows.extend(rows[:-2])
            val_rows.append(rows[-2])
            test_rows.append(rows[-1])

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, val_df, test_df


def build_datasets_and_loaders(
    ml1m_ratings_path: str,
    batch_size: int = 1024,
    num_workers: int = 0,
):
    """
    旧接口保留（给你其他模型用）：
    - 加载 ratings.dat
    - 映射 ID
    - 划分 train/val/test
    - 构建 Dataset + DataLoader
    """
    ratings_df = load_ml1m_ratings(ml1m_ratings_path)
    user2idx, item2idx, idx2user, idx2item = build_id_mappings(ratings_df)

    train_df, val_df, test_df = split_by_user_time(ratings_df)

    def encode(df_part):
        user_indices = df_part["userId"].map(user2idx).values
        item_indices = df_part["movieId"].map(item2idx).values
        ratings = df_part["rating"].values.astype("float32")
        return user_indices, item_indices, ratings

    train_u, train_i, train_r = encode(train_df)
    val_u, val_i, val_r = encode(val_df)
    test_u, test_i, test_r = encode(test_df)

    train_dataset = RatingsDataset(train_u, train_i, train_r)
    val_dataset = RatingsDataset(val_u, val_i, val_r)
    test_dataset = RatingsDataset(test_u, test_i, test_r)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    meta = {
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2item": idx2item,
    }

    return train_loader, val_loader, test_loader, meta


# =========================
# (B) 新接口：Hybrid 训练数据（历史序列 + 负采样）
# =========================
class HybridTrainDataset(Dataset):
    """
    Hybrid 训练样本：
      (user_idx, pos_item_idx, rating, hist_items[max_len], hist_mask[max_len], neg_items[num_neg])

    - hist_items: 0-padding（padding id=0 不会影响，因为有 mask）
    - hist_mask: True 表示有效位置
    - neg_items: 每次 __getitem__ 动态负采样（避免过拟合固定负例）
    """
    def __init__(
        self,
        user_indices: np.ndarray,
        pos_item_indices: np.ndarray,
        ratings: np.ndarray,
        hist_items: np.ndarray,
        hist_mask: np.ndarray,
        user_pos_sets: Dict[int, set],
        n_items: int,
        num_neg: int = 1,
        seed: int = 42,
    ):
        assert len(user_indices) == len(pos_item_indices) == len(ratings) == len(hist_items) == len(hist_mask)

        self.user_indices = torch.LongTensor(user_indices)
        self.pos_item_indices = torch.LongTensor(pos_item_indices)
        self.ratings = torch.FloatTensor(ratings)
        self.hist_items = torch.LongTensor(hist_items)      # [N, L]
        self.hist_mask = torch.BoolTensor(hist_mask)        # [N, L]

        self.user_pos_sets = user_pos_sets
        self.n_items = int(n_items)
        self.num_neg = int(num_neg)

        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.ratings)

    def _sample_neg(self, user_idx: int) -> int:
        # 负采样：从 [0, n_items-1] 随机采一个用户未交互过的 item
        pos = self.user_pos_sets.get(int(user_idx), set())
        while True:
            j = int(self.rng.randint(0, self.n_items))
            if j not in pos:
                return j

    def __getitem__(self, idx):
        u = self.user_indices[idx]
        i = self.pos_item_indices[idx]
        r = self.ratings[idx]
        h = self.hist_items[idx]
        m = self.hist_mask[idx]

        if self.num_neg == 1:
            neg = torch.LongTensor([self._sample_neg(int(u.item()))]).squeeze(0)
        else:
            negs = [self._sample_neg(int(u.item())) for _ in range(self.num_neg)]
            neg = torch.LongTensor(negs)

        return u, i, r, h, m, neg


class HybridEvalDataset(Dataset):
    """
    Hybrid 验证/测试样本（无负采样）：
      (user_idx, item_idx, rating, hist_items[max_len], hist_mask[max_len])
    """
    def __init__(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        ratings: np.ndarray,
        hist_items: np.ndarray,
        hist_mask: np.ndarray,
    ):
        assert len(user_indices) == len(item_indices) == len(ratings) == len(hist_items) == len(hist_mask)
        self.user_indices = torch.LongTensor(user_indices)
        self.item_indices = torch.LongTensor(item_indices)
        self.ratings = torch.FloatTensor(ratings)
        self.hist_items = torch.LongTensor(hist_items)
        self.hist_mask = torch.BoolTensor(hist_mask)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.user_indices[idx],
            self.item_indices[idx],
            self.ratings[idx],
            self.hist_items[idx],
            self.hist_mask[idx],
        )


def _pad_hist(seq: List[int], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回:
      hist_items: [max_len] int64
      hist_mask:  [max_len] bool
    """
    if max_len <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.bool_)

    seq = seq[-max_len:]
    L = len(seq)
    hist_items = np.zeros((max_len,), dtype=np.int64)
    hist_mask = np.zeros((max_len,), dtype=np.bool_)
    if L > 0:
        hist_items[:L] = np.array(seq, dtype=np.int64)
        hist_mask[:L] = True
    return hist_items, hist_mask


def build_hybrid_datasets_and_loaders(
    ml1m_ratings_path: str,
    batch_size: int = 1024,
    num_workers: int = 0,
    max_hist_len: int = 50,
    num_neg: int = 1,
    seed: int = 42,
):
    """
    Hybrid 专用数据构建（推荐你 Hybrid 都用这个）：
    - 严格按 user->timestamp 顺序
    - 对每个用户：
        train: 0..n-3
        val: n-2
        test: n-1
    - 每条样本携带“当前交互之前的历史 item 序列”
    - train_loader 返回 (u, pos_i, r, hist, mask, neg_i)
    - val/test_loader 返回 (u, i, r, hist, mask)
    """
    ratings_df = load_ml1m_ratings(ml1m_ratings_path)
    user2idx, item2idx, idx2user, idx2item = build_id_mappings(ratings_df)

    # 排序后按用户分组
    ratings_df = ratings_df.sort_values(["userId", "timestamp"])

    # 用于负采样：建议把“所有交互(含val/test)”都当正例集合，避免采到未来正例
    user_pos_sets: Dict[int, set] = {}
    for user_id, g in ratings_df.groupby("userId"):
        uidx = user2idx[int(user_id)]
        items = set([item2idx[int(mid)] for mid in g["movieId"].values])
        user_pos_sets[uidx] = items

    # 采样构建：每个样本都带历史（在本用户序列中“当前位置之前”的 items）
    train_u, train_i, train_r, train_h, train_m = [], [], [], [], []
    val_u, val_i, val_r, val_h, val_m = [], [], [], [], []
    test_u, test_i, test_r, test_h, test_m = [], [], [], [], []

    for user_id, g in ratings_df.groupby("userId"):
        g = g.sort_values("timestamp")
        uidx = user2idx[int(user_id)]

        items_idx = [item2idx[int(mid)] for mid in g["movieId"].values.tolist()]
        ratings = g["rating"].values.astype("float32").tolist()

        n = len(items_idx)
        if n == 1:
            # 单条：放 train（基本不影响）
            hist_seq = []
            h, m = _pad_hist(hist_seq, max_hist_len)
            train_u.append(uidx); train_i.append(items_idx[0]); train_r.append(ratings[0])
            train_h.append(h); train_m.append(m)
        elif n == 2:
            # 1 train 1 test
            # train(0)
            hist_seq = []
            h, m = _pad_hist(hist_seq, max_hist_len)
            train_u.append(uidx); train_i.append(items_idx[0]); train_r.append(ratings[0])
            train_h.append(h); train_m.append(m)
            # test(1)
            hist_seq = [items_idx[0]]
            h, m = _pad_hist(hist_seq, max_hist_len)
            test_u.append(uidx); test_i.append(items_idx[1]); test_r.append(ratings[1])
            test_h.append(h); test_m.append(m)
        else:
            # >=3: train 0..n-3, val n-2, test n-1
            # train
            hist_seq = []
            for t in range(0, n - 2):
                h, m = _pad_hist(hist_seq, max_hist_len)
                train_u.append(uidx); train_i.append(items_idx[t]); train_r.append(ratings[t])
                train_h.append(h); train_m.append(m)
                hist_seq.append(items_idx[t])

            # val
            h, m = _pad_hist(hist_seq, max_hist_len)
            val_u.append(uidx); val_i.append(items_idx[n - 2]); val_r.append(ratings[n - 2])
            val_h.append(h); val_m.append(m)
            hist_seq.append(items_idx[n - 2])

            # test
            h, m = _pad_hist(hist_seq, max_hist_len)
            test_u.append(uidx); test_i.append(items_idx[n - 1]); test_r.append(ratings[n - 1])
            test_h.append(h); test_m.append(m)

    # 转 numpy
    train_u = np.array(train_u, dtype=np.int64)
    train_i = np.array(train_i, dtype=np.int64)
    train_r = np.array(train_r, dtype=np.float32)
    train_h = np.stack(train_h, axis=0).astype(np.int64)
    train_m = np.stack(train_m, axis=0).astype(np.bool_)

    val_u = np.array(val_u, dtype=np.int64)
    val_i = np.array(val_i, dtype=np.int64)
    val_r = np.array(val_r, dtype=np.float32)
    val_h = np.stack(val_h, axis=0).astype(np.int64)
    val_m = np.stack(val_m, axis=0).astype(np.bool_)

    test_u = np.array(test_u, dtype=np.int64)
    test_i = np.array(test_i, dtype=np.int64)
    test_r = np.array(test_r, dtype=np.float32)
    test_h = np.stack(test_h, axis=0).astype(np.int64)
    test_m = np.stack(test_m, axis=0).astype(np.bool_)

    train_ds = HybridTrainDataset(
        train_u, train_i, train_r, train_h, train_m,
        user_pos_sets=user_pos_sets, n_items=len(item2idx), num_neg=num_neg, seed=seed
    )
    val_ds = HybridEvalDataset(val_u, val_i, val_r, val_h, val_m)
    test_ds = HybridEvalDataset(test_u, test_i, test_r, test_h, test_m)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "n_users": len(user2idx),
        "n_items": len(item2idx),
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2item": idx2item,
        "max_hist_len": int(max_hist_len),
        "num_neg": int(num_neg),
    }
    return train_loader, val_loader, test_loader, meta
