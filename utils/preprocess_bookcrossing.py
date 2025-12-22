import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class BookCrossingDataset(Dataset):
    """
    简单的评分数据集，返回: user_idx, item_idx, rating
    """
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.from_numpy(user_ids).long()
        self.item_ids = torch.from_numpy(item_ids).long()
        self.ratings = torch.from_numpy(ratings).float()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


def load_bookcrossing_raw(data_dir: str):
    """
    读取 Book-Crossing 三个原始 CSV 表。
    """
    ratings_path = os.path.join(data_dir, "BX-Book-Ratings.csv")
    users_path = os.path.join(data_dir, "BX-Users.csv")
    books_path = os.path.join(data_dir, "BX-Books.csv")

    ratings = pd.read_csv(
        ratings_path, sep=';', encoding='latin-1', on_bad_lines='skip'
    )
    users = pd.read_csv(
        users_path, sep=';', encoding='latin-1', on_bad_lines='skip'
    )
    books = pd.read_csv(
        books_path, sep=';', encoding='latin-1', on_bad_lines='skip'
    )

    return ratings, users, books


def _parse_country_from_location(loc: str) -> str:
    """
    从 Location 字符串里粗略解析出国家：
    典型格式: "nyc, new york, usa"
    规则：按逗号分割，取最后一段，strip + 小写。
    """
    if not isinstance(loc, str) or not loc:
        return "unknown"
    parts = [p.strip().lower() for p in loc.split(",") if p.strip()]
    if not parts:
        return "unknown"
    return parts[-1]


def build_user_features(
    users: pd.DataFrame,
    user2idx: dict,
    n_users: int,
    top_k_countries: int = 20,
):
    """
    基于 BX-Users 里的 Age + Location（国家）构造用户特征矩阵：
    - Age: 过滤 [5,100]，NaN 用均值填补，再归一化到 [0,1]
    - Country: 解析 Location 的最后一段为 country 字符串
      * 统计频次，选 top_k_countries 个国家单独 one-hot
      * 其他国家 + 无法解析合并到 "other" 一类
    返回:
      user_features: [n_users, 1 + (top_k_countries + 1)]
      meta_user_feat: 包含国家映射等信息（可选）
    """
    users_local = users.copy()

    # ---- Age 处理 ----
    users_local["Age"] = pd.to_numeric(users_local["Age"], errors="coerce")
    age = users_local["Age"]
    # 合理范围 [5,100]，其余视为 NaN
    age = age.where((age >= 5) & (age <= 100))
    mean_age = age.mean()
    age_filled = age.fillna(mean_age)
    age_norm = (age_filled - 5.0) / (100.0 - 5.0)
    age_norm = age_norm.clip(0.0, 1.0)
    users_local["age_norm"] = age_norm

    # ---- Country 解析 ----
    users_local["country"] = users_local["Location"].apply(_parse_country_from_location)

    # 统计国家频次，选 top_k_countries
    country_counts = users_local["country"].value_counts()
    top_countries = list(country_counts.head(top_k_countries).index)
    # "other" 作为兜底一类
    country_to_idx = {c: i for i, c in enumerate(top_countries)}  # 0..top_k-1
    other_idx = top_k_countries  # 最后一维留给 "other"

    # 以 User-ID 为索引，方便用 user2idx 映射
    users_local = users_local.set_index("User-ID")

    feat_dim = 1 + (top_k_countries + 1)  # age_norm(1) + country_one_hot(top_k + other)
    user_features = np.zeros((n_users, feat_dim), dtype=np.float32)

    for raw_uid, idx in user2idx.items():
        if raw_uid in users_local.index:
            row = users_local.loc[raw_uid]

            # 年龄
            age_v = float(row["age_norm"])
            user_features[idx, 0] = age_v

            # 国家
            c = row["country"]
            if c in country_to_idx:
                c_idx = country_to_idx[c]
            else:
                c_idx = other_idx
            user_features[idx, 1 + c_idx] = 1.0
        else:
            # 极少数 ratings 里有用户，但 user 表里没有：年龄用均值，国家放 other
            user_features[idx, 0] = float((mean_age - 5.0) / (100.0 - 5.0))
            user_features[idx, 1 + other_idx] = 1.0

    meta_user_feat = {
        "top_countries": top_countries,
        "country_to_idx": country_to_idx,
        "other_idx": other_idx,
        "feat_dim": feat_dim,
    }

    return user_features, meta_user_feat


def prepare_bookcrossing_interactions(
    data_dir: str,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
    drop_implicit_zeros: bool = True,
):
    """
    从原始 BX-Book-Ratings 中构造 (user_idx, item_idx, rating) DataFrame，
    并返回一些元信息（用户数、物品数、id 映射、用户特征矩阵）。
    """
    ratings, users, books = load_bookcrossing_raw(data_dir)

    df = ratings.copy()

    # 1) 去掉隐式的 0 分
    if drop_implicit_zeros:
        df = df[df["Book-Rating"] > 0]

    # 2) 1~10 映射到 1~5：ceil(r / 2)
    df["rating"] = np.ceil(df["Book-Rating"].astype(np.float32) / 2.0)
    df["rating"] = df["rating"].clip(1, 5)

    # 3) 过滤少交互的用户和图书
    user_counts = df["User-ID"].value_counts()
    item_counts = df["ISBN"].value_counts()

    df = df[df["User-ID"].isin(user_counts[user_counts >= min_user_ratings].index)]
    df = df[df["ISBN"].isin(item_counts[item_counts >= min_item_ratings].index)]

    # 4) 映射到连续索引
    unique_users = df["User-ID"].unique()
    unique_items = df["ISBN"].unique()

    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {isbn: j for j, isbn in enumerate(unique_items)}

    df["user_idx"] = df["User-ID"].map(user2idx)
    df["item_idx"] = df["ISBN"].map(item2idx)

    n_users = len(unique_users)
    n_items = len(unique_items)

    # 5) 构造用户年龄 + 国家特征矩阵
    user_features, meta_user_feat = build_user_features(
        users, user2idx, n_users, top_k_countries=20
    )

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "user_features": user_features,      # [n_users, feat_dim]
        "user_feat_meta": meta_user_feat,    # 记录国家映射等信息
    }

    return df[["user_idx", "item_idx", "rating"]].reset_index(drop=True), meta


def train_val_test_split(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    n_total = len(df)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def df_to_dataset(df: pd.DataFrame) -> BookCrossingDataset:
    user = df["user_idx"].to_numpy(dtype=np.int64)
    item = df["item_idx"].to_numpy(dtype=np.int64)
    rating = df["rating"].to_numpy(dtype=np.float32)
    return BookCrossingDataset(user, item, rating)


def build_datasets_and_loaders_bookcrossing(
    data_dir: str,
    batch_size: int = 512,
    num_workers: int = 0,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    对 Book-Crossing 构建 train/val/test 的 DataLoader，
    返回 (train_loader, val_loader, test_loader, meta)
    其中 meta 里包含 user_features.
    """
    interactions, meta = prepare_bookcrossing_interactions(
        data_dir=data_dir,
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
    )

    train_df, val_df, test_df = train_val_test_split(
        interactions,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=42,
    )

    train_ds = df_to_dataset(train_df)
    val_ds = df_to_dataset(val_df)
    test_ds = df_to_dataset(test_df)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, meta

def train_val_test_split_coldstart(
    df: pd.DataFrame,
    cold_user_ratio: float = 0.1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    冷启动划分：
    - 随机选一部分用户作为冷启动用户 (cold_user_ratio)
    - 冷启动用户的所有交互只出现在 test_cold，不出现在 train/val
    - 对剩余“暖用户”做普通的 train/val/test 划分
    返回：
      train_df, val_df, test_all_df, test_cold_df, cold_users
    """
    rng = np.random.RandomState(seed)

    # 1) 随机挑选冷启动用户
    unique_users = df["user_idx"].unique()
    rng.shuffle(unique_users)
    n_cold = max(1, int(len(unique_users) * cold_user_ratio))
    cold_users = unique_users[:n_cold]

    cold_mask = df["user_idx"].isin(cold_users)
    df_cold = df[cold_mask].reset_index(drop=True)      # 所有冷启动用户交互
    df_warm = df[~cold_mask].reset_index(drop=True)     # 其余暖用户交互

    # 2) 对暖用户交互做普通划分
    train_df, val_df, test_warm_df = train_val_test_split(
        df_warm,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # 3) 总 test = 暖用户 test + 冷启动用户全部
    test_all_df = pd.concat([test_warm_df, df_cold], ignore_index=True)
    test_cold_df = df_cold  # 单独冷启动 test 集

    return train_df, val_df, test_all_df, test_cold_df, cold_users


def build_datasets_and_loaders_bookcrossing_coldstart(
    data_dir: str,
    batch_size: int = 512,
    num_workers: int = 0,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
    cold_user_ratio: float = 0.1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
):
    """
    冷启动版本的数据加载：
    返回:
      train_loader, val_loader, test_all_loader, test_cold_loader, meta_cold
    其中 meta_cold 在原 meta 基础上增加:
      - "cold_users": 冷启动用户的 user_idx 列表
    """
    interactions, meta = prepare_bookcrossing_interactions(
        data_dir=data_dir,
        min_user_ratings=min_user_ratings,
        min_item_ratings=min_item_ratings,
    )

    train_df, val_df, test_all_df, test_cold_df, cold_users = train_val_test_split_coldstart(
        interactions,
        cold_user_ratio=cold_user_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=42,
    )

    train_ds = df_to_dataset(train_df)
    val_ds = df_to_dataset(val_df)
    test_all_ds = df_to_dataset(test_all_df)
    test_cold_ds = df_to_dataset(test_cold_df)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_all_loader = DataLoader(
        test_all_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_cold_loader = DataLoader(
        test_cold_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    meta_cold = dict(meta)
    meta_cold["cold_users"] = cold_users

    return train_loader, val_loader, test_all_loader, test_cold_loader, meta_cold
