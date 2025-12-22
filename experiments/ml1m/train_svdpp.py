import os
import sys
import time
import yaml
from typing import Dict, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ================= 路径设置：从 experiments/ml1m 回到项目根目录 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# CURRENT_DIR = .../recommender_project/experiments/ml1m
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
# PROJECT_ROOT = .../recommender_project

UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 先尝试从 utils 目录直接 import preprocess.py，和你之前其它脚本保持一致
try:
    from preprocess import load_ml1m_ratings, build_id_mappings, split_by_user_time
except ImportError:
    from utils.preprocess import load_ml1m_ratings, build_id_mappings, split_by_user_time

# 同理，SVD++ 模型：优先从 models 目录直接 import
try:
    from svdpp_model import SVDPP
except ImportError:
    from models.svdpp_model import SVDPP
# =====================================================================


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


class RatingDataset(Dataset):
    """简单的评分数据集，输出 (user_idx, item_idx, rating)"""

    def __init__(self, user_idx, item_idx, rating):
        assert len(user_idx) == len(item_idx) == len(rating)
        self.user_idx = np.array(user_idx, dtype=np.int64)
        self.item_idx = np.array(item_idx, dtype=np.int64)
        self.rating = np.array(rating, dtype=np.float32)

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        u = int(self.user_idx[idx])
        i = int(self.item_idx[idx])
        r = float(self.rating[idx])
        return (
            torch.LongTensor([u]).squeeze(0),
            torch.LongTensor([i]).squeeze(0),
            torch.FloatTensor([r]).squeeze(0),
        )


def build_user_interactions(user_idx, item_idx, num_users: int) -> Dict[int, Set[int]]:
    """
    从训练集构建用户隐式反馈集合 N(u)：
      N(u) = { 该用户在训练集中出现过的所有物品 i }
    """
    user_interactions: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
    for u, i in zip(user_idx, item_idx):
        user_interactions[int(u)].add(int(i))
    return user_interactions


def train_epoch(model, loader, optimizer, device, reg_lambda: float = 0.0):
    model.train()
    total_loss = 0.0
    n = 0

    mse = nn.MSELoss(reduction="sum")

    for user_idx, item_idx, rating in loader:
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        rating = rating.to(device)

        optimizer.zero_grad()
        pred = model(user_idx, item_idx)  # [B]

        loss_mse = mse(pred, rating)
        loss = loss_mse

        # 可选的 L2 正则（主要给 embedding 做一点约束）
        if reg_lambda > 0.0:
            l2_reg = 0.0
            for param in model.parameters():
                l2_reg = l2_reg + torch.sum(param.pow(2))
            loss = loss + reg_lambda * l2_reg

        loss.backward()
        optimizer.step()

        batch_size = rating.size(0)
        total_loss += loss_mse.item()  # 这里只统计 MSE 部分，便于直观
        n += batch_size

    return total_loss / n


def evaluate_rmse_mae(model, loader, device):
    model.eval()
    mse_loss = 0.0
    mae_loss = 0.0
    n = 0

    with torch.no_grad():
        for user_idx, item_idx, rating in loader:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            rating = rating.to(device)

            pred = model(user_idx, item_idx)
            mse_loss += torch.sum((pred - rating) ** 2).item()
            mae_loss += torch.sum(torch.abs(pred - rating)).item()
            n += rating.size(0)

    rmse = (mse_loss / n) ** 0.5
    mae = mae_loss / n
    return rmse, mae


def main():
    config_path = os.path.join(PROJECT_ROOT, "utils", "config.yaml")
    cfg = load_config(config_path)

    # 这里做一次绝对路径修正
    ratings_cfg_path = cfg["data"]["ml1m_ratings"]
    if os.path.isabs(ratings_cfg_path):
        ratings_path = ratings_cfg_path
    else:
        ratings_path = os.path.join(PROJECT_ROOT, ratings_cfg_path)

    batch_size = cfg["train"]["batch_size"]
    # 给 SVD++ 单独预留一个 lr / reg，如果没配就用默认
    lr = cfg["train"].get("svdpp_lr", cfg["train"]["lr"])
    reg_lambda = float(cfg["train"].get("svdpp_reg", 0.0))
    epochs = cfg["train"]["epochs"]

    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    embedding_dim = int(cfg["model"].get("mf_dim", 32))

    print(f"Using device: {device}")
    print(f"Ratings path: {ratings_path}")
    print("Loading ML-1M ratings and building splits...")
    # 后面保持不变：
    ratings_df = load_ml1m_ratings(ratings_path)


    # 1. 读取评分数据 & 映射 & 划分
    ratings_df = load_ml1m_ratings(ratings_path)
    user2idx, item2idx, idx2user, idx2item = build_id_mappings(ratings_df)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"n_users={n_users}, n_items={n_items}")

    train_df, val_df, test_df = split_by_user_time(ratings_df)

    train_u = train_df["userId"].map(user2idx).values
    train_i = train_df["movieId"].map(item2idx).values
    train_r = train_df["rating"].values.astype("float32")

    val_u = val_df["userId"].map(user2idx).values
    val_i = val_df["movieId"].map(item2idx).values
    val_r = val_df["rating"].values.astype("float32")

    test_u = test_df["userId"].map(user2idx).values
    test_i = test_df["movieId"].map(item2idx).values
    test_r = test_df["rating"].values.astype("float32")

    global_mean = float(train_r.mean())
    print(f"Global mean rating (train) = {global_mean:.4f}")

    # 2. 构建用户隐式反馈集合 N(u)
    user_interactions = build_user_interactions(train_u, train_i, n_users)

    # 3. DataLoader
    train_ds = RatingDataset(train_u, train_i, train_r)
    val_ds = RatingDataset(val_u, val_i, val_r)
    test_ds = RatingDataset(test_u, test_i, test_r)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # 4. 构建 SVD++ 模型
    print("Building SVD++ model...")
    model = SVDPP(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=embedding_dim,
        user_interactions=user_interactions,
        global_mean=global_mean,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    best_val_rmse = float("inf")
    best_state = None

    # 5. 训练循环
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, device, reg_lambda=reg_lambda
        )
        val_rmse, val_mae = evaluate_rmse_mae(model, val_loader, device)
        dt = time.time() - t0

        print(
            f"[SVD++][Epoch {epoch:03d}] train_MSE={train_loss:.4f} "
            f"val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f} ({dt:.1f}s)"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()

    # 6. 用最佳验证模型在 test 上评估
    if best_state is not None:
        model.load_state_dict(best_state)

    test_rmse, test_mae = evaluate_rmse_mae(model, test_loader, device)
    print(f"[SVD++][Test] RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

    # 7. 保存模型
    save_path = os.path.join(PROJECT_ROOT, "svdpp_ml1m_best.pth")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_users": n_users,
            "n_items": n_items,
            "config": cfg,
            "embedding_dim": embedding_dim,
            "global_mean": global_mean,
        },
        save_path,
    )
    print(f"Saved best SVD++ model to {save_path}")


if __name__ == "__main__":
    main()
