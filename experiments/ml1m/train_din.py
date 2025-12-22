# experiments/ml1m/train_din.py
import os
import sys
import time
import yaml
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ========= 路径设置：把工程根目录 / utils / models 加入 sys.path =========
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../recommender_project/experiments/ml1m
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))      # .../recommender_project

UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
# =====================================================================

# 预处理函数（先尝试从 preprocess.py 导入，找不到再从 utils.preprocess 导入）
try:
    from preprocess import (
        load_ml1m_ratings,
        build_id_mappings,
        split_by_user_time,
    )
except ImportError:
    from utils.preprocess import (
        load_ml1m_ratings,
        build_id_mappings,
        split_by_user_time,
    )

# DIN 模型：models/din_model.py
from din_model import DIN


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


class DINDataset(Dataset):
    """
    DIN 用的数据集：
      每条样本包含：
        - user_idx: 用户索引
        - hist_items: 历史 item 序列（补到 max_len）
        - hist_len: 真实历史长度
        - target_item: 当前预测的 item
        - rating: 当前评分（显式反馈）
    """
    def __init__(self, samples: List[Dict[str, Any]], max_hist_len: int = 50):
        super().__init__()
        self.samples = samples
        self.max_hist_len = max_hist_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        user_idx = int(s["user_idx"])
        target_item = int(s["target_item"])
        rating = float(s["rating"])

        hist = list(s["hist_items"])
        # 截断：只保留最近 max_hist_len 个
        if len(hist) > self.max_hist_len:
            hist = hist[-self.max_hist_len:]

        hist_len = len(hist)
        # 左侧 padding 为 0（与 DIN 模型中的 padding_idx=0 对应）
        pad_len = self.max_hist_len - hist_len
        hist_padded = [0] * pad_len + hist  # 长度 = max_hist_len

        return (
            torch.LongTensor([user_idx]).squeeze(0),          # user_idx
            torch.LongTensor(hist_padded),                    # hist_items [L]
            torch.LongTensor([hist_len]).squeeze(0),          # hist_len
            torch.LongTensor([target_item]).squeeze(0),       # target_item
            torch.FloatTensor([rating]).squeeze(0),           # rating
        )


def build_din_samples(
    full_df: pd.DataFrame,
    target_split: str,
    min_hist_len: int = 1,
) -> List[Dict[str, Any]]:
    """
    通用序列构造函数：

    full_df:  包含所有交互（train/val/test），列至少有：
              user_idx, item_idx, rating, timestamp, split in {"train","val","test"}
    target_split: 指定生成样本的 split（"train"/"val"/"test"）
    min_hist_len: 最小历史长度（< 该长度的样本会被跳过）

    逻辑：
      对每个用户，按时间排序其所有交互。
      遍历时间顺序：
        - 当前记录若属于 target_split 且之前历史长度 >= min_hist_len，
          则生成一条样本：(history -> 当前 target)。
        - 无论属于哪个 split，当前记录都会加入历史，用于后续样本。
      这样：
        - train 样本的历史只包含更早时间的行为（不会用未来的 val/test 记录）。
        - val/test 样本可以看到 train + 更早的 val/test 记录作为历史。
    """
    samples: List[Dict[str, Any]] = []

    # 按 user_idx, timestamp 排序
    full_df_sorted = full_df.sort_values(["user_idx", "timestamp"])

    for user_idx, df_u in full_df_sorted.groupby("user_idx"):
        hist_items: List[int] = []
        hist_ratings: List[float] = []

        for _, row in df_u.iterrows():
            split = row["split"]
            item_idx = int(row["item_idx"])
            rating = float(row["rating"])

            # 若当前记录用作目标
            if split == target_split and len(hist_items) >= min_hist_len:
                samples.append(
                    {
                        "user_idx": int(user_idx),
                        "hist_items": hist_items.copy(),
                        "target_item": item_idx,
                        "rating": rating,
                    }
                )

            # 当前记录加入历史（供后续使用）
            hist_items.append(item_idx)
            hist_ratings.append(rating)

    return samples


def train_one_epoch(
    model: DIN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        user_idx, hist_items, hist_len, target_item, rating = batch

        user_idx = user_idx.to(device)
        hist_items = hist_items.to(device)
        hist_len = hist_len.to(device)
        target_item = target_item.to(device)
        rating = rating.to(device)

        optimizer.zero_grad()
        preds = model(user_idx, hist_items, hist_len, target_item)
        loss = criterion(preds, rating)
        loss.backward()
        optimizer.step()

        batch_size = rating.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size

    return total_loss / max(1, n)


def evaluate_rmse_mae(
    model: DIN,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    mse_loss = 0.0
    mae_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            user_idx, hist_items, hist_len, target_item, rating = batch

            user_idx = user_idx.to(device)
            hist_items = hist_items.to(device)
            hist_len = hist_len.to(device)
            target_item = target_item.to(device)
            rating = rating.to(device)

            preds = model(user_idx, hist_items, hist_len, target_item)
            mse_loss += torch.sum((preds - rating) ** 2).item()
            mae_loss += torch.sum(torch.abs(preds - rating)).item()
            n += rating.size(0)

    rmse = (mse_loss / n) ** 0.5 if n > 0 else 0.0
    mae = mae_loss / n if n > 0 else 0.0
    return rmse, mae


def main():
    # 1. 配置与设备
    config_path = os.path.join(PROJECT_ROOT, "utils", "config.yaml")
    cfg = load_config(config_path)

    ratings_path = cfg["data"]["ml1m_ratings"]
    if not os.path.isabs(ratings_path):
        ratings_path = os.path.join(PROJECT_ROOT, ratings_path)

    batch_size = cfg["train"]["batch_size"]
    lr = cfg["train"]["lr"]
    epochs = cfg["train"]["epochs"]
    device_str = cfg["train"]["device"]
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Ratings path: {ratings_path}")

    # 2. 加载评分数据 & 构建 ID 映射 & 按时间划分
    print("Loading ML-1M ratings and building splits...")
    ratings_df = load_ml1m_ratings(ratings_path)
    user2idx, item2idx, idx2user, idx2item = build_id_mappings(ratings_df)

    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"n_users={n_users}, n_items={n_items}")

    train_df, val_df, test_df = split_by_user_time(ratings_df)

    # 标记 split，方便后续统一处理
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # 合并：full_df 是所有交互的集合
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 映射到索引（用户索引 0..n_users-1；物品索引 1..n_items，用 0 做 padding）
    full_df["user_idx"] = full_df["userId"].map(user2idx).astype("int64")
    full_df["item_idx_original"] = full_df["movieId"].map(item2idx).astype("int64")

    # 为了 padding_idx=0，把所有 item 索引整体 +1
    full_df["item_idx"] = full_df["item_idx_original"] + 1
    num_items_for_din = n_items + 1  # 0 留给 padding

    # 确保 timestamp 存在
    if "timestamp" not in full_df.columns:
        raise ValueError("ratings DataFrame must contain 'timestamp' column for sequential split.")

    # 3. 构造 DIN 序列样本
    max_hist_len = 50  # 之后愿意可以放到 config 里

    print("Building DIN training / validation / test sequences...")
    samples_train = build_din_samples(full_df, target_split="train", min_hist_len=1)
    samples_val = build_din_samples(full_df, target_split="val", min_hist_len=1)
    samples_test = build_din_samples(full_df, target_split="test", min_hist_len=1)

    print(
        f"DIN samples: train={len(samples_train)}, "
        f"val={len(samples_val)}, test={len(samples_test)}"
    )

    train_dataset = DINDataset(samples_train, max_hist_len=max_hist_len)
    val_dataset = DINDataset(samples_val, max_hist_len=max_hist_len)
    test_dataset = DINDataset(samples_test, max_hist_len=max_hist_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 4. 构建 DIN 模型
    print("Building DIN model...")
    embed_dim = cfg["model"].get("mf_dim", 32)  # 复用 NeuMF 的 mf_dim
    mlp_layers = cfg["model"].get("mlp_layers", [128, 64, 32])

    model = DIN(
        num_users=n_users,
        num_items=num_items_for_din,
        embed_dim=embed_dim,
        att_hidden_sizes=(64, 32, 16),
        fc_hidden_sizes=tuple(mlp_layers),
        max_history_len=max_hist_len,
        dropout=cfg["model"].get("dropout", 0.0),
        use_dice=True,
    ).to(device)

    criterion = nn.MSELoss()

    # 处理 weight_decay 可能写成字符串的情况
    weight_decay_raw = cfg["train"].get("weight_decay", 0.0)
    try:
        weight_decay = float(weight_decay_raw)
    except (TypeError, ValueError):
        print(f"[WARN] Invalid weight_decay={weight_decay_raw}, fallback to 0.0")
        weight_decay = 0.0

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 5. 训练循环
    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_rmse, val_mae = evaluate_rmse_mae(model, val_loader, device)
        dt = time.time() - t0

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f} ({dt:.1f}s)"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()

    # 6. 用最佳模型在 test 上评估
    if best_state is not None:
        model.load_state_dict(best_state)

    test_rmse, test_mae = evaluate_rmse_mae(model, test_loader, device)
    print(f"[Test] RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

    # 7. 保存模型
    save_path = os.path.join(PROJECT_ROOT, "din_ml1m_best.pth")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_users": n_users,
            "n_items": num_items_for_din,  # 注意这里是 +1 后的数量（含 padding）
            "config": cfg,
            "max_hist_len": max_hist_len,
        },
        save_path,
    )
    print(f"Saved best DIN model to {save_path}")


if __name__ == "__main__":
    main()
