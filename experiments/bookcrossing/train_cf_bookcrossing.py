import os
import time
import yaml
import sys

import torch
import torch.nn as nn
from torch.optim import Adam

# ================= 路径设置：适用于 experiments/bookcrossing/ 下 =================
# 本文件所在目录: .../recommender_project/experiments/bookcrossing
EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录: .../recommender_project
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.append(p)
# =====================================================================

from models.cf_model import NeuMF
from preprocess_bookcrossing import (
    build_datasets_and_loaders_bookcrossing,
)


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for user_idx, item_idx, rating in loader:
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        rating = rating.to(device)

        optimizer.zero_grad()
        pred = model(user_idx, item_idx)
        loss = criterion(pred, rating)
        loss.backward()
        optimizer.step()

        batch_size = rating.size(0)
        total_loss += loss.item() * batch_size
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
    # 配置文件在项目根目录的 utils 下
    config_path = os.path.join(PROJECT_ROOT, "utils", "config_bookcrossing.yaml")
    cfg = load_config(config_path)

    # 数据目录同样从 PROJECT_ROOT 开始
    data_dir = os.path.join(PROJECT_ROOT, cfg["data"]["bookcrossing_dir"])
    batch_size = cfg["train"]["batch_size"]
    lr = cfg["train"]["lr"]
    epochs = cfg["train"]["epochs"]
    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("Loading Book-Crossing data...")
    train_loader, val_loader, test_loader, meta = build_datasets_and_loaders_bookcrossing(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        min_user_ratings=cfg["data"].get("min_user_ratings", 5),
        min_item_ratings=cfg["data"].get("min_item_ratings", 5),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
        test_ratio=cfg["data"].get("test_ratio", 0.1),
    )
    n_users = meta["n_users"]
    n_items = meta["n_items"]
    print(f"Book-Crossing: n_users={n_users}, n_items={n_items}")

    print("Building NeuMF model...")
    model = NeuMF(
        num_users=n_users,
        num_items=n_items,
        mf_dim=cfg["model"]["mf_dim"],
        mlp_layer_sizes=tuple(cfg["model"]["mlp_layers"]),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)

    criterion = nn.MSELoss()

    # 安全处理 weight_decay，避免 YAML 里写成字符串导致报错
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

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_rmse, val_mae = evaluate_rmse_mae(model, val_loader, device)
        dt = time.time() - t0

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f} ({dt:.1f}s)"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_rmse, test_mae = evaluate_rmse_mae(model, test_loader, device)
    print(f"[Test] RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

    save_path = os.path.join(PROJECT_ROOT, "neuMF_bookcrossing_best.pth")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_users": n_users,
            "n_items": n_items,
            "config": cfg,
        },
        save_path,
    )
    print(f"Saved best NeuMF model to {save_path}")


if __name__ == "__main__":
    main()
