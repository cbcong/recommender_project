import os
import time
import yaml
import sys

import torch
import torch.nn as nn
from torch.optim import Adam

# ================= 路径设置（适用于 experiments/bookcrossing/ 下） =================
# 本文件所在目录: .../recommender_project/experiments/bookcrossing
EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录: .../recommender_project
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.append(p)
# ==========================================================================

from models.cf_model import NeuMF
from models.cf_user_model import NCFUserFeat
from preprocess_bookcrossing import (
    build_datasets_and_loaders_bookcrossing_coldstart,
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


def train_and_select_best(model, train_loader, val_loader, device, cfg, tag: str):
    criterion = nn.MSELoss()

    weight_decay_raw = cfg["train"].get("weight_decay", 0.0)
    try:
        weight_decay = float(weight_decay_raw)
    except (TypeError, ValueError):
        print(f"[WARN] Invalid weight_decay={weight_decay_raw}, fallback to 0.0")
        weight_decay = 0.0

    optimizer = Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=weight_decay,
    )

    epochs = cfg["train"]["epochs"]
    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_rmse, val_mae = evaluate_rmse_mae(model, val_loader, device)
        dt = time.time() - t0

        print(
            f"[{tag}][Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f} ({dt:.1f}s)"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_rmse


def main():
    # 注意：配置文件在项目根目录的 utils 下
    config_path = os.path.join(PROJECT_ROOT, "utils", "config_bookcrossing.yaml")
    cfg = load_config(config_path)

    # 数据目录同样从 PROJECT_ROOT 开始
    data_dir = os.path.join(PROJECT_ROOT, cfg["data"]["bookcrossing_dir"])
    batch_size = cfg["train"]["batch_size"]
    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("Loading Book-Crossing data with COLD-START split ...")
    (
        train_loader,
        val_loader,
        test_all_loader,
        test_cold_loader,
        meta,
    ) = build_datasets_and_loaders_bookcrossing_coldstart(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        min_user_ratings=cfg["data"].get("min_user_ratings", 5),
        min_item_ratings=cfg["data"].get("min_item_ratings", 5),
        cold_user_ratio=cfg["data"].get("cold_user_ratio", 0.1),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
        test_ratio=cfg["data"].get("test_ratio", 0.1),
    )

    n_users = meta["n_users"]
    n_items = meta["n_items"]
    user_features_full = meta["user_features"]  # [n_users, feat_dim]
    print(
        f"Book-Crossing (cold-start): n_users={n_users}, n_items={n_items}, "
        f"user_feat_dim={user_features_full.shape[1]}, "
        f"n_cold_users={len(meta['cold_users'])}"
    )

    # -------- 1) NeuMF（无用户特征）---------
    print("\n========== Train NeuMF (ID only) under cold-start ==========")
    model_neumf = NeuMF(
        num_users=n_users,
        num_items=n_items,
        mf_dim=cfg["model"]["mf_dim"],
        mlp_layer_sizes=tuple(cfg["model"]["mlp_layers"]),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)

    model_neumf, best_val_rmse_neumf = train_and_select_best(
        model_neumf, train_loader, val_loader, device, cfg, tag="NeuMF"
    )

    test_all_rmse, test_all_mae = evaluate_rmse_mae(
        model_neumf, test_all_loader, device
    )
    test_cold_rmse, test_cold_mae = evaluate_rmse_mae(
        model_neumf, test_cold_loader, device
    )

    print(f"[NeuMF][Val-Best] RMSE={best_val_rmse_neumf:.4f}")
    print(f"[NeuMF][Test-ALL]  RMSE={test_all_rmse:.4f}, MAE={test_all_mae:.4f}")
    print(f"[NeuMF][Test-COLD] RMSE={test_cold_rmse:.4f}, MAE={test_cold_mae:.4f}")

    torch.save(
        {
            "state_dict": model_neumf.state_dict(),
            "n_users": n_users,
            "n_items": n_items,
            "config": cfg,
        },
        os.path.join(PROJECT_ROOT, "neumf_bookcrossing_coldstart_best.pth"),
    )

    # -------- 2) NCFUserFeat (Age only) ---------
    print("\n========== Train NCFUserFeat (AGE ONLY) under cold-start ==========")
    # Age 在 user_features_full 的第 0 列
    user_features_age = user_features_full[:, :1]
    user_features_age_t = torch.from_numpy(user_features_age).float().to(device)

    model_age = NCFUserFeat(
        num_users=n_users,
        num_items=n_items,
        user_feat_dim=user_features_age.shape[1],  # =1
        emb_dim=cfg["model"].get("emb_dim", 32),
        mlp_layer_sizes=tuple(cfg["model"]["mlp_layers"]),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    model_age.set_user_features(user_features_age_t)

    model_age, best_val_rmse_age = train_and_select_best(
        model_age, train_loader, val_loader, device, cfg, tag="NCFUserFeat-AGE"
    )

    age_all_rmse, age_all_mae = evaluate_rmse_mae(
        model_age, test_all_loader, device
    )
    age_cold_rmse, age_cold_mae = evaluate_rmse_mae(
        model_age, test_cold_loader, device
    )

    print(f"[NCFUserFeat-AGE][Val-Best] RMSE={best_val_rmse_age:.4f}")
    print(
        f"[NCFUserFeat-AGE][Test-ALL]  RMSE={age_all_rmse:.4f}, MAE={age_all_mae:.4f}"
    )
    print(
        f"[NCFUserFeat-AGE][Test-COLD] RMSE={age_cold_rmse:.4f}, MAE={age_cold_mae:.4f}"
    )

    torch.save(
        {
            "state_dict": model_age.state_dict(),
            "n_users": n_users,
            "n_items": n_items,
            "config": cfg,
        },
        os.path.join(PROJECT_ROOT, "ncf_user_age_bookcrossing_coldstart_best.pth"),
    )

    # -------- 3) NCFUserFeat (Age + Country) ---------
    print("\n========== Train NCFUserFeat (AGE + COUNTRY) under cold-start ==========")
    user_features_full_t = torch.from_numpy(user_features_full).float().to(device)

    model_full = NCFUserFeat(
        num_users=n_users,
        num_items=n_items,
        user_feat_dim=user_features_full.shape[1],  # 例如 22
        emb_dim=cfg["model"].get("emb_dim", 32),
        mlp_layer_sizes=tuple(cfg["model"]["mlp_layers"]),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    model_full.set_user_features(user_features_full_t)

    model_full, best_val_rmse_full = train_and_select_best(
        model_full,
        train_loader,
        val_loader,
        device,
        cfg,
        tag="NCFUserFeat-AGE+COUNTRY",
    )

    full_all_rmse, full_all_mae = evaluate_rmse_mae(
        model_full, test_all_loader, device
    )
    full_cold_rmse, full_cold_mae = evaluate_rmse_mae(
        model_full, test_cold_loader, device
    )

    print(f"[NCFUserFeat-AGE+COUNTRY][Val-Best] RMSE={best_val_rmse_full:.4f}")
    print(
        f"[NCFUserFeat-AGE+COUNTRY][Test-ALL]  RMSE={full_all_rmse:.4f}, MAE={full_all_mae:.4f}"
    )
    print(
        f"[NCFUserFeat-AGE+COUNTRY][Test-COLD] RMSE={full_cold_rmse:.4f}, MAE={full_cold_mae:.4f}"
    )

    torch.save(
        {
            "state_dict": model_full.state_dict(),
            "n_users": n_users,
            "n_items": n_items,
            "config": cfg,
        },
        os.path.join(
            PROJECT_ROOT, "ncf_user_age_country_bookcrossing_coldstart_best.pth"
        ),
    )

    print("\n========== Cold-start experiment on Book-Crossing finished. ==========")


if __name__ == "__main__":
    main()
