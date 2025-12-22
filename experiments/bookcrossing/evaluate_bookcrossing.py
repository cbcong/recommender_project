import os
import sys
import time
import yaml
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ================= 路径与 import 设置 =================
# 当前文件：.../recommender_project/experiments/bookcrossing/evaluate_bookcrossing.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.append(p)

# models
try:
    from models.cf_model import NeuMF
except ImportError:
    from cf_model import NeuMF

try:
    from models.cf_user_model import NCFUserFeat
except ImportError:
    from cf_user_model import NCFUserFeat

# utils / preprocess_bookcrossing
try:
    from utils.preprocess_bookcrossing import (
        prepare_bookcrossing_interactions,
        train_val_test_split,
        df_to_dataset,
    )
except ImportError:
    from preprocess_bookcrossing import (
        prepare_bookcrossing_interactions,
        train_val_test_split,
        df_to_dataset,
    )
# =====================================================


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_loaders_for_eval(df_train, df_val, df_test, batch_size: int = 1024):
    """把 df 转成 DataLoader，方便算 RMSE/MAE。"""
    train_ds = df_to_dataset(df_train)
    val_ds = df_to_dataset(df_val)
    test_ds = df_to_dataset(df_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


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


def build_user_item_sets(train_df, val_df, test_df, n_users: int):
    """
    构建：
      - train_items_by_user[u] : 训练+验证集中该用户出现过的 item 集合（用于排除）
      - test_items_by_user[u]  : 测试集中该用户的正样本 item 集合
    """
    train_items_by_user = defaultdict(set)
    test_items_by_user = defaultdict(set)

    # 训练 + 验证都视为“已看过”，在 Top-K 排序时需要排除
    for df in [train_df, val_df]:
        for u, i in zip(df["user_idx"].values, df["item_idx"].values):
            train_items_by_user[int(u)].add(int(i))

    for u, i in zip(test_df["user_idx"].values, test_df["item_idx"].values):
        test_items_by_user[int(u)].add(int(i))

    return train_items_by_user, test_items_by_user


def evaluate_topk(
    model,
    train_df,
    val_df,
    test_df,
    n_users: int,
    n_items: int,
    device,
    k: int = 10,
):
    """
    对每个在 test 中出现的用户：
      - 对所有 item 打分（排除 train+val 中出现过的 item）
      - 取 Top-K，计算 Recall@K / NDCG@K / HitRate@K
    """
    model.eval()

    train_items_by_user, test_items_by_user = build_user_item_sets(
        train_df, val_df, test_df, n_users
    )

    all_items = torch.arange(n_items, dtype=torch.long, device=device)

    hit_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0
    user_cnt = 0

    # 预先计算一个 log2 序列，加速 NDCG 计算
    log2_table = 1.0 / np.log2(np.arange(k) + 2.0)  # 长度 k，对应 rank 0..k-1

    with torch.no_grad():
        for u in range(n_users):
            test_items = test_items_by_user.get(u, None)
            if not test_items:
                continue  # 该用户在 test 没有正样本，跳过

            test_items = set(test_items)
            user_cnt += 1

            # user u 对所有 item 的打分
            user_idx_vec = torch.full(
                (n_items,), u, dtype=torch.long, device=device
            )
            scores = model(user_idx_vec, all_items).detach().cpu().numpy()

            # 排除训练+验证中已经交互过的 items（不能再推荐）
            seen_items = train_items_by_user.get(u, set())
            for i in seen_items:
                if 0 <= i < n_items:
                    scores[i] = -1e9

            # Top-K 索引（先 argpartition，再按分数降序排序）
            topk_indices = np.argpartition(scores, -k)[-k:]
            topk_indices = topk_indices[np.argsort(scores[topk_indices])[::-1]]

            topk_set = set(int(i) for i in topk_indices)

            # HitRate@K
            hit = 1.0 if len(topk_set & test_items) > 0 else 0.0
            hit_sum += hit

            # Recall@K
            recall = len(topk_set & test_items) / float(len(test_items))
            recall_sum += recall

            # NDCG@K
            dcg = 0.0
            for rank, item in enumerate(topk_indices):
                if int(item) in test_items:
                    dcg += log2_table[rank]  # 1/log2(rank+2)
            # 理论最优 DCG（IDCG）：正样本数与 K 中取较小值
            m = min(len(test_items), k)
            if m > 0:
                idcg = np.sum(log2_table[:m])
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            ndcg_sum += ndcg

    if user_cnt == 0:
        return 0.0, 0.0, 0.0

    hit_rate = hit_sum / user_cnt
    recall = recall_sum / user_cnt
    ndcg = ndcg_sum / user_cnt

    return hit_rate, recall, ndcg


def load_neumf_for_eval(ckpt_path: str, n_users: int, n_items: int, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    n_users_ckpt = ckpt["n_users"]
    n_items_ckpt = ckpt["n_items"]
    if n_users_ckpt != n_users or n_items_ckpt != n_items:
        print(
            f"[WARN] NeuMF ckpt n_users/n_items ({n_users_ckpt},{n_items_ckpt}) "
            f"!= current meta ({n_users},{n_items}), 仍尝试加载..."
        )

    mlp_layers = cfg["model"]["mlp_layers"]  # 例如 [64, 32]
    model = NeuMF(
        num_users=n_users,
        num_items=n_items,
        mf_dim=cfg["model"]["mf_dim"],
        mlp_layer_sizes=tuple(mlp_layers),    # ✅ 正确写法
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model



def load_ncf_userfeat_for_eval(
    ckpt_path: str,
    user_features_np: np.ndarray,
    n_users: int,
    n_items: int,
    device,
):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    n_users_ckpt = ckpt["n_users"]
    n_items_ckpt = ckpt["n_items"]
    if n_users_ckpt != n_users or n_items_ckpt != n_items:
        print(
            f"[WARN] NCFUserFeat ckpt n_users/n_items ({n_users_ckpt},{n_items_ckpt}) "
            f"!= current meta ({n_users},{n_items}), 仍尝试加载..."
        )

    user_features_t = torch.from_numpy(user_features_np).float().to(device)

    model = NCFUserFeat(
        num_users=n_users,
        num_items=n_items,
        user_feat_dim=user_features_np.shape[1],
        emb_dim=cfg["model"].get("emb_dim", 32),
        mlp_layer_sizes=tuple(cfg["model"]["mlp_layers"]),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    model.set_user_features(user_features_t)
    model.load_state_dict(ckpt["state_dict"])
    return model


def main():
    # 1. 读取配置和数据
    config_path = os.path.join(PROJECT_ROOT, "utils", "config_bookcrossing.yaml")
    cfg = load_config(config_path)

    data_dir_cfg = cfg["data"]["bookcrossing_dir"]
    # 相对路径 -> 绝对路径
    if not os.path.isabs(data_dir_cfg):
        data_dir = os.path.join(PROJECT_ROOT, data_dir_cfg)
    else:
        data_dir = data_dir_cfg

    batch_size = cfg["train"]["batch_size"]
    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    print("Preparing Book-Crossing interactions (for Top-K eval)...")
    interactions, meta = prepare_bookcrossing_interactions(
        data_dir=data_dir,
        min_user_ratings=cfg["data"].get("min_user_ratings", 5),
        min_item_ratings=cfg["data"].get("min_item_ratings", 5),
    )

    n_users = meta["n_users"]
    n_items = meta["n_items"]
    user_features_full = meta["user_features"]  # [n_users, feat_dim]

    print(
        f"Book-Crossing: n_users={n_users}, n_items={n_items}, "
        f"user_feat_dim={user_features_full.shape[1]}"
    )

    # 划分 train/val/test（随机划分，种子固定）
    train_df, val_df, test_df = train_val_test_split(
        interactions,
        val_ratio=cfg["data"].get("val_ratio", 0.1),
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        seed=42,
    )

    print(
        f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    train_loader, val_loader, test_loader = build_loaders_for_eval(
        train_df, val_df, test_df, batch_size=batch_size
    )

    K = 10

    # ========= 2. 评估 NeuMF =========
    neu_ckpt_path = os.path.join(PROJECT_ROOT, "neuMF_bookcrossing_best.pth")
    if os.path.exists(neu_ckpt_path):
        print("\n========== Evaluating NeuMF (ID only) ==========")
        model_neu = load_neumf_for_eval(neu_ckpt_path, n_users, n_items, device)

        rmse, mae = evaluate_rmse_mae(model_neu, test_loader, device)
        print(f"NeuMF - RMSE={rmse:.4f}, MAE={mae:.4f}")

        hit, rec, ndcg = evaluate_topk(
            model_neu,
            train_df,
            val_df,
            test_df,
            n_users,
            n_items,
            device,
            k=K,
        )
        print(
            f"NeuMF - HitRate@{K}={hit:.4f}, Recall@{K}={rec:.4f}, NDCG@{K}={ndcg:.4f}"
        )
    else:
        print(f"\n[WARN] NeuMF ckpt not found at {neu_ckpt_path}, skip NeuMF eval.")

    # ========= 3. 评估 NCFUserFeat (AGE ONLY) =========
    age_ckpt_path = os.path.join(
        PROJECT_ROOT, "ncf_user_age_only_bookcrossing_best.pth"
    )
    if os.path.exists(age_ckpt_path):
        print("\n========== Evaluating NCFUserFeat (AGE ONLY) ==========")
        # Age 在 full features 的第 0 列
        user_features_age = user_features_full[:, :1]
        model_age = load_ncf_userfeat_for_eval(
            age_ckpt_path,
            user_features_age,
            n_users,
            n_items,
            device,
        )

        rmse, mae = evaluate_rmse_mae(model_age, test_loader, device)
        print(f"NCFUserFeat-AGE - RMSE={rmse:.4f}, MAE={mae:.4f}")

        hit, rec, ndcg = evaluate_topk(
            model_age,
            train_df,
            val_df,
            test_df,
            n_users,
            n_items,
            device,
            k=K,
        )
        print(
            f"NCFUserFeat-AGE - HitRate@{K}={hit:.4f}, "
            f"Recall@{K}={rec:.4f}, NDCG@{K}={ndcg:.4f}"
        )
    else:
        print(
            f"\n[INFO] AGE-only ckpt not found at {age_ckpt_path}, "
            f"如果需要该结果，请先运行 Age-only 训练脚本。"
        )

    # ========= 4. 评估 NCFUserFeat (AGE + COUNTRY) =========
    full_ckpt_path = os.path.join(PROJECT_ROOT, "ncf_userfeat_bookcrossing_best.pth")
    if os.path.exists(full_ckpt_path):
        print("\n========== Evaluating NCFUserFeat (AGE + COUNTRY) ==========")
        model_full = load_ncf_userfeat_for_eval(
            full_ckpt_path,
            user_features_full,
            n_users,
            n_items,
            device,
        )

        rmse, mae = evaluate_rmse_mae(model_full, test_loader, device)
        print(f"NCFUserFeat-AGE+COUNTRY - RMSE={rmse:.4f}, MAE={mae:.4f}")

        hit, rec, ndcg = evaluate_topk(
            model_full,
            train_df,
            val_df,
            test_df,
            n_users,
            n_items,
            device,
            k=K,
        )
        print(
            f"NCFUserFeat-AGE+COUNTRY - HitRate@{K}={hit:.4f}, "
            f"Recall@{K}={rec:.4f}, NDCG@{K}={ndcg:.4f}"
        )
    else:
        print(
            f"\n[WARN] NCFUserFeat (AGE+COUNTRY) ckpt not found at {full_ckpt_path}, "
            f"请确认已跑过 train_cf_bookcrossing_userfeat.py。"
        )


if __name__ == "__main__":
    main()
