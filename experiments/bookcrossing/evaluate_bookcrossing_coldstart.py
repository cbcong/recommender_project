import os
import sys
import time
import yaml
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ===================== 路径设置 =====================

# 本文件所在目录： .../recommender_project/experiments/bookcrossing
EXP_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录：.../recommender_project  （往上两级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))

UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ====================================================

from models.cf_model import NeuMF
from models.cf_user_model import NCFUserFeat
from preprocess_bookcrossing import (
    prepare_bookcrossing_interactions,
    train_val_test_split_coldstart,
    df_to_dataset,
)



def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_loader_from_df(df, batch_size: int = 1024):
    ds = df_to_dataset(df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


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


def build_user_item_sets(train_df, val_df, test_df):
    """
    构建：
      - train_items_by_user[u] : 训练+验证集中该用户出现过的 item 集合（用于排除）
      - test_items_by_user[u]  : 测试集中该用户的正样本 item 集合
    """
    from collections import defaultdict

    train_items_by_user = defaultdict(set)
    test_items_by_user = defaultdict(set)

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
    target_users=None,
):
    """
    Top-K 评估：
      - train_df + val_df 用来构建“已交互集合”（推荐时要排除）
      - test_df 中的正样本作为评估目标
      - target_users:
          None  -> 对 test 中所有出现的用户评估 (Test-ALL)
          set   -> 只对指定用户子集评估 (例如冷启动用户集合)
    """
    model.eval()

    train_items_by_user, test_items_by_user = build_user_item_sets(
        train_df, val_df, test_df
    )

    if target_users is not None:
        target_users = set(int(u) for u in target_users)

    all_items = torch.arange(n_items, dtype=torch.long, device=device)

    hit_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0
    user_cnt = 0

    log2_table = 1.0 / np.log2(np.arange(k) + 2.0)  # rank 0..k-1 -> 1/log2(rank+2)

    with torch.no_grad():
        for u, items in test_items_by_user.items():
            if not items:
                continue

            if target_users is not None and u not in target_users:
                continue

            user_cnt += 1
            test_items = set(items)

            # 对所有 item 打分
            user_idx_vec = torch.full(
                (n_items,), u, dtype=torch.long, device=device
            )
            scores = model(user_idx_vec, all_items).detach().cpu().numpy()

            # 排除 train+val 里看过的 item
            seen_items = train_items_by_user.get(u, set())
            for i in seen_items:
                if 0 <= i < n_items:
                    scores[i] = -1e9

            # 取 Top-K
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
                    dcg += log2_table[rank]
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


def load_neumf_ckpt(ckpt_path: str, n_users: int, n_items: int, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    n_users_ckpt = ckpt["n_users"]
    n_items_ckpt = ckpt["n_items"]
    if n_users_ckpt != n_users or n_items_ckpt != n_items:
        print(
            f"[WARN] NeuMF ckpt n_users/n_items ({n_users_ckpt},{n_items_ckpt}) "
            f"!= current meta ({n_users},{n_items}), 仍尝试加载..."
        )

    model = NeuMF(
        num_users=n_users,
        num_items=n_items,
        mf_dim=cfg["model"]["mf_dim"],
        mlp_layer_sizes=tuple(cfg["model"]["mlp_layers"]),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model


def load_ncf_userfeat_ckpt(
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
    # 1. 读配置 & 数据
    config_path = os.path.join(PROJECT_ROOT, "utils", "config_bookcrossing.yaml")
    cfg = load_config(config_path)

    data_dir = os.path.join(PROJECT_ROOT, cfg["data"]["bookcrossing_dir"])
    batch_size = cfg["train"]["batch_size"]
    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    print("Preparing Book-Crossing interactions (COLD-START split)...")
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

    # 冷启动划分：与 train_bookcrossing_coldstart_all.py 保持一致 (seed=42)
    (
        train_df,
        val_df,
        test_all_df,
        test_cold_df,
        cold_users,
    ) = train_val_test_split_coldstart(
        interactions,
        cold_user_ratio=cfg["data"].get("cold_user_ratio", 0.1),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        seed=42,
    )

    print(
        f"Split sizes (cold-start): "
        f"train={len(train_df)}, val={len(val_df)}, "
        f"test_all={len(test_all_df)}, test_cold={len(test_cold_df)}, "
        f"n_cold_users={len(cold_users)}"
    )

    test_all_loader = build_loader_from_df(test_all_df, batch_size=batch_size)
    test_cold_loader = build_loader_from_df(test_cold_df, batch_size=batch_size)

    K = 10

    # ========= 2. NeuMF (ID only) =========
    neu_ckpt_path = os.path.join(
        PROJECT_ROOT, "neumf_bookcrossing_coldstart_best.pth"
    )
    if os.path.exists(neu_ckpt_path):
        print("\n========== Evaluating NeuMF (ID only) under COLD-START ==========")
        model_neu = load_neumf_ckpt(neu_ckpt_path, n_users, n_items, device)

        # RMSE/MAE
        rmse_all, mae_all = evaluate_rmse_mae(model_neu, test_all_loader, device)
        rmse_cold, mae_cold = evaluate_rmse_mae(model_neu, test_cold_loader, device)

        print(f"NeuMF - Test-ALL  RMSE={rmse_all:.4f}, MAE={mae_all:.4f}")
        print(f"NeuMF - Test-COLD RMSE={rmse_cold:.4f}, MAE={mae_cold:.4f}")

        # Top-K (ALL)
        hit_all, rec_all, ndcg_all = evaluate_topk(
            model_neu,
            train_df,
            val_df,
            test_all_df,
            n_users,
            n_items,
            device,
            k=K,
            target_users=None,
        )
        print(
            f"NeuMF - Test-ALL  HitRate@{K}={hit_all:.4f}, "
            f"Recall@{K}={rec_all:.4f}, NDCG@{K}={ndcg_all:.4f}"
        )

        # Top-K (COLD 用户子集)
        hit_cold, rec_cold, ndcg_cold = evaluate_topk(
            model_neu,
            train_df,
            val_df,
            test_all_df,
            n_users,
            n_items,
            device,
            k=K,
            target_users=cold_users,
        )
        print(
            f"NeuMF - Test-COLD HitRate@{K}={hit_cold:.4f}, "
            f"Recall@{K}={rec_cold:.4f}, NDCG@{K}={ndcg_cold:.4f}"
        )
    else:
        print(f"\n[WARN] NeuMF ckpt not found at {neu_ckpt_path}, skip NeuMF eval.")

    # ========= 3. NCFUserFeat (AGE ONLY) =========
    age_ckpt_path = os.path.join(
        PROJECT_ROOT, "ncf_user_age_bookcrossing_coldstart_best.pth"
    )
    if os.path.exists(age_ckpt_path):
        print(
            "\n========== Evaluating NCFUserFeat (AGE ONLY) under COLD-START =========="
        )
        user_features_age = user_features_full[:, :1]
        model_age = load_ncf_userfeat_ckpt(
            age_ckpt_path,
            user_features_age,
            n_users,
            n_items,
            device,
        )

        rmse_all, mae_all = evaluate_rmse_mae(model_age, test_all_loader, device)
        rmse_cold, mae_cold = evaluate_rmse_mae(model_age, test_cold_loader, device)

        print(f"NCFUserFeat-AGE - Test-ALL  RMSE={rmse_all:.4f}, MAE={mae_all:.4f}")
        print(f"NCFUserFeat-AGE - Test-COLD RMSE={rmse_cold:.4f}, MAE={mae_cold:.4f}")

        hit_all, rec_all, ndcg_all = evaluate_topk(
            model_age,
            train_df,
            val_df,
            test_all_df,
            n_users,
            n_items,
            device,
            k=K,
            target_users=None,
        )
        print(
            f"NCFUserFeat-AGE - Test-ALL  HitRate@{K}={hit_all:.4f}, "
            f"Recall@{K}={rec_all:.4f}, NDCG@{K}={ndcg_all:.4f}"
        )

        hit_cold, rec_cold, ndcg_cold = evaluate_topk(
            model_age,
            train_df,
            val_df,
            test_all_df,
            n_users,
            n_items,
            device,
            k=K,
            target_users=cold_users,
        )
        print(
            f"NCFUserFeat-AGE - Test-COLD HitRate@{K}={hit_cold:.4f}, "
            f"Recall@{K}={rec_cold:.4f}, NDCG@{K}={ndcg_cold:.4f}"
        )
    else:
        print(
            f"\n[INFO] AGE-only cold-start ckpt not found at {age_ckpt_path}, "
            f"如需该结果，请先运行 train_bookcrossing_coldstart_all.py。"
        )

    # ========= 4. NCFUserFeat (AGE + COUNTRY) =========
    full_ckpt_path = os.path.join(
        PROJECT_ROOT, "ncf_user_age_country_bookcrossing_coldstart_best.pth"
    )
    if os.path.exists(full_ckpt_path):
        print(
            "\n========== Evaluating NCFUserFeat (AGE+COUNTRY) under COLD-START =========="
        )
        model_full = load_ncf_userfeat_ckpt(
            full_ckpt_path,
            user_features_full,
            n_users,
            n_items,
            device,
        )

        rmse_all, mae_all = evaluate_rmse_mae(model_full, test_all_loader, device)
        rmse_cold, mae_cold = evaluate_rmse_mae(model_full, test_cold_loader, device)

        print(
            f"NCFUserFeat-AGE+COUNTRY - Test-ALL  RMSE={rmse_all:.4f}, MAE={mae_all:.4f}"
        )
        print(
            f"NCFUserFeat-AGE+COUNTRY - Test-COLD RMSE={rmse_cold:.4f}, MAE={mae_cold:.4f}"
        )

        hit_all, rec_all, ndcg_all = evaluate_topk(
            model_full,
            train_df,
            val_df,
            test_all_df,
            n_users,
            n_items,
            device,
            k=K,
            target_users=None,
        )
        print(
            f"NCFUserFeat-AGE+COUNTRY - Test-ALL  HitRate@{K}={hit_all:.4f}, "
            f"Recall@{K}={rec_all:.4f}, NDCG@{K}={ndcg_all:.4f}"
        )

        hit_cold, rec_cold, ndcg_cold = evaluate_topk(
            model_full,
            train_df,
            val_df,
            test_all_df,
            n_users,
            n_items,
            device,
            k=K,
            target_users=cold_users,
        )
        print(
            f"NCFUserFeat-AGE+COUNTRY - Test-COLD HitRate@{K}={hit_cold:.4f}, "
            f"Recall@{K}={rec_cold:.4f}, NDCG@{K}={ndcg_cold:.4f}"
        )
    else:
        print(
            f"\n[WARN] NCFUserFeat (AGE+COUNTRY) cold-start ckpt not found at "
            f"{full_ckpt_path}, 请确认 train_bookcrossing_coldstart_all.py 已成功运行。"
        )

    print("\n========== COLD-START Top-K evaluation on Book-Crossing finished. ==========")


if __name__ == "__main__":
    main()
