import os
import sys
import math
import yaml
import numpy as np

import torch

# ========= 路径设置：保证从 experiments/ml1m 下也能找到根目录 / utils / models =========

import os
import sys

# ====== Path setup：把根目录设成 recommender_project ======
# 当前文件路径：.../recommender_project/experiments/ml1m/xxx.py
# 先取当前文件所在目录，再往上两级 -> 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
# ==========================================================


# ========= 项目内模块导入（注意不带 utils./models. 前缀） =========

from preprocess import (
    load_ml1m_ratings,
    build_id_mappings,
    split_by_user_time,
)
from cf_model import NeuMF
from gnn_model import LightGCN
from hybrid_model import HybridNCF
from content_model import ItemContentEncoder


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_user_pos_dict(user_idx, item_idx, num_users):
    """
    根据 (u, i) 交互构建每个用户的正样本集合 dict: u -> set(items)
    """
    user_pos = {u: set() for u in range(num_users)}
    for u, i in zip(user_idx, item_idx):
        user_pos[int(u)].add(int(i))
    return user_pos


def build_normalized_adj(num_users, num_items, user_idx, item_idx, device):
    """
    构建 LightGCN 使用的归一化邻接矩阵 A_hat (稀疏张量)，
    逻辑与 train_gnn.py 中保持一致。
    """
    num_nodes = num_users + num_items

    user_nodes = np.array(user_idx, dtype=np.int64)
    item_nodes = np.array(item_idx, dtype=np.int64) + num_users

    rows = np.concatenate([user_nodes, item_nodes], axis=0)
    cols = np.concatenate([item_nodes, user_nodes], axis=0)

    values = np.ones(len(rows), dtype=np.float32)

    degree = np.bincount(rows, minlength=num_nodes).astype(np.float32)
    degree[degree == 0.0] = 1.0

    norm_values = values / np.sqrt(degree[rows] * degree[cols])

    indices = np.vstack((rows, cols))
    indices_tensor = torch.from_numpy(indices).long()
    values_tensor = torch.from_numpy(norm_values).float()

    adj = torch.sparse_coo_tensor(
        indices_tensor,
        values_tensor,
        torch.Size([num_nodes, num_nodes]),
    )
    adj = adj.coalesce().to(device)
    return adj


def ndcg_at_k(rank_list, ground_truth_set, k):
    """
    计算单个用户的 NDCG@K（二元相关性：命中为1，否则0）
    rank_list: [item_id0, item_id1, ...] 排序后的物品列表
    ground_truth_set: 该用户的真实正样本集合（set）
    """
    dcg = 0.0
    for rank, item in enumerate(rank_list[:k], start=1):
        if item in ground_truth_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(ground_truth_set), k)
    if ideal_hits == 0:
        return 0.0

    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(rank + 1)

    return dcg / idcg if idcg > 0 else 0.0


# ----------------- NeuMF 相关评估 -----------------


def load_neumf_model(model_path, device):
    """
    从 neuMF_ml1m_best.pth 加载 NeuMF 模型和配置。
    """
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    n_users = ckpt["n_users"]
    n_items = ckpt["n_items"]

    mf_dim = cfg["model"]["mf_dim"]
    mlp_layers = cfg["model"]["mlp_layers"]
    dropout = cfg["model"].get("dropout", 0.0)

    model = NeuMF(
        num_users=n_users,
        num_items=n_items,
        mf_dim=mf_dim,
        mlp_layer_sizes=tuple(mlp_layers),
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, n_users, n_items


def evaluate_neumf_rmse_mae(model, test_u, test_i, test_r, device):
    """
    使用 NeuMF 对测试集评分做 RMSE / MAE 评估。
    """
    model.eval()
    mse_loss = 0.0
    mae_loss = 0.0
    n = 0

    batch_size = 4096
    num_samples = len(test_r)
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            u = torch.LongTensor(test_u[start:end]).to(device)
            i = torch.LongTensor(test_i[start:end]).to(device)
            r = torch.FloatTensor(test_r[start:end]).to(device)

            pred = model(u, i)
            mse_loss += torch.sum((pred - r) ** 2).item()
            mae_loss += torch.sum(torch.abs(pred - r)).item()
            n += r.size(0)

    rmse = (mse_loss / n) ** 0.5
    mae = mae_loss / n
    return rmse, mae


def evaluate_neumf_topk(model, train_user_pos, test_user_pos, n_items, device, k=10):
    """
    使用 NeuMF 打分做 Top-K 推荐评估，计算 Recall@K 和 NDCG@K。
    """
    model.eval()
    test_users = list(test_user_pos.keys())

    hits = 0
    total = 0
    ndcg_sum = 0.0

    with torch.no_grad():
        for u in test_users:
            user_tensor = torch.full((n_items,), u, dtype=torch.long, device=device)
            item_tensor = torch.arange(n_items, dtype=torch.long, device=device)

            scores = model(user_tensor, item_tensor)  # [n_items]

            train_items = train_user_pos.get(u, set())
            if train_items:
                scores[list(train_items)] = -1e9

            _, topk_idx = torch.topk(scores, k)
            topk_list = topk_idx.cpu().numpy().tolist()
            topk_set = set(topk_list)

            gt_items = test_user_pos[u]
            hit_count = sum(1 for i in gt_items if i in topk_set)
            hits += hit_count
            total += len(gt_items)

            ndcg_sum += ndcg_at_k(topk_list, gt_items, k)

    recall = hits / total if total > 0 else 0.0
    ndcg = ndcg_sum / len(test_users) if len(test_users) > 0 else 0.0
    return recall, ndcg


# ----------------- LightGCN 相关评估 -----------------


def load_lightgcn_model(model_path, adj, device):
    """
    从 lightgcn_ml1m_best.pth 加载 LightGCN 模型。
    """
    ckpt = torch.load(model_path, map_location=device)
    n_users = ckpt["n_users"]
    n_items = ckpt["n_items"]
    embedding_dim = ckpt["embedding_dim"]
    num_layers = ckpt["num_layers"]

    model = LightGCN(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, n_users, n_items


def evaluate_lightgcn_rmse_mae(model, test_u, test_i, test_r, device):
    """
    用 LightGCN embedding 的点积当“伪评分”，计算 RMSE / MAE（参考指标）。
    """
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.get_user_item_embeddings()
        user_emb = user_emb.to(device)
        item_emb = item_emb.to(device)

        test_u_t = torch.LongTensor(test_u).to(device)
        test_i_t = torch.LongTensor(test_i).to(device)
        test_r_t = torch.FloatTensor(test_r).to(device)

        u_e = user_emb[test_u_t]   # [N, d]
        i_e = item_emb[test_i_t]   # [N, d]
        pred = (u_e * i_e).sum(dim=-1)

        mse_loss = torch.mean((pred - test_r_t) ** 2).item()
        mae_loss = torch.mean(torch.abs(pred - test_r_t)).item()

    rmse = mse_loss ** 0.5
    mae = mae_loss
    return rmse, mae


def evaluate_lightgcn_topk(model, train_user_pos, test_user_pos, device, k=10):
    """
    使用 LightGCN embedding 的点积做 Top-K 评估，计算 Recall@K 和 NDCG@K。
    """
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.get_user_item_embeddings()
        user_emb = user_emb.to(device)
        item_emb = item_emb.to(device)

        test_users = list(test_user_pos.keys())
        hits = 0
        total = 0
        ndcg_sum = 0.0

        n_items = item_emb.size(0)

        for u in test_users:
            u_e = user_emb[u]  # [d]
            scores = torch.matmul(item_emb, u_e)  # [n_items]

            train_items = train_user_pos.get(u, set())
            if train_items:
                scores[list(train_items)] = -1e9

            _, topk_idx = torch.topk(scores, k)
            topk_list = topk_idx.cpu().numpy().tolist()
            topk_set = set(topk_list)

            gt_items = test_user_pos[u]
            hit_count = sum(1 for i in gt_items if i in topk_set)
            hits += hit_count
            total += len(gt_items)

            ndcg_sum += ndcg_at_k(topk_list, gt_items, k)

        recall = hits / total if total > 0 else 0.0
        ndcg = ndcg_sum / len(test_users) if len(test_users) > 0 else 0.0

    return recall, ndcg


# ----------------- HybridNCF 相关评估 -----------------


def load_hybrid_model(model_path, ratings_path, device):
    """
    从 hybrid_ml1m_best.pth 加载 HybridNCF 模型，并重建内容编码器。
    """
    ckpt = torch.load(model_path, map_location=device)
    n_users = ckpt["n_users"]
    n_items = ckpt["n_items"]
    cfg = ckpt.get("config", None)

    features_dir = os.path.join(PROJECT_ROOT, "data", "features")
    text_feat_path = os.path.join(features_dir, "ml1m_text_embeddings_64.npy")
    text_index_path = os.path.join(features_dir, "ml1m_text_index.csv")
    image_feat_path = os.path.join(features_dir, "ml1m_image_embeddings_64.npy")
    image_index_path = os.path.join(features_dir, "ml1m_image_index.csv")

    content_encoder = ItemContentEncoder(
        ratings_path=ratings_path,
        text_feat_path=text_feat_path,
        text_index_path=text_index_path,
        image_feat_path=image_feat_path,
        image_index_path=image_index_path,
        use_text=True,
        use_image=True,
    )

    model = HybridNCF(
        num_users=n_users,
        num_items=n_items,
        content_encoder=content_encoder,
        gmf_dim=32,
        mlp_dim=32,
        content_proj_dim=32,
        mlp_layer_sizes=(128, 64, 32),
        dropout=cfg["model"].get("dropout", 0.0) if cfg is not None else 0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, n_users, n_items


def evaluate_hybrid_rmse_mae(model, test_u, test_i, test_r, device):
    """
    用 HybridNCF 的输出评分，计算 RMSE / MAE（平均值）。
    """
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0

    batch_size = 4096
    num_samples = len(test_r)

    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            u = torch.LongTensor(test_u[start:end]).to(device)
            i = torch.LongTensor(test_i[start:end]).to(device)
            r = torch.FloatTensor(test_r[start:end]).to(device)

            pred = model(u, i)

            # 保险：很多模型会输出 [B,1]，统一成 [B]
            pred = pred.view(-1)
            r = r.view(-1)

            diff = pred - r
            mse_sum += torch.sum(diff * diff).item()
            mae_sum += torch.sum(torch.abs(diff)).item()
            n += r.size(0)

    rmse = (mse_sum / n) ** 0.5
    mae = mae_sum / n
    return rmse, mae



def evaluate_hybrid_topk(model, train_user_pos, test_user_pos, n_items, device, k=10):
    """
    使用 HybridNCF 做 Top-K 推荐评估，计算 Recall@K 和 NDCG@K。
    """
    model.eval()
    test_users = list(test_user_pos.keys())

    hits = 0
    total = 0
    ndcg_sum = 0.0

    with torch.no_grad():
        for u in test_users:
            user_tensor = torch.full((n_items,), u, dtype=torch.long, device=device)
            item_tensor = torch.arange(n_items, dtype=torch.long, device=device)

            scores = model(user_tensor, item_tensor)  # [n_items]

            train_items = train_user_pos.get(u, set())
            if train_items:
                scores[list(train_items)] = -1e9

            _, topk_idx = torch.topk(scores, k)
            topk_list = topk_idx.cpu().numpy().tolist()
            topk_set = set(topk_list)

            gt_items = test_user_pos[u]
            hit_count = sum(1 for i in gt_items if i in topk_set)
            hits += hit_count
            total += len(gt_items)

            ndcg_sum += ndcg_at_k(topk_list, gt_items, k)

    recall = hits / total if total > 0 else 0.0
    ndcg = ndcg_sum / len(test_users) if len(test_users) > 0 else 0.0
    return recall, ndcg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 读取配置和评分数据
    config_path = os.path.join(UTILS_DIR, "config.yaml")
    cfg = load_config(config_path)

    ratings_cfg = cfg["data"]["ml1m_ratings"]
    if os.path.isabs(ratings_cfg):
        ratings_path = ratings_cfg
    else:
        ratings_path = os.path.join(PROJECT_ROOT, ratings_cfg)

    print("Loading ratings and building splits...")
    ratings_df = load_ml1m_ratings(ratings_path)
    user2idx, item2idx, idx2user, idx2item = build_id_mappings(ratings_df)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"n_users={n_users}, n_items={n_items}")

    train_df, val_df, test_df = split_by_user_time(ratings_df)

    train_u = train_df["userId"].map(user2idx).values
    train_i = train_df["movieId"].map(item2idx).values

    val_u = val_df["userId"].map(user2idx).values
    val_i = val_df["movieId"].map(item2idx).values

    test_u = test_df["userId"].map(user2idx).values
    test_i = test_df["movieId"].map(item2idx).values
    test_r = test_df["rating"].values.astype("float32")

    train_user_pos = build_user_pos_dict(train_u, train_i, n_users)
    val_user_pos = build_user_pos_dict(val_u, val_i, n_users)
    test_user_pos = build_user_pos_dict(test_u, test_i, n_users)

    # ========== NeuMF ==========
    print("\n========== Evaluating NeuMF ==========")
    neumf_path = os.path.join(PROJECT_ROOT, "neuMF_ml1m_best.pth")
    if not os.path.exists(neumf_path):
        print(f"[Warn] NeuMF checkpoint not found at {neumf_path}, skip NeuMF evaluation.")
    else:
        neumf_model, neumf_n_users, neumf_n_items = load_neumf_model(neumf_path, device)

        neumf_rmse, neumf_mae = evaluate_neumf_rmse_mae(
            neumf_model, test_u, test_i, test_r, device
        )
        print(f"NeuMF - RMSE={neumf_rmse:.4f}, MAE={neumf_mae:.4f}")

        neumf_recall10, neumf_ndcg10 = evaluate_neumf_topk(
            neumf_model,
            train_user_pos=train_user_pos,
            test_user_pos=test_user_pos,
            n_items=neumf_n_items,
            device=device,
            k=10,
        )
        print(f"NeuMF - Recall@10={neumf_recall10:.4f}, NDCG@10={neumf_ndcg10:.4f}")

    # ========== LightGCN ==========
    print("\n========== Evaluating LightGCN ==========")
    lightgcn_path = os.path.join(PROJECT_ROOT, "lightgcn_ml1m_best.pth")
    if not os.path.exists(lightgcn_path):
        print(f"[Warn] LightGCN checkpoint not found at {lightgcn_path}, skip LightGCN evaluation.")
    else:
        print("Rebuilding normalized adjacency matrix for LightGCN...")
        adj = build_normalized_adj(n_users, n_items, train_u, train_i, device=device)

        lightgcn_model, lg_n_users, lg_n_items = load_lightgcn_model(
            lightgcn_path, adj, device
        )

        lg_rmse, lg_mae = evaluate_lightgcn_rmse_mae(
            lightgcn_model, test_u, test_i, test_r, device
        )
        print(f"LightGCN - RMSE={lg_rmse:.4f}, MAE={lg_mae:.4f}")

        lg_recall10, lg_ndcg10 = evaluate_lightgcn_topk(
            lightgcn_model,
            train_user_pos=train_user_pos,
            test_user_pos=test_user_pos,
            device=device,
            k=10,
        )
        print(f"LightGCN - Recall@10={lg_recall10:.4f}, NDCG@10={lg_ndcg10:.4f}")

    # ========== HybridNCF ==========
    print("\n========== Evaluating HybridNCF (CF + Content) ==========")
    hybrid_path = os.path.join(PROJECT_ROOT, "hybrid_ml1m_best.pth")
    if not os.path.exists(hybrid_path):
        print(f"[Warn] Hybrid checkpoint not found at {hybrid_path}, skip Hybrid evaluation.")
    else:
        hybrid_model, hy_n_users, hy_n_items = load_hybrid_model(
            hybrid_path, ratings_path, device
        )

        hy_rmse, hy_mae = evaluate_hybrid_rmse_mae(
            hybrid_model, test_u, test_i, test_r, device
        )
        print(f"HybridNCF - RMSE={hy_rmse:.4f}, MAE={hy_mae:.4f}")

        hy_recall10, hy_ndcg10 = evaluate_hybrid_topk(
            hybrid_model,
            train_user_pos=train_user_pos,
            test_user_pos=test_user_pos,
            n_items=hy_n_items,
            device=device,
            k=10,
        )
        print(f"HybridNCF - Recall@10={hy_recall10:.4f}, NDCG@10={hy_ndcg10:.4f}")


if __name__ == "__main__":
    main()
