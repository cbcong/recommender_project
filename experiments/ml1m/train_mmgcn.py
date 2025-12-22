# experiments/ml1m/train_mmgcn.py
import os
import sys
import time
from typing import Dict, Any, Tuple

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ========== 路径设置：工程根目录 / utils / models ==========
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../recommender_project/experiments/ml1m
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))      # .../recommender_project

UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ========== 导入预处理函数 ==========
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

from mmgcn_model import MMGCN


# ========== 工具函数 ==========

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


class BPRDataset(Dataset):
    """
    BPR 训练用数据集：
      - 存一份 (user, pos_item) 列表
      - 每次 __getitem__ 时随机采一个 neg_item
    """
    def __init__(self, user_indices, item_indices, num_items, user_pos_dict):
        assert len(user_indices) == len(item_indices)
        self.user_indices = np.array(user_indices, dtype=np.int64)
        self.item_indices = np.array(item_indices, dtype=np.int64)
        self.num_items = num_items
        self.user_pos_dict = user_pos_dict

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        u = int(self.user_indices[idx])
        pos_i = int(self.item_indices[idx])

        # 采一个负样本（直到不是该用户的正样本）
        while True:
            neg_i = np.random.randint(0, self.num_items)
            if neg_i not in self.user_pos_dict[u]:
                break

        return (
            torch.LongTensor([u]).squeeze(0),
            torch.LongTensor([pos_i]).squeeze(0),
            torch.LongTensor([neg_i]).squeeze(0),
        )


def build_user_pos_dict(user_idx, item_idx, num_users: int):
    user_pos = {u: set() for u in range(num_users)}
    for u, i in zip(user_idx, item_idx):
        user_pos[int(u)].add(int(i))
    return user_pos


def build_normalized_adj(num_users, num_items, user_idx, item_idx, device):
    """
    构建 LightGCN / NGCF / MMGCN 使用的归一化邻接矩阵 A_hat (稀疏张量)：

    节点编号:
      - 用户节点: 0 ~ num_users-1
      - 物品节点: num_users ~ num_users+num_items-1
    边:
      - 对每个 (u, i) 加双向边 (u, i') 和 (i', u)
    归一化:
      A_hat = D^{-1/2} A D^{-1/2}
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


def bpr_loss(u_e, pos_e, neg_e, reg_lambda: float):
    """
    BPR 损失：
      L = -log(σ( (u·i_pos) - (u·i_neg) )) + L2 正则
    """
    pos_scores = torch.sum(u_e * pos_e, dim=-1)
    neg_scores = torch.sum(u_e * neg_e, dim=-1)

    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

    reg_loss = reg_lambda * (
        u_e.norm(2).pow(2) +
        pos_e.norm(2).pow(2) +
        neg_e.norm(2).pow(2)
    ) / u_e.size(0)

    return loss + reg_loss


def ndcg_at_k(rank_list, ground_truth_set, k: int):
    """
    单用户 NDCG@K（二元相关性）
    """
    dcg = 0.0
    for rank, item in enumerate(rank_list[:k], start=1):
        if item in ground_truth_set:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(ground_truth_set), k)
    if ideal_hits == 0:
        return 0.0

    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / np.log2(rank + 1)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_topk(
    model: MMGCN,
    train_user_pos,
    test_user_pos,
    device,
    k: int = 10,
) -> Tuple[float, float]:
    """
    Top-K 评估：Recall@K / NDCG@K
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
            u_e = user_emb[u]                             # [d]
            scores = torch.matmul(item_emb, u_e)          # [n_items]

            train_items = train_user_pos.get(u, set())
            if train_items:
                scores[list(train_items)] = -1e9          # 屏蔽训练集中已看过的 item

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


def load_item_features(
    feat_path: str,
    index_path: str,
    item2idx: Dict[int, int],
) -> np.ndarray:
    """
    从 .npy + .csv 中加载 item 文本/图像特征，并对齐到 [0, n_items) 的索引空间。

    假定 index CSV 至少包含一列是 item 原始 ID（例如 movieId 或 item_id），
    可能还包含一列 index（指向 npy 行号）。如果没有 index 列，则默认 DataFrame 行号即特征行号。
    """
    if (not os.path.exists(feat_path)) or (not os.path.exists(index_path)):
        raise FileNotFoundError(
            f"Feature file or index file not found: {feat_path} / {index_path}"
        )

    feats = np.load(feat_path)  # [N_feat, D]
    index_df = pd.read_csv(index_path)

    cols = list(index_df.columns)
    if "movieId" in cols:
        key_col = "movieId"
    elif "item_id" in cols:
        key_col = "item_id"
    else:
        key_col = cols[0]

    use_index_col = "index" in cols

    n_items = len(item2idx)
    dim = feats.shape[1]
    out = np.zeros((n_items, dim), dtype=np.float32)

    for df_row_idx, row in index_df.iterrows():
        raw_id = int(row[key_col])
        if raw_id not in item2idx:
            continue
        feat_idx = int(row["index"]) if use_index_col else df_row_idx
        if feat_idx < 0 or feat_idx >= feats.shape[0]:
            continue
        out[item2idx[raw_id]] = feats[feat_idx]

    return out


# ========== 主函数 ==========

def main():
    config_path = os.path.join(PROJECT_ROOT, "utils", "config.yaml")
    cfg = load_config(config_path)

    ratings_path = cfg["data"]["ml1m_ratings"]
    if not os.path.isabs(ratings_path):
        ratings_path = os.path.join(PROJECT_ROOT, ratings_path)

    # MMGCN 超参数（可根据需要再调）
    embedding_dim = 64
    num_layers = 3
    batch_size = 2048
    epochs = 50
    lr = 0.001
    reg_lambda = 1e-4
    top_k = 10

    device = cfg["train"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    print(f"Using device: {device}")
    print(f"Ratings path: {ratings_path}")

    # 1. 加载 ML-1M 评分数据 & 构建映射
    print("Loading ML-1M ratings and building splits...")
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

    train_user_pos = build_user_pos_dict(train_u, train_i, n_users)
    val_user_pos = build_user_pos_dict(val_u, val_i, n_users)
    test_user_pos = build_user_pos_dict(test_u, test_i, n_users)

    # 2. 构建归一化邻接矩阵（三个模态共用同一结构）
    print("Building normalized adjacency matrices for MMGCN...")
    adj_id = build_normalized_adj(n_users, n_items, train_u, train_i, device=device)
    adj_text = adj_id  # 如需更精细的模态图，可以在此构造独立矩阵
    adj_image = adj_id

    # 3. 加载文本 / 图像特征，并对齐到 item 索引
    print("Loading item multi-modal features (text + image)...")
    features_dir = os.path.join(PROJECT_ROOT, "data", "features")
    text_feat_path = os.path.join(features_dir, "ml1m_text_embeddings_64.npy")
    text_index_path = os.path.join(features_dir, "ml1m_text_index.csv")
    image_feat_path = os.path.join(features_dir, "ml1m_image_embeddings_64.npy")
    image_index_path = os.path.join(features_dir, "ml1m_image_index.csv")

    item_text_np = load_item_features(text_feat_path, text_index_path, item2idx)
    item_image_np = load_item_features(image_feat_path, image_index_path, item2idx)

    item_text_t = torch.from_numpy(item_text_np).float().to(device)
    item_image_t = torch.from_numpy(item_image_np).float().to(device)

    print(
        f"Text feat dim={item_text_t.size(1)}, "
        f"Image feat dim={item_image_t.size(1)}"
    )

    # 4. BPR 数据集 / DataLoader
    train_dataset = BPRDataset(train_u, train_i, n_items, train_user_pos)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # 5. 构建 MMGCN 模型
    print("Building MMGCN model...")
    model = MMGCN(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        adj_id=adj_id,
        adj_text=adj_text,
        adj_image=adj_image,
        item_text_feats=item_text_t,
        item_image_feats=item_image_t,
        content_hidden_dim=64,
        dropout=0.0,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    best_val_recall = 0.0
    best_state = None

    # 6. 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            users, pos_items, neg_items = batch
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()
            u_e, pos_e, neg_e = model(users, pos_items, neg_items)
            loss = bpr_loss(u_e, pos_e, neg_e, reg_lambda)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # 每个 epoch 在验证集上评估 Recall@K / NDCG@K
        val_recall, val_ndcg = evaluate_topk(
            model,
            train_user_pos=train_user_pos,
            test_user_pos=val_user_pos,
            device=device,
            k=top_k,
        )

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}] loss={avg_loss:.4f} "
            f"val_Recall@{top_k}={val_recall:.4f} "
            f"val_NDCG@{top_k}={val_ndcg:.4f} ({dt:.1f}s)"
        )

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_state = {
                "model_state": model.state_dict(),
                "n_users": n_users,
                "n_items": n_items,
                "embedding_dim": embedding_dim,
                "num_layers": num_layers,
            }

    # 7. 用最佳模型在测试集上评估
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    test_recall, test_ndcg = evaluate_topk(
        model,
        train_user_pos=train_user_pos,
        test_user_pos=test_user_pos,
        device=device,
        k=top_k,
    )
    print(
        f"[Test] Recall@{top_k}={test_recall:.4f}, "
        f"NDCG@{top_k}={test_ndcg:.4f}"
    )

    # 8. 保存模型
    save_path = os.path.join(PROJECT_ROOT, "mmgcn_ml1m_best.pth")
    torch.save(
        {
            "model_state": model.state_dict(),
            "n_users": n_users,
            "n_items": n_items,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "config": cfg,
        },
        save_path,
    )
    print(f"Saved best MMGCN model to {save_path}")


if __name__ == "__main__":
    main()
