import os
import sys
import time
import yaml
import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ================= 路径设置 =================
# 当前文件在: .../recommender_project/experiments/ml1m/train_ngcf.py
# 项目根目录: .../recommender_project
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 预处理函数（兼容 utils.preprocess / preprocess 两种导入方式）
try:
    from utils.preprocess import (
        load_ml1m_ratings,
        build_id_mappings,
        split_by_user_time,
    )
except ImportError:
    from preprocess import (
        load_ml1m_ratings,
        build_id_mappings,
        split_by_user_time,
    )

# NGCF 模型
try:
    from models.ngcf_model import NGCF
except ImportError:
    from ngcf_model import NGCF
# =====================================================


def load_config(config_path: str):
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
    构建 NGCF 使用的归一化邻接矩阵 A_hat (稀疏张量)：
      - 节点:
          用户: 0 ~ num_users-1
          物品: num_users ~ num_users+num_items-1
      - 边: 对每个 (u, i) 加双向边 (u, i') 和 (i', u)
      - 归一化: A_hat = D^{-1/2} A D^{-1/2}
    """
    num_nodes = num_users + num_items

    user_nodes = np.array(user_idx, dtype=np.int64)
    item_nodes = np.array(item_idx, dtype=np.int64) + num_users

    # 双向边
    rows = np.concatenate([user_nodes, item_nodes], axis=0)
    cols = np.concatenate([item_nodes, user_nodes], axis=0)

    # 初始边权全为 1
    values = np.ones(len(rows), dtype=np.float32)

    # 节点度数
    degree = np.bincount(rows, minlength=num_nodes).astype(np.float32)
    degree[degree == 0.0] = 1.0  # 防止除零

    # 每条边的归一化权重: 1 / sqrt(deg[i] * deg[j])
    norm_values = values / np.sqrt(degree[rows] * degree[cols])

    indices = np.vstack((rows, cols))  # [2, E]
    indices_tensor = torch.from_numpy(indices).long()
    values_tensor = torch.from_numpy(norm_values).float()

    adj = torch.sparse_coo_tensor(
        indices_tensor,
        values_tensor,
        torch.Size([num_nodes, num_nodes]),
    )
    adj = adj.coalesce().to(device)
    return adj


def bpr_loss(u_e, pos_e, neg_e, reg_lambda):
    """
    BPR 损失：
      L = -log σ( (u·i_pos) - (u·i_neg) ) + L2 正则
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


def ndcg_at_k(rank_list, ground_truth_set, k):
    """
    计算单个用户的 NDCG@K（二元相关性）
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


def evaluate_topk(model, train_user_pos, test_user_pos, num_users, num_items, device, k=10):
    """
    使用 NGCF 的 embedding 做 Top-K 评估，计算 Recall@K / NDCG@K。
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

        for u in test_users:
            u_e = user_emb[u]  # [d]
            # scores: [num_items]
            scores = torch.matmul(item_emb, u_e)  # [n_items]

            # 屏蔽训练集中的正样本（不推荐已经看过的）
            train_items = train_user_pos.get(u, set())
            if train_items:
                scores[list(train_items)] = -1e9

            # Top-K
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
    # 1. 读取配置
    config_path = os.path.join(PROJECT_ROOT, "utils", "config.yaml")
    cfg = load_config(config_path)

    ratings_path = cfg["data"]["ml1m_ratings"]
    # 确保这里已经是绝对路径（你之前已经改过一次了）
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Ratings path: {ratings_path}")

    device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Final device: {device}")

    # NGCF 超参数（你可以之后挪进 config.yaml）
    embedding_dim = 64
    num_layers = 3
    batch_size = 2048
    epochs = 50
    lr = 0.001
    reg_lambda = 1e-4

    # 2. 加载 ML-1M 评分并划分 train/val/test
    print("Loading ML-1M ratings and building splits...")
    ratings_df = load_ml1m_ratings(ratings_path)
    user2idx, item2idx, idx2user, idx2item = build_id_mappings(ratings_df)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"n_users={n_users}, n_items={n_items}")

    train_df, val_df, test_df = split_by_user_time(ratings_df)

    # 编码为 [0, n_users) / [0, n_items)
    train_u = train_df["userId"].map(user2idx).values
    train_i = train_df["movieId"].map(item2idx).values

    val_u = val_df["userId"].map(user2idx).values
    val_i = val_df["movieId"].map(item2idx).values

    test_u = test_df["userId"].map(user2idx).values
    test_i = test_df["movieId"].map(item2idx).values

    # user -> 正样本集合
    train_user_pos = build_user_pos_dict(train_u, train_i, n_users)
    val_user_pos = build_user_pos_dict(val_u, val_i, n_users)
    test_user_pos = build_user_pos_dict(test_u, test_i, n_users)

    # 3. 构建归一化邻接矩阵（只用 train 集）
    print("Building normalized adjacency matrix for NGCF...")
    adj = build_normalized_adj(n_users, n_items, train_u, train_i, device=device)

    # 4. 构建 BPR 数据集 / DataLoader
    train_dataset = BPRDataset(train_u, train_i, n_items, train_user_pos)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # 5. 构建 NGCF 模型
    print("Building NGCF model...")
    model = NGCF(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        adj=adj,
        dropout=0.1,
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

        # 每个 epoch 之后在验证集上评估 Recall@10 / NDCG@10
        val_recall, val_ndcg = evaluate_topk(
            model,
            train_user_pos=train_user_pos,
            test_user_pos=val_user_pos,
            num_users=n_users,
            num_items=n_items,
            device=device,
            k=10,
        )

        dt = time.time() - t0
        print(
            f"[NGCF][Epoch {epoch:03d}] loss={avg_loss:.4f} "
            f"val_Recall@10={val_recall:.4f} val_NDCG@10={val_ndcg:.4f} ({dt:.1f}s)"
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

    # 7. 用最佳验证模型在测试集上评估
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    test_recall, test_ndcg = evaluate_topk(
        model,
        train_user_pos=train_user_pos,
        test_user_pos=test_user_pos,
        num_users=n_users,
        num_items=n_items,
        device=device,
        k=10,
    )
    print(f"[NGCF][Test] Recall@10={test_recall:.4f}, NDCG@10={test_ndcg:.4f}")

    # 8. 保存最佳模型
    save_path = os.path.join(PROJECT_ROOT, "ngcf_ml1m_best.pth")
    torch.save(best_state, save_path)
    print(f"Saved best NGCF model to {save_path}")


if __name__ == "__main__":
    main()
