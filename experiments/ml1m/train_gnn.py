# experiments/ml1m/train_gnn.py

import os
import sys
import time
import yaml
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

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

# ===================================================

# 优先从 utils 导入，如果你还有老版本的 preprocess.py / gnn_model.py 在根目录，也兼容
try:
    from utils.preprocess import load_ml1m_ratings, build_id_mappings, split_by_user_time
except ImportError:
    from preprocess import load_ml1m_ratings, build_id_mappings, split_by_user_time

try:
    from models.gnn_model import LightGCN
except ImportError:
    from gnn_model import LightGCN


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
    根据训练集 (u, i) 构建每个用户的正样本集合 dict: u -> set(items)
    """
    user_pos = {u: set() for u in range(num_users)}
    for u, i in zip(user_idx, item_idx):
        user_pos[int(u)].add(int(i))
    return user_pos


def build_normalized_adj(num_users, num_items, user_idx, item_idx, device):
    """
    构建 LightGCN 使用的归一化邻接矩阵 A_hat (稀疏张量)：

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

    # 双向边
    rows = np.concatenate([user_nodes, item_nodes], axis=0)
    cols = np.concatenate([item_nodes, user_nodes], axis=0)

    # 初始权重全为 1
    values = np.ones(len(rows), dtype=np.float32)

    # 每个节点的度数（行求和）
    degree = np.bincount(rows, minlength=num_nodes).astype(np.float32)
    degree[degree == 0.0] = 1.0  # 防止除零

    # A_hat 的每条边权重: 1 / sqrt(deg[i] * deg[j])
    norm_values = values / np.sqrt(degree[rows] * degree[cols])

    indices = np.vstack((rows, cols))  # [2, E]
    indices_tensor = torch.from_numpy(indices).long()
    values_tensor = torch.from_numpy(norm_values).float()

    adj = torch.sparse_coo_tensor(
        indices_tensor,
        values_tensor,
        torch.Size([num_nodes, num_nodes])
    )
    adj = adj.coalesce().to(device)
    return adj


def bpr_loss(u_e, pos_e, neg_e, reg_lambda):
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


def evaluate_topk(model, train_user_pos, test_user_pos, num_users, num_items, device, k=10):
    """
    简单的 Top-K 评估（Recall@K）：
    - 对每个在 test 集中出现的用户 u:
        * 计算对所有物品的打分
        * 去掉训练集中的正样本
        * 取 Top-K
        * 看 test 中的正样本有多少在 Top-K 中
    - 返回整体 Recall@K
    """
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.get_user_item_embeddings()
        user_emb = user_emb.to(device)
        item_emb = item_emb.to(device)

        test_users = list(test_user_pos.keys())
        hits = 0
        total = 0

        for u in test_users:
            u_e = user_emb[u]  # [d]
            scores = torch.mv(item_emb, u_e)  # [n_items]

            # 屏蔽掉训练集中的正样本（不推荐用户已经看过的）
            train_items = train_user_pos.get(u, set())
            if train_items:
                scores[list(train_items)] = -1e9

            # Top-K
            _, topk_idx = torch.topk(scores, k)
            topk_set = set(topk_idx.cpu().numpy().tolist())

            gt_items = test_user_pos[u]
            hit_count = sum(1 for i in gt_items if i in topk_set)
            hits += hit_count
            total += len(gt_items)

        recall = hits / total if total > 0 else 0.0
        return recall


def main():
    # 1. 读取配置（只用 data.ml1m_ratings）
    config_path = os.path.join(UTILS_DIR, "config.yaml")
    cfg = load_config(config_path)

    ratings_path = cfg["data"]["ml1m_ratings"]
    # 把相对路径转成绝对路径，确保从 experiments/ml1m 运行也能找到
    if not os.path.isabs(ratings_path):
        ratings_path = os.path.join(PROJECT_ROOT, ratings_path)

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using ratings file: {ratings_path}")

    # LightGCN 的超参数（你愿意可以挪到 config.yaml 里）
    embedding_dim = 64
    num_layers = 3
    batch_size = 2048
    epochs = 50
    lr = 0.001
    reg_lambda = 1e-4

    # 2. 加载评分数据，并划分 train/val/test（跟 NeuMF 一样的策略）
    print("Loading ratings...")
    ratings_df = load_ml1m_ratings(ratings_path)
    user2idx, item2idx, idx2user, idx2item = build_id_mappings(ratings_df)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"n_users={n_users}, n_items={n_items}")

    train_df, val_df, test_df = split_by_user_time(ratings_df)

    # 编码为 [0, n_user) / [0, n_item)
    train_u = train_df["userId"].map(user2idx).values
    train_i = train_df["movieId"].map(item2idx).values

    val_u = val_df["userId"].map(user2idx).values
    val_i = val_df["movieId"].map(item2idx).values

    test_u = test_df["userId"].map(user2idx).values
    test_i = test_df["movieId"].map(item2idx).values

    # 训练集 user->正样本集合
    train_user_pos = build_user_pos_dict(train_u, train_i, n_users)

    # 验证 / 测试用的 user->正样本集合
    val_user_pos = build_user_pos_dict(val_u, val_i, n_users)
    test_user_pos = build_user_pos_dict(test_u, test_i, n_users)

    # 3. 构建归一化邻接矩阵
    print("Building normalized adjacency matrix...")
    adj = build_normalized_adj(n_users, n_items, train_u, train_i, device=device)

    # 4. 构建 BPR 数据集和 DataLoader（只用训练集）
    train_dataset = BPRDataset(train_u, train_i, n_items, train_user_pos)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # 5. 构建 LightGCN 模型
    print("Building LightGCN model...")
    model = LightGCN(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        adj=adj,
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

        # 每个 epoch 之后做一次验证集 Recall@10
        val_recall = evaluate_topk(
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
            f"[Epoch {epoch:03d}] loss={avg_loss:.4f} "
            f"val_Recall@10={val_recall:.4f} ({dt:.1f}s)"
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

    # 7. 用最佳验证模型在测试集上评估 Recall@10
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    test_recall = evaluate_topk(
        model,
        train_user_pos=train_user_pos,
        test_user_pos=test_user_pos,
        num_users=n_users,
        num_items=n_items,
        device=device,
        k=10,
    )
    print(f"[Test] Recall@10={test_recall:.4f}")

    # 8. 保存最佳模型到「项目根目录」（方便 evaluate 复用）
    save_path = os.path.join(PROJECT_ROOT, "lightgcn_ml1m_best.pth")
    torch.save(best_state, save_path)
    print(f"Saved best LightGCN model to {save_path}")


if __name__ == "__main__":
    main()
