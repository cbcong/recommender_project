# experiments/ml1m/train_multivae.py
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Set

import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ------------------- 路径设置 -------------------
# 当前文件：.../recommender_project/experiments/ml1m/train_multivae.py
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# 工程根目录：.../recommender_project
PROJECT_ROOT = os.path.dirname(os.path.dirname(CUR_DIR))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 预处理函数：优先从本目录导入，失败再从 utils 导入，兼容你的工程结构
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

# Multi-VAE 模型（你已经把我给你的实现放在 models/vae_model.py 里）
from vae_model import MultiVAE
# ------------------------------------------------


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class UserInteractionDataset(Dataset):
    """
    用户级数据集：每个样本是一整行 user 的多热向量 x_u ∈ {0,1}^{n_items}
    """

    def __init__(self, mat: np.ndarray):
        """
        mat: [n_users, n_items], float32, 0/1
        """
        assert mat.ndim == 2
        self.mat = mat

    def __len__(self):
        return self.mat.shape[0]

    def __getitem__(self, idx):
        x = self.mat[idx]  # np.ndarray
        return torch.from_numpy(x)  # [n_items], float32


def build_user_pos_dict(
    user_idx: np.ndarray,
    item_idx: np.ndarray,
) -> Dict[int, Set[int]]:
    """
    根据 (user_idx, item_idx) 构建每个用户的正样本集合: u -> set(items)
    """
    user_pos: Dict[int, Set[int]] = defaultdict(set)
    for u, i in zip(user_idx, item_idx):
        user_pos[int(u)].add(int(i))
    return user_pos


def multivae_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.2,
) -> torch.Tensor:
    """
    Multi-VAE 的重构 + KL 损失：
      - 重构项：对 logits 做 log_softmax，然后只在正反馈 (x=1) 上计算负对数似然
      - KL 项：标准 Gaussian VAE 的 KL
      - 总损失：recon_loss + beta * kl_loss

    logits, x: [B, n_items]
    mu, logvar: [B, z_dim]
    """
    x = x.float()  # 确保是浮点

    # 重构损失（Liang et al. MultiVAE 中的 multinomial likelihood 写法）
    log_softmax = torch.log_softmax(logits, dim=1)      # [B, n_items]
    # x 是 0/1 的多热向量，只在正样本上累积
    recon_loss = -torch.sum(log_softmax * x, dim=1).mean()   # 标量

    # KL = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [B]
    kl_loss = kl.mean()

    return recon_loss + beta * kl_loss


def train_epoch(
    model: MultiVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta_kl: float = 0.2,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x in loader:
        # [B, n_items]
        batch_x = batch_x.to(device).float()

        optimizer.zero_grad()
        logits, mu, logvar = model(batch_x)
        loss = multivae_loss(logits, batch_x, mu, logvar, beta=beta_kl)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def evaluate_topk(
    model: MultiVAE,
    user_matrix: np.ndarray,
    train_user_pos: Dict[int, Set[int]],
    eval_user_pos: Dict[int, Set[int]],
    device: torch.device,
    k: int = 10,
):
    """
    使用 Multi-VAE 的输出 logits 进行 Top-K 评估 (HitRate@K, Recall@K, NDCG@K)

    - user_matrix: 训练矩阵 X_train, shape=[n_users, n_items]，只包含 train 交互
    - train_user_pos: 训练集 u->items，用于屏蔽“已看过”的物品
    - eval_user_pos:   评估用交互 (val 或 test) u->items，作为 ground-truth
    """
    model.eval()
    n_users, n_items = user_matrix.shape

    with torch.no_grad():
        user_tensor = torch.from_numpy(user_matrix).float().to(device)
        batch_size = 512
        all_scores = []

        # 先对所有用户算一遍 logits，并收集到 CPU
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            x_batch = user_tensor[start:end]  # [B, n_items]
            logits, _, _ = model(x_batch)     # [B, n_items]
            all_scores.append(logits.cpu())
        scores_all = torch.cat(all_scores, dim=0)  # [n_users, n_items]

    K = k
    # 预生成 1/log2(rank+2) 表，加速 NDCG 计算
    log2_table = 1.0 / np.log2(np.arange(K) + 2.0)

    hit_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0
    user_cnt = 0

    for u, gt_items in eval_user_pos.items():
        if not gt_items:
            continue

        user_cnt += 1
        gt_set = set(int(i) for i in gt_items)

        # 取出该用户的得分向量，并复制一份（避免原 tensor 被修改）
        scores_u = scores_all[u].clone()  # [n_items]

        # 屏蔽训练集中已经看过的 items：不再推荐
        train_items = train_user_pos.get(u, set())
        if train_items:
            mask_idx = torch.tensor(list(train_items), dtype=torch.long)
            scores_u[mask_idx] = -1e9

        # Top-K 物品索引
        _, topk_idx = torch.topk(scores_u, K)
        topk = topk_idx.numpy()
        topk_set = set(int(i) for i in topk)

        # HitRate@K
        hit = 1.0 if len(topk_set & gt_set) > 0 else 0.0
        hit_sum += hit

        # Recall@K
        recall = len(topk_set & gt_set) / float(len(gt_set))
        recall_sum += recall

        # NDCG@K
        dcg = 0.0
        for rank, item in enumerate(topk):
            if int(item) in gt_set:
                dcg += log2_table[rank]
        m = min(len(gt_set), K)
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


def main():
    # 1. 配置与设备
    config_path = os.path.join(PROJECT_ROOT, "utils", "config.yaml")
    cfg = load_config(config_path)

    ratings_path = cfg["data"]["ml1m_ratings"]
    if not os.path.isabs(ratings_path):
        ratings_path = os.path.join(PROJECT_ROOT, ratings_path)

    device_cfg = cfg["train"].get("device", "cuda")
    device = torch.device(
        "cuda" if device_cfg == "cuda" and torch.cuda.is_available() else "cpu"
    )

    batch_size = cfg["train"]["batch_size"]
    lr = cfg["train"]["lr"]
    epochs = cfg["train"]["epochs"]

    weight_decay_raw = cfg["train"].get("weight_decay", 0.0)
    try:
        weight_decay = float(weight_decay_raw)
    except (TypeError, ValueError):
        print(f"[WARN] Invalid weight_decay={weight_decay_raw}, fallback to 0.0")
        weight_decay = 0.0

    beta_kl = 0.2   # KL 权重
    K_EVAL = 10     # Top-K 评估的 K

    print(f"Using device: {device}")
    print(f"Ratings path: {ratings_path}")
    print("Loading ML-1M ratings and building user-item matrix for Multi-VAE...")

    # 2. 加载评分 & 编码
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

    # 3. 构建训练矩阵 X_train (implicit，多热向量)
    X_train = np.zeros((n_users, n_items), dtype=np.float32)
    for u, i in zip(train_u, train_i):
        X_train[int(u), int(i)] = 1.0

    # user -> 正样本集合（train/val/test，用于 Top-K 评估）
    train_user_pos = build_user_pos_dict(train_u, train_i)
    val_user_pos = build_user_pos_dict(val_u, val_i)
    test_user_pos = build_user_pos_dict(test_u, test_i)

    # 4. DataLoader（用户级）
    train_dataset = UserInteractionDataset(X_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # 5. 构建 MultiVAE 模型
    print("Building MultiVAE model...")

    # ---------- 5.1 设定 p_dims ----------
    # p_dims 是 decoder 的各层维度，[z_dim, h1, ..., input_dim]
    # 最后一维必须等于 n_items（物品数）
    p_dims = cfg["model"].get("multivae_p_dims", [200, 600, n_items])

    # 如果 config 里写错了最后一维，我们强制改成 n_items，避免维度不匹配
    if p_dims[-1] != n_items:
        print(f"[WARN] multivae_p_dims[-1]={p_dims[-1]} != n_items={n_items}，自动改为 n_items")
        p_dims[-1] = n_items

    # ---------- 5.2 设定 q_dims ----------
    # q_dims 是 encoder 的各层维度，[input_dim, ..., z_dim]
    # 若 config 没写，就自动反转 p_dims
    q_dims_cfg = cfg["model"].get("multivae_q_dims", None)
    if q_dims_cfg is None:
        q_dims = p_dims[::-1]
    else:
        q_dims = list(q_dims_cfg)
        if q_dims[0] != n_items:
            print(f"[WARN] multivae_q_dims[0]={q_dims[0]} != n_items={n_items}，自动改为 n_items")
            q_dims[0] = n_items
        if q_dims[-1] != p_dims[0]:
            print(f"[WARN] multivae_q_dims[-1]={q_dims[-1]} != p_dims[0]={p_dims[0]}，自动改为 p_dims[0]")
            q_dims[-1] = p_dims[0]

    # ---------- 5.3 dropout ----------
    dropout = float(cfg["model"].get("multivae_dropout", cfg["model"].get("dropout", 0.5)))

    # ---------- 5.4 构建 MultiVAE ----------
    # 注意：MultiVAE 的 __init__ 现在是 (p_dims, q_dims=None, dropout=0.5)
    model = MultiVAE(
        p_dims=p_dims,
        q_dims=q_dims,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_recall = 0.0
    best_state = None

    # 6. 训练循环
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta_kl=beta_kl,
        )
        # 用 Recall@K 在验证集上做模型选择
        val_hit, val_rec, val_ndcg = evaluate_topk(
            model,
            X_train,
            train_user_pos=train_user_pos,
            eval_user_pos=val_user_pos,
            device=device,
            k=K_EVAL,
        )
        dt = time.time() - t0

        print(
            f"[Epoch {epoch:03d}] loss/batch={train_loss:.4f} "
            f"val_Hit@{K_EVAL}={val_hit:.4f} "
            f"val_Recall@{K_EVAL}={val_rec:.4f} "
            f"val_NDCG@{K_EVAL}={val_ndcg:.4f} "
            f"({dt:.1f}s)"
        )

        if val_rec > best_val_recall:
            best_val_recall = val_rec
            best_state = model.state_dict()

    # 7. 测试集评估（使用验证集上 Recall@K 最优的模型）
    if best_state is not None:
        model.load_state_dict(best_state)

    test_hit, test_rec, test_ndcg = evaluate_topk(
        model,
        X_train,
        train_user_pos=train_user_pos,
        eval_user_pos=test_user_pos,
        device=device,
        k=K_EVAL,
    )
    print(
        f"[Test] HitRate@{K_EVAL}={test_hit:.4f}, "
        f"Recall@{K_EVAL}={test_rec:.4f}, "
        f"NDCG@{K_EVAL}={test_ndcg:.4f}"
    )

    # 8. 保存最佳模型（给 evaluate_beyond_ml1m.py 用）
    save_path = os.path.join(PROJECT_ROOT, "multivae_ml1m_best.pth")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "p_dims": p_dims,
            "q_dims": q_dims,
            "n_users": n_users,
            "n_items": n_items,
            "config": cfg,
        },
        save_path,
    )
    print(f"Saved best MultiVAE model to {save_path}")


if __name__ == "__main__":
    main()
