import os
import sys
import time
import math
import random
import itertools
from collections import defaultdict
from typing import Dict, List

import pandas as pd
import yaml
import numpy as np
import torch

# ================== 路径设置 ==================
# 当前文件: recommender_project/experiments/ml1m/evaluate_beyond_ml1m.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 这里用“扁平 import”，配合上面对 sys.path 的设置
from preprocess import load_ml1m_ratings, build_id_mappings, split_by_user_time
from cf_model import NeuMF
# ====== 检查 DIN 是否可用 ======
try:
    import din_model  # 只用来判断 din_model.py 是否存在
    HAS_DIN = True
except ImportError:
    HAS_DIN = False

# 下面几个模型用 try/except，避免你暂时没这个文件时整个脚本挂掉
try:
    from gnn_model import LightGCN
except ImportError:
    LightGCN = None

try:
    from ngcf_model import NGCF
except ImportError:
    NGCF = None

# ===== Multi-VAE (MultiVAE) =====
try:
    # 如果你后来还是建了 vae_model.py，就用这个
    from multivae_model import MultiVAE
    HAS_MULTIVAE = True
except ImportError:
    try:
        from vae_model import MultiVAE
        HAS_MULTIVAE = True
    except ImportError:
        MultiVAE = None
        HAS_MULTIVAE = False
        print(
            "[WARN] MultiVAE class not found (vae_model.py / vae_model.py 都没找到)，"
            "跳过 Multi-VAE。"
        )


try:
    from din_model import DIN
except ImportError:
    DIN = None

try:
    from mmgcn_model import MMGCN
except ImportError:
    MMGCN = None


# =====================================================
#                 通用工具函数
# =====================================================

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def prepare_ml1m_data():
    """
    读取 ML-1M，构建 train/val/test 划分，并映射成 [0, n_users) / [0, n_items) 索引。
    同时构建：
      - train_items_by_user: dict[u] = set(items)，只统计 train 集（给“过滤已看过”用）
      - val_items_by_user:   dict[u] = set(items)，只统计 val 集（MMGCN 想过滤 val 可用）
      - test_items_by_user:  dict[u] = set(items)，只统计 test 集（确定 eval_users）
      - item_popularity: np.ndarray[n_items]，统计 train+val 中的交互次数
      - item_users: dict[i] = set(users)，基于 train+val
      - eval_users: 在 test 中出现过的用户列表
      - train_u, train_i: 训练集中 (u, i) 的索引数组（给图模型重建邻接用）
      - user2idx, item2idx: 原始 ID -> 索引映射（MMGCN 构图时用）
    """
    config_path = os.path.join(PROJECT_ROOT, "utils", "config.yaml")
    cfg = load_config(config_path)

    ratings_path = cfg["data"]["ml1m_ratings"]
    if not os.path.isabs(ratings_path):
        ratings_path = os.path.join(PROJECT_ROOT, ratings_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Ratings path: {ratings_path}")
    print("Loading ML-1M ratings and building splits...")

    ratings_df = load_ml1m_ratings(ratings_path)
    user2idx, item2idx, idx2user, idx2item = build_id_mappings(ratings_df)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"n_users={n_users}, n_items={n_items}")

    train_df, val_df, test_df = split_by_user_time(ratings_df)

    # 映射为整数索引
    for df in [train_df, val_df, test_df]:
        df["user_idx"] = df["userId"].map(user2idx).astype("int64")
        df["item_idx"] = df["movieId"].map(item2idx).astype("int64")

    # ------- 各种统计结构 -------
    train_items_by_user_list = [set() for _ in range(n_users)]
    val_items_by_user = defaultdict(set)
    test_items_by_user = defaultdict(set)

    item_users_list = [set() for _ in range(n_items)]
    item_popularity = np.zeros(n_items, dtype=np.int64)

    # 1) train：给“过滤已看过”用
    for u, i in zip(train_df["user_idx"].values, train_df["item_idx"].values):
        u = int(u)
        i = int(i)
        train_items_by_user_list[u].add(i)

    # 2) val：单独记录，MMGCN 如需过滤 val 也可以用
    for u, i in zip(val_df["user_idx"].values, val_df["item_idx"].values):
        u = int(u)
        i = int(i)
        val_items_by_user[u].add(i)

    # 3) item_popularity / item_users：用 train+val
    for df in [train_df, val_df]:
        for u, i in zip(df["user_idx"].values, df["item_idx"].values):
            u = int(u)
            i = int(i)
            item_popularity[i] += 1
            item_users_list[i].add(u)

    # 4) test：只用来确定 eval_users
    for u, i in zip(test_df["user_idx"].values, test_df["item_idx"].values):
        u = int(u)
        i = int(i)
        test_items_by_user[u].add(i)

    train_items_by_user = {u: s for u, s in enumerate(train_items_by_user_list)}
    item_users = {i: s for i, s in enumerate(item_users_list)}
    eval_users = sorted(test_items_by_user.keys())

    train_u = train_df["user_idx"].values
    train_i = train_df["item_idx"].values

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_items_by_user": train_items_by_user,
        "val_items_by_user": val_items_by_user,
        "test_items_by_user": test_items_by_user,
        "item_popularity": item_popularity,
        "item_users": item_users,
        "eval_users": eval_users,
        "train_u": train_u,
        "train_i": train_i,
        "device": device,
        "config": cfg,
        "user2idx": user2idx,
        "item2idx": item2idx,
    }
    return meta



# =====================================================
#            Beyond-Accuracy 相关指标函数
# =====================================================

def compute_item_coverage(recs, n_items: int):
    """ItemCoverage@K：推荐列表中被推荐到的不同 item 占总 item 的比例。"""
    all_items = set()
    for items in recs.values():
        all_items.update(items)
    return len(all_items) / float(n_items)


def build_long_tail_mask(item_popularity: np.ndarray, head_ratio=0.8):
    """
    根据流行度划分 head / tail：
      - 按流行度降序排序
      - 找到最小的 head 集合，使得 head 中的交互数 >= head_ratio * 全部交互数
      - 返回 tail_mask: True 表示是长尾物品
    """
    n_items = item_popularity.shape[0]
    idx_sorted = np.argsort(item_popularity)[::-1]
    pop_sorted = item_popularity[idx_sorted]
    total_pop = pop_sorted.sum()
    if total_pop <= 0:
        return np.ones(n_items, dtype=bool)

    cumsum = np.cumsum(pop_sorted)
    head_cut = np.searchsorted(cumsum, head_ratio * total_pop)
    head_cut = min(head_cut, n_items - 1)

    head_items = idx_sorted[: head_cut + 1]
    tail_mask = np.ones(n_items, dtype=bool)
    tail_mask[head_items] = False
    return tail_mask


def compute_longtail_share(recs, tail_mask: np.ndarray):
    """LongTailShare@K：Top-K 推荐中落在长尾集合 T 的比例。"""
    tail_hits = 0
    total = 0
    for items in recs.values():
        for i in items:
            if 0 <= i < len(tail_mask) and tail_mask[i]:
                tail_hits += 1
            total += 1
    return tail_hits / float(total) if total > 0 else 0.0


def compute_novelty(recs, item_popularity: np.ndarray):
    """
    Novelty@K：基于 self-information 的新颖度。
      novelty(i) = -log2(pop(i) / sum_j pop(j))
    对所有用户、所有推荐物品取平均。
    """
    total_events = item_popularity.sum()
    n_items = item_popularity.shape[0]
    if total_events <= 0:
        return 0.0

    probs = item_popularity.astype("float64") / float(total_events)
    probs[probs <= 0] = 1e-12
    info = -np.log2(probs)  # [n_items]

    s = 0.0
    n = 0
    for items in recs.values():
        for i in items:
            if 0 <= i < n_items:
                s += info[i]
                n += 1
    return s / float(n) if n > 0 else 0.0


def compute_ild(recs, item_users: dict):
    """
    Intra-list Diversity@K（ILD）：
      - 对每个用户 u 的推荐列表 L_u:
          * 枚举所有无序物品对 (i,j)
          * 基于用户共现算 Jaccard 相似度 sim(i,j)
          * dist(i,j) = 1 - sim(i,j)
        得到该用户的 ILD(u)
      - 最后对所有用户做平均
    """
    total_ild = 0.0
    user_cnt = 0

    for _, items in recs.items():
        if len(items) <= 1:
            continue

        pairs = list(itertools.combinations(items, 2))
        if not pairs:
            continue

        s = 0.0
        m = 0
        for i, j in pairs:
            users_i = item_users.get(i, set())
            users_j = item_users.get(j, set())
            if not users_i or not users_j:
                sim = 0.0
            else:
                inter = len(users_i & users_j)
                union = len(users_i) + len(users_j) - inter
                sim = inter / union if union > 0 else 0.0
            s += (1.0 - sim)
            m += 1

        if m > 0:
            total_ild += s / m
            user_cnt += 1

    return total_ild / user_cnt if user_cnt > 0 else 0.0


def compute_personalization(recs, sample_users=500):
    """
    Personalization@K：用户间列表差异。
      - 随机抽一部分用户
      - 计算所有用户对之间的 Jaccard(L_u, L_v)
      - 个性化度 = 1 - 平均 Jaccard
    """
    users = list(recs.keys())
    if len(users) <= 1:
        return 0.0

    if len(users) > sample_users:
        random.seed(42)
        users = random.sample(users, sample_users)

    sets = {u: set(recs[u]) for u in users}
    n = len(users)
    s = 0.0
    m = 0

    for i in range(n):
        for j in range(i + 1, n):
            a = sets[users[i]]
            b = sets[users[j]]
            inter = len(a & b)
            union = len(a | b)
            sim = inter / union if union > 0 else 0.0
            s += sim
            m += 1

    avg_sim = s / m if m > 0 else 0.0
    return 1.0 - avg_sim


def compute_avg_popularity(recs, item_popularity: np.ndarray):
    """平均流行度：所有推荐物品的平均 pop(i) 值。"""
    s = 0.0
    n = 0
    for items in recs.values():
        for i in items:
            if 0 <= i < len(item_popularity):
                s += item_popularity[i]
                n += 1
    return s / float(n) if n > 0 else 0.0


# =====================================================
#                 NeuMF Top-K 推荐
# =====================================================

def generate_topk_neumf(
    ckpt_path: str,
    n_users: int,
    n_items: int,
    train_items_by_user: dict,
    eval_users,
    device,
    K: int = 10,
):
    """
    使用 NeuMF 的打分产出 Top-K 推荐：
      返回 recs: dict[u] = [item0, item1, ..., item(K-1)]
    """
    if not os.path.exists(ckpt_path):
        print(f"[WARN] NeuMF checkpoint not found at {ckpt_path}, skip NeuMF.")
        return None

    print("\n========== Generating Top-K for NeuMF ==========")
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = ckpt["config"]
    n_users_ckpt = ckpt["n_users"]
    n_items_ckpt = ckpt["n_items"]
    if n_users_ckpt != n_users or n_items_ckpt != n_items:
        print(
            f"[WARN] NeuMF ckpt (n_users={n_users_ckpt}, n_items={n_items_ckpt}) "
            f"!= current meta (n_users={n_users}, n_items={n_items}), 仍继续尝试。"
        )

    model = NeuMF(
        num_users=n_users,
        num_items=n_items,
        mf_dim=cfg["model"]["mf_dim"],
        mlp_layer_sizes=tuple(cfg["model"]["mlp_layers"]),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_items = torch.arange(n_items, dtype=torch.long, device=device)
    recs = {}

    t0 = time.time()
    with torch.no_grad():
        for idx, u in enumerate(eval_users, start=1):
            u_int = int(u)
            user_tensor = torch.full((n_items,), u_int, dtype=torch.long, device=device)
            scores = model(user_tensor, all_items)  # [n_items]

            # 屏蔽训练+验证中已经看过的物品
            seen_items = train_items_by_user.get(u_int, set())
            if seen_items:
                scores[list(seen_items)] = -1e9

            _, topk_idx = torch.topk(scores, K)
            recs[u_int] = topk_idx.cpu().numpy().tolist()

            if idx % 500 == 0:
                elapsed = time.time() - t0
                print(f"  processed {idx}/{len(eval_users)} users ({elapsed:.1f}s)")

    print(f"NeuMF Top-K generation done for {len(eval_users)} users.")
    return recs


# =====================================================
#          图类模型邻接矩阵 + 通用 Top-K
# =====================================================

def build_normalized_adj(num_users, num_items, user_idx, item_idx, device):
    """
    构建 LightGCN/NGCF 使用的归一化邻接矩阵 A_hat (稀疏张量)：
      节点编号:
        - 用户: [0, num_users)
        - 物品: [num_users, num_users + num_items)
      对每个 (u, i) 加双向边，A_hat = D^{-1/2} A D^{-1/2}
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
        torch.Size([num_nodes, num_nodes])
    )
    adj = adj.coalesce().to(device)
    return adj


def generate_topk_from_embeddings(
    model,
    n_items: int,
    train_items_by_user: dict,
    eval_users,
    device,
    K: int = 10,
    model_name: str = "Model",
):
    """
    通用 Top-K：适用于有 get_user_item_embeddings() 的模型（LightGCN / NGCF / MMGCN）。
    """
    print(f"\n========== Generating Top-K for {model_name} ==========")
    recs = {}
    t0 = time.time()
    with torch.no_grad():
        user_emb, item_emb = model.get_user_item_embeddings()
        user_emb = user_emb.to(device)
        item_emb = item_emb.to(device)

        for idx, u in enumerate(eval_users, start=1):
            u_int = int(u)
            u_e = user_emb[u_int]  # [d]
            scores = torch.matmul(item_emb, u_e)  # [n_items]

            seen_items = train_items_by_user.get(u_int, set())
            if seen_items:
                scores[list(seen_items)] = -1e9

            _, topk_idx = torch.topk(scores, K)
            recs[u_int] = topk_idx.cpu().numpy().tolist()

            if idx % 500 == 0:
                elapsed = time.time() - t0
                print(f"  processed {idx}/{len(eval_users)} users ({elapsed:.1f}s)")

    print(f"{model_name} Top-K generation done for {len(eval_users)} users.")
    return recs


# =====================================================
#          LightGCN / NGCF / MMGCN 加载 + Top-K
# =====================================================

def load_lightgcn_model(model_path, adj, device):
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


def generate_topk_lightgcn(
    ckpt_path,
    n_users,
    n_items,
    train_u,
    train_i,
    train_items_by_user,
    eval_users,
    device,
    K=10,
):
    if LightGCN is None:
        print("\n[WARN] LightGCN class not found (gnn_model.py 缺失？)，跳过 LightGCN。")
        return None
    if not os.path.exists(ckpt_path):
        print(f"\n[WARN] LightGCN checkpoint not found at {ckpt_path}，跳过 LightGCN。")
        return None

    print("\n========== Preparing LightGCN for Top-K ==========")
    adj = build_normalized_adj(n_users, n_items, train_u, train_i, device)
    model, lg_n_users, lg_n_items = load_lightgcn_model(ckpt_path, adj, device)
    if lg_n_users != n_users or lg_n_items != n_items:
        print(
            f"[WARN] LightGCN ckpt (n_users={lg_n_users}, n_items={lg_n_items}) "
            f"!= current meta (n_users={n_users}, n_items={n_items})，仍继续尝试。"
        )

    return generate_topk_from_embeddings(
        model=model,
        n_items=n_items,
        train_items_by_user=train_items_by_user,
        eval_users=eval_users,
        device=device,
        K=K,
        model_name="LightGCN",
    )


def load_ngcf_model(model_path, adj, device):
    ckpt = torch.load(model_path, map_location=device)
    n_users = ckpt["n_users"]
    n_items = ckpt["n_items"]
    embedding_dim = ckpt.get("embedding_dim", ckpt.get("emb_dim", 64))

    # 从 ckpt 里读层数；兼容不同命名
    num_layers = ckpt.get("num_layers", ckpt.get("n_layers", 3))

    # 不再传 node_dropout / message_dropout，按实际 NGCF 定义来
    model = NGCF(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        adj=adj,
    ).to(device)

    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()
    return model, n_users, n_items


def generate_topk_ngcf(
    ckpt_path,
    n_users,
    n_items,
    train_u,
    train_i,
    train_items_by_user,
    eval_users,
    device,
    K=10,
):
    if NGCF is None:
        print("\n[WARN] NGCF class not found (ngcf_model.py 缺失？)，跳过 NGCF。")
        return None
    if not os.path.exists(ckpt_path):
        print(f"\n[WARN] NGCF checkpoint not found at {ckpt_path}，跳过 NGCF。")
        return None

    print("\n========== Preparing NGCF for Top-K ==========")
    adj = build_normalized_adj(n_users, n_items, train_u, train_i, device)
    model, g_n_users, g_n_items = load_ngcf_model(ckpt_path, adj, device)
    if g_n_users != n_users or g_n_items != n_items:
        print(
            f"[WARN] NGCF ckpt (n_users={g_n_users}, n_items={g_n_items}) "
            f"!= current meta (n_users={n_users}, n_items={n_items})，仍继续尝试。"
        )

    return generate_topk_from_embeddings(
        model=model,
        n_items=n_items,
        train_items_by_user=train_items_by_user,
        eval_users=eval_users,
        device=device,
        K=K,
        model_name="NGCF",
    )

def generate_topk_mmgcn(
    ckpt_path: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
    n_users: int,
    n_items: int,
    train_items_by_user: Dict[int, set],
    val_items_by_user: Dict[int, set],
    test_items_by_user: Dict[int, set],
    device: torch.device,
    topk: int = 10,
) -> Dict[int, np.ndarray]:
    """
    使用 MMGCN 生成 Top-K 推荐列表。

    recs 返回格式：{ user_idx(int): np.ndarray(shape=[K], dtype=int) }
    """
    # 1. 构建模型（含邻接矩阵和多模态特征）
    model = load_mmgcn_model(
        ckpt_path=ckpt_path,
        train_df=train_df,
        user2idx=user2idx,
        item2idx=item2idx,
        n_users=n_users,
        n_items=n_items,
        device=device,
    )

    model.eval()
    recs: Dict[int, np.ndarray] = {}

    with torch.no_grad():
        # 和 train_mmgcn.evaluate_topk 中一样，通过 get_user_item_embeddings 拿最终 embedding
        user_emb, item_emb = model.get_user_item_embeddings()
        user_emb = user_emb.to(device)
        item_emb = item_emb.to(device)

        eval_users = sorted(test_items_by_user.keys())
        n_eval = len(eval_users)

        for idx, u in enumerate(eval_users, start=1):
            u_e = user_emb[u]                      # [d]
            scores = torch.matmul(item_emb, u_e)   # [n_items]

            # 屏蔽 train / val / test 中已经交互过的 item
            seen_items = set()
            seen_items |= train_items_by_user.get(u, set())
            seen_items |= val_items_by_user.get(u, set())


            if seen_items:
                seen_idx = torch.tensor(
                    list(seen_items), dtype=torch.long, device=device
                )
                scores[seen_idx] = -1e9

            _, topk_idx = torch.topk(scores, topk)
            recs[u] = topk_idx.cpu().numpy()

            if idx % 500 == 0:
                print(f"  [MMGCN] processed {idx}/{n_eval} users")

    print(f"MMGCN Top-K generation done for {len(recs)} users.")
    return recs



# =====================================================
#            Multi-VAE 加载 + Top-K
# =====================================================

def build_user_item_matrix(train_df, val_df, n_users, n_items):
    """
    Multi-VAE 用的用户-物品多热矩阵：
      - 行：用户
      - 列：物品
      - 值：是否在 train+val 中出现（这里用 0/1）
    """
    X = np.zeros((n_users, n_items), dtype=np.float32)
    for df in [train_df, val_df]:
        for u, i in zip(df["user_idx"].values, df["item_idx"].values):
            X[int(u), int(i)] = 1.0
    return X


def load_multivae_model(model_path, n_items, device):
    """
    假定 ckpt 至少包含:
      - "model_state" 或 "state_dict"
      - 可选: "p_dims", "q_dims", "dropout"
    如果缺少 p_dims，就用一套常见结构 [600, 200, n_items]。
    """
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        p_dims = ckpt.get("p_dims", [600, 200, n_items])
        q_dims = ckpt.get("q_dims", None)
        dropout = ckpt.get("dropout", 0.5)
    else:
        # 兼容直接保存 state_dict 的情况
        state_dict = ckpt
        p_dims = [600, 200, n_items]
        q_dims = None
        dropout = 0.5

    model = MultiVAE(p_dims=p_dims, q_dims=q_dims, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_topk_multivae(
    ckpt_path,
    n_users,
    n_items,
    train_df,
    val_df,
    train_items_by_user,
    eval_users,
    device,
    K=10,
):
    if MultiVAE is None:
        print("\n[WARN] MultiVAE class not found (vae_model.py 缺失？)，跳过 Multi-VAE。")
        return None
    if not os.path.exists(ckpt_path):
        print(f"\n[WARN] MultiVAE checkpoint not found at {ckpt_path}，跳过 Multi-VAE。")
        return None

    print("\n========== Preparing Multi-VAE for Top-K ==========")
    # 用户-物品矩阵（train + val）
    X = build_user_item_matrix(train_df, val_df, n_users, n_items)
    X_torch = torch.from_numpy(X).float().to(device)

    model = load_multivae_model(ckpt_path, n_items=n_items, device=device)

    recs = {}
    t0 = time.time()
    with torch.no_grad():
        for idx, u in enumerate(eval_users, start=1):
            u_int = int(u)
            x_u = X_torch[u_int : u_int + 1]  # [1, n_items]

            # Multi-VAE forward: 假定返回 (logits, mu, logvar)
            out = model(x_u)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            scores = logits.view(-1)  # [n_items]

            seen_items = train_items_by_user.get(u_int, set())
            if seen_items:
                scores[list(seen_items)] = -1e9

            _, topk_idx = torch.topk(scores, K)
            recs[u_int] = topk_idx.cpu().numpy().tolist()

            if idx % 500 == 0:
                elapsed = time.time() - t0
                print(f"  processed {idx}/{len(eval_users)} users ({elapsed:.1f}s)")

    print(f"Multi-VAE Top-K generation done for {len(eval_users)} users.")
    return recs


# =====================================================
#            DIN 加载 + Top-K
# =====================================================

def build_user_histories_for_din(train_df, max_seq_len: int):
    """
    DIN 的用户历史序列（只用 train 集）：
      - 对每个用户按 timestamp 升序排序
      - 截取最近 max_seq_len 个 item
    """
    # 确保有 timestamp 列；如果没有，就按原顺序
    if "timestamp" in train_df.columns:
        df_sorted = train_df.sort_values(["user_idx", "timestamp"])
    else:
        df_sorted = train_df.sort_values(["user_idx"])

    histories = {}
    for u, group in df_sorted.groupby("user_idx"):
        items = group["item_idx"].tolist()
        if len(items) == 0:
            histories[int(u)] = []
        else:
            histories[int(u)] = items[-max_seq_len:]
    return histories


# ===== DIN: 加载模型 =====
def load_din_model(ckpt_path: str, device: torch.device):
    """
    从 din_ml1m_best.pth 里还原 DIN 模型。

    ckpt 中我们当初保存了：
      - state_dict
      - n_users
      - n_items（注意：这里是 num_items_for_din = 数据集 n_items + 1，0 为 padding）
      - config
      - max_hist_len
    """
    import torch
    from din_model import DIN  # 已经在 models 目录下

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    n_users = ckpt["n_users"]
    din_n_items = ckpt["n_items"]          # = 原始 n_items + 1
    max_hist_len = ckpt.get("max_hist_len", 50)

    # 当初在 train_din.py 里是用 NeuMF 的 mf_dim 和 mlp_layers
    embed_dim = cfg["model"].get("mf_dim", 32)
    mlp_layers = cfg["model"].get("mlp_layers", [128, 64, 32])
    dropout = cfg["model"].get("dropout", 0.0)

    model = DIN(
        num_users=n_users,
        num_items=din_n_items,
        embed_dim=embed_dim,
        att_hidden_sizes=(64, 32, 16),
        fc_hidden_sizes=tuple(mlp_layers),
        max_history_len=max_hist_len,
        dropout=dropout,
        use_dice=True,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, max_hist_len, n_users, din_n_items

# ===== DIN: 生成 Top-K 推荐列表 =====
def generate_topk_din(
    ckpt_path: str,
    n_users: int,
    n_items: int,
    train_df,
    val_df,
    train_items_by_user: Dict[int, set],
    test_users: List[int],
    K: int,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """
    基于已经训练好的 DIN 模型，为每个 test 用户生成 Top-K 推荐列表。

    参数说明（和你 main() 里调用一致）：
      - ckpt_path: din_ml1m_best.pth 路径
      - n_users:   数据集用户数（6040）
      - n_items:   数据集物品数（3706，注意：不含 padding）
      - train_df:  含有 train 交互的 DataFrame（包含 user_idx, item_idx, timestamp）
      - val_df:    暂时没用到，但保留参数以兼容已有代码
      - train_items_by_user: {user_idx -> 该用户训练集中交互过的 item_idx 集合}
      - test_users: 需要评估的用户列表
      - K:         Top-K
      - device:    cuda/cpu

    返回：
      - recs: {user_idx -> 长度为 K 的 item_idx 数组（0..n_items-1）}
    """
    import os

    if (not ckpt_path) or (not os.path.exists(ckpt_path)):
        print(f"[WARN] DIN ckpt not found at {ckpt_path}, skip DIN.")
        return {}

    # ---------- 1. 加载 DIN 模型 ----------
    model, max_hist_len, d_n_users, d_n_items = load_din_model(ckpt_path, device)
    if d_n_users != n_users:
        print(
            f"[WARN] DIN ckpt (n_users={d_n_users}) != current meta (n_users={n_users})，仍继续尝试。"
        )
    if d_n_items not in (n_items, n_items + 1):
        print(
            f"[WARN] DIN ckpt n_items={d_n_items} 和当前 n_items={n_items} 不一致，"
            f"默认认为 ckpt 中是 n_items+1（包含 padding=0）。"
        )

    # ---------- 2. 为 DIN 构造每个用户的历史序列（基于 train_df） ----------
    # 这里假设 train_df 已经有：user_idx, item_idx, timestamp
    # 每个用户按时间排序，把 item_idx + 1 当作 DIN 的 item embedding 索引（0 留给 padding）
    hist_seq_by_user: Dict[int, List[int]] = {}

    train_sorted = train_df.sort_values(["user_idx", "timestamp"])
    for u, df_u in train_sorted.groupby("user_idx"):
        items = df_u["item_idx"].astype("int64").tolist()
        # 转为 DIN 使用的索引空间：+1
        seq = [i + 1 for i in items]
        hist_seq_by_user[int(u)] = seq

    # ---------- 3. 准备候选 item 与各种缓存 ----------
    recs: Dict[int, np.ndarray] = {}

    # DIN embedding 里的候选 item 索引：1..d_n_items-1
    all_item_ids_din = torch.arange(1, d_n_items, dtype=torch.long, device=device)
    # 映射回数据集的 item_idx: 0..n_items-1
    all_item_indices_dataset = all_item_ids_din.cpu().numpy() - 1

    # 一次评估多少个候选 item，防止显存爆炸
    chunk_size = 512

    # ---------- 4. 逐用户生成 Top-K ----------
    model.eval()
    with torch.no_grad():
        num_eval_users = 0
        for idx, u in enumerate(test_users):
            u = int(u)

            # 该用户在 train 中的历史序列
            seq = hist_seq_by_user.get(u, [])
            if len(seq) == 0:
                # 没有历史，DIN 不能建模兴趣，直接跳过
                continue

            # 截断到 max_hist_len，只保留最近的交互
            if len(seq) > max_hist_len:
                seq = seq[-max_hist_len:]

            L = len(seq)
            pad_len = max_hist_len - L
            # 左侧 padding = 0
            hist_padded = [0] * pad_len + seq  # 长度 = max_hist_len

            # [1, L]
            hist_items_single = torch.LongTensor(hist_padded).unsqueeze(0).to(device)
            # [1]
            hist_len_single = torch.LongTensor([L]).to(device)

            # -------- 对所有候选 item 打分 --------
            score_list = []
            for start in range(0, len(all_item_ids_din), chunk_size):
                end = min(start + chunk_size, len(all_item_ids_din))
                items_batch = all_item_ids_din[start:end]  # [B]
                B = items_batch.size(0)

                user_batch = torch.full((B,), u, dtype=torch.long, device=device)  # [B]
                hist_batch = hist_items_single.expand(B, -1)  # [B, L]
                len_batch = hist_len_single.expand(B)        # [B]

                preds = model(user_batch, hist_batch, len_batch, items_batch)  # [B]
                score_list.append(preds.detach().cpu().numpy())

            scores_din = np.concatenate(score_list, axis=0)  # len = d_n_items - 1

            # -------- 把分数映射回 “数据集 item_idx 空间” --------
            # scores_full 长度 = n_items，默认 -1e9（视为禁止推荐）
            scores_full = np.full(n_items, -1e9, dtype=np.float32)

            valid_mask = (all_item_indices_dataset >= 0) & (all_item_indices_dataset < n_items)
            scores_full[all_item_indices_dataset[valid_mask]] = scores_din[valid_mask]

            # 过滤掉训练集中已经交互过的 item（不再推荐）
            seen_items = train_items_by_user.get(u, set())
            for it in seen_items:
                if 0 <= it < n_items:
                    scores_full[it] = -1e9

            # 如果所有 item 分数都是 -1e9，说明没有可推荐的 item，跳过该用户
            if np.all(scores_full == -1e9):
                continue

            # Top-K 索引（0..n_items-1）
            topk_indices = np.argpartition(scores_full, -K)[-K:]
            topk_indices = topk_indices[np.argsort(scores_full[topk_indices])[::-1]]

            recs[u] = topk_indices
            num_eval_users += 1

            if (idx + 1) % 500 == 0:
                print(f"  [DIN] processed {idx + 1}/{len(test_users)} users")

        print(f"DIN Top-K generation done for {num_eval_users} users.")

    return recs

# =====================================================
#                     主流程
# =====================================================

def build_normalized_adj_for_mmgcn(
    num_users: int,
    num_items: int,
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    和 train_mmgcn.py 中的 build_normalized_adj 完全一致，
    但这里起名为 build_normalized_adj_for_mmgcn，避免和其它模型冲突。
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


def load_item_features_for_mmgcn(
    feat_path: str,
    index_path: str,
    item2idx: Dict[int, int],
) -> np.ndarray:
    """
    从 .npy + .csv 中加载 item 文本/图像特征，并对齐到 [0, n_items) 的索引空间。
    这就是 train_mmgcn.py 里的 load_item_features，一字不差搬过来的。
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

def load_mmgcn_model(
    ckpt_path: str,
    train_df: pd.DataFrame,
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
    n_users: int,
    n_items: int,
    device: torch.device,
) -> MMGCN:
    """
    评估阶段构建 MMGCN：
      - 用 train_df 构建 adj_id / adj_text / adj_image（和 train_mmgcn 一样）
      - 从 data/features/ 加载文本 + 图像特征并对齐
      - 按 ckpt 中保存的 embedding_dim / num_layers 实例化 MMGCN
      - 再 load_state_dict
    """

    # 1. 用训练集交互构建邻接矩阵（三个模态共用）
    train_u = train_df["userId"].map(user2idx).values
    train_i = train_df["movieId"].map(item2idx).values

    adj_id = build_normalized_adj_for_mmgcn(
        num_users=n_users,
        num_items=n_items,
        user_idx=train_u,
        item_idx=train_i,
        device=device,
    )
    adj_text = adj_id
    adj_image = adj_id

    # 2. 加载文本 / 图像特征（路径与 train_mmgcn.py 保持一致）
    features_dir = os.path.join(PROJECT_ROOT, "data", "features")
    text_feat_path = os.path.join(features_dir, "ml1m_text_embeddings_64.npy")
    text_index_path = os.path.join(features_dir, "ml1m_text_index.csv")
    image_feat_path = os.path.join(features_dir, "ml1m_image_embeddings_64.npy")
    image_index_path = os.path.join(features_dir, "ml1m_image_index.csv")

    item_text_np = load_item_features_for_mmgcn(
        text_feat_path, text_index_path, item2idx
    )
    item_image_np = load_item_features_for_mmgcn(
        image_feat_path, image_index_path, item2idx
    )

    item_text_t = torch.from_numpy(item_text_np).float().to(device)
    item_image_t = torch.from_numpy(item_image_np).float().to(device)

    # 3. 从 ckpt 里取超参数（训练时存进去的）
    ckpt = torch.load(ckpt_path, map_location=device)
    embedding_dim = ckpt.get("embedding_dim", 64)
    num_layers = ckpt.get("num_layers", 3)

    # 4. 构建 MMGCN（参数完全照 train_mmgcn.py）
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

    # 5. 加载参数（train_mmgcn 保存的是 "model_state"）
    state_key = "model_state" if "model_state" in ckpt else "state_dict"
    model.load_state_dict(ckpt[state_key])
    model.eval()

    return model



def print_beyond_metrics(model_name, recs, K, n_items, tail_mask, item_popularity, item_users):
    if recs is None:
        return

    item_coverage = compute_item_coverage(recs, n_items)
    longtail_share = compute_longtail_share(recs, tail_mask)
    novelty = compute_novelty(recs, item_popularity)
    ild = compute_ild(recs, item_users)
    personalization = compute_personalization(recs, sample_users=500)
    avg_pop = compute_avg_popularity(recs, item_popularity)

    print(f"\nModel: {model_name}")
    print(f"  ItemCoverage@{K:2d}    = {item_coverage:.4f}")
    print(f"  LongTailShare@{K:2d}   = {longtail_share:.4f}")
    print(f"  Novelty@{K:2d}         = {novelty:.4f}")
    print(f"  ILD@{K:2d}             = {ild:.4f}")
    print(f"  Personalization@{K:2d} = {personalization:.4f}")
    print(f"  AvgPopularity@{K:2d}   = {avg_pop:.4f}")

def eval_and_print_beyond_metrics(
    recs_by_user,
    test_items_by_user,
    item_popularity,
    n_items,
    model_name: str,
    K: int,
):
    """
    统一计算并打印 Beyond-Accuracy 指标：
      - ItemCoverage@K
      - LongTailShare@K
      - Novelty@K
      - ILD@K
      - Personalization@K
      - AvgPopularity@K

    参数：
      recs_by_user: {user_idx: [item_idx1, item_idx2, ...]} 或 numpy 数组
      test_items_by_user: {user_idx: set(真实测试 item)}（当前函数里没用到，只是接口保持统一）
      item_popularity: 可以是 dict 或 numpy 数组（一维）
      n_items: 物品总数（不含 padding）
      model_name: 模型名称（打印用）
      K: Top-K
    """

    # ---- 0) 统一 item_popularity 为 dict: {item_idx: count} ----
    # NeuMF / LightGCN / NGCF 的调用里传进来一般是 dict，
    # DIN 那里传进来的是 numpy.ndarray，这里做统一转换。
    if isinstance(item_popularity, dict):
        item_pop_dict = item_popularity
    else:
        # 视为 array-like，按索引 0..len-1 生成字典
        arr = np.asarray(item_popularity, dtype=np.float64)
        item_pop_dict = {i: float(c) for i, c in enumerate(arr)}
    item_popularity = item_pop_dict

    # ---- 1) Item Coverage ----
    recommended_items = set()
    for u, recs in recs_by_user.items():
        if recs is None:
            continue
        # recs 可能是 list 或 numpy array，都统一处理
        recs_list = list(recs)
        if len(recs_list) == 0:
            continue
        recommended_items.update(recs_list[:K])

    if n_items > 0:
        item_coverage = len(recommended_items) / float(n_items)
    else:
        item_coverage = 0.0

    # ---- 2) Long-Tail Share + 3) Novelty ----
    pops = np.array(list(item_popularity.values()), dtype=np.float64)
    if pops.size == 0:
        long_tail_share = 0.0
        novelty = 0.0
    else:
        # 按流行度从高到低排序，取前 20% 作为 head，其余为 long-tail
        sorted_pops = np.sort(pops)[::-1]
        cutoff_idx = int(0.2 * len(sorted_pops))
        cutoff_idx = max(1, cutoff_idx)
        pop_threshold = sorted_pops[cutoff_idx - 1]

        long_tail_items = {i for i, cnt in item_popularity.items() if cnt <= pop_threshold}

        lt_count = 0
        total_recs = 0
        for recs in recs_by_user.values():
            if recs is None:
                continue
            recs_list = list(recs)
            for iid in recs_list[:K]:
                total_recs += 1
                if iid in long_tail_items:
                    lt_count += 1
        long_tail_share = (lt_count / float(total_recs)) if total_recs > 0 else 0.0

        # Novelty：用 -log2(popularity 归一化后的概率)
        total_cnt = pops.sum()
        novelty_sum = 0.0
        novelty_recs = 0
        if total_cnt > 0:
            for recs in recs_by_user.values():
                if recs is None:
                    continue
                recs_list = list(recs)
                for iid in recs_list[:K]:
                    cnt = float(item_popularity.get(iid, 0.5))
                    p_i = cnt / total_cnt
                    if p_i <= 0:
                        continue
                    novelty_sum += -np.log2(p_i)
                    novelty_recs += 1
        novelty = (novelty_sum / novelty_recs) if novelty_recs > 0 else 0.0

    # ---- 4) ILD（Intra-list Diversity）----
    # 用基于流行度排名的简单“距离”：dist(i,j) = |rank_i - rank_j|（归一化）
    item_ids = list(item_popularity.keys())
    if len(item_ids) > 1:
        item_ids_sorted = sorted(item_ids, key=lambda i: item_popularity[i], reverse=True)
        Np = len(item_ids_sorted)
        rank_norm = {
            iid: idx / (Np - 1) if Np > 1 else 0.0
            for idx, iid in enumerate(item_ids_sorted)
        }
        ild_sum = 0.0
        pair_cnt = 0

        for recs in recs_by_user.values():
            if recs is None:
                continue
            items = list(recs)[:K]
            L = len(items)
            if L <= 1:
                continue
            for i in range(L):
                for j in range(i + 1, L):
                    ii = items[i]
                    jj = items[j]
                    ri = rank_norm.get(ii, 0.5)
                    rj = rank_norm.get(jj, 0.5)
                    dist = abs(ri - rj)
                    ild_sum += dist
                    pair_cnt += 1
        ild = (ild_sum / pair_cnt) if pair_cnt > 0 else 0.0
    else:
        ild = 0.0

    # ---- 5) Personalization ----
    user_ids = list(recs_by_user.keys())
    n_users_eval = len(user_ids)

    max_users_for_personal = 1000
    if n_users_eval > max_users_for_personal:
        user_ids = user_ids[:max_users_for_personal]
        n_users_eval = max_users_for_personal

    if n_users_eval <= 1:
        personalization = 0.0
    else:
        sum_jaccard = 0.0
        pair_cnt = 0
        topk_sets = {}
        for u in user_ids:
            recs = recs_by_user[u]
            if recs is None:
                topk_sets[u] = set()
            else:
                topk_sets[u] = set(list(recs)[:K])

        for i in range(n_users_eval):
            ui = user_ids[i]
            set_i = topk_sets[ui]
            for j in range(i + 1, n_users_eval):
                uj = user_ids[j]
                set_j = topk_sets[uj]
                if not set_i and not set_j:
                    continue
                inter = len(set_i & set_j)
                union = len(set_i | set_j)
                if union == 0:
                    continue
                jacc = inter / union
                sum_jaccard += jacc
                pair_cnt += 1
        avg_jaccard = (sum_jaccard / pair_cnt) if pair_cnt > 0 else 0.0
        personalization = 1.0 - avg_jaccard

    # ---- 6) Avg Popularity ----
    pop_sum = 0.0
    pop_cnt = 0
    for recs in recs_by_user.values():
        if recs is None:
            continue
        recs_list = list(recs)
        for iid in recs_list[:K]:
            pop_sum += float(item_popularity.get(iid, 0.0))
            pop_cnt += 1
    avg_popularity = (pop_sum / pop_cnt) if pop_cnt > 0 else 0.0

    # ---- 打印结果 ----
    print(f"\nModel: {model_name}")
    print(f"  ItemCoverage@{K}    = {item_coverage:.4f}")
    print(f"  LongTailShare@{K}   = {long_tail_share:.4f}")
    print(f"  Novelty@{K}         = {novelty:.4f}")
    print(f"  ILD@{K}             = {ild:.4f}")
    print(f"  Personalization@{K} = {personalization:.4f}")
    print(f"  AvgPopularity@{K}   = {avg_popularity:.4f}")


def main():
    # 1. 数据与统计
    meta = prepare_ml1m_data()
    n_users = meta["n_users"]
    n_items = meta["n_items"]
    train_df = meta["train_df"]
    val_df = meta["val_df"]
    test_df = meta["test_df"]
    train_items_by_user = meta["train_items_by_user"]
    val_items_by_user = meta["val_items_by_user"]      # 新增
    test_items_by_user = meta["test_items_by_user"]
    item_popularity = meta["item_popularity"]
    item_users = meta["item_users"]
    eval_users = meta["eval_users"]
    train_u = meta["train_u"]
    train_i = meta["train_i"]
    device = meta["device"]
    user2idx = meta["user2idx"]                        # 新增
    item2idx = meta["item2idx"]                        # 新增


    print(f"Number of eval users (with test interactions): {len(eval_users)}")

    tail_mask = build_long_tail_mask(item_popularity)
    K = 10

    # 2. NeuMF
    neu_ckpt_path = os.path.join(PROJECT_ROOT, "neuMF_ml1m_best.pth")
    neu_recs = generate_topk_neumf(
        neu_ckpt_path,
        n_users=n_users,
        n_items=n_items,
        train_items_by_user=train_items_by_user,
        eval_users=eval_users,
        device=device,
        K=K,
    )

    print(f"\n========== Beyond-Accuracy on ML-1M (K={K}) ==========")
    print_beyond_metrics(
        model_name="NeuMF",
        recs=neu_recs,
        K=K,
        n_items=n_items,
        tail_mask=tail_mask,
        item_popularity=item_popularity,
        item_users=item_users,
    )

    # 3. LightGCN
    lightgcn_path = os.path.join(PROJECT_ROOT, "lightgcn_ml1m_best.pth")
    lg_recs = generate_topk_lightgcn(
        lightgcn_path,
        n_users=n_users,
        n_items=n_items,
        train_u=train_u,
        train_i=train_i,
        train_items_by_user=train_items_by_user,
        eval_users=eval_users,
        device=device,
        K=K,
    )
    print_beyond_metrics(
        model_name="LightGCN",
        recs=lg_recs,
        K=K,
        n_items=n_items,
        tail_mask=tail_mask,
        item_popularity=item_popularity,
        item_users=item_users,
    )

    # 4. NGCF
    ngcf_path = os.path.join(PROJECT_ROOT, "ngcf_ml1m_best.pth")
    ngcf_recs = generate_topk_ngcf(
        ngcf_path,
        n_users=n_users,
        n_items=n_items,
        train_u=train_u,
        train_i=train_i,
        train_items_by_user=train_items_by_user,
        eval_users=eval_users,
        device=device,
        K=K,
    )
    print_beyond_metrics(
        model_name="NGCF",
        recs=ngcf_recs,
        K=K,
        n_items=n_items,
        tail_mask=tail_mask,
        item_popularity=item_popularity,
        item_users=item_users,
    )

    # 5. Multi-VAE
    multivae_path = os.path.join(PROJECT_ROOT, "multivae_ml1m_best.pth")
    multivae_recs = generate_topk_multivae(
        multivae_path,
        n_users=n_users,
        n_items=n_items,
        train_df=train_df,
        val_df=val_df,
        train_items_by_user=train_items_by_user,
        eval_users=eval_users,
        device=device,
        K=K,
    )
    print_beyond_metrics(
        model_name="Multi-VAE",
        recs=multivae_recs,
        K=K,
        n_items=n_items,
        tail_mask=tail_mask,
        item_popularity=item_popularity,
        item_users=item_users,
    )

    # 6. DIN
    # ========= DIN =========
    din_ckpt_path = os.path.join(PROJECT_ROOT, "din_ml1m_best.pth")
    if os.path.exists(din_ckpt_path) and HAS_DIN:
        print("\n========== Preparing DIN for Top-K ==========")
        din_recs = generate_topk_din(
            ckpt_path=din_ckpt_path,
            n_users=n_users,
            n_items=n_items,
            train_df=train_df,
            val_df=val_df,
            train_items_by_user=train_items_by_user,
            test_users=eval_users,
            K=K,
            device=device,
        )

        print("\nModel: DIN")
        eval_and_print_beyond_metrics(
            recs_by_user=din_recs,
            test_items_by_user=test_items_by_user,
            item_popularity=item_popularity,
            n_items=n_items,
            model_name="DIN",
            K=K,
        )
    else:
        print("[WARN] DIN class not found (din_model.py 缺失？)，跳过 DIN。")

    # 7. MMGCN
    # 7. MMGCN
    print("\n========== Preparing MMGCN for Top-K ==========")
    mmgcn_ckpt_path = os.path.join(PROJECT_ROOT, "mmgcn_ml1m_best.pth")
    mmgcn_recs = {}

    if os.path.exists(mmgcn_ckpt_path) and MMGCN is not None:
        try:
            mmgcn_recs = generate_topk_mmgcn(
                ckpt_path=mmgcn_ckpt_path,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                user2idx=user2idx,
                item2idx=item2idx,
                n_users=n_users,
                n_items=n_items,
                train_items_by_user=train_items_by_user,
                val_items_by_user=val_items_by_user,
                test_items_by_user=test_items_by_user,
                device=device,
                topk=K,
            )

            print_beyond_metrics(
                model_name="MMGCN",
                recs=mmgcn_recs,
                K=K,
                n_items=n_items,
                tail_mask=tail_mask,
                item_popularity=item_popularity,
                item_users=item_users,
            )
        except Exception as e:
            print(f"[WARN] Failed to evaluate MMGCN: {e}")
    else:
        print(f"[WARN] MMGCN checkpoint not found at {mmgcn_ckpt_path} 或 MMGCN 类缺失，跳过 MMGCN。")


if __name__ == "__main__":
    main()
