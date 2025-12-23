# recommender_project/experiments/ml1m/tune_v3_rerank_ml1m.py
import os
import sys
import time
import math
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import yaml
import numpy as np
import pandas as pd
import torch

# =========================================================
# Path bootstrap
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# preprocess
try:
    from utils.preprocess import load_ml1m_ratings, build_id_mappings, split_by_user_time
except Exception:
    from preprocess import load_ml1m_ratings, build_id_mappings, split_by_user_time

# rerank
try:
    from rerank_model import AdaptiveMMRReranker, RerankWeights
except Exception as e:
    raise RuntimeError("Cannot import models/rerank_model.py (AdaptiveMMRReranker, RerankWeights).") from e

# models
try:
    from gnn_model import LightGCN
except Exception:
    LightGCN = None

try:
    from ngcf_model import NGCF
except Exception:
    NGCF = None

try:
    from mmgcn_model import MMGCN
except Exception:
    MMGCN = None

try:
    from hybrid_model import HybridNCF
    from content_model import ItemContentEncoder
except Exception:
    HybridNCF = None
    ItemContentEncoder = None


try:
    from hybrid_acc_model import HybridNCFAcc
except Exception:
    HybridNCFAcc = None

try:
    from hybrid_tail_model import HybridNCFTail
except Exception:
    HybridNCFTail = None

try:
    from svdpp_model import SVDPP
except Exception:
    SVDPP = None

try:
    from vae_model import MultiVAE
except Exception:
    MultiVAE = None

try:
    from din_model import DIN
except Exception:
    DIN = None


# =========================================================
# Config helpers
# =========================================================
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(path: str, cfg: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def _abs_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def ensure_rerank_sections(cfg: dict) -> dict:
    """
    你现有 YAML 使用：
      rerank.weights / rerank.per_model
    同时兼容旧字段：
      rerank.default_weights / rerank.per_model_weights
    """
    cfg.setdefault("rerank", {})
    rr = cfg["rerank"]

    rr.setdefault("weights", {
        "w_rel": 0.92,
        "w_novel": 0.02,
        "w_tail": 0.02,
        "w_div": 0.04,
        "cold_boost": 2.0,
        "hist_ref": 50,
    })
    rr.setdefault("per_model", {})

    # 兼容旧字段（如果其它脚本还在读）
    rr.setdefault("default_weights", dict(rr["weights"]))
    rr.setdefault("per_model_weights", dict(rr["per_model"]))

    rr.setdefault("tune", {})
    return cfg


def get_kn_from_cfg(cfg: dict, default_k=10, default_n=200) -> Tuple[int, int]:
    """
    tune 使用优先级：
      1) tune_v3_rerank.K/topN
      2) rerank.K/N
      3) default
    """
    tcfg = cfg.get("tune_v3_rerank", {}) or {}
    if "K" in tcfg or "topN" in tcfg:
        K = int(tcfg.get("K", default_k))
        topN = int(tcfg.get("topN", default_n))
        return K, topN

    rr = cfg.get("rerank", {}) or {}
    K = int(rr.get("K", default_k))
    topN = int(rr.get("N", default_n))
    return K, topN


def write_back_best_weights(
    cfg_path: str,
    model_tag: str,
    best_w: dict,
    best_val: dict,
    best_test: Optional[dict] = None,
    objective: str = "NDCG",
) -> None:
    cfg = load_config(cfg_path)
    cfg = ensure_rerank_sections(cfg)

    rr = cfg["rerank"]
    rr.setdefault("per_model", {})
    rr.setdefault("per_model_weights", {})

    payload = {
        "w_rel": float(best_w["w_rel"]),
        "w_novel": float(best_w["w_novel"]),
        "w_tail": float(best_w["w_tail"]),
        "w_div": float(best_w["w_div"]),
        "cold_boost": float(best_w.get("cold_boost", rr["weights"].get("cold_boost", 2.0))),
        "hist_ref": int(best_w.get("hist_ref", rr["weights"].get("hist_ref", 50))),
        "tuned_on": "val",
        "objective": objective,
        "val_recall": float(best_val.get("Recall", 0.0)),
        "val_ndcg": float(best_val.get("NDCG", 0.0)),
        "val_coverage": float(best_val.get("Coverage", 0.0)),
        "val_longtail": float(best_val.get("LongTailShare", 0.0)),
        "val_novelty": float(best_val.get("Novelty", 0.0)),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if best_test is not None:
        payload.update({
            "test_recall": float(best_test.get("Recall", 0.0)),
            "test_ndcg": float(best_test.get("NDCG", 0.0)),
            "test_coverage": float(best_test.get("Coverage", 0.0)),
            "test_longtail": float(best_test.get("LongTailShare", 0.0)),
            "test_novelty": float(best_test.get("Novelty", 0.0)),
        })

    # 新字段
    rr["per_model"][model_tag] = payload
    # 同步旧字段
    rr["per_model_weights"][model_tag] = dict(payload)

    save_config(cfg_path, cfg)
    print(f"[Config] Updated rerank.per_model['{model_tag}'] in: {cfg_path}")


# =========================================================
# Torch load helpers
# =========================================================
def torch_load_safe_weights(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def torch_load_full(path: str, map_location: str):
    return torch.load(path, map_location=map_location)


# =========================================================
# Artifact dirs
# =========================================================
def make_artifact_dirs(project_root: str, dataset: str, exp_name: str):
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    art_dir = os.path.join(project_root, "artifacts", dataset, exp_name, run_tag)
    os.makedirs(art_dir, exist_ok=True)
    return art_dir


# =========================================================
# Metrics
# =========================================================
def build_user_pos_dict(user_idx: np.ndarray, item_idx: np.ndarray, n_users: int) -> Dict[int, set]:
    user_pos = {u: set() for u in range(n_users)}
    for u, i in zip(user_idx, item_idx):
        user_pos[int(u)].add(int(i))
    return user_pos


def build_long_tail_mask(item_popularity: np.ndarray, head_ratio: float = 0.8) -> np.ndarray:
    n_items = item_popularity.shape[0]
    idx_sorted = np.argsort(item_popularity)[::-1]
    pop_sorted = item_popularity[idx_sorted]
    total = pop_sorted.sum()
    if total <= 0:
        return np.ones(n_items, dtype=bool)
    cumsum = np.cumsum(pop_sorted)
    head_cut = np.searchsorted(cumsum, head_ratio * total)
    head_cut = min(head_cut, n_items - 1)
    head_items = idx_sorted[: head_cut + 1]
    tail_mask = np.ones(n_items, dtype=bool)
    tail_mask[head_items] = False
    return tail_mask


def ndcg_at_k(rank_list: List[int], gt_set: set, k: int) -> float:
    if not gt_set:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(rank_list[:k], start=1):
        if item in gt_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = min(len(gt_set), k)
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics_topk(
    recs: Dict[int, List[int]],
    user_pos: Dict[int, set],
    n_items: int,
    item_popularity: np.ndarray,
    tail_mask: np.ndarray,
    K: int,
    item_vectors: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    users = list(user_pos.keys())
    if len(users) == 0:
        return {
            "Recall": 0.0,
            "NDCG": 0.0,
            "Coverage": 0.0,
            "LongTailShare": 0.0,
            "Novelty": 0.0,
            "ILAD": 0.0,
        }

    hits = 0
    total = 0
    ndcg_sum = 0.0

    all_items = set()
    total_events = float(item_popularity.sum())
    probs = item_popularity.astype("float64") / total_events if total_events > 0 else np.zeros_like(item_popularity, dtype="float64")
    probs[probs <= 0] = 1e-12
    info = -np.log2(probs)

    lt_cnt, rec_cnt = 0, 0
    nov_sum, nov_cnt = 0.0, 0
    ilad_sum, ilad_users = 0.0, 0

    vec = None
    if item_vectors is not None and item_vectors.size > 0:
        vec = item_vectors

    for u in users:
        gt = user_pos[u]
        r = recs.get(u, [])
        if not r:
            continue
        rK = r[:K]
        all_items.update(rK)

        hit_u = sum(1 for it in gt if it in set(rK))
        hits += hit_u
        total += len(gt)
        ndcg_sum += ndcg_at_k(rK, gt, K)

        for it in rK:
            rec_cnt += 1
            if 0 <= it < n_items and tail_mask[it]:
                lt_cnt += 1
            if 0 <= it < n_items:
                nov_sum += float(info[it])
                nov_cnt += 1

        if vec is not None and len(rK) > 1:
            emb = vec[rK]
            sim = emb @ emb.T
            m = sim.shape[0]
            triu_idx = np.triu_indices(m, k=1)
            pair_sim = sim[triu_idx]
            pair_div = 1.0 - pair_sim
            ilad_sum += float(pair_div.mean())
            ilad_users += 1

    recall = hits / total if total > 0 else 0.0
    ndcg = ndcg_sum / float(len(users)) if len(users) > 0 else 0.0
    coverage = len(all_items) / float(n_items) if n_items > 0 else 0.0
    longtail = lt_cnt / float(rec_cnt) if rec_cnt > 0 else 0.0
    novelty = nov_sum / float(nov_cnt) if nov_cnt > 0 else 0.0

    ilad = ilad_sum / float(ilad_users) if ilad_users > 0 else 0.0

    return {
        "Recall": recall,
        "NDCG": ndcg,
        "Coverage": coverage,
        "LongTailShare": longtail,
        "Novelty": novelty,
        "ILAD": ilad,
    }


# =========================================================
# Adjacency + item vectors
# =========================================================
def build_normalized_adj(num_users: int, num_items: int, user_idx: np.ndarray, item_idx: np.ndarray, device) -> torch.Tensor:
    num_nodes = num_users + num_items
    user_nodes = np.array(user_idx, dtype=np.int64)
    item_nodes = np.array(item_idx, dtype=np.int64) + num_users

    rows = np.concatenate([user_nodes, item_nodes], axis=0)
    cols = np.concatenate([item_nodes, user_nodes], axis=0)
    vals = np.ones(len(rows), dtype=np.float32)

    degree = np.bincount(rows, minlength=num_nodes).astype(np.float32)
    degree[degree == 0.0] = 1.0
    norm_vals = vals / np.sqrt(degree[rows] * degree[cols])

    idx = np.vstack([rows, cols])
    adj = torch.sparse_coo_tensor(
        torch.from_numpy(idx).long(),
        torch.from_numpy(norm_vals).float(),
        torch.Size([num_nodes, num_nodes]),
    )
    return adj.coalesce().to(device)


def load_item_vectors_aligned_ml1m(cfg: dict, n_items: int, item2idx: Dict[int, int]) -> np.ndarray:
    paths = cfg.get("paths", {}) or {}
    feat_dir = paths.get("features_dir", "data/features")

    text_feat_path = _abs_path(paths.get("ml1m_text_emb", os.path.join(feat_dir, "ml1m_text_embeddings_64.npy")))
    text_index_path = _abs_path(paths.get("ml1m_text_index", os.path.join(feat_dir, "ml1m_text_index.csv")))
    img_feat_path = _abs_path(paths.get("ml1m_image_emb", os.path.join(feat_dir, "ml1m_image_embeddings_64.npy")))
    img_index_path = _abs_path(paths.get("ml1m_image_index", os.path.join(feat_dir, "ml1m_image_index.csv")))

    def _load_one(feat_path: str, index_path: str) -> np.ndarray:
        feats = np.load(feat_path).astype(np.float32)
        df = pd.read_csv(index_path)

        if "movieId" in df.columns:
            key_col = "movieId"
        elif "movie_id" in df.columns:
            key_col = "movie_id"
        elif "item_id" in df.columns:
            key_col = "item_id"
        else:
            key_col = df.columns[0]

        use_index_col = "index" in df.columns
        out = np.zeros((n_items, feats.shape[1]), dtype=np.float32)

        for ridx, row in df.iterrows():
            mid = int(row[key_col])
            if mid not in item2idx:
                continue
            fidx = int(row["index"]) if use_index_col else ridx
            if 0 <= fidx < feats.shape[0]:
                out[item2idx[mid]] = feats[fidx]
        return out

    text = _load_one(text_feat_path, text_index_path)
    img = _load_one(img_feat_path, img_index_path)
    return np.concatenate([text, img], axis=1)


# =========================================================
# Data prep (val tuning)
# =========================================================
def prepare_ml1m_data(device: str):
    cfg_path = os.path.join(UTILS_DIR, "config.yaml")
    cfg = load_config(cfg_path)
    cfg = ensure_rerank_sections(cfg)

    ratings_path = cfg["data"]["ml1m_ratings"]
    ratings_path = _abs_path(ratings_path)

    ratings_df = load_ml1m_ratings(ratings_path)
    user2idx, item2idx, idx2user, idx2item = build_id_mappings(ratings_df)

    train_df, val_df, test_df = split_by_user_time(ratings_df)

    for df in [train_df, val_df, test_df]:
        df["user_idx"] = df["userId"].map(user2idx).astype("int64")
        df["item_idx"] = df["movieId"].map(item2idx).astype("int64")

    n_users = len(user2idx)
    n_items = len(item2idx)

    train_user_pos = build_user_pos_dict(train_df["user_idx"].values, train_df["item_idx"].values, n_users)
    val_user_pos = build_user_pos_dict(val_df["user_idx"].values, val_df["item_idx"].values, n_users)
    test_user_pos = build_user_pos_dict(test_df["user_idx"].values, test_df["item_idx"].values, n_users)

    # seen：train（用于 val 评估）与 train+val（用于 test 评估）
    seen_train = {u: set() for u in range(n_users)}
    for u, i in zip(train_df["user_idx"].values, train_df["item_idx"].values):
        seen_train[int(u)].add(int(i))

    seen_trainval = {u: set(seen_train[u]) for u in range(n_users)}
    for u, i in zip(val_df["user_idx"].values, val_df["item_idx"].values):
        seen_trainval[int(u)].add(int(i))

    # item popularity 用 train+val（更贴近最终推荐环境）
    item_pop = np.zeros(n_items, dtype=np.int64)
    for u in range(n_users):
        for it in seen_trainval[u]:
            item_pop[it] += 1

    tail_mask = build_long_tail_mask(item_pop, head_ratio=0.8)
    global_mean = float(train_df["rating"].mean()) if "rating" in train_df.columns else 0.0

    item_vectors = load_item_vectors_aligned_ml1m(cfg, n_items=n_items, item2idx=item2idx)
    norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    item_vectors_norm = item_vectors / norms

    return {
        "cfg_path": cfg_path,
        "cfg": cfg,
        "ratings_path": ratings_path,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2item": idx2item,
        "n_users": n_users,
        "n_items": n_items,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_user_pos": train_user_pos,
        "val_user_pos": val_user_pos,
        "test_user_pos": test_user_pos,
        "seen_train": seen_train,
        "seen_trainval": seen_trainval,
        "item_popularity": item_pop,
        "tail_mask": tail_mask,
        "global_mean": global_mean,
        "train_u": train_df["user_idx"].values,
        "train_i": train_df["item_idx"].values,
        "device": device,
        "item_vectors": item_vectors,
        "item_vectors_norm": item_vectors_norm,
    }


# =========================================================
# Model loaders (与 evaluate 同风格)
# =========================================================
def load_lightgcn(path: str, adj: torch.Tensor, device: str):
    ckpt = torch_load_safe_weights(path, map_location=device)
    model = LightGCN(
        num_users=ckpt["n_users"],
        num_items=ckpt["n_items"],
        embedding_dim=ckpt["embedding_dim"],
        num_layers=ckpt["num_layers"],
        adj=adj,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_ngcf(path: str, adj: torch.Tensor, device: str):
    ckpt = torch_load_safe_weights(path, map_location=device)
    emb = ckpt.get("embedding_dim", 64)
    layers = ckpt.get("num_layers", 3)
    model = NGCF(
        num_users=ckpt["n_users"],
        num_items=ckpt["n_items"],
        embedding_dim=emb,
        num_layers=layers,
        adj=adj,
    ).to(device)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_mmgcn(path: str, cfg: dict, meta: dict):
    ckpt = torch_load_safe_weights(path, map_location=meta["device"])
    n_users = meta["n_users"]
    n_items = meta["n_items"]
    device = meta["device"]

    adj = build_normalized_adj(n_users, n_items, meta["train_u"], meta["train_i"], device=device)
    adj_id = adj_text = adj_image = adj

    paths = cfg.get("paths", {}) or {}
    feat_dir = _abs_path(paths.get("features_dir", "data/features"))
    text_feat_path = _abs_path(paths.get("ml1m_text_emb", os.path.join(feat_dir, "ml1m_text_embeddings_64.npy")))
    img_feat_path = _abs_path(paths.get("ml1m_image_emb", os.path.join(feat_dir, "ml1m_image_embeddings_64.npy")))

    text_feat = np.load(text_feat_path).astype(np.float32)
    img_feat = np.load(img_feat_path).astype(np.float32)

    if text_feat.shape[0] != n_items:
        text_feat = text_feat[:n_items]
    if img_feat.shape[0] != n_items:
        pad = np.zeros((n_items, img_feat.shape[1]), dtype=np.float32)
        L = min(n_items, img_feat.shape[0])
        pad[:L] = img_feat[:L]
        img_feat = pad

    item_text = torch.from_numpy(text_feat).float().to(device)
    item_img = torch.from_numpy(img_feat).float().to(device)

    model = MMGCN(
        num_users=n_users,
        num_items=n_items,
        embedding_dim=ckpt.get("embedding_dim", 64),
        num_layers=ckpt.get("num_layers", 3),
        adj_id=adj_id,
        adj_text=adj_text,
        adj_image=adj_image,
        item_text_feats=item_text,
        item_image_feats=item_img,
        content_hidden_dim=64,
        dropout=0.0,
    ).to(device)

    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_hybrid_robust(path: str, cfg: dict, ratings_path: str, device: str):
    ckpt = torch_load_full(path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state", ckpt))

    if isinstance(ckpt, dict) and "n_users" in ckpt and "n_items" in ckpt:
        n_users = int(ckpt["n_users"])
        n_items = int(ckpt["n_items"])
    else:
        n_users = int(state["user_embedding_gmf.weight"].shape[0])
        n_items = int(state["item_embedding_gmf.weight"].shape[0])

    gmf_dim = int(state["user_embedding_gmf.weight"].shape[1])
    mlp_dim = int(state["user_embedding_mlp.weight"].shape[1])

    if "text_proj.weight" in state:
        content_proj_dim = int(state["text_proj.weight"].shape[0])
    elif "image_proj.weight" in state:
        content_proj_dim = int(state["image_proj.weight"].shape[0])
    else:
        content_proj_dim = mlp_dim

    ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    hycfg = {}
    if isinstance(ckpt_cfg, dict) and isinstance(ckpt_cfg.get("hybrid", None), dict):
        hycfg = ckpt_cfg["hybrid"]
    else:
        hycfg = cfg.get("hybrid", {}) or {}

    dropout = float(hycfg.get("dropout", 0.10))
    use_history = bool(hycfg.get("use_history", True))
    max_hist_len = int(hycfg.get("max_hist_len", 50))
    n_heads = int(hycfg.get("n_heads", 4))
    n_transformer_layers = int(hycfg.get("n_transformer_layers", 2))
    rating_min = float(hycfg.get("rating_min", 1.0))
    rating_max = float(hycfg.get("rating_max", 5.0))
    global_mean = float(hycfg.get("global_mean", 0.0))
    mlp_layer_sizes = hycfg.get("mlp_layer_sizes", (512, 256, 128))
    if isinstance(mlp_layer_sizes, list):
        mlp_layer_sizes = tuple(mlp_layer_sizes)

    paths = cfg.get("paths", {}) or {}
    feat_dir = _abs_path(paths.get("features_dir", "data/features"))
    text_feat_path = _abs_path(paths.get("ml1m_text_emb", os.path.join(feat_dir, "ml1m_text_embeddings_64.npy")))
    text_index_path = _abs_path(paths.get("ml1m_text_index", os.path.join(feat_dir, "ml1m_text_index.csv")))
    img_feat_path = _abs_path(paths.get("ml1m_image_emb", os.path.join(feat_dir, "ml1m_image_embeddings_64.npy")))
    img_index_path = _abs_path(paths.get("ml1m_image_index", os.path.join(feat_dir, "ml1m_image_index.csv")))

    content_encoder = ItemContentEncoder(
        ratings_path=ratings_path,
        text_feat_path=text_feat_path,
        text_index_path=text_index_path,
        image_feat_path=img_feat_path,
        image_index_path=img_index_path,
        use_text=True,
        use_image=True,
    )

    model = HybridNCF(
        num_users=n_users,
        num_items=n_items,
        content_encoder=content_encoder,
        gmf_dim=gmf_dim,
        mlp_dim=mlp_dim,
        content_proj_dim=content_proj_dim,
        mlp_layer_sizes=mlp_layer_sizes,
        dropout=dropout,
        use_history=use_history,
        max_hist_len=max_hist_len,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
        rating_min=rating_min,
        rating_max=rating_max,
        global_mean=global_mean,
    ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_hybrid_acc_robust(path: str, cfg: dict, ratings_path: str, device: str):
    if HybridNCFAcc is None:
        raise RuntimeError("HybridNCFAcc not importable.")

    ckpt = torch_load_full(path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state", ckpt))

    if isinstance(ckpt, dict) and "n_users" in ckpt and "n_items" in ckpt:
        n_users = int(ckpt["n_users"])
        n_items = int(ckpt["n_items"])
    else:
        n_users = int(state["user_embedding_gmf.weight"].shape[0])
        n_items = int(state["item_embedding_gmf.weight"].shape[0])

    gmf_dim = int(state["user_embedding_gmf.weight"].shape[1])
    mlp_dim = int(state["user_embedding_mlp.weight"].shape[1])

    if "text_proj.weight" in state:
        content_proj_dim = int(state["text_proj.weight"].shape[0])
    elif "image_proj.weight" in state:
        content_proj_dim = int(state["image_proj.weight"].shape[0])
    else:
        content_proj_dim = mlp_dim

    hycfg = cfg.get("hybrid_acc", cfg.get("hybrid", {})) if isinstance(cfg, dict) else {}

    dropout = float(hycfg.get("dropout", 0.10))
    use_text = bool(hycfg.get("use_text", True))
    use_image = bool(hycfg.get("use_image", True))
    use_history = bool(hycfg.get("use_history", True))
    max_hist_len = int(hycfg.get("max_hist_len", 50))
    n_heads = int(hycfg.get("n_heads", 4))
    n_transformer_layers = int(hycfg.get("n_transformer_layers", 2))
    rating_min = float(hycfg.get("rating_min", 1.0))
    rating_max = float(hycfg.get("rating_max", 5.0))
    global_mean = float(ckpt.get("global_mean", 0.0))

    mlp_layer_sizes = hycfg.get("mlp_layer_sizes", (512, 256, 128))
    if isinstance(mlp_layer_sizes, list):
        mlp_layer_sizes = tuple(mlp_layer_sizes)

    feat_dir = os.path.join(PROJECT_ROOT, "data", "features")
    text_feat_path = os.path.join(feat_dir, "ml1m_text_embeddings_64.npy")
    text_index_path = os.path.join(feat_dir, "ml1m_text_index.csv")
    img_feat_path = os.path.join(feat_dir, "ml1m_image_embeddings_64.npy")
    img_index_path = os.path.join(feat_dir, "ml1m_image_index.csv")

    content_encoder = ItemContentEncoder(
        ratings_path=ratings_path,
        text_feat_path=text_feat_path,
        text_index_path=text_index_path,
        image_feat_path=img_feat_path,
        image_index_path=img_index_path,
        use_text=use_text,
        use_image=use_image,
    )

    model = HybridNCFAcc(
        num_users=n_users,
        num_items=n_items,
        content_encoder=content_encoder,
        gmf_dim=gmf_dim,
        mlp_dim=mlp_dim,
        content_proj_dim=content_proj_dim,
        mlp_layer_sizes=mlp_layer_sizes,
        dropout=dropout,
        use_history=use_history,
        max_hist_len=max_hist_len,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
        rating_min=rating_min,
        rating_max=rating_max,
        global_mean=global_mean,
    ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_hybrid_tail_robust(path: str, cfg: dict, ratings_path: str, device: str, item_popularity: Optional[np.ndarray] = None):
    if HybridNCFTail is None:
        raise RuntimeError("HybridNCFTail not importable.")

    ckpt = torch_load_full(path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state", ckpt))

    if isinstance(ckpt, dict) and "n_users" in ckpt and "n_items" in ckpt:
        n_users = int(ckpt["n_users"])
        n_items = int(ckpt["n_items"])
    else:
        n_users = int(state["user_embedding_gmf.weight"].shape[0])
        n_items = int(state["item_embedding_gmf.weight"].shape[0])

    gmf_dim = int(state["user_embedding_gmf.weight"].shape[1])
    mlp_dim = int(state["user_embedding_mlp.weight"].shape[1])

    if "text_proj.weight" in state:
        content_proj_dim = int(state["text_proj.weight"].shape[0])
    elif "image_proj.weight" in state:
        content_proj_dim = int(state["image_proj.weight"].shape[0])
    else:
        content_proj_dim = mlp_dim

    hycfg = cfg.get("hybrid_tail", cfg.get("hybrid", {})) if isinstance(cfg, dict) else {}

    dropout = float(hycfg.get("dropout", 0.10))
    use_text = bool(hycfg.get("use_text", True))
    use_image = bool(hycfg.get("use_image", True))
    use_history = bool(hycfg.get("use_history", True))
    max_hist_len = int(hycfg.get("max_hist_len", 50))
    n_heads = int(hycfg.get("n_heads", 4))
    n_transformer_layers = int(hycfg.get("n_transformer_layers", 2))
    rating_min = float(hycfg.get("rating_min", 1.0))
    rating_max = float(hycfg.get("rating_max", 5.0))
    global_mean = float(ckpt.get("global_mean", 0.0))

    pop_alpha = float(ckpt.get("pop_alpha", hycfg.get("pop_alpha", 0.30)))
    pop_mode = str(ckpt.get("pop_mode", hycfg.get("pop_mode", "log_norm")))
    learnable_pop_alpha = bool(ckpt.get("learnable_pop_alpha", hycfg.get("learnable_pop_alpha", False)))
    user_pop_scaling = bool(ckpt.get("user_pop_scaling", hycfg.get("user_pop_scaling", False)))
    user_pop_scale_range = ckpt.get("user_pop_scale_range", hycfg.get("user_pop_scale_range", (0.5, 1.5)))
    if not isinstance(user_pop_scale_range, (list, tuple)) or len(user_pop_scale_range) != 2:
        user_pop_scale_range = (0.5, 1.5)

    mlp_layer_sizes = hycfg.get("mlp_layer_sizes", (512, 256, 128))
    if isinstance(mlp_layer_sizes, list):
        mlp_layer_sizes = tuple(mlp_layer_sizes)

    feat_dir = os.path.join(PROJECT_ROOT, "data", "features")
    text_feat_path = os.path.join(feat_dir, "ml1m_text_embeddings_64.npy")
    text_index_path = os.path.join(feat_dir, "ml1m_text_index.csv")
    img_feat_path = os.path.join(feat_dir, "ml1m_image_embeddings_64.npy")
    img_index_path = os.path.join(feat_dir, "ml1m_image_index.csv")

    content_encoder = ItemContentEncoder(
        ratings_path=ratings_path,
        text_feat_path=text_feat_path,
        text_index_path=text_index_path,
        image_feat_path=img_feat_path,
        image_index_path=img_index_path,
        use_text=use_text,
        use_image=use_image,
    )

    if item_popularity is None and isinstance(ckpt, dict) and "item_pop_train" in ckpt:
        item_popularity = np.array(ckpt["item_pop_train"], dtype=np.float32)

    model = HybridNCFTail(
        num_users=n_users,
        num_items=n_items,
        content_encoder=content_encoder,
        gmf_dim=gmf_dim,
        mlp_dim=mlp_dim,
        content_proj_dim=content_proj_dim,
        mlp_layer_sizes=mlp_layer_sizes,
        dropout=dropout,
        use_history=use_history,
        max_hist_len=max_hist_len,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
        rating_min=rating_min,
        rating_max=rating_max,
        global_mean=global_mean,
        item_popularity=item_popularity,
        pop_alpha=pop_alpha,
        pop_mode=pop_mode,
        learnable_pop_alpha=learnable_pop_alpha,
        user_pop_scaling=user_pop_scaling,
        user_pop_scale_range=tuple(float(x) for x in user_pop_scale_range),
    ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_svdpp(path: str, user_interactions: Dict[int, set], global_mean: float, device: str):
    ckpt = torch_load_safe_weights(path, map_location=device)
    emb_dim = int(ckpt.get("embedding_dim", 64))
    model = SVDPP(
        num_users=int(ckpt["n_users"]),
        num_items=int(ckpt["n_items"]),
        embedding_dim=emb_dim,
        user_interactions=user_interactions,
        global_mean=global_mean,
    ).to(device)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_multivae(path: str, n_items: int, device: str):
    ckpt = torch_load_safe_weights(path, map_location=device)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))

    if "p_dims" in ckpt:
        p_dims = list(ckpt["p_dims"])
    elif "config" in ckpt and isinstance(ckpt["config"], dict):
        p_dims = ckpt["config"].get("model", {}).get("p_dims", [200, 600, n_items])
    else:
        p_dims = [200, 600, n_items]
    p_dims = list(p_dims)
    p_dims[-1] = n_items

    dropout = 0.5
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        dropout = float(ckpt["config"].get("model", {}).get("dropout", dropout))

    model = MultiVAE(p_dims=p_dims, dropout=dropout).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _extract_state_dict(ckpt: dict) -> dict:
    for k in ["state_dict", "model_state_dict", "model", "model_state"]:
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt


def _infer_mlp_hidden_sizes(state: dict, prefix: str):
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    pairs = []
    for k, v in state.items():
        m = pat.match(k)
        if m and isinstance(v, torch.Tensor) and v.dim() == 2:
            idx = int(m.group(1))
            out_dim = int(v.shape[0])
            pairs.append((idx, out_dim))
    pairs.sort(key=lambda x: x[0])
    return tuple(out for _, out in pairs)


def load_din(ckpt_path: str, device: str = "cpu"):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    state = _extract_state_dict(ckpt)

    embed_dim = int(state["user_embedding.weight"].shape[1])
    num_users_ckpt = int(state["user_embedding.weight"].shape[0])
    num_items_ckpt = int(state["item_embedding.weight"].shape[0])

    fc_hidden_sizes = _infer_mlp_hidden_sizes(state, "fc.mlp")
    att_hidden_sizes = _infer_mlp_hidden_sizes(state, "attention.mlp")

    use_dice = any(k.startswith("fc.mlp") and k.endswith(".alpha") for k in state.keys())

    from din_model import DIN as DIN_CLASS
    model = DIN_CLASS(
        num_users=num_users_ckpt,
        num_items=num_items_ckpt,
        embed_dim=embed_dim,
        att_hidden_sizes=att_hidden_sizes if len(att_hidden_sizes) > 0 else (64, 32, 16),
        fc_hidden_sizes=fc_hidden_sizes if len(fc_hidden_sizes) > 0 else (64, 32),
        max_history_len=ckpt.get("max_history_len", 50) if isinstance(ckpt, dict) else getattr(ckpt, "max_history_len", 50),
        dropout=0.0,
        use_dice=use_dice,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def infer_din_item_offset(model: "DIN", n_items: int) -> int:
    try:
        if int(model.item_embedding.num_embeddings) == n_items + 1:
            return 1
    except Exception:
        pass
    return 0


def build_user_hist_sequences(trainval_df, n_users: int, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if "timestamp" not in trainval_df.columns:
        trainval_df = trainval_df.copy()
        trainval_df["timestamp"] = np.arange(len(trainval_df), dtype=np.int64)
    grp = trainval_df.sort_values(["user_idx", "timestamp"]).groupby("user_idx")["item_idx"].apply(list)

    hist_items = np.zeros((n_users, max_len), dtype=np.int64)
    hist_lens = np.zeros((n_users,), dtype=np.int64)
    for u in range(n_users):
        seq = grp.get(u, [])
        if not seq:
            continue
        seq = seq[-max_len:]
        L = len(seq)
        hist_lens[u] = L
        hist_items[u, :L] = np.array(seq, dtype=np.int64)
    return hist_items, hist_lens


# =========================================================
# Candidate precompute
# =========================================================
@torch.no_grad()
def precompute_topn_for_embedding_model(
    model,
    seen: Dict[int, set],
    eval_users: List[int],
    topN: int,
    device: str,
    model_name: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    t0 = time.time()
    user_emb, item_emb = model.get_user_item_embeddings()
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    topn_items_by_user = {}
    topn_scores_by_user = {}

    for idx, u in enumerate(eval_users, start=1):
        u_e = user_emb[u]
        scores = torch.matmul(item_emb, u_e)
        s = seen.get(u, set())
        if s:
            scores[list(s)] = -1e9
        vals, inds = torch.topk(scores, topN)
        topn_items_by_user[u] = inds.detach().cpu().numpy().astype(np.int64)
        topn_scores_by_user[u] = vals.detach().cpu().numpy().astype(np.float32)
        if idx % 500 == 0:
            print(f"[{model_name}-TopN] {idx}/{len(eval_users)} done, {time.time()-t0:.1f}s")
    return topn_items_by_user, topn_scores_by_user


@torch.no_grad()
def precompute_topn_for_forward_model(
    model,
    seen: Dict[int, set],
    eval_users: List[int],
    n_items: int,
    topN: int,
    device: str,
    model_name: str,
    chunk: int = 1024,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    t0 = time.time()
    all_items = torch.arange(n_items, dtype=torch.long, device=device)
    topn_items_by_user = {}
    topn_scores_by_user = {}

    for idx, u in enumerate(eval_users, start=1):
        user = torch.full((n_items,), int(u), dtype=torch.long, device=device)
        scores = torch.empty((n_items,), dtype=torch.float32, device=device)
        for st in range(0, n_items, chunk):
            ed = min(st + chunk, n_items)
            scores[st:ed] = model(user[st:ed], all_items[st:ed]).view(-1)

        s = seen.get(u, set())
        if s:
            scores[list(s)] = -1e9

        vals, inds = torch.topk(scores, topN)
        topn_items_by_user[u] = inds.detach().cpu().numpy().astype(np.int64)
        topn_scores_by_user[u] = vals.detach().cpu().numpy().astype(np.float32)

        if idx % 500 == 0:
            print(f"[{model_name}-TopN] {idx}/{len(eval_users)} done, {time.time()-t0:.1f}s")

    return topn_items_by_user, topn_scores_by_user


@torch.no_grad()
def precompute_topn_for_multivae(
    model: "MultiVAE",
    seen: Dict[int, set],
    eval_users: List[int],
    n_items: int,
    topN: int,
    device: str,
    model_name: str,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    t0 = time.time()
    topn_items_by_user = {}
    topn_scores_by_user = {}

    for idx, u in enumerate(eval_users, start=1):
        x = torch.zeros((1, n_items), dtype=torch.float32, device=device)
        s = seen.get(u, set())
        if s:
            x[0, list(s)] = 1.0
        logits, *_ = model(x)
        scores = logits.view(-1)
        if s:
            scores[list(s)] = -1e9

        vals, inds = torch.topk(scores, topN)
        topn_items_by_user[u] = inds.detach().cpu().numpy().astype(np.int64)
        topn_scores_by_user[u] = vals.detach().cpu().numpy().astype(np.float32)

        if idx % 500 == 0:
            print(f"[{model_name}-TopN] {idx}/{len(eval_users)} done, {time.time()-t0:.1f}s")

    return topn_items_by_user, topn_scores_by_user


@torch.no_grad()
def precompute_topn_for_din(
    model: "DIN",
    seen: Dict[int, set],
    eval_users: List[int],
    n_items: int,
    topN: int,
    device: str,
    model_name: str,
    hist_items: np.ndarray,
    hist_lens: np.ndarray,
    item_offset: int,
    chunk: int = 1024,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    t0 = time.time()
    all_items = torch.arange(n_items, dtype=torch.long, device=device)
    topn_items_by_user = {}
    topn_scores_by_user = {}

    for idx, u in enumerate(eval_users, start=1):
        hi = hist_items[u].copy()
        hl = int(hist_lens[u])
        if item_offset == 1:
            hi = np.where(hi > 0, hi + 1, hi)

        hist_t = torch.tensor(hi, dtype=torch.long, device=device).unsqueeze(0)
        lens_t = torch.tensor([hl], dtype=torch.long, device=device)
        user_t = torch.tensor([u], dtype=torch.long, device=device)

        scores = torch.empty((n_items,), dtype=torch.float32, device=device)
        for st in range(0, n_items, chunk):
            ed = min(st + chunk, n_items)
            cand = all_items[st:ed]
            cand_in = cand + 1 if item_offset == 1 else cand
            uu = user_t.expand(ed - st)
            hh = hist_t.expand(ed - st, -1)
            ll = lens_t.expand(ed - st)
            s = model(uu, hh, ll, cand_in).view(-1)
            scores[st:ed] = s

        sset = seen.get(u, set())
        if sset:
            scores[list(sset)] = -1e9

        vals, inds = torch.topk(scores, topN)
        topn_items_by_user[u] = inds.detach().cpu().numpy().astype(np.int64)
        topn_scores_by_user[u] = vals.detach().cpu().numpy().astype(np.float32)

        if idx % 500 == 0:
            print(f"[{model_name}-TopN] {idx}/{len(eval_users)} done, {time.time()-t0:.1f}s")

    return topn_items_by_user, topn_scores_by_user


# =========================================================
# Tuning core (方案1：单纯形搜索)
# =========================================================
def rerank_all_users(
    reranker: "AdaptiveMMRReranker",
    eval_users: List[int],
    seen: Dict[int, set],
    topn_items_by_user: Dict[int, np.ndarray],
    topn_scores_by_user: Dict[int, np.ndarray],
    K: int,
    weights: "RerankWeights",
) -> Dict[int, List[int]]:
    recs = {}
    for u in eval_users:
        cand_items = topn_items_by_user[u].tolist()
        cand_scores = topn_scores_by_user[u].tolist()
        hist_len = len(seen.get(u, set()))
        recs[u] = reranker.rerank_user(
            user_id=int(u),
            cand_items=cand_items,
            cand_scores=cand_scores,
            seen_items=seen.get(u, set()),
            K=K,
            hist_len=hist_len,
            weights=weights,
        )
    return recs


def select_best_by_val(rows: List[dict]) -> dict:
    """
    目标：最大化 val_NDCG，其次 val_Recall，再次覆盖与多样性（ILAD）
    """
    if not rows:
        raise ValueError("No tuning rows to select from.")

    def key_fn(r: dict):
        return (
            float(r.get("val_NDCG", 0.0)),
            float(r.get("val_Recall", 0.0)),
            float(r.get("val_Coverage", 0.0)),
            float(r.get("val_ILAD", 0.0)),
        )

    return max(rows, key=key_fn)


# =========================================================
# Main
# =========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = prepare_ml1m_data(device=device)

    cfg = meta["cfg"]
    cfg_path = meta["cfg_path"]

    n_users = meta["n_users"]
    n_items = meta["n_items"]

    item_pop = meta["item_popularity"]
    tail_mask = meta["tail_mask"]

    # K/topN from config
    K, topN = get_kn_from_cfg(cfg, default_k=10, default_n=200)
    print(f"[Config] Tune: K={K}, topN={topN}")

    ART_DIR = make_artifact_dirs(PROJECT_ROOT, dataset="ml1m", exp_name="tune_v3_rerank_ml1m")
    print(f"[Artifacts] ART_DIR = {ART_DIR}")

    # rerank item vectors
    item_vec = load_item_vectors_aligned_ml1m(cfg, n_items=n_items, item2idx=meta["item2idx"])

    reranker = AdaptiveMMRReranker(
        item_popularity=item_pop,
        tail_mask=tail_mask,
        item_vectors=item_vec,
        device=device,
    )

    # ---------- grid from config (方案1) ----------
    tune_cfg = cfg.get("tune_v3_rerank", {}) or {}
    w_rel_list = tune_cfg.get("w_rel_list", [0.70, 0.80, 0.85, 0.90, 0.95])
    w_div_list = tune_cfg.get("w_div_list", [0.00, 0.03, 0.05, 0.08])
    w_nov_list = tune_cfg.get("w_nov_list", [0.00, 0.03, 0.05, 0.08])

    # cold_boost / hist_ref 从 rerank.weights 取
    default_w = (cfg.get("rerank", {}) or {}).get("weights", {}) or {}
    cold_boost = float(default_w.get("cold_boost", 2.0))
    hist_ref = int(default_w.get("hist_ref", 50))

    # ---------- eval users ----------
    val_users = sorted(list(meta["val_user_pos"].keys()))
    test_users = sorted(list(meta["test_user_pos"].keys()))
    print(f"n_users={n_users}, n_items={n_items}, val_users={len(val_users)}, test_users={len(test_users)}")

    # ---------- helper: evaluate base topK ----------
    def eval_base_topk_from_topn(topn_items_by_user: Dict[int, np.ndarray], users: List[int], user_pos: Dict[int, set]) -> Dict[str, float]:
        recs = {}
        for u in users:
            recs[u] = topn_items_by_user[u][:K].tolist()
        return compute_metrics_topk(
            recs,
            user_pos=user_pos,
            n_items=n_items,
            item_popularity=item_pop,
            tail_mask=tail_mask,
            K=K,
            item_vectors=meta["item_vectors_norm"],
        )

    # ---------- tune one model ----------
    def tune_one_model(
        model_tag: str,
        topn_items_val: Dict[int, np.ndarray],
        topn_scores_val: Dict[int, np.ndarray],
        topn_items_test: Dict[int, np.ndarray],
        topn_scores_test: Dict[int, np.ndarray],
        seen_for_val: Dict[int, set],
        seen_for_test: Dict[int, set],
    ):
        print(f"\n========== Tuning {model_tag} (Unified AdaptiveMMR / Simplex Grid) ==========")

        base_val = eval_base_topk_from_topn(topn_items_val, val_users, meta["val_user_pos"])
        base_test = eval_base_topk_from_topn(topn_items_test, test_users, meta["test_user_pos"])
        print(f"[{model_tag}-Base][VAL]  Recall={base_val['Recall']:.4f} NDCG={base_val['NDCG']:.4f} Cov={base_val['Coverage']:.4f} LT={base_val['LongTailShare']:.4f} Nov={base_val['Novelty']:.3f} ILAD={base_val['ILAD']:.4f}")
        print(f"[{model_tag}-Base][TEST] Recall={base_test['Recall']:.4f} NDCG={base_test['NDCG']:.4f} Cov={base_test['Coverage']:.4f} LT={base_test['LongTailShare']:.4f} Nov={base_test['Novelty']:.3f} ILAD={base_test['ILAD']:.4f}")

        rows = []
        t0_all = time.time()

        combo_cnt = 0
        for w_rel in w_rel_list:
            for w_div in w_div_list:
                for w_nov in w_nov_list:
                    w_tail = 1.0 - float(w_rel) - float(w_div) - float(w_nov)
                    if w_tail < 0:
                        continue

                    weights = RerankWeights(
                        w_rel=float(w_rel),
                        w_novel=float(w_nov),
                        w_tail=float(w_tail),
                        w_div=float(w_div),
                        cold_boost=float(cold_boost),
                        hist_ref=int(hist_ref),
                    )

                    t0 = time.time()
                    recs_val = rerank_all_users(
                        reranker=reranker,
                        eval_users=val_users,
                        seen=seen_for_val,
                        topn_items_by_user=topn_items_val,
                        topn_scores_by_user=topn_scores_val,
                        K=K,
                        weights=weights,
                    )
                    m_val = compute_metrics_topk(
                        recs=recs_val,
                        user_pos=meta["val_user_pos"],
                        n_items=n_items,
                        item_popularity=item_pop,
                        tail_mask=tail_mask,
                        K=K,
                        item_vectors=meta["item_vectors_norm"],
                    )
                    dt = time.time() - t0
                    combo_cnt += 1

                    row = {
                        "model": model_tag,
                        "K": K,
                        "topN": topN,
                        "w_rel": float(w_rel),
                        "w_div": float(w_div),
                        "w_novel": float(w_nov),
                        "w_tail": float(w_tail),
                        "cold_boost": float(cold_boost),
                        "hist_ref": int(hist_ref),
                        "val_Recall": float(m_val["Recall"]),
                        "val_NDCG": float(m_val["NDCG"]),
                        "val_Coverage": float(m_val["Coverage"]),
                        "val_LongTailShare": float(m_val["LongTailShare"]),
                        "val_Novelty": float(m_val["Novelty"]),
                        "val_ILAD": float(m_val["ILAD"]),
                        "seconds_val": float(dt),
                    }
                    rows.append(row)

                    print(
                        f"[{model_tag}][VAL] "
                        f"w_rel={w_rel:.2f} w_div={w_div:.2f} w_nov={w_nov:.2f} w_tail={w_tail:.2f} | "
                        f"Recall={m_val['Recall']:.4f} NDCG={m_val['NDCG']:.4f} "
                        f"Cov={m_val['Coverage']:.4f} LT={m_val['LongTailShare']:.4f} Nov={m_val['Novelty']:.3f} ILAD={m_val['ILAD']:.4f} "
                        f"({dt:.1f}s)"
                    )

        best = select_best_by_val(rows)
        best_w = {
            "w_rel": best["w_rel"],
            "w_div": best["w_div"],
            "w_novel": best["w_novel"],
            "w_tail": best["w_tail"],
            "cold_boost": best["cold_boost"],
            "hist_ref": best["hist_ref"],
        }

        weights_best = RerankWeights(
            w_rel=float(best_w["w_rel"]),
            w_novel=float(best_w["w_novel"]),
            w_tail=float(best_w["w_tail"]),
            w_div=float(best_w["w_div"]),
            cold_boost=float(best_w["cold_boost"]),
            hist_ref=int(best_w["hist_ref"]),
        )
        recs_test = rerank_all_users(
            reranker=reranker,
            eval_users=test_users,
            seen=seen_for_test,
            topn_items_by_user=topn_items_test,
            topn_scores_by_user=topn_scores_test,
            K=K,
            weights=weights_best,
        )
        m_test = compute_metrics_topk(
            recs=recs_test,
            user_pos=meta["test_user_pos"],
            n_items=n_items,
            item_popularity=item_pop,
            tail_mask=tail_mask,
            K=K,
            item_vectors=meta["item_vectors_norm"],
        )

        print(f"\n[{model_tag}] BEST (by VAL NDCG) => w_rel={best_w['w_rel']:.2f} w_div={best_w['w_div']:.2f} w_nov={best_w['w_novel']:.2f} w_tail={best_w['w_tail']:.2f}")
        print(f"[{model_tag}-BEST][VAL]  Recall={best['val_Recall']:.4f} NDCG={best['val_NDCG']:.4f} Cov={best['val_Coverage']:.4f} LT={best['val_LongTailShare']:.4f} Nov={best['val_Novelty']:.3f} ILAD={best.get('val_ILAD',0.0):.4f}")
        print(f"[{model_tag}-BEST][TEST] Recall={m_test['Recall']:.4f} NDCG={m_test['NDCG']:.4f} Cov={m_test['Coverage']:.4f} LT={m_test['LongTailShare']:.4f} Nov={m_test['Novelty']:.3f} ILAD={m_test['ILAD']:.4f}")

        write_back_best_weights(
            cfg_path=cfg_path,
            model_tag=model_tag,
            best_w=best_w,
            best_val={
                "Recall": best["val_Recall"],
                "NDCG": best["val_NDCG"],
                "Coverage": best["val_Coverage"],
                "LongTailShare": best["val_LongTailShare"],
                "Novelty": best["val_Novelty"],
            },
            best_test=m_test,
            objective="NDCG",
        )

        df_rows = pd.DataFrame(rows)
        out_csv = os.path.join(ART_DIR, f"tune_rows_{model_tag.replace('/', '_')}.csv")
        df_rows.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[Saved] {out_csv}")
        print(f"[{model_tag}] tuning done in {time.time() - t0_all:.1f}s (combos={combo_cnt})")

        return {
            "model": model_tag,
            "best_w_rel": best_w["w_rel"],
            "best_w_div": best_w["w_div"],
            "best_w_novel": best_w["w_novel"],
            "best_w_tail": best_w["w_tail"],
            "val_recall": best["val_Recall"],
            "val_ndcg": best["val_NDCG"],
            "test_recall": m_test["Recall"],
            "test_ndcg": m_test["NDCG"],
        }

    # -------------------------------------------------
    # ckpt paths (优先从 config.paths 读取)
    # -------------------------------------------------
    paths = cfg.get("paths", {}) or {}

    def _ckpt(name_key: str, default_name: str) -> str:
        p = paths.get(name_key, default_name)
        return _abs_path(p)

    summaries = []

    # LightGCN
    lg_path = _ckpt("lightgcn_ckpt", "lightgcn_ml1m_best.pth")
    if os.path.exists(lg_path) and LightGCN is not None:
        adj = build_normalized_adj(n_users, n_items, meta["train_u"], meta["train_i"], device=device)
        lg = load_lightgcn(lg_path, adj, device=device)

        topn_items_val, topn_scores_val = precompute_topn_for_embedding_model(
            lg, meta["seen_train"], val_users, topN, device, "LightGCN(VAL)"
        )
        topn_items_test, topn_scores_test = precompute_topn_for_embedding_model(
            lg, meta["seen_trainval"], test_users, topN, device, "LightGCN(TEST)"
        )

        summaries.append(tune_one_model(
            "LightGCN-Rerank",
            topn_items_val, topn_scores_val,
            topn_items_test, topn_scores_test,
            meta["seen_train"], meta["seen_trainval"]
        ))
    else:
        print("[SKIP] LightGCN ckpt or class not found:", lg_path)

    # NGCF
    ng_path = _ckpt("ngcf_ckpt", "ngcf_ml1m_best.pth")
    if os.path.exists(ng_path) and NGCF is not None:
        adj = build_normalized_adj(n_users, n_items, meta["train_u"], meta["train_i"], device=device)
        ng = load_ngcf(ng_path, adj, device=device)

        topn_items_val, topn_scores_val = precompute_topn_for_embedding_model(
            ng, meta["seen_train"], val_users, topN, device, "NGCF(VAL)"
        )
        topn_items_test, topn_scores_test = precompute_topn_for_embedding_model(
            ng, meta["seen_trainval"], test_users, topN, device, "NGCF(TEST)"
        )

        summaries.append(tune_one_model(
            "NGCF-Rerank",
            topn_items_val, topn_scores_val,
            topn_items_test, topn_scores_test,
            meta["seen_train"], meta["seen_trainval"]
        ))
    else:
        print("[SKIP] NGCF ckpt or class not found:", ng_path)

    # MMGCN
    mm_path = _ckpt("mmgcn_ckpt", "mmgcn_ml1m_best.pth")
    if os.path.exists(mm_path) and MMGCN is not None:
        mm = load_mmgcn(mm_path, cfg, meta)

        topn_items_val, topn_scores_val = precompute_topn_for_embedding_model(
            mm, meta["seen_train"], val_users, topN, device, "MMGCN(VAL)"
        )
        topn_items_test, topn_scores_test = precompute_topn_for_embedding_model(
            mm, meta["seen_trainval"], test_users, topN, device, "MMGCN(TEST)"
        )

        summaries.append(tune_one_model(
            "MMGCN-Rerank",
            topn_items_val, topn_scores_val,
            topn_items_test, topn_scores_test,
            meta["seen_train"], meta["seen_trainval"]
        ))
    else:
        print("[SKIP] MMGCN ckpt or class not found:", mm_path)

    # HybridNCF
    hy_path = _ckpt("hybrid_ckpt", "hybrid_ml1m_best.pth")
    if not os.path.exists(hy_path):
        hy_path = _ckpt("hybrid_ckpt_safe", "hybrid_ml1m_best_safe.pth")

    if os.path.exists(hy_path) and HybridNCF is not None and ItemContentEncoder is not None:
        hy = load_hybrid_robust(hy_path, cfg=cfg, ratings_path=meta["ratings_path"], device=device)

        topn_items_val, topn_scores_val = precompute_topn_for_forward_model(
            hy, meta["seen_train"], val_users, n_items, topN, device, "HybridNCF(VAL)"
        )
        topn_items_test, topn_scores_test = precompute_topn_for_forward_model(
            hy, meta["seen_trainval"], test_users, n_items, topN, device, "HybridNCF(TEST)"
        )

        summaries.append(tune_one_model(
            "HybridNCF-Rerank",
            topn_items_val, topn_scores_val,
            topn_items_test, topn_scores_test,
            meta["seen_train"], meta["seen_trainval"]
        ))
    else:
        print("[SKIP] Hybrid ckpt or class not found:", hy_path)

    # HybridAccNCF
    hyacc_path = _ckpt("hybrid_acc_ckpt", "hybrid_acc_ml1m_best.pth")
    hyacc_path = hyacc_path if os.path.isabs(hyacc_path) else os.path.join(PROJECT_ROOT, hyacc_path)

    if HybridNCFAcc is not None and os.path.exists(hyacc_path):
        print("[Tune] HybridAccNCF:", hyacc_path)
        hyacc = load_hybrid_acc_robust(hyacc_path, cfg=cfg, ratings_path=meta["ratings_path"], device=device)

        topn_items_val, topn_scores_val = precompute_topn_for_forward_model(
            hyacc, meta["seen_train"], val_users, meta["n_items"], topN, device, model_name="HybridAcc val"
        )
        topn_items_test, topn_scores_test = precompute_topn_for_forward_model(
            hyacc, meta["seen_trainval"], test_users, meta["n_items"], topN, device, model_name="HybridAcc test"
        )

        summaries.append(
            tune_one_model(
                "HybridAcc-Rerank",
                topn_items_val, topn_scores_val,
                topn_items_test, topn_scores_test,
                meta["seen_train"],  # seen_for_val
                meta["seen_trainval"],  # seen_for_test
            )
        )

    # HybridTailNCF
    hytail_path = _ckpt("hybrid_tail_ckpt", "hybrid_tail_ml1m_best.pth")
    hytail_path = hytail_path if os.path.isabs(hytail_path) else os.path.join(PROJECT_ROOT, hytail_path)

    if HybridNCFTail is not None and os.path.exists(hytail_path):
        print("[Tune] HybridTailNCF:", hytail_path)
        hytail = load_hybrid_tail_robust(
            hytail_path, cfg=cfg, ratings_path=meta["ratings_path"], device=device,
            item_popularity=item_pop.astype("float32")
        )

        topn_items_val, topn_scores_val = precompute_topn_for_forward_model(
            hytail, meta["seen_train"], val_users, meta["n_items"], topN, device, model_name="HybridTail val"
        )
        topn_items_test, topn_scores_test = precompute_topn_for_forward_model(
            hytail, meta["seen_trainval"], test_users, meta["n_items"], topN, device, model_name="HybridTail test"
        )

        summaries.append(
            tune_one_model(
                "HybridTail-Rerank",
                topn_items_val, topn_scores_val,
                topn_items_test, topn_scores_test,
                meta["seen_train"],  # seen_for_val
                meta["seen_trainval"],  # seen_for_test
            )
        )

    # SVDPP
    svdpp_path = _ckpt("svdpp_ckpt", "svdpp_ml1m_best.pth")
    if os.path.exists(svdpp_path) and SVDPP is not None:
        user_interactions = {u: set() for u in range(n_users)}
        for u, i in zip(meta["train_df"]["user_idx"].values, meta["train_df"]["item_idx"].values):
            user_interactions[int(u)].add(int(i))
        svdpp = load_svdpp(svdpp_path, user_interactions, meta["global_mean"], device=device)

        topn_items_val, topn_scores_val = precompute_topn_for_forward_model(
            svdpp, meta["seen_train"], val_users, n_items, topN, device, "SVDPP(VAL)"
        )
        topn_items_test, topn_scores_test = precompute_topn_for_forward_model(
            svdpp, meta["seen_trainval"], test_users, n_items, topN, device, "SVDPP(TEST)"
        )

        summaries.append(tune_one_model(
            "SVDPP-Rerank",
            topn_items_val, topn_scores_val,
            topn_items_test, topn_scores_test,
            meta["seen_train"], meta["seen_trainval"]
        ))
    else:
        print("[SKIP] SVDPP ckpt or class not found:", svdpp_path)

    # MultiVAE
    vae_path = _ckpt("multivae_ckpt", "multivae_ml1m_best.pth")
    if os.path.exists(vae_path) and MultiVAE is not None:
        vae = load_multivae(vae_path, n_items=n_items, device=device)

        topn_items_val, topn_scores_val = precompute_topn_for_multivae(
            vae, meta["seen_train"], val_users, n_items, topN, device, "MultiVAE(VAL)"
        )
        topn_items_test, topn_scores_test = precompute_topn_for_multivae(
            vae, meta["seen_trainval"], test_users, n_items, topN, device, "MultiVAE(TEST)"
        )

        summaries.append(tune_one_model(
            "MultiVAE-Rerank",
            topn_items_val, topn_scores_val,
            topn_items_test, topn_scores_test,
            meta["seen_train"], meta["seen_trainval"]
        ))
    else:
        print("[SKIP] MultiVAE ckpt or class not found:", vae_path)

    # DIN
    din_path = _ckpt("din_ckpt", "din_ml1m_best.pth")
    if os.path.exists(din_path) and DIN is not None:
        din = load_din(din_path, device=device)
        item_offset = infer_din_item_offset(din, n_items=n_items)

        max_len = int(getattr(din, "max_history_len", 50))

        # VAL: 仅用 train history
        hist_items_train, hist_lens_train = build_user_hist_sequences(meta["train_df"], n_users=n_users, max_len=max_len)

        # TEST: 用 train+val history
        trainval_df = pd.concat([meta["train_df"], meta["val_df"]], ignore_index=True)
        hist_items_trainval, hist_lens_trainval = build_user_hist_sequences(trainval_df, n_users=n_users, max_len=max_len)

        topn_items_val, topn_scores_val = precompute_topn_for_din(
            din, meta["seen_train"], val_users, n_items, topN, device, "DIN(VAL)",
            hist_items=hist_items_train, hist_lens=hist_lens_train, item_offset=item_offset, chunk=1024
        )
        topn_items_test, topn_scores_test = precompute_topn_for_din(
            din, meta["seen_trainval"], test_users, n_items, topN, device, "DIN(TEST)",
            hist_items=hist_items_trainval, hist_lens=hist_lens_trainval, item_offset=item_offset, chunk=1024
        )

        summaries.append(tune_one_model(
            "DIN-Rerank",
            topn_items_val, topn_scores_val,
            topn_items_test, topn_scores_test,
            meta["seen_train"], meta["seen_trainval"]
        ))
    else:
        print("[SKIP] DIN ckpt or class not found:", din_path)

    # Summary
    if summaries:
        df_sum = pd.DataFrame(summaries)
        out_sum = os.path.join(ART_DIR, "tune_best_summary.csv")
        df_sum.to_csv(out_sum, index=False, encoding="utf-8-sig")
        print("\n========== Tune Best Summary ==========")
        print(df_sum.to_string(index=False))
        print(f"\nSaved: {out_sum}")

    print("\n[Done] Tune finished. Now run evaluate_v3_rerank_ml1m.py to get final test table using updated config.")


if __name__ == "__main__":
    main()
