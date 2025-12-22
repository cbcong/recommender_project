# recommender_project/experiments/ml1m/evaluate_v3_rerank_ml1m.py
import os
import sys
import time
import math
import inspect
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------- preprocess ----------
try:
    from utils.preprocess import load_ml1m_ratings, build_id_mappings, split_by_user_time
except Exception:
    from preprocess import load_ml1m_ratings, build_id_mappings, split_by_user_time

# ---------- base models ----------
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

# ---------- fusion model ----------
try:
    from hybrid_model import HybridNCF
    from content_model import ItemContentEncoder

    HAS_HYBRID = True
except Exception:
    HybridNCF = None
    ItemContentEncoder = None
    HAS_HYBRID = False

# ---------- hybrid variants ----------
try:
    from hybrid_acc_model import HybridNCFAcc
except Exception:
    HybridNCFAcc = None

try:
    from hybrid_tail_model import HybridNCFTail
except Exception:
    HybridNCFTail = None

# ---------- classic / deep models ----------
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

# ---------- rerank ----------
try:
    from rerank_model import AdaptiveMMRReranker, RerankWeights
except Exception as e:
    raise RuntimeError("Cannot import models/rerank_model.py (AdaptiveMMRReranker, RerankWeights).") from e


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def torch_load_safe_weights(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def torch_load_full(path: str, map_location: str):
    # 需要读 config/idx2item 等非 tensor 信息时用 full
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_user_pos_dict(user_idx: np.ndarray, item_idx: np.ndarray, n_users: int) -> Dict[int, set]:
    d = {u: set() for u in range(n_users)}
    for u, i in zip(user_idx, item_idx):
        d[int(u)].add(int(i))
    return d


def build_long_tail_mask(item_popularity: np.ndarray, head_ratio: float = 0.8) -> np.ndarray:
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


def recall_ndcg_at_k(rank_list: List[int], gt: set, k: int) -> Tuple[float, float]:
    if not gt:
        return 0.0, 0.0
    r = rank_list[:k]
    hit = sum(1 for x in gt if x in set(r))
    recall = hit / float(len(gt))

    dcg = 0.0
    for idx, it in enumerate(r, start=1):
        if it in gt:
            dcg += 1.0 / math.log2(idx + 1)
    ideal = min(len(gt), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1)) if ideal > 0 else 0.0
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return recall, ndcg


def evaluate_accuracy(recs: Dict[int, List[int]], test_user_pos: Dict[int, set], k: int) -> Tuple[float, float]:
    rs, ns, cnt = 0.0, 0.0, 0
    for u, gt in test_user_pos.items():
        if u not in recs:
            continue
        r, n = recall_ndcg_at_k(recs[u], gt, k)
        rs += r
        ns += n
        cnt += 1
    return (rs / cnt if cnt else 0.0), (ns / cnt if cnt else 0.0)


def compute_item_coverage(recs: Dict[int, List[int]], n_items: int) -> float:
    s = set()
    for items in recs.values():
        s.update(items)
    return len(s) / float(n_items) if n_items > 0 else 0.0


def compute_longtail_share(recs: Dict[int, List[int]], tail_mask: np.ndarray) -> float:
    tail_hits, total = 0, 0
    for items in recs.values():
        for i in items:
            if 0 <= i < len(tail_mask) and tail_mask[i]:
                tail_hits += 1
            total += 1
    return tail_hits / float(total) if total > 0 else 0.0


def compute_novelty(recs: Dict[int, List[int]], item_popularity: np.ndarray) -> float:
    total_events = item_popularity.sum()
    if total_events <= 0:
        return 0.0
    probs = item_popularity.astype("float64") / float(total_events)
    probs[probs <= 0] = 1e-12
    info = -np.log2(probs)
    s, n = 0.0, 0
    for items in recs.values():
        for i in items:
            if 0 <= i < len(info):
                s += info[i]
                n += 1
    return s / float(n) if n > 0 else 0.0


def evaluate_hit_rate_at_n(cand_items: Dict[int, List[int]], test_user_pos: Dict[int, set], N: int) -> float:
    hit, cnt = 0, 0
    for u, gt in test_user_pos.items():
        items = cand_items.get(u, [])
        if not items:
            continue
        topn = set(items[:N])
        if len(topn & gt) > 0:
            hit += 1
        cnt += 1
    return hit / cnt if cnt > 0 else 0.0


def build_base_topk_from_candidates(cand: Dict[int, Tuple[List[int], List[float]]], K: int) -> Dict[int, List[int]]:
    recs = {}
    for u, (items, _scores) in cand.items():
        recs[u] = items[:K]
    return recs


def get_rerank_kn(cfg: dict, default_k: int = 10, default_n: int = 200) -> Tuple[int, int]:
    rr = cfg.get("rerank", {}) or {}
    ev = cfg.get("eval_v3_rerank", {}) or {}

    K = int(ev.get("topK", rr.get("K", default_k)))
    N = int(ev.get("candidate_N", rr.get("N", default_n)))
    return K, N


def get_rerank_weights_from_cfg(cfg: dict, model_tag: str) -> "RerankWeights":
    rr = cfg.get("rerank", {}) or {}
    base = rr.get("weights", None)
    if base is None:
        base = rr.get("default_weights", {}) or {}
    per = rr.get("per_model", None)
    if per is None:
        per = rr.get("per_model_weights", {}) or {}

    w = dict(base) if isinstance(base, dict) else {}
    if model_tag in per and isinstance(per[model_tag], dict):
        w.update(per[model_tag])

    return RerankWeights(
        w_rel=float(w.get("w_rel", 1.0)),
        w_novel=float(w.get("w_novel", 0.15)),
        w_tail=float(w.get("w_tail", 0.15)),
        w_div=float(w.get("w_div", 0.25)),
        cold_boost=float(w.get("cold_boost", 2.0)),
        hist_ref=int(w.get("hist_ref", 50)),
    )


def build_normalized_adj(num_users, num_items, user_idx, item_idx, device):
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
    adj = torch.sparse_coo_tensor(
        torch.from_numpy(indices).long(),
        torch.from_numpy(norm_values).float(),
        torch.Size([num_nodes, num_nodes]),
    ).coalesce().to(device)
    return adj


def load_item_vectors_aligned_ml1m(n_items: int, item2idx: Dict[int, int]) -> np.ndarray:
    feat_dir = os.path.join(PROJECT_ROOT, "data", "features")
    text_feat_path = os.path.join(feat_dir, "ml1m_text_embeddings_64.npy")
    text_index_path = os.path.join(feat_dir, "ml1m_text_index.csv")
    img_feat_path = os.path.join(feat_dir, "ml1m_image_embeddings_64.npy")
    img_index_path = os.path.join(feat_dir, "ml1m_image_index.csv")

    def _load_one(feat_path: str, index_path: str) -> np.ndarray:
        feats = np.load(feat_path).astype(np.float32)
        df = pd.read_csv(index_path)

        if "movieId" in df.columns:
            key_col = "movieId"
        elif "movie_id" in df.columns:
            key_col = "movie_id"
        else:
            key_col = df.columns[0]

        out = np.zeros((n_items, feats.shape[1]), dtype=np.float32)
        movie_to_row = {int(m): i for i, m in enumerate(df[key_col].values)}
        for mid, idx in item2idx.items():
            ridx = movie_to_row.get(int(mid), None)
            if ridx is None:
                continue
            if 0 <= ridx < feats.shape[0]:
                out[int(idx)] = feats[int(ridx)]
        return out

    text = _load_one(text_feat_path, text_index_path)
    img = _load_one(img_feat_path, img_index_path)
    return np.concatenate([text, img], axis=1)


@torch.no_grad()
def topn_from_embeddings(model, eval_users: List[int], seen: Dict[int, set], N: int, device: str):
    model.eval()
    recs = {}
    user_emb, item_emb = model.get_user_item_embeddings()
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    for u in eval_users:
        u_e = user_emb[u]
        scores = torch.matmul(item_emb, u_e)
        s = seen.get(u, set())
        if s:
            scores[list(s)] = -1e9
        topv, topi = torch.topk(scores, k=min(N, scores.numel()))
        recs[u] = (topi.detach().cpu().numpy().tolist(),
                   topv.detach().cpu().numpy().tolist())
    return recs


@torch.no_grad()
def topn_pointwise_anyscore(model, eval_users: List[int], seen: Dict[int, set], n_items: int, N: int, device: str,
                            chunk: int = 1024):
    """
    关键升级：如果模型实现了 score_logits，则用 logits 做候选排序（避免 sigmoid 压缩导致排序弱）。
    否则回退到 forward。
    """
    model.eval()
    recs = {}
    all_items = torch.arange(n_items, dtype=torch.long, device=device)

    use_logits = callable(getattr(model, "score_logits", None))

    for u in eval_users:
        u_t = torch.full((n_items,), int(u), dtype=torch.long, device=device)
        scores = torch.empty((n_items,), dtype=torch.float32, device=device)

        for st in range(0, n_items, chunk):
            ed = min(st + chunk, n_items)
            if use_logits:
                s = model.score_logits(u_t[st:ed], all_items[st:ed])
            else:
                s = model(u_t[st:ed], all_items[st:ed])
            scores[st:ed] = s.view(-1)

        sset = seen.get(u, set())
        if sset:
            scores[list(sset)] = -1e9
        topv, topi = torch.topk(scores, k=min(N, scores.numel()))
        recs[u] = (topi.detach().cpu().numpy().tolist(),
                   topv.detach().cpu().numpy().tolist())
    return recs


def build_user_hist_tensors_from_df(trainval_df: pd.DataFrame, n_users: int, max_hist_len: int, pad_idx: int):
    """
    统一用 user_idx/item_idx 构造历史；左对齐。
    """
    if "timestamp" not in trainval_df.columns:
        tmp = trainval_df.copy()
        tmp["timestamp"] = np.arange(len(tmp), dtype=np.int64)
        trainval_df = tmp

    trainval_df = trainval_df.sort_values(["user_idx", "timestamp"])
    grp = trainval_df.groupby("user_idx")["item_idx"].apply(list)

    hist_items = np.full((n_users, max_hist_len), fill_value=pad_idx, dtype=np.int64)
    hist_lens = np.zeros((n_users,), dtype=np.int64)

    for u in range(n_users):
        seq = grp.get(u, [])
        if not seq:
            continue
        seq = seq[-max_hist_len:]
        L = len(seq)
        hist_lens[u] = L
        hist_items[u, :L] = np.array(seq, dtype=np.int64)

    return torch.from_numpy(hist_items), torch.from_numpy(hist_lens)


def maybe_refresh_hybrid_cache(model: "HybridNCF", hist_df: pd.DataFrame, n_users: int, device: str):
    if not callable(getattr(model, "set_user_histories", None)) or not callable(
            getattr(model, "refresh_user_cache", None)):
        return
    max_len = int(getattr(model, "max_hist_len", 50))
    pad_idx = int(getattr(model, "PAD_IDX", model.num_items))
    hist_items, hist_lens = build_user_hist_tensors_from_df(hist_df, n_users=n_users, max_hist_len=max_len,
                                                            pad_idx=pad_idx)
    model.set_user_histories(hist_items, hist_lens)
    model.refresh_user_cache(device=device)


def load_lightgcn(ckpt_path: str, adj, device: str):
    if LightGCN is None:
        raise RuntimeError("LightGCN not importable.")
    ckpt = torch_load_safe_weights(ckpt_path, map_location=device)
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


def load_ngcf(ckpt_path: str, adj, device: str):
    if NGCF is None:
        raise RuntimeError("NGCF not importable.")
    ckpt = torch_load_safe_weights(ckpt_path, map_location=device)
    model = NGCF(
        num_users=ckpt["n_users"],
        num_items=ckpt["n_items"],
        embedding_dim=ckpt.get("embedding_dim", 64),
        num_layers=ckpt.get("num_layers", 3),
        adj=adj,
    ).to(device)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_mmgcn(ckpt_path: str, meta: dict):
    if MMGCN is None:
        raise RuntimeError("MMGCN not importable.")
    device = meta["device"]
    ckpt = torch_load_safe_weights(ckpt_path, map_location=device)

    n_users, n_items = meta["n_users"], meta["n_items"]
    train_df = meta["train_df"]

    train_u = train_df["user_idx"].values
    train_i = train_df["item_idx"].values
    adj = build_normalized_adj(n_users, n_items, train_u, train_i, device=device)
    adj_id = adj_text = adj_image = adj

    feat_dir = os.path.join(PROJECT_ROOT, "data", "features")
    text_feat = np.load(os.path.join(feat_dir, "ml1m_text_embeddings_64.npy")).astype(np.float32)
    img_feat = np.load(os.path.join(feat_dir, "ml1m_image_embeddings_64.npy")).astype(np.float32)

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

    state_key = "model_state" if "model_state" in ckpt else "state_dict"
    model.load_state_dict(ckpt[state_key], strict=False)
    model.eval()
    return model


def load_hybrid_robust(ckpt_path: str, ratings_path: str, idx2item: np.ndarray, n_items: int, device: str):
    if not HAS_HYBRID:
        raise RuntimeError("HybridNCF not importable.")

    ckpt = torch_load_full(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state", ckpt))

    if isinstance(ckpt, dict) and "n_users" in ckpt and "n_items" in ckpt:
        n_users = int(ckpt["n_users"])
        n_items_ckpt = int(ckpt["n_items"])
    else:
        n_users = int(state["user_embedding_gmf.weight"].shape[0])
        n_items_ckpt = int(state["item_embedding_gmf.weight"].shape[0])

    # 以数据侧为准（防止 ckpt 存了别的版本）
    n_items_use = int(n_items)
    if n_items_ckpt != n_items_use:
        print(f"[HybridNCF][Warn] ckpt n_items={n_items_ckpt} != data n_items={n_items_use}, use data-side n_items.")

    gmf_dim = int(state["user_embedding_gmf.weight"].shape[1])
    mlp_dim = int(state["user_embedding_mlp.weight"].shape[1])

    if "text_proj.weight" in state:
        content_proj_dim = int(state["text_proj.weight"].shape[0])
    elif "image_proj.weight" in state:
        content_proj_dim = int(state["image_proj.weight"].shape[0])
    else:
        content_proj_dim = mlp_dim

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    hycfg = cfg.get("hybrid", {}) if isinstance(cfg, dict) else {}

    dropout = float(hycfg.get("dropout", 0.10))
    use_history = bool(hycfg.get("use_history", True))
    max_hist_len = int(hycfg.get("max_hist_len", 50))
    n_heads = int(hycfg.get("n_heads", 4))
    n_transformer_layers = int(hycfg.get("n_transformer_layers", 2))
    rating_min = float(hycfg.get("rating_min", 1.0))
    rating_max = float(hycfg.get("rating_max", 5.0))
    global_mean = float(ckpt.get("global_mean", hycfg.get("global_mean", 0.0)))

    mlp_layer_sizes = hycfg.get("mlp_layer_sizes", (512, 256, 128))
    if isinstance(mlp_layer_sizes, list):
        mlp_layer_sizes = tuple(mlp_layer_sizes)

    feat_dir = os.path.join(PROJECT_ROOT, "data", "features")
    text_feat_path = os.path.join(feat_dir, "ml1m_text_embeddings_64.npy")
    text_index_path = os.path.join(feat_dir, "ml1m_text_index.csv")
    img_feat_path = os.path.join(feat_dir, "ml1m_image_embeddings_64.npy")
    img_index_path = os.path.join(feat_dir, "ml1m_image_index.csv")

    # 强制 idx2item 对齐
    content_encoder = ItemContentEncoder(
        ratings_path=ratings_path,
        text_feat_path=text_feat_path,
        text_index_path=text_index_path,
        image_feat_path=img_feat_path,
        image_index_path=img_index_path,
        use_text=True,
        use_image=True,
        idx2item=idx2item,
        n_items=n_items_use,
    )

    model = HybridNCF(
        num_users=n_users,
        num_items=n_items_use,
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

    incompat = model.load_state_dict(state, strict=False)
    if getattr(incompat, "missing_keys", None) or getattr(incompat, "unexpected_keys", None):
        if incompat.missing_keys:
            print("[HybridNCF] missing_keys(head):", incompat.missing_keys[:20])
        if incompat.unexpected_keys:
            print("[HybridNCF] unexpected_keys(head):", incompat.unexpected_keys[:20])

    model.eval()
    return model


def load_hybrid_acc_robust(ckpt_path: str, ratings_path: str, idx2item: np.ndarray, n_items: int, device: str):
    """
    与 load_hybrid_robust 基本一致，只是实例化 HybridNCFAcc（准确性优先变体）。
    """
    if HybridNCFAcc is None:
        raise RuntimeError("HybridNCFAcc not importable.")

    ckpt = torch_load_full(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state", ckpt))

    if isinstance(ckpt, dict) and "n_users" in ckpt and "n_items" in ckpt:
        n_users = int(ckpt["n_users"])
        n_items_ckpt = int(ckpt["n_items"])
    else:
        n_users = int(state["user_embedding_gmf.weight"].shape[0])
        n_items_ckpt = int(state["item_embedding_gmf.weight"].shape[0])

    n_items_use = int(n_items)
    if n_items_ckpt != n_items_use:
        print(f"[HybridAcc][Warn] ckpt n_items={n_items_ckpt} != data n_items={n_items_use}, use data-side n_items.")

    gmf_dim = int(state["user_embedding_gmf.weight"].shape[1])
    mlp_dim = int(state["user_embedding_mlp.weight"].shape[1])

    if "text_proj.weight" in state:
        content_proj_dim = int(state["text_proj.weight"].shape[0])
    elif "image_proj.weight" in state:
        content_proj_dim = int(state["image_proj.weight"].shape[0])
    else:
        content_proj_dim = mlp_dim

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    hycfg = cfg.get("hybrid_acc", cfg.get("hybrid", {})) if isinstance(cfg, dict) else {}

    dropout = float(hycfg.get("dropout", 0.10))
    use_history = bool(hycfg.get("use_history", True))
    max_hist_len = int(hycfg.get("max_hist_len", 50))
    n_heads = int(hycfg.get("n_heads", 4))
    n_transformer_layers = int(hycfg.get("n_transformer_layers", 2))
    rating_min = float(hycfg.get("rating_min", 1.0))
    rating_max = float(hycfg.get("rating_max", 5.0))
    global_mean = float(ckpt.get("global_mean", hycfg.get("global_mean", 0.0)))

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
        use_text=True,
        use_image=True,
        idx2item=idx2item,
        n_items=n_items_use,
    )

    model = HybridNCFAcc(
        num_users=n_users,
        num_items=n_items_use,
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

    incompat = model.load_state_dict(state, strict=False)
    if getattr(incompat, "missing_keys", None) or getattr(incompat, "unexpected_keys", None):
        if incompat.missing_keys:
            print("[HybridAcc] missing_keys(head):", incompat.missing_keys[:20])
        if incompat.unexpected_keys:
            print("[HybridAcc] unexpected_keys(head):", incompat.unexpected_keys[:20])

    model.eval()
    return model


def load_hybrid_tail_robust(
        ckpt_path: str,
        ratings_path: str,
        idx2item: np.ndarray,
        n_items: int,
        device: str,
        item_popularity: Optional[np.ndarray] = None,
):
    """
    HybridNCFTail：在 score_logits 里加流行度惩罚；item_popularity 可由评估侧覆盖（建议用 train+val）。
    """
    if HybridNCFTail is None:
        raise RuntimeError("HybridNCFTail not importable.")

    ckpt = torch_load_full(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state", ckpt))

    if isinstance(ckpt, dict) and "n_users" in ckpt and "n_items" in ckpt:
        n_users = int(ckpt["n_users"])
        n_items_ckpt = int(ckpt["n_items"])
    else:
        n_users = int(state["user_embedding_gmf.weight"].shape[0])
        n_items_ckpt = int(state["item_embedding_gmf.weight"].shape[0])

    n_items_use = int(n_items)
    if n_items_ckpt != n_items_use:
        print(f"[HybridTail][Warn] ckpt n_items={n_items_ckpt} != data n_items={n_items_use}, use data-side n_items.")

    gmf_dim = int(state["user_embedding_gmf.weight"].shape[1])
    mlp_dim = int(state["user_embedding_mlp.weight"].shape[1])

    if "text_proj.weight" in state:
        content_proj_dim = int(state["text_proj.weight"].shape[0])
    elif "image_proj.weight" in state:
        content_proj_dim = int(state["image_proj.weight"].shape[0])
    else:
        content_proj_dim = mlp_dim

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    hycfg = cfg.get("hybrid_tail", cfg.get("hybrid", {})) if isinstance(cfg, dict) else {}

    dropout = float(hycfg.get("dropout", 0.10))
    use_history = bool(hycfg.get("use_history", True))
    max_hist_len = int(hycfg.get("max_hist_len", 50))
    n_heads = int(hycfg.get("n_heads", 4))
    n_transformer_layers = int(hycfg.get("n_transformer_layers", 2))
    rating_min = float(hycfg.get("rating_min", 1.0))
    rating_max = float(hycfg.get("rating_max", 5.0))
    global_mean = float(ckpt.get("global_mean", hycfg.get("global_mean", 0.0)))

    pop_alpha = float(ckpt.get("pop_alpha", hycfg.get("pop_alpha", 0.30)))
    pop_mode = str(ckpt.get("pop_mode", hycfg.get("pop_mode", "log_norm")))

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
        use_text=True,
        use_image=True,
        idx2item=idx2item,
        n_items=n_items_use,
    )

    # item_popularity 优先用评估侧传入（通常= train+val）
    if item_popularity is None and isinstance(ckpt, dict) and "item_pop_train" in ckpt:
        item_popularity = np.array(ckpt["item_pop_train"], dtype=np.float32)

    model = HybridNCFTail(
        num_users=n_users,
        num_items=n_items_use,
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
    ).to(device)

    incompat = model.load_state_dict(state, strict=False)
    if getattr(incompat, "missing_keys", None) or getattr(incompat, "unexpected_keys", None):
        if incompat.missing_keys:
            print("[HybridTail] missing_keys(head):", incompat.missing_keys[:20])
        if incompat.unexpected_keys:
            print("[HybridTail] unexpected_keys(head):", incompat.unexpected_keys[:20])

    model.eval()
    return model


def prepare_ml1m():
    cfg = load_config(os.path.join(PROJECT_ROOT, "utils", "config.yaml"))
    ratings_path = cfg["data"]["ml1m_ratings"]
    if not os.path.isabs(ratings_path):
        ratings_path = os.path.join(PROJECT_ROOT, ratings_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ratings_df = load_ml1m_ratings(ratings_path)
    user2idx, item2idx, idx2user, idx2item = build_id_mappings(ratings_df)
    n_users, n_items = len(user2idx), len(item2idx)

    train_df, val_df, test_df = split_by_user_time(ratings_df)
    for df in [train_df, val_df, test_df]:
        df["user_idx"] = df["userId"].map(user2idx).astype("int64")
        df["item_idx"] = df["movieId"].map(item2idx).astype("int64")

    train_user_pos = build_user_pos_dict(train_df["user_idx"].values, train_df["item_idx"].values, n_users)
    val_user_pos = build_user_pos_dict(val_df["user_idx"].values, val_df["item_idx"].values, n_users)
    test_user_pos = build_user_pos_dict(test_df["user_idx"].values, test_df["item_idx"].values, n_users)

    trainval_seen = {u: (train_user_pos[u] | val_user_pos[u]) for u in range(n_users)}

    item_pop = np.zeros(n_items, dtype=np.int64)
    for u in range(n_users):
        for it in trainval_seen[u]:
            item_pop[it] += 1

    global_mean = float(train_df["rating"].mean()) if "rating" in train_df.columns else 0.0

    return {
        "cfg": cfg,
        "ratings_path": ratings_path,
        "ratings_df": ratings_df,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": np.array(idx2user, dtype=np.int64),
        "idx2item": np.array(idx2item, dtype=np.int64),
        "n_users": n_users,
        "n_items": n_items,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_user_pos": train_user_pos,
        "trainval_seen": trainval_seen,
        "test_user_pos": test_user_pos,
        "item_pop": item_pop,
        "tail_mask": build_long_tail_mask(item_pop, head_ratio=0.8),
        "device": device,
        "global_mean": global_mean,
    }


def main():
    meta = prepare_ml1m()
    cfg = meta["cfg"]

    device = meta["device"]
    n_users = meta["n_users"]
    n_items = meta["n_items"]

    train_df = meta["train_df"]
    val_df = meta["val_df"]
    test_user_pos = meta["test_user_pos"]
    trainval_seen = meta["trainval_seen"]
    item_pop = meta["item_pop"]
    tail_mask = meta["tail_mask"]
    ratings_path = meta["ratings_path"]

    eval_users = sorted(list(test_user_pos.keys()))
    K, N = get_rerank_kn(cfg, default_k=10, default_n=200)

    print(f"[Config] Rerank: K={K}, N={N}")

    item_vec = load_item_vectors_aligned_ml1m(n_items=n_items, item2idx=meta["item2idx"])

    reranker = AdaptiveMMRReranker(
        item_popularity=item_pop,
        tail_mask=tail_mask,
        item_vectors=item_vec,
        device=device,
    )

    results = []

    def _eval_and_log(tag: str, recs_items: Dict[int, List[int]]):
        rK, nK = evaluate_accuracy(recs_items, test_user_pos, k=K)
        cov = compute_item_coverage(recs_items, n_items)
        lt = compute_longtail_share(recs_items, tail_mask)
        nov = compute_novelty(recs_items, item_pop)
        results.append((tag, rK, nK, cov, lt, nov))
        print(f"[{tag}] Recall@{K}={rK:.4f} NDCG@{K}={nK:.4f} Cov@{K}={cov:.4f} LT@{K}={lt:.4f} Nov@{K}={nov:.4f}")

    def _eval_base_and_log(tag: str, cand: Dict[int, Tuple[List[int], List[float]]]):
        base_recs = build_base_topk_from_candidates(cand, K=K)
        base_items_only = {u: items for u, (items, _s) in cand.items()}
        hitN = evaluate_hit_rate_at_n(base_items_only, test_user_pos, N=N)
        rK, nK = evaluate_accuracy(base_recs, test_user_pos, k=K)
        cov = compute_item_coverage(base_recs, n_items)
        lt = compute_longtail_share(base_recs, tail_mask)
        nov = compute_novelty(base_recs, item_pop)
        results.append((f"{tag}-Base", rK, nK, cov, lt, nov))
        print(f"[{tag}-Base] Recall@{K}={rK:.4f} NDCG@{K}={nK:.4f} Hit@{N}={hitN:.4f}")
        print(f"[{tag}-Base] Recall@{K}={rK:.4f} NDCG@{K}={nK:.4f} Cov@{K}={cov:.4f} LT@{K}={lt:.4f} Nov@{K}={nov:.4f}")

    def _rerank(tag: str, cand: Dict[int, Tuple[List[int], List[float]]]):
        weights = get_rerank_weights_from_cfg(cfg, tag)
        print(f"[Config] Weights for {tag}: {weights}")

        recs = {}
        t0 = time.time()
        for idx, u in enumerate(eval_users, 1):
            items, scores = cand[u]
            hist_len = len(trainval_seen.get(u, set()))
            recs[u] = reranker.rerank_user(
                user_id=u,
                cand_items=items,
                cand_scores=scores,
                seen_items=trainval_seen.get(u, set()),
                K=K,
                hist_len=hist_len,
                weights=weights,
            )
            if idx % 500 == 0:
                print(f"[{tag}] {idx}/{len(eval_users)} reranked, {time.time() - t0:.1f}s")
        _eval_and_log(tag, recs)

    # ========= LightGCN =========
    lg_path = os.path.join(PROJECT_ROOT, "lightgcn_ml1m_best.pth")
    if os.path.exists(lg_path) and LightGCN is not None:
        adj = build_normalized_adj(n_users, n_items, train_df["user_idx"].values, train_df["item_idx"].values,
                                   device=device)
        lg = load_lightgcn(lg_path, adj, device=device)
        cand = topn_from_embeddings(lg, eval_users, trainval_seen, N=N, device=device)
        _eval_base_and_log("LightGCN-Rerank", cand)
        _rerank("LightGCN-Rerank", cand)

    # ========= NGCF =========
    ng_path = os.path.join(PROJECT_ROOT, "ngcf_ml1m_best.pth")
    if os.path.exists(ng_path) and NGCF is not None:
        adj = build_normalized_adj(n_users, n_items, train_df["user_idx"].values, train_df["item_idx"].values,
                                   device=device)
        ng = load_ngcf(ng_path, adj, device=device)
        cand = topn_from_embeddings(ng, eval_users, trainval_seen, N=N, device=device)
        _eval_base_and_log("NGCF-Rerank", cand)
        _rerank("NGCF-Rerank", cand)

    # ========= MMGCN =========
    mm_path = os.path.join(PROJECT_ROOT, "mmgcn_ml1m_best.pth")
    if os.path.exists(mm_path) and MMGCN is not None:
        mm = load_mmgcn(mm_path, meta)
        cand = topn_from_embeddings(mm, eval_users, trainval_seen, N=N, device=device)
        _eval_base_and_log("MMGCN-Rerank", cand)
        _rerank("MMGCN-Rerank", cand)

    # ========= HybridNCF =========
    hy_path = cfg.get("paths", {}).get("hybrid_ckpt", "hybrid_ml1m_best.pth")
    hy_path = hy_path if os.path.isabs(hy_path) else os.path.join(PROJECT_ROOT, hy_path)
    if not os.path.exists(hy_path):
        hy_path2 = cfg.get("paths", {}).get("hybrid_ckpt_safe", "hybrid_ml1m_best_safe.pth")
        hy_path2 = hy_path2 if os.path.isabs(hy_path2) else os.path.join(PROJECT_ROOT, hy_path2)
        hy_path = hy_path2

    if os.path.exists(hy_path) and HAS_HYBRID:
        hy = load_hybrid_robust(
            hy_path,
            ratings_path=ratings_path,
            idx2item=meta["idx2item"],
            n_items=n_items,
            device=device,
        )
        # 关键：test 推荐时历史应为 train+val
        trainval_df = pd.concat([train_df, val_df], ignore_index=True)
        maybe_refresh_hybrid_cache(hy, hist_df=trainval_df, n_users=n_users, device=device)

        cand = topn_pointwise_anyscore(hy, eval_users, trainval_seen, n_items=n_items, N=N, device=device, chunk=1024)
        _eval_base_and_log("HybridNCF-Rerank", cand)
        _rerank("HybridNCF-Rerank", cand)

    # ========= HybridAccNCF =========
    hyacc_path = cfg.get("paths", {}).get("hybrid_acc_ckpt", "hybrid_acc_ml1m_best.pth")
    hyacc_path = hyacc_path if os.path.isabs(hyacc_path) else os.path.join(PROJECT_ROOT, hyacc_path)
    if not os.path.exists(hyacc_path):
        hyacc_path2 = cfg.get("paths", {}).get("hybrid_acc_ckpt_safe", "hybrid_acc_ml1m_best_safe.pth")
        hyacc_path2 = hyacc_path2 if os.path.isabs(hyacc_path2) else os.path.join(PROJECT_ROOT, hyacc_path2)
        hyacc_path = hyacc_path2

    if os.path.exists(hyacc_path) and HAS_HYBRID and HybridNCFAcc is not None:
        hyacc = load_hybrid_acc_robust(
            hyacc_path,
            ratings_path=ratings_path,
            idx2item=meta["idx2item"],
            n_items=n_items,
            device=device,
        )
        trainval_df = pd.concat([train_df, val_df], ignore_index=True)
        maybe_refresh_hybrid_cache(hyacc, hist_df=trainval_df, n_users=n_users, device=device)

        cand = topn_pointwise_anyscore(hyacc, eval_users, trainval_seen, n_items=n_items, N=N, device=device,
                                       chunk=1024)
        _eval_base_and_log("HybridAcc-Rerank", cand)
        _rerank("HybridAcc-Rerank", cand)

    # ========= HybridTailNCF =========
    hytail_path = cfg.get("paths", {}).get("hybrid_tail_ckpt", "hybrid_tail_ml1m_best.pth")
    hytail_path = hytail_path if os.path.isabs(hytail_path) else os.path.join(PROJECT_ROOT, hytail_path)
    if not os.path.exists(hytail_path):
        hytail_path2 = cfg.get("paths", {}).get("hybrid_tail_ckpt_safe", "hybrid_tail_ml1m_best_safe.pth")
        hytail_path2 = hytail_path2 if os.path.isabs(hytail_path2) else os.path.join(PROJECT_ROOT, hytail_path2)
        hytail_path = hytail_path2

    if os.path.exists(hytail_path) and HAS_HYBRID and HybridNCFTail is not None:
        # 评估端用 train+val 的 item_pop（与 tail_mask / novelty 一致）
        hytail = load_hybrid_tail_robust(
            hytail_path,
            ratings_path=ratings_path,
            idx2item=meta["idx2item"],
            n_items=n_items,
            device=device,
            item_popularity=item_pop.astype("float32"),
        )
        trainval_df = pd.concat([train_df, val_df], ignore_index=True)
        maybe_refresh_hybrid_cache(hytail, hist_df=trainval_df, n_users=n_users, device=device)

        cand = topn_pointwise_anyscore(hytail, eval_users, trainval_seen, n_items=n_items, N=N, device=device,
                                       chunk=1024)
        _eval_base_and_log("HybridTail-Rerank", cand)
        _rerank("HybridTail-Rerank", cand)
    # ========= Save =========
    out_csv = os.path.join(PROJECT_ROOT, "results_v3_ml1m_rerank.csv")
    df = pd.DataFrame(results, columns=["Model", f"Recall@{K}", f"NDCG@{K}", f"Coverage@{K}", f"LongTailShare@{K}",
                                        f"Novelty@{K}"])
    print("\n========== V3 Rerank Results (ML-1M) ==========")
    print(df.to_string(index=False))
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
