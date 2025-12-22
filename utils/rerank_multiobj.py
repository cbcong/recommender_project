# utils/rerank_multiobj.py
# 多目标重排（MMR风格）：相关性 + 新颖度 + 长尾 + 多样性
# 支持：per-user 归一化、anchor_L 精度保护（前L个不重排）

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Tuple

import numpy as np


def minmax_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-Max 归一化到 [0,1]；若常数向量则返回全 0。"""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if abs(xmax - xmin) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin + eps)


def zscore_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Z-score 归一化；若方差很小则返回全 0。"""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    std = float(np.std(x))
    if std < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mu) / (std + eps)


def safe_unit_norm_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """把二维矩阵每行单位化，用于余弦相似度。"""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def compute_self_information_novelty(item_popularity: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Novelty(i) = -log2(pop(i) / sum pop)
    pop=0 时做平滑。
    """
    pop = np.asarray(item_popularity, dtype=np.float64)
    total = float(np.sum(pop))
    if total <= 0:
        return np.zeros_like(pop, dtype=np.float32)

    p = pop / total
    p[p <= 0] = eps
    info = -np.log2(p)
    return info.astype(np.float32)


def compute_longtail_reward(item_popularity: np.ndarray, power: float = 1.0) -> np.ndarray:
    """
    长尾奖励：pop 越小 reward 越大。
    一个稳健的选择：reward = 1 / (pop + 1)^power
    """
    pop = np.asarray(item_popularity, dtype=np.float32)
    return 1.0 / np.power(pop + 1.0, power)


@dataclass
class RerankWeights:
    w_rel: float = 0.85
    w_novelty: float = 0.05
    w_longtail: float = 0.05
    w_div: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        return {
            "w_rel": float(self.w_rel),
            "w_novelty": float(self.w_novelty),
            "w_longtail": float(self.w_longtail),
            "w_div": float(self.w_div),
        }


def mmr_rerank(
    candidate_items: Sequence[int],
    rel_scores: Sequence[float],
    novelty_scores: Optional[Sequence[float]] = None,
    longtail_scores: Optional[Sequence[float]] = None,
    item_emb: Optional[np.ndarray] = None,
    K: int = 10,
    weights: Optional[RerankWeights] = None,
    anchor_L: int = 0,
    normalize: str = "minmax",   # "minmax" | "zscore" | "none"
    div_mode: str = "max",       # "max" | "mean"
) -> List[int]:
    """
    多目标 MMR 重排：
      score(i | S) = w_rel * rel(i) + w_nov * nov(i) + w_lt * lt(i) + w_div * div(i,S)

    - candidate_items: 一般是 base 模型 Top-N 候选（N>=K），按 rel 降序会更合理
    - rel_scores: 与 candidate_items 对齐的 base 分数
    - novelty_scores / longtail_scores: 与 candidate_items 对齐
    - item_emb: [n_items, d] 的全量 embedding，用于多样性（可为空则不启用 div）
    - anchor_L: 前 L 个直接保留（精度保护），剩余位置做 MMR 选择
    """
    if weights is None:
        weights = RerankWeights()

    cand = np.asarray(list(candidate_items), dtype=np.int64)
    rel = np.asarray(list(rel_scores), dtype=np.float32)
    if cand.size == 0:
        return []

    K = int(K)
    if K <= 0:
        return []

    # 保底：候选不足就直接截断
    if cand.size <= K:
        return cand[:K].tolist()

    # 归一化函数
    if normalize == "minmax":
        norm_fn = minmax_norm
    elif normalize == "zscore":
        norm_fn = zscore_norm
    else:
        norm_fn = lambda x: np.asarray(x, dtype=np.float32)

    nov = np.zeros_like(rel, dtype=np.float32) if novelty_scores is None else np.asarray(novelty_scores, dtype=np.float32)
    lt = np.zeros_like(rel, dtype=np.float32) if longtail_scores is None else np.asarray(longtail_scores, dtype=np.float32)

    rel_n = norm_fn(rel)
    nov_n = norm_fn(nov)
    lt_n = norm_fn(lt)

    # anchor：直接保留前 L 个（默认按 candidate 的前 L 个）
    anchor_L = max(0, int(anchor_L))
    selected: List[int] = []
    if anchor_L > 0:
        anchor_L_eff = min(anchor_L, K, cand.size)
        selected = cand[:anchor_L_eff].tolist()
        # 从候选集中移除已选
        mask = np.ones(cand.size, dtype=bool)
        mask[:anchor_L_eff] = False
        cand_rem = cand[mask]
        rel_n_rem = rel_n[mask]
        nov_n_rem = nov_n[mask]
        lt_n_rem = lt_n[mask]
    else:
        cand_rem = cand
        rel_n_rem = rel_n
        nov_n_rem = nov_n
        lt_n_rem = lt_n

    need = K - len(selected)
    if need <= 0:
        return selected[:K]

    # 多样性项准备
    use_div = (item_emb is not None) and (weights.w_div is not None) and (float(weights.w_div) != 0.0)
    if use_div:
        emb_all = np.asarray(item_emb, dtype=np.float32)
        # 注意：candidate 是 item idx（0..n_items-1）
        emb_cand = emb_all[cand_rem]  # [Nc, d]
        emb_cand = safe_unit_norm_rows(emb_cand)
    else:
        emb_cand = None

    # base 部分（与 S 无关）
    base = (
        float(weights.w_rel) * rel_n_rem +
        float(weights.w_novelty) * nov_n_rem +
        float(weights.w_longtail) * lt_n_rem
    )

    # 逐个选择
    chosen_mask = np.zeros(cand_rem.size, dtype=bool)

    # 若启用 div，需要维护 selected 的 embedding（unit norm）
    if use_div:
        sel_emb_list: List[np.ndarray] = []
        if len(selected) > 0:
            sel_emb = safe_unit_norm_rows(emb_all[np.asarray(selected, dtype=np.int64)])
            for i in range(sel_emb.shape[0]):
                sel_emb_list.append(sel_emb[i])

    for _ in range(need):
        remain_idx = np.where(~chosen_mask)[0]
        if remain_idx.size == 0:
            break

        scores = base[remain_idx].copy()

        if use_div and len(selected) > 0 and emb_cand is not None and len(sel_emb_list) > 0:
            # 计算 div：1 - max_cos 或 1 - mean_cos
            sel_mat = np.stack(sel_emb_list, axis=0)  # [|S|, d]
            cand_mat = emb_cand[remain_idx]           # [R, d]
            sims = cand_mat @ sel_mat.T               # [R, |S|]

            if div_mode == "mean":
                sim_agg = np.mean(sims, axis=1)
            else:
                sim_agg = np.max(sims, axis=1)

            div = 1.0 - sim_agg                        # [-? , 2]
            # 映射到 [0,1]：余弦相似度在 [-1,1]，1-sim 在 [0,2]，除以2
            div01 = np.clip(div / 2.0, 0.0, 1.0).astype(np.float32)

            scores += float(weights.w_div) * div01

        best_local = int(remain_idx[int(np.argmax(scores))])
        chosen_mask[best_local] = True
        chosen_item = int(cand_rem[best_local])
        selected.append(chosen_item)

        if use_div and emb_cand is not None:
            sel_emb_list.append(emb_cand[best_local])

        if len(selected) >= K:
            break

    return selected[:K]


def rerank_one_user_topn(
    topn_items: np.ndarray,
    topn_scores: np.ndarray,
    item_novelty_all: Optional[np.ndarray],
    item_longtail_all: Optional[np.ndarray],
    item_emb_all: Optional[np.ndarray],
    K: int,
    weights: RerankWeights,
    anchor_L: int,
    normalize: str,
    div_mode: str,
) -> List[int]:
    """给 tune 脚本用的便捷包装：输入 Top-N，输出 Top-K。"""
    nov = None if item_novelty_all is None else item_novelty_all[topn_items]
    lt = None if item_longtail_all is None else item_longtail_all[topn_items]
    return mmr_rerank(
        candidate_items=topn_items,
        rel_scores=topn_scores,
        novelty_scores=nov,
        longtail_scores=lt,
        item_emb=item_emb_all,
        K=K,
        weights=weights,
        anchor_L=anchor_L,
        normalize=normalize,
        div_mode=div_mode,
    )
