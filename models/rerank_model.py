# recommender_project/models/rerank_model.py
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch


def _minmax_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """把 1D tensor 归一化到 [0, 1]."""
    x_min = torch.min(x)
    x_max = torch.max(x)
    denom = (x_max - x_min).clamp_min(eps)
    return (x - x_min) / denom


@dataclass
class RerankWeights:
    # 基础权重（warm 用户）
    w_rel: float = 1.00       # 相关性（base score）
    w_novel: float = 0.10     # 新颖性
    w_tail: float = 0.10      # 长尾奖励
    w_div: float = 0.10       # 多样性（MMR 的相似度惩罚）
    # 冷启动增强系数（cold 用户会把 novel/tail/div 放大）
    cold_boost: float = 2.00
    # 冷启动判定：hist_len >= hist_ref 认为“暖”
    hist_ref: int = 50


class AdaptiveMMRReranker:
    """
    自适应多目标重排序（候选集上做 Top-K）
    - relevance: 来自 base 模型的打分
    - novelty:   -log2(pop / sum_pop)
    - tail:      长尾 mask
    - diversity: MMR（用内容向量或 item embedding 的 cosine 相似度）

    你可以把它作为 V3 的“方法贡献模块”写进论文，且可消融：
      - 去掉冷启动自适应（cold_boost=1）
      - 去掉多样性项（w_div=0）
      - 去掉长尾项（w_tail=0）等
    """
    def __init__(
        self,
        item_popularity: np.ndarray,
        tail_mask: np.ndarray,
        item_vectors: Optional[np.ndarray] = None,
        device: str = "cpu",
        eps: float = 1e-12,
    ):
        assert item_popularity.ndim == 1
        assert tail_mask.ndim == 1
        self.device = torch.device(device)
        self.eps = eps

        pop = torch.tensor(item_popularity, dtype=torch.float32, device=self.device)
        pop = pop.clamp_min(0.0)

        total = float(torch.sum(pop).item())
        if total <= 0:
            novelty = torch.zeros_like(pop)
        else:
            p = (pop / total).clamp_min(eps)
            novelty = -torch.log2(p)

        self.popularity = pop
        self.novelty = novelty
        self.tail = torch.tensor(tail_mask.astype(np.float32), device=self.device)

        # 预处理 item vectors（用于多样性）
        if item_vectors is not None:
            vec = torch.tensor(item_vectors, dtype=torch.float32, device=self.device)
            # L2 normalize for cosine similarity
            norm = torch.norm(vec, dim=1, keepdim=True).clamp_min(eps)
            self.vec = vec / norm
        else:
            self.vec = None

    def rerank_user(
        self,
        user_id: int,
        cand_items: List[int],
        cand_scores: List[float],
        seen_items: Optional[set],
        K: int,
        hist_len: int,
        weights: RerankWeights,
    ) -> List[int]:
        """
        输入：
          cand_items/cand_scores: base 模型的 Top-N 候选（N 100~300）
          seen_items: 训练/验证已交互 item，用于安全过滤
          hist_len: 用户历史长度（用来决定 cold_factor）
        输出：
          Top-K list
        """
        if len(cand_items) == 0:
            return []

        cand_items_t = torch.tensor(cand_items, dtype=torch.long, device=self.device)
        rel = torch.tensor(cand_scores, dtype=torch.float32, device=self.device)

        # 安全：过滤 seen_items（候选阶段一般已过滤，但这里兜底）
        if seen_items:
            mask = torch.ones(len(cand_items), dtype=torch.bool, device=self.device)
            seen = torch.tensor(list(seen_items), dtype=torch.long, device=self.device)
            # seen 可能很大，这里用 isin 的简化写法（小规模 N 候选足够）
            for s in seen:
                mask = mask & (cand_items_t != s)
            cand_items_t = cand_items_t[mask]
            rel = rel[mask]
            if cand_items_t.numel() == 0:
                return []

        # 归一化各项到 [0,1]，保证加权可控
        rel_n = _minmax_norm(rel, eps=self.eps)
        nov_n = _minmax_norm(self.novelty[cand_items_t], eps=self.eps)
        tail = self.tail[cand_items_t]

        # cold_factor：越冷越接近 1，越暖越接近 0
        cold_factor = 1.0 - min(float(hist_len) / float(max(1, weights.hist_ref)), 1.0)
        boost = 1.0 + weights.cold_boost * cold_factor

        w_rel = weights.w_rel
        w_nov = weights.w_novel * boost
        w_tail = weights.w_tail * boost
        w_div = weights.w_div * boost

        selected: List[int] = []
        selected_idx: List[int] = []

        # 多样性需要向量
        if self.vec is None:
            # 如果没有向量，就退化为“只做 rel/nov/tail”的可复现策略
            score = w_rel * rel_n + w_nov * nov_n + w_tail * tail
            topk = torch.topk(score, k=min(K, score.numel())).indices
            return cand_items_t[topk].detach().cpu().numpy().tolist()

        cand_vec = self.vec[cand_items_t]  # [N, d]

        N = cand_items_t.numel()
        K_eff = min(K, N)

        remaining = torch.ones(N, dtype=torch.bool, device=self.device)

        for _ in range(K_eff):
            if len(selected_idx) == 0:
                max_sim = torch.zeros(N, dtype=torch.float32, device=self.device)
            else:
                sel_vec = cand_vec[selected_idx]  # [t, d]
                # cosine sim: [N, t]
                sim = cand_vec @ sel_vec.T
                max_sim = torch.max(sim, dim=1).values

            mmr_score = (w_rel * rel_n) + (w_nov * nov_n) + (w_tail * tail) - (w_div * max_sim)

            mmr_score = torch.where(remaining, mmr_score, torch.tensor(-1e9, device=self.device))
            pick = int(torch.argmax(mmr_score).item())

            selected_idx.append(pick)
            selected.append(int(cand_items_t[pick].item()))
            remaining[pick] = False

        return selected
