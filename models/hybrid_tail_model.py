# models/hybrid_tail_model.py
# -*- coding: utf-8 -*-
"""
HybridNCFTail: 以“长尾/新颖性/覆盖”等 beyond-accuracy 指标优先为目标的 HybridNCF 变体。

核心思路（尽量不改 HybridNCF 的结构与训练方式）：
- 在打分 logits 上加入“流行度惩罚项”，让模型更倾向于推荐长尾物品；
- 惩罚项形式可控，默认 log(1+pop) 并做归一化；
- 训练与推理使用同一 score_logits 逻辑，避免 train/test 不一致。

logit' = logit_base - α * g(pop(item))
其中：
  g(pop) = log(1+pop) / log(1+pop_max)  （默认）
  α 为强度（pop_alpha）
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from models.hybrid_model import HybridNCF
except Exception:
    from hybrid_model import HybridNCF


class HybridNCFTail(HybridNCF):
    def __init__(
        self,
        *args,
        item_popularity: Optional[Union[np.ndarray, torch.Tensor]] = None,
        pop_alpha: float = 0.30,
        pop_mode: str = "log_norm",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pop_alpha = float(pop_alpha)
        self.pop_mode = str(pop_mode)

        # buffer: [n_items]，用于推理与训练一致
        n_items = int(self.num_items)
        if item_popularity is None:
            pop = torch.zeros(n_items, dtype=torch.float32)
        else:
            if isinstance(item_popularity, np.ndarray):
                pop = torch.from_numpy(item_popularity.astype("float32"))
            elif torch.is_tensor(item_popularity):
                pop = item_popularity.detach().float().cpu()
            else:
                raise TypeError("item_popularity must be np.ndarray or torch.Tensor")
            if pop.numel() != n_items:
                # 尝试裁剪/补齐
                pop2 = torch.zeros(n_items, dtype=torch.float32)
                L = min(n_items, int(pop.numel()))
                pop2[:L] = pop.view(-1)[:L]
                pop = pop2

        self.register_buffer("item_popularity", pop, persistent=True)

    @torch.no_grad()
    def set_item_popularity(self, item_popularity: Union[np.ndarray, torch.Tensor]):
        n_items = int(self.num_items)
        if isinstance(item_popularity, np.ndarray):
            pop = torch.from_numpy(item_popularity.astype("float32"))
        elif torch.is_tensor(item_popularity):
            pop = item_popularity.detach().float().cpu()
        else:
            raise TypeError("item_popularity must be np.ndarray or torch.Tensor")
        if pop.numel() != n_items:
            pop2 = torch.zeros(n_items, dtype=torch.float32)
            L = min(n_items, int(pop.numel()))
            pop2[:L] = pop.view(-1)[:L]
            pop = pop2
        self.item_popularity.data.copy_(pop)

    def _pop_penalty(self, item_idx: torch.LongTensor) -> torch.Tensor:
        """
        返回 [B] 惩罚项（非负）。
        """
        if self.pop_alpha <= 0:
            return torch.zeros(item_idx.size(0), device=item_idx.device, dtype=torch.float32)

        pop = self.item_popularity.to(item_idx.device)[item_idx.long()].float()  # [B]
        # g(pop)
        if self.pop_mode == "raw":
            g = pop
        elif self.pop_mode == "sqrt":
            g = torch.sqrt(torch.clamp(pop, min=0.0))
        else:
            # 默认 log_norm：log(1+pop) / log(1+pop_max)
            logp = torch.log1p(torch.clamp(pop, min=0.0))
            pop_max = torch.max(self.item_popularity).to(item_idx.device).float()
            denom = torch.log1p(torch.clamp(pop_max, min=1.0))
            denom = torch.clamp(denom, min=1e-6)
            g = logp / denom

        return self.pop_alpha * g

    def score_logits(self, user_idx: torch.LongTensor, item_idx: torch.LongTensor) -> torch.Tensor:
        base = super().score_logits(user_idx, item_idx)
        penalty = self._pop_penalty(item_idx)
        return base - penalty
