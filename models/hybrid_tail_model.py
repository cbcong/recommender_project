# models/hybrid_tail_model.py
# -*- coding: utf-8 -*-
"""
HybridNCFTail: 以“长尾/新颖性/覆盖”等 beyond-accuracy 指标优先为目标的 HybridNCF 变体。

核心思路（尽量不改 HybridNCF 的结构与训练方式）：
- 在打分 logits 上加入“流行度惩罚项”，让模型更倾向于推荐长尾物品；
- 惩罚项形式可控，默认 log(1+pop) 并做归一化；
- 支持 learnable α、用户级缩放（基于历史平均流行度）与平滑动量，避免惩罚过度震荡；
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
        learnable_pop_alpha: bool = False,
        user_pop_scaling: bool = False,
        user_pop_scale_range: tuple[float, float] = (0.5, 1.5),
        user_pop_pref_momentum: float = 0.0,
        pop_penalty_cap: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.pop_mode = str(pop_mode)
        self.user_pop_scaling = bool(user_pop_scaling)
        self.user_pop_scale_range = tuple(user_pop_scale_range)
        self.user_pop_pref_momentum = float(user_pop_pref_momentum)
        self.pop_penalty_cap = None if pop_penalty_cap is None else float(pop_penalty_cap)

        if learnable_pop_alpha:
            self.pop_alpha = nn.Parameter(torch.tensor(float(pop_alpha), dtype=torch.float32))
        else:
            self.register_buffer("pop_alpha", torch.tensor(float(pop_alpha), dtype=torch.float32))

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
        self.register_buffer("user_popularity_pref", torch.zeros(self.num_users, dtype=torch.float32), persistent=False)

    def _current_pop_alpha(self, device: torch.device) -> torch.Tensor:
        alpha = self.pop_alpha
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(float(alpha), device=device)
        else:
            alpha = alpha.to(device)
        return torch.clamp(alpha, min=0.0)

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
        # popularity 变更后，按需刷新用户偏好
        if self._user_hist_items is not None and self._user_hist_lens is not None:
            self._refresh_user_pop_pref(self._user_hist_items, self._user_hist_lens)

    def _refresh_user_pop_pref(self, hist_items: torch.LongTensor, hist_lens: torch.LongTensor):
        """
        根据用户历史点击的流行度均值，生成用户级缩放因子（归一化到 [0,1]）。
        """
        if not self.user_pop_scaling:
            return
        if hist_items is None or hist_lens is None:
            self.user_popularity_pref.zero_()
            return

        device = hist_items.device
        pop = self.item_popularity.to(device)
        pad_mask = hist_items != self.PAD_IDX
        if pad_mask.numel() == 0:
            self.user_popularity_pref.zero_()
            return

        # 避免 PAD 越界
        safe_items = torch.where(pad_mask, hist_items.clamp(min=0, max=pop.numel() - 1), torch.zeros_like(hist_items))
        pop_seq = pop[safe_items] * pad_mask.float()

        hist_len = hist_lens.to(device).float().clamp(min=1.0)
        mean_pop = pop_seq.sum(dim=1) / hist_len

        if self.pop_mode == "raw":
            norm_pop = mean_pop
        elif self.pop_mode == "sqrt":
            norm_pop = torch.sqrt(torch.clamp(mean_pop, min=0.0))
        else:
            logp = torch.log1p(torch.clamp(mean_pop, min=0.0))
            denom = torch.log1p(torch.clamp(self.item_popularity.max().to(device), min=1.0))
            denom = torch.clamp(denom, min=1e-6)
            norm_pop = logp / denom

        norm_pop = torch.clamp(norm_pop, min=0.0, max=1.0)
        dest_dev = self.user_popularity_pref.device
        norm_pop = norm_pop.to(dest_dev)

        if self.user_pop_pref_momentum > 0:
            mom = float(self.user_pop_pref_momentum)
            self.user_popularity_pref.mul_(mom).add_(norm_pop * (1.0 - mom))
        else:
            self.user_popularity_pref.data.copy_(norm_pop)

    def _pop_penalty(self, item_idx: torch.LongTensor, user_idx: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        返回 [B] 惩罚项（非负）。
        """
        alpha = self._current_pop_alpha(item_idx.device)
        if torch.all(alpha <= 0):
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

        penalty = alpha * g

        if self.user_pop_scaling and user_idx is not None:
            low, high = self.user_pop_scale_range
            low = float(low)
            high = float(high)
            pref = self.user_popularity_pref.to(item_idx.device)[user_idx.long()].float()
            scale = low + (1.0 - pref) * (high - low)
            scale = torch.clamp(scale, min=min(low, high), max=max(low, high))
            penalty = penalty * scale

        if self.pop_penalty_cap is not None:
            penalty = torch.clamp(penalty, max=float(self.pop_penalty_cap))

        return penalty

    def set_user_histories(self, user_hist_items: torch.LongTensor, user_hist_lens: torch.LongTensor):
        super().set_user_histories(user_hist_items, user_hist_lens)
        self._refresh_user_pop_pref(user_hist_items, user_hist_lens)

    def score_logits(self, user_idx: torch.LongTensor, item_idx: torch.LongTensor) -> torch.Tensor:
        base = super().score_logits(user_idx, item_idx)
        penalty = self._pop_penalty(item_idx, user_idx=user_idx)
        return base - penalty
