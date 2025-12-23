# models/hybrid_acc_model.py
# -*- coding: utf-8 -*-
"""
HybridNCFAcc: 准确性优先（Recall/NDCG 优先）的 HybridNCF 变体。

核心思想：
- 不改 HybridNCF 的主打分/融合结构；
- 仅在训练阶段额外加入一个“内容-协同表征对齐”辅助目标（InfoNCE），
  强化内容特征对协同表示的贡献，从而提升 Top-K 准确性。
- 该对齐损失由 train_hybrid_acc.py 调用（model.has_align / model.align_loss）。

注意：
HybridNCF 本体未显式保存 self.mlp_dim，因此本文件会在 __init__ 中补齐该属性，
否则外部使用会触发 AttributeError（你刚遇到的报错）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from hybrid_model import HybridNCF
except Exception:
    from models.hybrid_model import HybridNCF


class HybridNCFAcc(HybridNCF):
    """
    训练时可配合 InfoNCE 对齐损失：
        L_align = 0.5 * (CE(S, y) + CE(S^T, y)),
    其中 S_ij = (z_cf(i) · z_ct(j)) / tau，y = [0,1,...,B-1]。

    - z_cf(i): item 的协同表示（item_embedding_mlp）
    - z_ct(i): item 的内容表示（content_encoder 输出经线性投影）
    """

    def __init__(
        self,
        *args,
        align_proj_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # ---------- 关键修复：HybridNCF 本体不一定有 self.mlp_dim ----------
        # 以 embedding 的维度为准（最可靠）。
        try:
            self.mlp_dim = int(self.user_embedding_mlp.embedding_dim)
        except Exception:
            # 极端兜底：从权重 shape 推断
            self.mlp_dim = int(self.user_embedding_mlp.weight.shape[1])

        # ---------- 对齐模块 ----------
        self.align_dim = int(align_proj_dim) if align_proj_dim is not None else int(self.mlp_dim)

        if int(getattr(self, "content_dim", 0)) > 0:
            # CF item -> align space
            self.item_align_proj = nn.Linear(self.mlp_dim, self.align_dim, bias=False) if self.align_dim != self.mlp_dim else nn.Identity()
            # content(raw) -> align space
            self.content_align_proj = nn.Linear(int(self.content_dim), self.align_dim, bias=False)
            self._has_align = True
        else:
            self.item_align_proj = None
            self.content_align_proj = None
            self._has_align = False

    def has_align(self) -> bool:
        return bool(self._has_align)

    def align_loss(
        self,
        item_idx: torch.LongTensor,
        temperature: float = 0.2,
        sample_size: int | None = None,
    ) -> torch.Tensor:
        """
        InfoNCE 对齐损失（建议在训练脚本对子采样后调用，避免 O(B^2) 过慢）。

        参数:
            item_idx: [B] item indices
            temperature: tau
            sample_size: 可选，随机子采样一个子批次以降低显存/计算量
        返回:
            标量 loss
        """
        if not self.has_align():
            return torch.tensor(0.0, device=item_idx.device)

        item_idx = item_idx.long()
        if sample_size is not None and sample_size > 0 and sample_size < item_idx.numel():
            perm = torch.randperm(item_idx.numel(), device=item_idx.device)[:sample_size]
            item_idx = item_idx[perm]

        # CF 表征（mlp item embedding）
        z_cf = self.item_embedding_mlp(item_idx)  # [B, mlp_dim]
        z_cf = self.item_align_proj(z_cf)         # [B, align_dim]

        # 内容表征（raw content -> align_dim）
        z_ct = self.content_encoder(item_idx)     # [B, content_dim]
        z_ct = self.content_align_proj(z_ct)      # [B, align_dim]

        z_cf = F.normalize(z_cf, dim=-1)
        z_ct = F.normalize(z_ct, dim=-1)

        # logits: [B,B]
        logits = (z_cf @ z_ct.t()) / float(temperature)
        labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)

        loss_i2c = F.cross_entropy(logits, labels)
        loss_c2i = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i2c + loss_c2i)
