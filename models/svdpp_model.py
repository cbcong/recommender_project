import math
from typing import Dict, Iterable, List, Set

import torch
import torch.nn as nn


class SVDPP(nn.Module):
    """
    经典 SVD++ 实现（带用户/物品偏置 + 隐式反馈 y_j）：

    r_hat(u, i) = μ + b_u + b_i + q_i^T ( p_u + |N(u)|^{-1/2} * sum_{j∈N(u)} y_j )

    - num_users: 用户数量
    - num_items: 物品数量
    - embedding_dim: 隐向量维度
    - user_interactions: dict[u] = set(items)，只用训练集构建
    - global_mean: 训练集评分均值
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        user_interactions: Dict[int, Iterable[int]],
        global_mean: float = 0.0,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # 主隐向量 p_u、q_i
        self.user_factors = nn.Embedding(num_users, embedding_dim)
        self.item_factors = nn.Embedding(num_items, embedding_dim)

        # 隐式反馈向量 y_j
        self.item_implicit_factors = nn.Embedding(num_items, embedding_dim)

        # 用户/物品偏置
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # 全局均值（可以设为可训练，也可以设为常量）
        self.global_mean = nn.Parameter(
            torch.tensor(float(global_mean), dtype=torch.float32),
            requires_grad=False,  # 简化：直接当常量使用
        )

        # Python 字典，记录每个用户在训练集中交互过的物品集合
        # 纯 Python 对象，不会成为参数
        self.user_interactions: Dict[int, List[int]] = {}
        for u, items in user_interactions.items():
            # 统一存成 list，避免 set 的遍历顺序不稳定
            self.user_interactions[int(u)] = list(set(int(i) for i in items))

        self._init_parameters()

    def _init_parameters(self):
        # 参考 MF 的常规初始化方法
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.normal_(self.item_implicit_factors.weight, std=0.01)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def _aggregate_implicit(self, user_idx: torch.LongTensor) -> torch.Tensor:
        """
        对一个 batch 的用户，计算：
        |N(u)|^{-1/2} * sum_{j∈N(u)} y_j

        返回形状 [batch_size, embedding_dim]
        """
        device = self.item_implicit_factors.weight.device
        user_idx = user_idx.to(device)

        # 找出 batch 中的 unique 用户
        unique_users, inverse_indices = torch.unique(
            user_idx, return_inverse=True
        )  # unique_users: [U], inverse_indices: [B]

        # 为每个 unique 用户计算一遍 sum_y，再映射回 batch
        U = unique_users.size(0)
        agg_implicit = torch.zeros(
            (U, self.embedding_dim), dtype=torch.float32, device=device
        )

        for idx, u_tensor in enumerate(unique_users):
            u = int(u_tensor.item())
            items = self.user_interactions.get(u, None)
            if not items:
                # 该用户在训练集中无交互（极少数情况），就当 0 处理
                continue

            item_ids = torch.tensor(items, dtype=torch.long, device=device)
            y_j = self.item_implicit_factors(item_ids)  # [n_u, dim]
            norm = math.sqrt(len(items))
            agg_implicit[idx] = y_j.sum(dim=0) / norm

        # 按 inverse_indices 映射回 batch
        batch_implicit = agg_implicit[inverse_indices]  # [B, dim]
        return batch_implicit

    def forward(self, user_idx: torch.LongTensor, item_idx: torch.LongTensor) -> torch.Tensor:
        """
        user_idx: [B]
        item_idx: [B]
        返回预测评分 [B]
        """
        device = self.user_factors.weight.device
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)

        p_u = self.user_factors(user_idx)          # [B, dim]
        q_i = self.item_factors(item_idx)          # [B, dim]
        b_u = self.user_bias(user_idx).squeeze(-1)  # [B]
        b_i = self.item_bias(item_idx).squeeze(-1)  # [B]

        implicit_u = self._aggregate_implicit(user_idx)  # [B, dim]

        # p_u + implicit_part
        user_rep = p_u + implicit_u                   # [B, dim]
        dot = (user_rep * q_i).sum(dim=-1)           # [B]

        pred = self.global_mean + b_u + b_i + dot    # [B]
        return pred
