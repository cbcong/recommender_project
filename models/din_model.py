# models/din_model.py
import torch
import torch.nn as nn


class Dice(nn.Module):
    """
    DIN 论文里的 Dice 激活函数：
    y = p * x + (1 - p) * alpha * x
    其中 p = sigmoid(BN(x))，alpha 是可学习参数（向量）。
    """
    def __init__(self, input_dim: int, eps: float = 1e-8):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=eps)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, C]

        # BatchNorm1d: 输入 [N, C]
        x_norm = self.bn(x)
        p = torch.sigmoid(x_norm)
        return p * x + (1 - p) * self.alpha * x


class AttentionUnit(nn.Module):
    """
    DIN 的局部激活单元（Local Activation Unit）：
    对于每个历史行为向量 h 和当前候选向量 a，构造：
      z = [h, a, h - a, h * a]
    然后通过若干层全连接 + Dice 激活得到一个标量注意力权重。
    """
    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(64, 32, 16),
        use_dice: bool = True,
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_sizes)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_dice:
                layers.append(Dice(dims[i + 1]))
            else:
                layers.append(nn.PReLU())
        self.mlp = nn.Sequential(*layers)

        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, input_dim]
        返回: [B, L, 1]
        """
        B, L, D = x.shape
        x_flat = x.view(B * L, D)  # [B*L, D]
        h = self.mlp(x_flat)      # [B*L, H]
        out = self.out(h)         # [B*L, 1]
        out = out.view(B, L, 1)
        return out


class FullyConnected(nn.Module):
    """
    DIN 输出侧的全连接网络：
    输入拼接 [e_user, e_hist, e_target]，再经过若干层 FC + Dice，输出标量评分。
    """
    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(200, 80),
        dropout: float = 0.0,
        use_dice: bool = True,
    ):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_sizes)

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_dice:
                layers.append(Dice(dims[i + 1]))
            else:
                layers.append(nn.PReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        返回: [B, 1]
        """
        h = self.mlp(x)
        out = self.out(h)
        return out


class DIN(nn.Module):
    """
    Deep Interest Network（适配显式评分场景）：

    输入：
      - user_ids:      [B]          用户索引
      - hist_item_ids: [B, L]       用户历史交互 item 序列（已补 0）
      - hist_lens:     [B]          每条序列真实长度（不含 padding）
      - target_item_ids: [B]        当前要预测评分的 item

    输出：
      - preds: [B]     对应评分预测（实数标量）
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int = 32,
        att_hidden_sizes=(64, 32, 16),
        fc_hidden_sizes=(200, 80),
        max_history_len: int = 50,
        dropout: float = 0.0,
        use_dice: bool = True,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_history_len = max_history_len

        # 用户 / 物品 Embedding
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        # item_embedding: padding_idx=0，用于历史序列的 0-padding
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)

        # 注意力模块（Local Activation Unit）
        # 输入是 [h, a, h - a, h * a]，维度 = 4 * embed_dim
        self.attention = AttentionUnit(
            input_dim=4 * embed_dim,
            hidden_sizes=att_hidden_sizes,
            use_dice=use_dice,
        )

        # 输出侧的全连接网络
        # 输入拼接 [e_user, e_hist_att, e_target] -> 3 * embed_dim
        self.fc = FullyConnected(
            input_dim=3 * embed_dim,
            hidden_sizes=fc_hidden_sizes,
            dropout=dropout,
            use_dice=use_dice,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        轻微初始化（可选）。
        """
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)
        if self.item_embedding.padding_idx is not None:
            with torch.no_grad():
                self.item_embedding.weight[self.item_embedding.padding_idx].fill_(0.0)

    def forward(
        self,
        user_ids: torch.Tensor,
        hist_item_ids: torch.Tensor,
        hist_lens: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向计算：
          1. 取用户 embedding: e_u
          2. 取历史 item embedding: E_hist [B, L, D]
          3. 取目标 item embedding: e_a [B, D] -> broadcast 为 [B, L, D]
          4. 构造 DIN 的局部特征：[E_hist, E_a, E_hist - E_a, E_hist * E_a]
          5. AttentionUnit 得到注意力权重，mask 掉 padding
          6. 得到加权历史兴趣向量 e_hist_att
          7. 拼接 [e_u, e_hist_att, e_a]，经过 FC 得到评分
        """
        # [B]
        user_ids = user_ids.long()
        target_item_ids = target_item_ids.long()
        hist_item_ids = hist_item_ids.long()
        hist_lens = hist_lens.long()

        # Embedding
        e_u = self.user_embedding(user_ids)          # [B, D]
        e_a = self.item_embedding(target_item_ids)   # [B, D]
        E_hist = self.item_embedding(hist_item_ids)  # [B, L, D]

        B, L, D = E_hist.shape

        # mask: padding 位置为 0 -> mask = 0
        mask = (hist_item_ids > 0).float()  # [B, L]

        # 扩展 e_a 为 [B, L, D]
        E_a = e_a.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]

        # 构造局部特征
        att_input = torch.cat(
            [
                E_hist,
                E_a,
                E_hist - E_a,
                E_hist * E_a,
            ],
            dim=-1,
        )  # [B, L, 4D]

        # 注意力得分: [B, L, 1] -> [B, L]
        att_scores = self.attention(att_input).squeeze(-1)  # [B, L]

        # mask 掉 padding 位置
        att_scores = att_scores.masked_fill(mask == 0, float("-inf"))

        # softmax 前，防止全部为 -inf 导致 NaN：用一个很小的偏置
        att_scores = att_scores + (mask + 1e-8).log()  # mask=0 的位置仍为 -inf

        # 注意力权重
        att_weights = torch.softmax(att_scores, dim=-1)  # [B, L]
        # 再乘一次 mask，防止数值误差
        att_weights = att_weights * mask
        # 归一化（防止全 0）
        att_sum = att_weights.sum(dim=-1, keepdim=True) + 1e-8
        att_weights = att_weights / att_sum  # [B, L]

        # 加权求和得到历史兴趣向量: [B, D]
        e_hist_att = torch.bmm(att_weights.unsqueeze(1), E_hist).squeeze(1)  # [B, D]

        # 拼接 [e_u, e_hist_att, e_a] -> [B, 3D]
        x = torch.cat([e_u, e_hist_att, e_a], dim=-1)

        # FC 输出评分: [B, 1] -> [B]
        out = self.fc(x).squeeze(-1)

        return out
