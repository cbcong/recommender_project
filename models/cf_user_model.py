import torch
import torch.nn as nn


class NCFUserFeat(nn.Module):
    """
    一个简单的“Neural CF + 用户特征”模型：
    - user_id / item_id 各自有 embedding
    - 用户特征（例如年龄）通过线性层映射到同维度
    - 三者拼接后过 MLP，输出评分
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_feat_dim: int,
        emb_dim: int = 32,
        mlp_layer_sizes=(64, 32, 16),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_feat_dim = user_feat_dim
        self.emb_dim = emb_dim

        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)

        # 用户特征映射到 emb_dim 维
        self.user_feat_proj = nn.Linear(user_feat_dim, emb_dim)

        mlp_input_dim = emb_dim * 3  # user_emb, item_emb, user_feat_emb
        layers = []
        in_dim = mlp_input_dim
        for h in mlp_layer_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

        # 存放 [num_users, user_feat_dim] 的特征表（非参数）
        self.register_buffer("user_features", None)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.user_feat_proj.weight)
        nn.init.zeros_(self.user_feat_proj.bias)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def set_user_features(self, user_feat_tensor: torch.Tensor):
        """
        设置用户特征矩阵：形状 [num_users, user_feat_dim]，放到同一 device 上。
        """
        if user_feat_tensor.shape[0] != self.num_users:
            raise ValueError(
                f"user_feat_tensor.shape[0]={user_feat_tensor.shape[0]} "
                f"but num_users={self.num_users}"
            )
        if user_feat_tensor.shape[1] != self.user_feat_dim:
            raise ValueError(
                f"user_feat_tensor.shape[1]={user_feat_tensor.shape[1]} "
                f"but user_feat_dim={self.user_feat_dim}"
            )
        # 使用 buffer 存储，不参与梯度更新
        self.user_features = user_feat_tensor

    def forward(self, user_idx, item_idx):
        u_emb = self.user_embedding(user_idx)   # [B, emb_dim]
        i_emb = self.item_embedding(item_idx)   # [B, emb_dim]

        if self.user_features is None:
            raise RuntimeError("User features not set. Call set_user_features() first.")

        # [B, user_feat_dim]
        u_feat = self.user_features[user_idx]
        u_feat_emb = self.user_feat_proj(u_feat)  # [B, emb_dim]

        x = torch.cat([u_emb, i_emb, u_feat_emb], dim=-1)  # [B, 3*emb_dim]
        x = self.mlp(x)
        x = self.output_layer(x)  # [B, 1]
        return x.squeeze(-1)
