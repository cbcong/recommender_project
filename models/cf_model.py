import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuMF(nn.Module):
    """
    经典 NeuMF：GMF + MLP 两路融合
    - GMF: element-wise product of user/item embeddings
    - MLP: concat user/item, 经过多层全连接
    - 最后拼接 GMF 输出和 MLP 输出，通过一层 FC 得到最终预测
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        mf_dim: int = 32,
        mlp_layer_sizes=(64, 32),
        dropout=0.0,
    ):
        super().__init__()

        # GMF 部分的 embedding
        self.user_embedding_gmf = nn.Embedding(num_users, mf_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, mf_dim)

        # MLP 部分的 embedding
        self.user_embedding_mlp = nn.Embedding(num_users, mlp_layer_sizes[0] // 2)
        self.item_embedding_mlp = nn.Embedding(num_items, mlp_layer_sizes[0] // 2)

        # MLP 全连接层
        mlp_layers = []
        input_size = mlp_layer_sizes[0]
        for hidden_size in mlp_layer_sizes[1:]:
            mlp_layers.append(nn.Dropout(dropout))
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            input_size = hidden_size
        self.mlp = nn.Sequential(*mlp_layers)

        # 输出层：GMF 输出 + MLP 输出 连接后 → 1 维评分
        self.output_layer = nn.Linear(mf_dim + mlp_layer_sizes[-1], 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, user_indices, item_indices):
        """
        user_indices, item_indices: LongTensor [batch_size]
        返回预测评分: FloatTensor [batch_size]
        """
        # GMF
        u_gmf = self.user_embedding_gmf(user_indices)
        i_gmf = self.item_embedding_gmf(item_indices)
        gmf_out = u_gmf * i_gmf  # element-wise product

        # MLP
        u_mlp = self.user_embedding_mlp(user_indices)
        i_mlp = self.item_embedding_mlp(item_indices)
        mlp_in = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_in)

        # 融合
        x = torch.cat([gmf_out, mlp_out], dim=-1)
        out = self.output_layer(x).squeeze(-1)  # [batch]
        return out
