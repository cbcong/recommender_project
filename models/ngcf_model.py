import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    """
    一个简化版的 NGCF 实现：
    - 以 LightGCN 的接口为参考：
        * __init__(num_users, num_items, embedding_dim, num_layers, adj, dropout)
        * forward(users, pos_items, neg_items) -> (u_e, pos_e, neg_e)
        * get_user_item_embeddings() -> (user_emb, item_emb)
    - 使用 normalized adj (稀疏) 做多层消息传递；
    - 每层包含：
        * 邻居聚合: side_embeddings = A_hat * ego_embeddings
        * 残差 + 双线性项: sum_emb = side + ego, bi_emb = side * ego
        * 线性变换 + LeakyReLU
        * Dropout
    - 最后把所有层（含初始）做平均，得到最终用户 / 物品嵌入。
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        adj: torch.Tensor = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if adj is None:
            raise ValueError("adj (normalized adjacency) must be provided for NGCF.")
        self.adj = adj  # 稀疏张量 [num_nodes, num_nodes]

        # 用户 / 物品 ID 嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # NGCF 每层的线性变换
        # neighbor part + bi-interaction part
        self.W_gc = nn.ModuleList()
        self.W_bi = nn.ModuleList()
        for _ in range(num_layers):
            self.W_gc.append(nn.Linear(embedding_dim, embedding_dim))
            self.W_bi.append(nn.Linear(embedding_dim, embedding_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self._init_weights()

    def _init_weights(self):
        # Xavier 初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        for layer in self.W_gc:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        for layer in self.W_bi:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def get_all_embeddings(self):
        """
        执行 K 层 NGCF 消息传递，返回最终的用户 / 物品嵌入。
        """
        # [num_nodes, d]
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )

        all_embeddings = [ego_embeddings]

        for layer in range(self.num_layers):
            # 邻居聚合
            side_embeddings = torch.sparse.mm(self.adj, ego_embeddings)  # [N, d]

            # 残差项 + 双线性交互
            sum_embeddings = side_embeddings + ego_embeddings
            bi_embeddings = side_embeddings * ego_embeddings

            # 线性变换 + LeakyReLU
            out = self.W_gc[layer](sum_embeddings) + self.W_bi[layer](bi_embeddings)
            out = self.leaky_relu(out)

            if self.dropout is not None:
                out = self.dropout(out)

            ego_embeddings = out
            all_embeddings.append(ego_embeddings)

        # 把所有层的 embedding 做平均
        all_embeddings = torch.stack(all_embeddings, dim=1)  # [N, L+1, d]
        all_embeddings = torch.mean(all_embeddings, dim=1)   # [N, d]

        user_emb, item_emb = torch.split(
            all_embeddings, [self.num_users, self.num_items], dim=0
        )
        return user_emb, item_emb

    def get_user_item_embeddings(self):
        return self.get_all_embeddings()

    def forward(self, users, pos_items, neg_items):
        """
        BPR 训练接口：
        - users: [B]
        - pos_items: [B]
        - neg_items: [B]
        返回这三者对应的 embedding。
        """
        user_emb, item_emb = self.get_all_embeddings()

        u_e = user_emb[users]       # [B, d]
        pos_e = item_emb[pos_items] # [B, d]
        neg_e = item_emb[neg_items] # [B, d]

        return u_e, pos_e, neg_e
