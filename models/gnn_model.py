import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """
    经典 LightGCN 实现（简化版）：
    - 节点 = 用户 + 物品
    - 使用预先构建好的归一化邻接矩阵 A_hat（稀疏）
    - K 层传播 + 均值聚合得到最终嵌入
    - 训练时用 BPR 损失
    """

    def __init__(self, num_users, num_items, embedding_dim, num_layers, adj):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # 节点嵌入（用户+物品）
        self.embedding = nn.Embedding(self.num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

        # 归一化后的稀疏邻接矩阵（torch.sparse_coo_tensor）
        self.adj = adj

    def computer(self):
        """
        执行 K 层图传播，返回最终所有节点嵌入。
        """
        all_emb = self.embedding.weight  # [N, d]
        embs = [all_emb]

        for _ in range(self.num_layers):
            # 稀疏矩阵乘法：A_hat @ all_emb
            all_emb = torch.sparse.mm(self.adj, all_emb)
            embs.append(all_emb)

        # [K+1, N, d] → 在层方向上做平均
        embs = torch.stack(embs, dim=0)
        out = torch.mean(embs, dim=0)  # [N, d]
        return out

    def get_user_item_embeddings(self):
        """
        返回用户嵌入矩阵和物品嵌入矩阵：
        - user_emb: [num_users, d]
        - item_emb: [num_items, d]
        """
        all_emb = self.computer()
        user_emb = all_emb[:self.num_users]
        item_emb = all_emb[self.num_users:]
        return user_emb, item_emb

    def forward(self, users, pos_items, neg_items):
        """
        BPR 训练用的前向：
        输入:
            users:     [batch]
            pos_items: [batch]
            neg_items: [batch]
        输出:
            u_e, pos_e, neg_e: 对应的嵌入向量
        """
        user_emb, item_emb = self.get_user_item_embeddings()
        u_e = user_emb[users]              # [batch, d]
        pos_e = item_emb[pos_items]        # [batch, d]
        neg_e = item_emb[neg_items]        # [batch, d]
        return u_e, pos_e, neg_e

    def predict(self, users, items):
        """
        用于评估打分：给定 (u, i) 返回预测偏好分数。
        """
        user_emb, item_emb = self.get_user_item_embeddings()
        u_e = user_emb[users]
        i_e = item_emb[items]
        scores = (u_e * i_e).sum(dim=-1)
        return scores
