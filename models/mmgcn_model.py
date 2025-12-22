# models/mmgcn_model.py
import torch
import torch.nn as nn


class MMGCN(nn.Module):
    """
    MMGCN：多模态图卷积推荐模型（ID / 文本 / 图像 三路 GCN + gating 融合）
    - 每个模态一套 user/item Embedding
    - 每个模态一张归一化的邻接矩阵（这里可以传同一张 ID 图，也可以传三张不同图）
    - item 侧使用文本 + 图像特征做内容 gating，学习每个 item 在三种模态上的权重
    - user 侧学习全局的模态权重（一个可学习的 3 维参数）
    - 打分用 user_final · item_final 的内积
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        adj_id,          # 稀疏邻接 [n_nodes, n_nodes]
        adj_text,        # 稀疏邻接 [n_nodes, n_nodes]
        adj_image,       # 稀疏邻接 [n_nodes, n_nodes]
        item_text_feats: torch.Tensor,   # [n_items, d_text]
        item_image_feats: torch.Tensor,  # [n_items, d_img]
        content_encoder=None,
        content_hidden_dim: int = 64,
        dropout: float = 0.0,
        **kwargs,
    ):

        super().__init__()
        self.content_encoder = kwargs.get("content_encoder", None)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # 邻接矩阵（假定外部已经 to(device)）
        self.adj_id = adj_id
        self.adj_text = adj_text
        self.adj_image = adj_image

        # ======== 三个模态的 user/item 嵌入 ========
        self.user_embedding_id = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_id = nn.Embedding(num_items, embedding_dim)

        self.user_embedding_text = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_text = nn.Embedding(num_items, embedding_dim)

        self.user_embedding_image = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_image = nn.Embedding(num_items, embedding_dim)

        self._reset_parameters()

        # ======== item 内容特征（文本 + 图像），用于 gating ========
        # 这里直接存成 register_buffer，方便跟随模型一起 to(device)，不参与梯度
        self.register_buffer("item_text_feats", item_text_feats)    # [n_items, d_text]
        self.register_buffer("item_image_feats", item_image_feats)  # [n_items, d_img]

        content_dim = 0
        if item_text_feats is not None and item_text_feats.numel() > 0:
            content_dim += item_text_feats.size(1)
        if item_image_feats is not None and item_image_feats.numel() > 0:
            content_dim += item_image_feats.size(1)

        if content_dim > 0:
            self.gate_mlp = nn.Sequential(
                nn.Linear(content_dim, content_hidden_dim),
                nn.ReLU(),
                nn.Linear(content_hidden_dim, 3),  # 三个模态：ID / TEXT / IMAGE
            )
        else:
            self.gate_mlp = None

        # user 端的模态权重（全局 3 维参数）
        self.user_modal_logits = nn.Parameter(torch.zeros(3))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def _reset_parameters(self):
        for emb in [
            self.user_embedding_id,
            self.item_embedding_id,
            self.user_embedding_text,
            self.item_embedding_text,
            self.user_embedding_image,
            self.item_embedding_image,
        ]:
            nn.init.normal_(emb.weight, std=0.01)

    # ======== 单模态 GCN 传播 ========
    def _propagate_single(
        self,
        user_emb: torch.Tensor,  # [n_users, d]
        item_emb: torch.Tensor,  # [n_items, d]
        adj,                     # 稀疏邻接 [n_nodes, n_nodes]
    ):
        """
        LightGCN 式传播：
          all_emb^0 = concat(user_emb, item_emb)
          all_emb^{k+1} = A_hat @ all_emb^k
          输出为多层 embeddings 的平均
        """
        all_emb = torch.cat([user_emb, item_emb], dim=0)  # [n_users+n_items, d]
        embs = [all_emb]

        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            if self.dropout is not None:
                all_emb = self.dropout(all_emb)
            embs.append(all_emb)

        # 多层平均
        all_out = torch.stack(embs, dim=0).mean(dim=0)  # [n_nodes, d]

        user_out = all_out[: self.num_users]
        item_out = all_out[self.num_users :]
        return user_out, item_out

    # ======== 计算三模态 + gating 融合后的 user/item 表征 ========
    def compute_embeddings(self):
        # 初始 Embedding
        u_id = self.user_embedding_id.weight    # [n_users, d]
        i_id = self.item_embedding_id.weight    # [n_items, d]

        u_text = self.user_embedding_text.weight
        i_text = self.item_embedding_text.weight

        u_img = self.user_embedding_image.weight
        i_img = self.item_embedding_image.weight

        # 三个模态各自做 GCN 传播
        u_id, i_id = self._propagate_single(u_id, i_id, self.adj_id)
        u_text, i_text = self._propagate_single(u_text, i_text, self.adj_text)
        u_img, i_img = self._propagate_single(u_img, i_img, self.adj_image)

        # ========= item 侧 gating（多模态融合） =========
        # item_stack: [n_items, 3, d]
        item_stack = torch.stack([i_id, i_text, i_img], dim=1)

        if (
            self.gate_mlp is not None
            and self.item_text_feats is not None
            and self.item_image_feats is not None
            and self.item_text_feats.numel() > 0
            and self.item_image_feats.numel() > 0
        ):
            content = torch.cat(
                [self.item_text_feats, self.item_image_feats], dim=1
            )  # [n_items, d_text+d_img]
            gate_logits = self.gate_mlp(content)        # [n_items, 3]
            gate_weights = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)  # [n_items, 3, 1]
        else:
            # 如果没有内容特征，则均匀权重
            gate_weights = torch.ones(
                (self.num_items, 3, 1), device=item_stack.device
            ) / 3.0

        # item_final: [n_items, d]
        item_final = (item_stack * gate_weights).sum(dim=1)

        # ========= user 侧 gating（全局模态权重） =========
        user_stack = torch.stack([u_id, u_text, u_img], dim=1)  # [n_users, 3, d]
        user_gate = torch.softmax(self.user_modal_logits, dim=0).view(1, 3, 1)
        user_final = (user_stack * user_gate).sum(dim=1)        # [n_users, d]

        return user_final, item_final

    # ======== BPR 训练前向 ========
    def forward(self, users, pos_items, neg_items):
        """
        BPR 训练前向：
          输入：
            users      [B]
            pos_items  [B]
            neg_items  [B]
          输出：
            u_e, pos_e, neg_e （embedding，用于外部计算 BPR loss）
        """
        user_emb, item_emb = self.compute_embeddings()
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        return u_e, pos_e, neg_e

    # ======== Top-K 评估用 ========
    def get_user_item_embeddings(self):
        """
        返回最终的 user / item embedding，用于 Top-K 评估。
        """
        return self.compute_embeddings()
