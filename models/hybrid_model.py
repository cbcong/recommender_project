# recommender_project/models/hybrid_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        return self.ln(x + h)


class HybridNCF(nn.Module):
    """
    HybridNCF
    - forward(): 显式评分（映射到 [rating_min, rating_max]）
    - score_logits(): 排序用 logit（不要 sigmoid 压缩）
    - 训练时：history Transformer 需要参与反传，因此不能只用 cache
    - 推理时：用 cache 加速
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        content_encoder,
        gmf_dim: int = 32,
        mlp_dim: int = 64,
        content_proj_dim: int = 64,
        mlp_layer_sizes=(256, 128, 64),
        dropout: float = 0.1,
        use_history: bool = True,
        max_hist_len: int = 50,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        modal_attn_heads: int = 2,
        modal_attn_dropout: float = 0.0,
        rating_min: float = 1.0,
        rating_max: float = 5.0,
        global_mean: float = 0.0,
        train_history_mode: str = "encode",  # "encode" or "cache"
    ):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)

        self.content_encoder = content_encoder
        self.content_dim = int(getattr(content_encoder, "content_dim", 0))
        self.text_dim = int(getattr(content_encoder, "text_dim", 0))
        self.image_dim = int(getattr(content_encoder, "image_dim", 0))

        self.use_history = bool(use_history)
        self.max_hist_len = int(max_hist_len)
        self.train_history_mode = str(train_history_mode)

        self.rating_min = float(rating_min)
        self.rating_max = float(rating_max)

        # global mean + bias
        self.global_mean = nn.Parameter(torch.tensor(float(global_mean), dtype=torch.float32))
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.item_bias = nn.Embedding(self.num_items, 1)

        # GMF
        self.user_embedding_gmf = nn.Embedding(self.num_users, gmf_dim)
        self.item_embedding_gmf = nn.Embedding(self.num_items, gmf_dim)

        # MLP base
        self.user_embedding_mlp = nn.Embedding(self.num_users, mlp_dim)
        self.item_embedding_mlp = nn.Embedding(self.num_items, mlp_dim)

        # content projections
        self.text_proj = nn.Linear(self.text_dim, content_proj_dim) if self.text_dim > 0 else None
        self.image_proj = nn.Linear(self.image_dim, content_proj_dim) if self.image_dim > 0 else None
        self.has_modal = (self.text_proj is not None) or (self.image_proj is not None)
        self.n_modal = int((self.text_proj is not None) + (self.image_proj is not None))

        if self.has_modal:
            self.modal_gate = nn.Sequential(
                nn.Linear(mlp_dim * 2, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, self.n_modal),
            )
            self.modal_ln = nn.LayerNorm(content_proj_dim)

            # 多模态注意力：让内容融合随用户/物品上下文自适应分配权重（SCI 级别增强）
            self.modal_attn_q = nn.Linear(mlp_dim * 2, content_proj_dim)
            self.modal_attn = nn.MultiheadAttention(
                embed_dim=content_proj_dim,
                num_heads=max(1, int(modal_attn_heads)),
                dropout=float(modal_attn_dropout),
                batch_first=True,
            )
            self.modal_attn_ln = nn.LayerNorm(content_proj_dim)
        else:
            self.modal_gate = None
            self.modal_ln = None
            self.modal_attn_q = None
            self.modal_attn = None
            self.modal_attn_ln = None

        self.content_to_mlp = None
        if self.has_modal and content_proj_dim != mlp_dim:
            self.content_to_mlp = nn.Linear(content_proj_dim, mlp_dim)

        self.cf_content_gate = nn.Sequential(
            nn.Linear(mlp_dim * 2, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 2),
        )
        self.item_fuse_ln = nn.LayerNorm(mlp_dim)

        # history Transformer
        self.PAD_IDX = self.num_items
        self.hist_item_embedding = nn.Embedding(self.num_items + 1, mlp_dim)
        self.pos_embedding = nn.Embedding(self.max_hist_len, mlp_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=mlp_dim,
            nhead=max(1, int(n_heads)),
            dim_feedforward=mlp_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.hist_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(n_transformer_layers)))
        self.hist_ln = nn.LayerNorm(mlp_dim)

        # cache（只用于 eval/infer）
        self.user_hist_cache = nn.Embedding(self.num_users, mlp_dim)
        self.user_hist_cache.weight.requires_grad_(False)

        self.user_gate = nn.Sequential(
            nn.Linear(mlp_dim * 2, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1),
        )

        # MLP head
        mlp_in = mlp_dim + mlp_dim + (content_proj_dim if self.has_modal else 0)
        self.mlp_in_proj = nn.Linear(mlp_in, int(mlp_layer_sizes[0]))
        self.mlp_in_ln = nn.LayerNorm(int(mlp_layer_sizes[0]))

        blocks = []
        dim0 = int(mlp_layer_sizes[0])
        for _ in range(2):
            blocks.append(ResidualMLPBlock(dim=dim0, hidden=dim0 * 2, dropout=float(dropout)))
        self.res_blocks = nn.Sequential(*blocks)

        tail_layers = []
        in_dim = dim0
        for h in mlp_layer_sizes[1:]:
            h = int(h)
            tail_layers.append(nn.Linear(in_dim, h))
            tail_layers.append(nn.ReLU())
            if dropout and dropout > 0:
                tail_layers.append(nn.Dropout(float(dropout)))
            in_dim = h
        self.mlp_tail = nn.Sequential(*tail_layers) if tail_layers else nn.Identity()

        fusion_dim = gmf_dim + int(mlp_layer_sizes[-1])
        self.predict_layer = nn.Linear(fusion_dim, 1)

        self._init_weights()

        # 全量 user histories（训练时 gather）
        self._user_hist_items = None  # [n_users, L]
        self._user_hist_lens = None   # [n_users]

    def _init_weights(self):
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.hist_item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_user_histories(self, user_hist_items: torch.LongTensor, user_hist_lens: torch.LongTensor):
        # 允许你在 train 脚本里直接传到 GPU，这里不强制搬运
        self._user_hist_items = user_hist_items
        self._user_hist_lens = user_hist_lens

    def _encode_histories(self, hist_items: torch.LongTensor, hist_lens: torch.LongTensor) -> torch.Tensor:
        """
        对齐方式无关：用 padding mask 找最后一个非 PAD 的位置
        """
        B, L = hist_items.shape
        device = hist_items.device

        x = self.hist_item_embedding(hist_items)  # [B,L,D]
        pos = self.pos_embedding(torch.arange(L, device=device)).unsqueeze(0).expand(B, L, -1)
        x = x + pos

        pad_mask = (hist_items == self.PAD_IDX)
        x = self.hist_encoder(x, src_key_padding_mask=pad_mask)  # [B,L,D]

        valid_mask = (~pad_mask).long()
        pos_idx = torch.arange(L, device=device).view(1, L).expand(B, L)
        last_pos = (valid_mask * pos_idx).max(dim=1).values  # [B]
        last = x[torch.arange(B, device=device), last_pos]    # [B,D]

        zero_mask = (hist_lens <= 0).view(-1, 1).float()
        last = last * (1.0 - zero_mask)
        return self.hist_ln(last)

    def _history_rep_for_users(self, user_idx: torch.LongTensor) -> torch.Tensor:
        """
        训练时可反传的 history representation：
        从全量 [n_users, L] histories 里 gather batch users
        """
        if self._user_hist_items is None or self._user_hist_lens is None:
            return torch.zeros(user_idx.size(0), self.user_embedding_mlp.embedding_dim, device=user_idx.device)

        hist_items = self._user_hist_items
        hist_lens = self._user_hist_lens
        if hist_items.device != user_idx.device:
            hist_items = hist_items.to(user_idx.device)
        if hist_lens.device != user_idx.device:
            hist_lens = hist_lens.to(user_idx.device)

        h_items = hist_items[user_idx]  # [B,L]
        h_lens = hist_lens[user_idx]    # [B]
        return self._encode_histories(h_items, h_lens)

    @torch.no_grad()
    def refresh_user_cache(self, device=None):
        """
        推理用 cache：不参与梯度
        """
        if not self.use_history:
            self.user_hist_cache.weight.zero_()
            return
        if self._user_hist_items is None or self._user_hist_lens is None:
            self.user_hist_cache.weight.zero_()
            return

        if device is None:
            device = self.user_hist_cache.weight.device

        hist_items = self._user_hist_items.to(device)
        hist_lens = self._user_hist_lens.to(device)

        n_users = hist_items.size(0)
        bs = 512
        out = torch.zeros(n_users, self.user_hist_cache.embedding_dim, device=device)
        for s in range(0, n_users, bs):
            e = min(n_users, s + bs)
            out[s:e] = self._encode_histories(hist_items[s:e], hist_lens[s:e])
        self.user_hist_cache.weight.data.copy_(out)

    def _fuse_content(self, u_mlp: torch.Tensor, i_mlp: torch.Tensor, content_raw: torch.Tensor):
        gates = {}
        if not self.has_modal or content_raw.numel() == 0:
            return None, i_mlp, gates

        experts = []
        if self.text_proj is not None and self.text_dim > 0:
            experts.append(self.text_proj(content_raw[:, : self.text_dim]))
        if self.image_proj is not None and self.image_dim > 0:
            experts.append(self.image_proj(content_raw[:, self.text_dim : self.text_dim + self.image_dim]))

        gate_in = torch.cat([u_mlp, i_mlp], dim=-1)

        # 注意力融合（多模态内容）
        if self.modal_attn is not None and len(experts) > 1:
            expert_stack = torch.stack(experts, dim=1)  # [B, M, D]
            q = self.modal_attn_q(gate_in).unsqueeze(1)  # [B,1,D]
            attn_out, attn_w = self.modal_attn(q, expert_stack, expert_stack, need_weights=True)
            content_shared = self.modal_attn_ln(attn_out.squeeze(1))
            gates["modal_attn_w"] = attn_w.squeeze(1)  # [B, M]
        else:
            modal_w = F.softmax(self.modal_gate(gate_in), dim=-1)
            gates["modal_w"] = modal_w

            content_shared = 0.0
            for k, ex in enumerate(experts):
                content_shared = content_shared + modal_w[:, k:k + 1] * ex
            content_shared = self.modal_ln(content_shared)

        content_to_mlp = content_shared
        if self.content_to_mlp is not None:
            content_to_mlp = self.content_to_mlp(content_shared)

        cf_w = F.softmax(self.cf_content_gate(gate_in), dim=-1)
        gates["cf_w"] = cf_w
        gates["use_content"] = cf_w[:, 1:2]

        item_fused = cf_w[:, 0:1] * i_mlp + cf_w[:, 1:2] * content_to_mlp
        item_fused = self.item_fuse_ln(item_fused)
        return content_shared, item_fused, gates

    def _build_user_rep(self, user_idx: torch.LongTensor) -> torch.Tensor:
        u = self.user_embedding_mlp(user_idx)

        if not self.use_history:
            return u

        # 关键：训练时用可反传 history；评估/推理用 cache
        if self.training and self.train_history_mode == "encode":
            h = self._history_rep_for_users(user_idx)
        else:
            h = self.user_hist_cache(user_idx)

        g = torch.sigmoid(self.user_gate(torch.cat([u, h], dim=-1)))
        return (1.0 - g) * u + g * h

    def score_logits(self, user_idx: torch.LongTensor, item_idx: torch.LongTensor) -> torch.Tensor:
        user_idx = user_idx.long()
        item_idx = item_idx.long()

        u_gmf = self.user_embedding_gmf(user_idx)
        i_gmf = self.item_embedding_gmf(item_idx)
        gmf_vec = u_gmf * i_gmf

        u_mlp = self._build_user_rep(user_idx)
        i_mlp = self.item_embedding_mlp(item_idx)

        if self.content_dim > 0:
            content_raw = self.content_encoder(item_idx)
        else:
            content_raw = torch.zeros(item_idx.size(0), 0, device=item_idx.device)

        content_shared, item_fused, _ = self._fuse_content(u_mlp, i_mlp, content_raw)

        if content_shared is not None:
            mlp_in = torch.cat([u_mlp, item_fused, content_shared], dim=-1)
        else:
            mlp_in = torch.cat([u_mlp, item_fused], dim=-1)

        h = self.mlp_in_ln(self.mlp_in_proj(mlp_in))
        h = self.res_blocks(h)
        h = self.mlp_tail(h)

        logit = self.predict_layer(torch.cat([gmf_vec, h], dim=-1)).squeeze(-1)
        logit = logit + self.global_mean + self.user_bias(user_idx).squeeze(-1) + self.item_bias(item_idx).squeeze(-1)
        return logit

    def forward_with_aux(self, user_idx: torch.LongTensor, item_idx: torch.LongTensor):
        logit = self.score_logits(user_idx, item_idx)
        pred = self.rating_min + (self.rating_max - self.rating_min) * torch.sigmoid(logit)
        return pred, {"logit": logit}

    def forward(self, user_idx: torch.LongTensor, item_idx: torch.LongTensor) -> torch.Tensor:
        pred, _ = self.forward_with_aux(user_idx, item_idx)
        return pred

    def predict(self, user_idx, item_idx):
        return self.forward(user_idx, item_idx)

    def load_state_dict(self, state_dict, strict: bool = True):
        has_hist = any(k.startswith("hist_encoder.") or k.startswith("user_hist_cache.") for k in state_dict.keys())
        if strict and not has_hist:
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)
