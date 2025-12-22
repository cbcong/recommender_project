# models/vae_model.py
import torch
import torch.nn as nn
from typing import List, Optional


class MultiDAE(nn.Module):
    """
    Multi-DAE: 只做自编码重建用的版本（如果你以后想用的话）。
    现在主要是为了保持和常见实现兼容，不影响 Multi-VAE 使用。
    """

    def __init__(self, p_dims: List[int], dropout: float = 0.5):
        """
        p_dims: 解码器层维度，例如 [200, 600, n_items]
                p_dims[0] 是 bottleneck 维度，p_dims[-1] 是输入维度（item 数）
        """
        super().__init__()
        self.p_dims = p_dims
        self.q_dims = p_dims[::-1]
        self.dropout = nn.Dropout(dropout)

        # Encoder: q_dims: [input_dim, ..., bottleneck]
        enc_dims = self.q_dims
        self.enc_layers = nn.ModuleList()
        for i in range(len(enc_dims) - 1):
            self.enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))

        # Decoder: p_dims: [bottleneck, ..., input_dim]
        dec_dims = self.p_dims
        self.dec_layers = nn.ModuleList()
        for i in range(len(dec_dims) - 1):
            self.dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))

        self._init_weights()

    def _init_weights(self):
        for m in list(self.enc_layers) + list(self.dec_layers):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim] multi-hot
        返回重建的 logits: [B, input_dim]
        """
        h = self.dropout(x)
        for i, layer in enumerate(self.enc_layers):
            h = torch.tanh(layer(h))
        for i, layer in enumerate(self.dec_layers):
            h = layer(h)
            if i != len(self.dec_layers) - 1:
                h = torch.tanh(h)
        return h


class MultiVAE(nn.Module):
    """
    Multi-VAE for collaborative filtering (Liang et al., WWW 2018)

    输入: x ∈ R^{#items} (multi-hot)
    编码: q(z|x)
    解码: p(x|z)
    """

    def __init__(
        self,
        p_dims: List[int],
        q_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
    ):
        """
        p_dims: 解码器各层维度，例如 [200, 600, n_items]
                其中 p_dims[0] = latent_dim, p_dims[-1] = 输入维度 (#items)
        q_dims: 编码器各层维度，若为 None，则自动设为 p_dims[::-1]
                要求 q_dims[0] = 输入维度, q_dims[-1] = latent_dim
        dropout: 输入上的 dropout 概率
        """
        super().__init__()
        if q_dims is None:
            q_dims = p_dims[::-1]

        assert q_dims[0] == p_dims[-1], (
            f"q_dims[0] (input dim) must equal p_dims[-1] (input dim), "
            f"got q_dims[0]={q_dims[0]}, p_dims[-1]={p_dims[-1]}"
        )

        self.p_dims = p_dims
        self.q_dims = q_dims
        self.dropout = nn.Dropout(dropout)

        # ----- Encoder -----
        # q_dims: [input_dim, h1, ..., hL, z_dim]
        enc_dims = self.q_dims
        self.enc_layers = nn.ModuleList()
        for i in range(len(enc_dims) - 2):
            self.enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
        # 最后一层拆分成 mu / logvar
        self.enc_mu = nn.Linear(enc_dims[-2], enc_dims[-1])
        self.enc_logvar = nn.Linear(enc_dims[-2], enc_dims[-1])

        # ----- Decoder -----
        # p_dims: [z_dim, hL, ..., input_dim]
        dec_dims = self.p_dims
        self.dec_layers = nn.ModuleList()
        for i in range(len(dec_dims) - 1):
            self.dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))

        self._init_weights()

    def _init_weights(self):
        for m in self.enc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_normal_(self.enc_mu.weight)
        nn.init.zeros_(self.enc_mu.bias)
        nn.init.xavier_normal_(self.enc_logvar.weight)
        nn.init.zeros_(self.enc_logvar.bias)

        for m in self.dec_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------- 编码 / 采样 / 解码 ----------
    def encode(self, x: torch.Tensor):
        """
        x: [B, input_dim]
        返回 mu, logvar: [B, z_dim]
        """
        h = self.dropout(x)
        for layer in self.enc_layers:
            h = torch.tanh(layer(h))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        reparameterization trick: z = mu + eps * sigma
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        """
        z: [B, z_dim]
        返回 logits: [B, input_dim]
        """
        h = z
        for i, layer in enumerate(self.dec_layers):
            h = layer(h)
            if i != len(self.dec_layers) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, x: torch.Tensor):
        """
        返回: logits, mu, logvar
        训练时用 logits + mu/logvar 计算 ELBO；
        推荐时只需要 logits。
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar
