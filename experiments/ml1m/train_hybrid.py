# recommender_project/experiments/ml1m/train_hybrid.py
import os
import sys
import time
from collections import defaultdict
from typing import Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# AMP
try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    autocast = None
    GradScaler = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
for p in [PROJECT_ROOT, UTILS_DIR, MODELS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from utils.preprocess import (
        build_datasets_and_loaders,
        load_ml1m_ratings,
        build_id_mappings,
        split_by_user_time,
    )
except Exception:
    from preprocess import (
        build_datasets_and_loaders,
        load_ml1m_ratings,
        build_id_mappings,
        split_by_user_time,
    )

try:
    from models.content_model import ItemContentEncoder
    from models.hybrid_model import HybridNCF
except Exception:
    from content_model import ItemContentEncoder
    from hybrid_model import HybridNCF


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def logits_to_rating(logit: torch.Tensor, rating_min: float, rating_max: float) -> torch.Tensor:
    # 与 HybridNCF.forward_with_aux() 一致：pred = min + (max-min)*sigmoid(logit)
    return float(rating_min) + (float(rating_max) - float(rating_min)) * torch.sigmoid(logit)


@torch.no_grad()
def evaluate_rmse_mae(model, loader, device, rating_min: float, rating_max: float, use_amp: bool):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0

    amp_enabled = bool(use_amp and device == "cuda" and autocast is not None)

    for u, i, r in loader:
        u = u.to(device, non_blocking=True)
        i = i.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True).view(-1)

        with autocast(enabled=amp_enabled) if autocast is not None else torch.no_grad():
            logit = model.score_logits(u, i).view(-1)
            p = logits_to_rating(logit, rating_min=rating_min, rating_max=rating_max).view(-1)

        d = (p - r).float()
        mse_sum += torch.sum(d * d).item()
        mae_sum += torch.sum(torch.abs(d)).item()
        n += r.size(0)

    rmse = (mse_sum / max(n, 1)) ** 0.5
    mae = mae_sum / max(n, 1)
    return rmse, mae


def build_user_pos_sets(train_df, user2idx, item2idx, n_users, pos_rating_threshold: float = 4.0):
    """
    只把高评分当作正反馈集合（用于 BPR 负采样排除）
    """
    user_pos = [set() for _ in range(n_users)]
    df = train_df
    if "rating" in df.columns:
        df = df[df["rating"] >= float(pos_rating_threshold)]
    for row in df.itertuples(index=False):
        u = user2idx[int(row.userId)]
        it = item2idx[int(row.movieId)]
        user_pos[u].add(it)
    return user_pos


def build_user_hist_tensors_left_aligned(train_df, user2idx, item2idx, n_users, max_hist_len, pad_idx):
    user_seq = defaultdict(list)
    train_df = train_df.sort_values(["userId", "timestamp"])
    for row in train_df.itertuples(index=False):
        u = user2idx[int(row.userId)]
        it = item2idx[int(row.movieId)]
        user_seq[u].append(it)

    hist_items = np.full((n_users, max_hist_len), fill_value=pad_idx, dtype=np.int64)
    hist_lens = np.zeros((n_users,), dtype=np.int64)

    for u in range(n_users):
        seq = user_seq.get(u, [])
        seq = seq[-max_hist_len:]
        L = len(seq)
        hist_lens[u] = L
        if L > 0:
            hist_items[u, :L] = np.array(seq, dtype=np.int64)

    return torch.from_numpy(hist_items), torch.from_numpy(hist_lens)


def sample_negatives(user_idx_batch, user_pos_sets, n_items, num_neg, device):
    """
    仍保持原始“排除正样本”的逻辑（不动核心行为），但尽量减少不必要的张量来回搬运。
    """
    u_np = user_idx_batch.detach().cpu().numpy()
    B = len(u_np)
    neg = np.empty((B, num_neg), dtype=np.int64)
    for b, u in enumerate(u_np):
        pos = user_pos_sets[int(u)]
        for k in range(num_neg):
            j = np.random.randint(0, n_items)
            while j in pos:
                j = np.random.randint(0, n_items)
            neg[b, k] = j
    return torch.from_numpy(neg).to(device, non_blocking=True)


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    user_pos_sets,
    n_items,
    num_neg,
    lambda_rating,
    lambda_bpr,
    pos_rating_threshold,
    grad_clip,
    rating_min: float,
    rating_max: float,
    scaler: Optional["GradScaler"],
    use_amp: bool,
):
    model.train()
    mse_fn = nn.MSELoss()
    total = 0.0
    n = 0

    amp_enabled = bool(use_amp and device == "cuda" and autocast is not None and scaler is not None)

    for u, i, r in loader:
        u = u.to(device, non_blocking=True)
        i = i.to(device, non_blocking=True)
        r = r.to(device, non_blocking=True).view(-1)

        optimizer.zero_grad(set_to_none=True)

        # 关键：只做一次 score_logits（原版 forward_with_aux 会额外再算一次）
        if autocast is not None:
            ctx = autocast(enabled=amp_enabled)
        else:
            ctx = torch.enable_grad()

        with ctx:
            logit_all = model.score_logits(u, i).view(-1)
            pred = logits_to_rating(logit_all, rating_min=rating_min, rating_max=rating_max).view(-1)
            loss_rating = mse_fn(pred, r)

            # BPR：只对高评分样本做；并且 pos_logit 直接复用 logit_all（避免再算一次）
            if lambda_bpr > 0 and num_neg > 0:
                pos_mask = (r >= float(pos_rating_threshold))
                if pos_mask.any():
                    u_pos = u[pos_mask]
                    # i_pos = i[pos_mask]  # 不再需要额外 score_logits
                    pos_logit = logit_all[pos_mask].unsqueeze(1)  # [Bp,1]

                    neg_items = sample_negatives(u_pos, user_pos_sets, n_items, num_neg, device)  # [Bp,K]
                    Bp, K = neg_items.shape
                    u_rep = u_pos.unsqueeze(1).expand(Bp, K).reshape(-1)
                    j_rep = neg_items.reshape(-1)

                    neg_logit = model.score_logits(u_rep, j_rep).view(Bp, K)
                    diff = pos_logit - neg_logit
                    loss_bpr = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
                else:
                    loss_bpr = torch.tensor(0.0, device=device)
            else:
                loss_bpr = torch.tensor(0.0, device=device)

            loss = float(lambda_rating) * loss_rating + float(lambda_bpr) * loss_bpr

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        bs = r.size(0)
        total += float(loss.item()) * bs
        n += bs

    return total / max(n, 1)


def main():
    cfg = load_config(os.path.join(PROJECT_ROOT, "utils", "config.yaml"))

    ratings_path = cfg["data"]["ml1m_ratings"]
    if not os.path.isabs(ratings_path):
        ratings_path = os.path.join(PROJECT_ROOT, ratings_path)

    device = cfg.get("train", {}).get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ---------- 性能相关：TF32 / benchmark ----------
    if device == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            # PyTorch 2.x
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    seed = int(cfg.get("train", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = int(cfg.get("train", {}).get("batch_size", 1024))
    lr = float(cfg.get("train", {}).get("lr", 1e-3))
    epochs = int(cfg.get("train", {}).get("epochs", 20))
    weight_decay = float(cfg.get("train", {}).get("weight_decay", 0.0))
    grad_clip = float(cfg.get("train", {}).get("grad_clip", 0.0))
    num_workers = int(cfg.get("train", {}).get("num_workers", 0))
    patience = int(cfg.get("train", {}).get("patience", 5))
    min_delta = float(cfg.get("train", {}).get("min_delta", 1e-4))

    # AMP 开关：默认 CUDA 下开启（不改你的核心训练目标，只是加速）
    use_amp = bool(cfg.get("train", {}).get("amp", True))

    print(f"Using device: {device}")
    print(f"Ratings path: {ratings_path}")
    print(f"batch_size={batch_size}, num_workers={num_workers}, amp={use_amp}")

    train_loader, val_loader, test_loader, meta = build_datasets_and_loaders(
        ratings_path, batch_size=batch_size, num_workers=num_workers
    )
    n_users = int(meta["n_users"])
    n_items = int(meta["n_items"])
    idx2item = meta.get("idx2item", None)

    ratings_df = load_ml1m_ratings(ratings_path)
    user2idx, item2idx, _, _ = build_id_mappings(ratings_df)
    train_df, val_df, test_df = split_by_user_time(ratings_df)

    global_mean = float(train_df["rating"].mean()) if "rating" in train_df.columns else 0.0

    feat_dir = os.path.join(PROJECT_ROOT, "data", "features")
    text_feat_path = os.path.join(feat_dir, "ml1m_text_embeddings_64.npy")
    text_index_path = os.path.join(feat_dir, "ml1m_text_index.csv")
    img_feat_path = os.path.join(feat_dir, "ml1m_image_embeddings_64.npy")
    img_index_path = os.path.join(feat_dir, "ml1m_image_index.csv")

    hy = cfg.get("hybrid", {})
    gmf_dim = int(hy.get("gmf_dim", 32))
    mlp_dim = int(hy.get("mlp_dim", 128))
    content_proj_dim = int(hy.get("content_proj_dim", 128))
    mlp_layer_sizes = tuple(hy.get("mlp_layer_sizes", [512, 256, 128]))
    dropout = float(hy.get("dropout", 0.10))

    use_text = bool(hy.get("use_text", True))
    use_image = bool(hy.get("use_image", True))
    use_history = bool(hy.get("use_history", True))
    max_hist_len = int(hy.get("max_hist_len", 50))
    n_heads = int(hy.get("n_heads", 4))
    n_transformer_layers = int(hy.get("n_transformer_layers", 2))

    num_neg = int(hy.get("num_neg", 4))
    lambda_rating = float(hy.get("lambda_rating", 1.0))
    lambda_bpr = float(hy.get("lambda_bpr", 0.5))
    pos_rating_threshold = float(hy.get("pos_rating_threshold", 4.0))

    rating_min = float(hy.get("rating_min", 1.0))
    rating_max = float(hy.get("rating_max", 5.0))

    content_encoder = ItemContentEncoder(
        ratings_path=ratings_path,
        text_feat_path=text_feat_path,
        text_index_path=text_index_path,
        image_feat_path=img_feat_path,
        image_index_path=img_index_path,
        use_text=use_text,
        use_image=use_image,
        idx2item=idx2item,
        n_items=n_items,
    )

    # 保持你原来的核心设置不变
    model = HybridNCF(
        num_users=n_users,
        num_items=n_items,
        content_encoder=content_encoder,
        gmf_dim=gmf_dim,
        mlp_dim=mlp_dim,
        content_proj_dim=content_proj_dim,
        mlp_layer_sizes=mlp_layer_sizes,
        dropout=dropout,
        use_history=use_history,
        max_hist_len=max_hist_len,
        n_heads=n_heads,
        n_transformer_layers=n_transformer_layers,
        rating_min=rating_min,
        rating_max=rating_max,
        global_mean=global_mean,
        train_history_mode="encode",  # 仍保持不动
    ).to(device)

    # 全量 history tensors（直接上 GPU）
    hist_items, hist_lens = build_user_hist_tensors_left_aligned(
        train_df=train_df,
        user2idx=user2idx,
        item2idx=item2idx,
        n_users=n_users,
        max_hist_len=max_hist_len,
        pad_idx=model.PAD_IDX,
    )
    model.set_user_histories(hist_items.to(device, non_blocking=True), hist_lens.to(device, non_blocking=True))

    # 初始化一次 cache（不改变训练，只保证 eval/保存一致性）
    model.refresh_user_cache(device=device)

    user_pos_sets = build_user_pos_sets(
        train_df=train_df,
        user2idx=user2idx,
        item2idx=item2idx,
        n_users=n_users,
        pos_rating_threshold=pos_rating_threshold,
    )

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    scaler = None
    if device == "cuda" and use_amp and GradScaler is not None:
        scaler = GradScaler(enabled=True)

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            user_pos_sets=user_pos_sets,
            n_items=n_items,
            num_neg=num_neg,
            lambda_rating=lambda_rating,
            lambda_bpr=lambda_bpr,
            pos_rating_threshold=pos_rating_threshold,
            grad_clip=grad_clip,
            rating_min=rating_min,
            rating_max=rating_max,
            scaler=scaler,
            use_amp=use_amp,
        )

        # 刷新 cache（用于 val/test 更稳定；如果你觉得仍慢，可后续改成每 N 个 epoch 刷一次）
        model.eval()
        model.refresh_user_cache(device=device)

        val_rmse, val_mae = evaluate_rmse_mae(
            model, val_loader, device, rating_min=rating_min, rating_max=rating_max, use_amp=use_amp
        )
        scheduler.step(val_rmse)

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] loss={tr_loss:.4f} val_RMSE={val_rmse:.4f} val_MAE={val_mae:.4f} ({dt:.1f}s)")

        if val_rmse + min_delta < best_val:
            best_val = val_rmse
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStop] no improvement for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        model.eval()
        model.refresh_user_cache(device=device)

    test_rmse, test_mae = evaluate_rmse_mae(
        model, test_loader, device, rating_min=rating_min, rating_max=rating_max, use_amp=use_amp
    )
    print(f"[Test] RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

    ckpt_name = cfg.get("paths", {}).get("hybrid_ckpt", "hybrid_ml1m_best.pth")
    ckpt_safe_name = cfg.get("paths", {}).get("hybrid_ckpt_safe", "hybrid_ml1m_best_safe.pth")
    save_path = ckpt_name if os.path.isabs(ckpt_name) else os.path.join(PROJECT_ROOT, ckpt_name)
    save_path_safe = ckpt_safe_name if os.path.isabs(ckpt_safe_name) else os.path.join(PROJECT_ROOT, ckpt_safe_name)

    payload_full = {
        "state_dict": model.state_dict(),
        "n_users": n_users,
        "n_items": n_items,
        "config": cfg,
        "seed": seed,
        "global_mean": float(global_mean),
        "idx2user": meta.get("idx2user", None),
        "idx2item": meta.get("idx2item", None),
    }
    payload_safe = {"state_dict": model.state_dict()}

    torch.save(payload_full, save_path)
    torch.save(payload_safe, save_path_safe)

    print(f"Saved full checkpoint to: {save_path}")
    print(f"Saved safe checkpoint to: {save_path_safe}")


if __name__ == "__main__":
    main()
