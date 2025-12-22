# recommender_project/experiments/ml1m/plot_v3_ml1m.py
import os
import re
import glob
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Paths
# -------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def find_latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]


def safe_read_csv(path: str) -> pd.DataFrame:
    if path is None or (not os.path.exists(path)):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_0_1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def pareto_front_2d(x: np.ndarray, y: np.ndarray, maximize_x=True, maximize_y=True) -> np.ndarray:
    """
    返回 Pareto 前沿点的布尔掩码。
    maximize_x=True 表示 x 越大越好；否则越小越好。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            better_x = (x[j] >= x[i]) if maximize_x else (x[j] <= x[i])
            better_y = (y[j] >= y[i]) if maximize_y else (y[j] <= y[i])
            strictly = ((x[j] > x[i]) if maximize_x else (x[j] < x[i])) or ((y[j] > y[i]) if maximize_y else (y[j] < y[i]))
            if better_x and better_y and strictly:
                mask[i] = False
                break
    return mask


def write_md(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


@dataclass
class PlotConfig:
    eval_csv: str
    base_csv: Optional[str]
    tune_csv: Optional[str]
    out_dir: str
    k: int
    fmt_png: bool = True
    fmt_pdf: bool = True


# -------------------------
# Loaders
# -------------------------
def load_eval_table(eval_csv: str) -> pd.DataFrame:
    df = safe_read_csv(eval_csv).copy()

    # 兼容列名：你的 evaluate 输出为 ["Model","Recall@10","NDCG@10","Coverage@10","LongTailShare@10","Novelty@10"]
    # 若用户改了列名，这里尽量识别
    if "Model" not in df.columns:
        raise ValueError("eval csv must have column: Model")

    # 推断 K
    k = None
    for col in df.columns:
        m = re.match(r"Recall@(\d+)", str(col))
        if m:
            k = int(m.group(1))
            break
    return df, k


def load_tune_table_auto() -> Optional[pd.DataFrame]:
    pattern = os.path.join(PROJECT_ROOT, "artifacts", "ml1m", "tune_v3_rerank_ml1m", "**", "results_v3_ml1m_rerank_tune.csv")
    latest = find_latest(pattern)
    if latest is None:
        return None
    df = safe_read_csv(latest)
    df.attrs["source_path"] = latest
    return df


# -------------------------
# Plot helpers
# -------------------------
def save_fig(out_dir: str, name: str, png: bool, pdf: bool) -> None:
    if png:
        plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
    if pdf:
        plt.savefig(os.path.join(out_dir, f"{name}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_bars(df: pd.DataFrame, k: int, out_dir: str, png=True, pdf=True) -> None:
    metrics = [f"Recall@{k}", f"NDCG@{k}", f"Coverage@{k}", f"LongTailShare@{k}", f"Novelty@{k}"]
    for m in metrics:
        if m not in df.columns:
            raise ValueError(f"Missing metric column in eval df: {m}")

    models = df["Model"].astype(str).tolist()

    # 每个指标一张专业柱状图（论文常见）
    for m in metrics:
        vals = df[m].values.astype(float)
        order = np.argsort(-vals)
        plt.figure(figsize=(10, 4.8))
        plt.bar([models[i] for i in order], vals[order])
        plt.ylabel(m)
        plt.title(f"Model Comparison on {m}")
        plt.xticks(rotation=25, ha="right")
        save_fig(out_dir, f"bar_{m.replace('@', '_at_')}", png, pdf)

    # 五指标同图（便于总览）
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.16
    for idx, m in enumerate(metrics):
        plt.bar(x + (idx - 2) * width, df[m].values.astype(float), width=width, label=m)
    plt.xticks(x, models, rotation=25, ha="right")
    plt.ylabel("Metric Value")
    plt.title(f"Overall Metrics Comparison (K={k})")
    plt.legend()
    save_fig(out_dir, f"bar_overall_metrics_k{k}", png, pdf)


def plot_radar(df: pd.DataFrame, k: int, out_dir: str, png=True, pdf=True) -> None:
    metrics = [f"Recall@{k}", f"NDCG@{k}", f"Coverage@{k}", f"LongTailShare@{k}", f"Novelty@{k}"]
    for m in metrics:
        if m not in df.columns:
            return

    # normalize each metric across models for radar
    vals = []
    for m in metrics:
        vals.append(normalize_0_1(df[m].values.astype(float)))
    vals = np.vstack(vals).T  # [n_models, n_metrics]

    labels = ["Recall", "NDCG", "Coverage", "LongTail", "Novelty"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(9, 7))
    ax = plt.subplot(111, polar=True)

    for i, model in enumerate(df["Model"].astype(str).tolist()):
        data = vals[i].tolist()
        data += data[:1]
        ax.plot(angles, data, linewidth=1.3, label=model)
        ax.fill(angles, data, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title(f"Normalized Radar (K={k})", y=1.07)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
    save_fig(out_dir, f"radar_normalized_k{k}", png, pdf)


def plot_tradeoffs(df: pd.DataFrame, k: int, out_dir: str, png=True, pdf=True) -> None:
    pairs = [
        (f"Recall@{k}", f"Novelty@{k}", True, True, f"tradeoff_recall_vs_novelty_k{k}"),
        (f"Recall@{k}", f"LongTailShare@{k}", True, True, f"tradeoff_recall_vs_longtail_k{k}"),
        (f"NDCG@{k}",  f"Novelty@{k}", True, True, f"tradeoff_ndcg_vs_novelty_k{k}"),
        (f"Coverage@{k}", f"Recall@{k}", True, True, f"tradeoff_coverage_vs_recall_k{k}"),
    ]

    for xcol, ycol, maxx, maxy, name in pairs:
        if xcol not in df.columns or ycol not in df.columns:
            continue
        x = df[xcol].values.astype(float)
        y = df[ycol].values.astype(float)
        models = df["Model"].astype(str).tolist()

        pf = pareto_front_2d(x, y, maximize_x=maxx, maximize_y=maxy)

        plt.figure(figsize=(7.2, 5.6))
        plt.scatter(x, y, s=40)
        for i, m in enumerate(models):
            plt.text(x[i], y[i], m, fontsize=8)

        # highlight Pareto
        plt.scatter(x[pf], y[pf], s=80, marker="x")
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.title(f"Trade-off: {xcol} vs {ycol} (Pareto highlighted)")
        save_fig(out_dir, name, png, pdf)


def plot_improvements(eval_df: pd.DataFrame, base_df: Optional[pd.DataFrame], k: int, out_dir: str, png=True, pdf=True) -> Optional[pd.DataFrame]:
    if base_df is None:
        return None

    # 期望 base_df 有相同 Model 列与指标列
    metrics = [f"Recall@{k}", f"NDCG@{k}", f"Coverage@{k}", f"LongTailShare@{k}", f"Novelty@{k}"]
    for m in metrics:
        if m not in base_df.columns:
            raise ValueError(f"base csv missing column: {m}")

    merged = eval_df.merge(base_df[["Model"] + metrics], on="Model", suffixes=("_rerank", "_base"))
    for m in metrics:
        merged[f"Delta_{m}"] = merged[f"{m}_rerank"] - merged[f"{m}_base"]
        # 论文常用：相对提升（对 0 值做保护）
        denom = merged[f"{m}_base"].replace(0, np.nan)
        merged[f"Rel_{m}"] = merged[f"Delta_{m}"] / denom

    # 作图：对 Recall/NDCG 做 delta
    for m in [f"Recall@{k}", f"NDCG@{k}"]:
        col = f"Delta_{m}"
        order = np.argsort(-merged[col].values.astype(float))
        plt.figure(figsize=(10, 4.8))
        plt.bar(merged["Model"].values[order], merged[col].values.astype(float)[order])
        plt.axhline(0.0, linewidth=1.0)
        plt.ylabel(f"Δ {m} (Rerank - Base)")
        plt.title(f"Improvement on {m} after Rerank")
        plt.xticks(rotation=25, ha="right")
        save_fig(out_dir, f"delta_{m.replace('@','_at_')}", png, pdf)

    return merged


def plot_tune_diagnostics(tune_df: pd.DataFrame, out_dir: str, k: int, png=True, pdf=True) -> None:
    """
    对 tune 网格结果做可解释分析：权重敏感性 + 相关性 + 热力图（以 w_div vs w_rel 为例）
    该函数假定 tune_df 包含：
      - model, w_rel, w_div, w_novelty/w_novel, w_longtail/w_tail, Recall@K, NDCG@K, Novelty@K, LongTailShare@K, Coverage@K
    """
    if tune_df is None or len(tune_df) == 0:
        return

    # 兼容不同权重字段名
    col_map = {}
    for c in tune_df.columns:
        lc = str(c).lower()
        if lc == "w_rel": col_map["w_rel"] = c
        if lc in ("w_div",): col_map["w_div"] = c
        if lc in ("w_novelty", "w_novel"): col_map["w_novel"] = c
        if lc in ("w_longtail", "w_tail"): col_map["w_tail"] = c

    need = ["w_rel", "w_div"]
    for n in need:
        if n not in col_map:
            return

    # 1) 每个模型：w_div vs Recall 的趋势散点（带线性拟合，用于解释“多样性权重升高是否牺牲准确率”）
    xw = col_map["w_div"]
    ycol = f"Recall@{k}" if f"Recall@{k}" in tune_df.columns else None
    if ycol:
        for model in sorted(tune_df["model"].unique()):
            sub = tune_df[tune_df["model"] == model].copy()
            x = sub[xw].values.astype(float)
            y = sub[ycol].values.astype(float)

            plt.figure(figsize=(7, 5.2))
            plt.scatter(x, y, s=18)
            if len(x) >= 2:
                # 简洁但有效：线性拟合，用于“方向性解释”
                a, b = np.polyfit(x, y, 1)
                xs = np.linspace(float(np.min(x)), float(np.max(x)), 50)
                plt.plot(xs, a * xs + b, linewidth=1.5)

            plt.xlabel("w_div")
            plt.ylabel(ycol)
            plt.title(f"{model}: Sensitivity of {ycol} to w_div")
            save_fig(out_dir, f"tune_sensitivity_{model}_wdiv_vs_recall_k{k}".replace("/", "_"), png, pdf)

    # 2) 权重与指标相关性（汇总表 + 热力图）
    metric_cols = [c for c in [f"Recall@{k}", f"NDCG@{k}", f"Novelty@{k}", f"LongTailShare@{k}", f"Coverage@{k}"] if c in tune_df.columns]
    weight_cols = [col_map["w_rel"], col_map["w_div"]]
    if "w_novel" in col_map: weight_cols.append(col_map["w_novel"])
    if "w_tail" in col_map: weight_cols.append(col_map["w_tail"])

    # 相关性按模型分别算，更专业（不同模型的权衡曲线差异很大）
    rows = []
    for model in sorted(tune_df["model"].unique()):
        sub = tune_df[tune_df["model"] == model].copy()
        for wc in weight_cols:
            for mc in metric_cols:
                if sub[wc].nunique() <= 1 or sub[mc].nunique() <= 1:
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(sub[wc].values.astype(float), sub[mc].values.astype(float))[0, 1])
                rows.append({"model": model, "weight": wc, "metric": mc, "pearson": corr})
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(os.path.join(out_dir, f"tune_weight_metric_correlation_k{k}.csv"), index=False, encoding="utf-8-sig")

    # 热力图（以全模型整体相关性均值展示，论文里可作为补充解释）
    pivot = corr_df.pivot_table(index="weight", columns="metric", values="pearson", aggfunc="mean")
    plt.figure(figsize=(9, 4.2))
    mat = pivot.values.astype(float)
    plt.imshow(mat, aspect="auto")
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns.tolist(), rotation=25, ha="right")
    plt.yticks(np.arange(pivot.shape[0]), pivot.index.tolist())
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = mat[i, j]
            txt = "nan" if (not np.isfinite(v)) else f"{v:.2f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)
    plt.title("Mean Pearson Correlation: Weight vs Metric (across models)")
    save_fig(out_dir, f"tune_corr_heatmap_k{k}", png, pdf)


def generate_report(eval_df: pd.DataFrame, k: int, out_dir: str, merged: Optional[pd.DataFrame], tune_df: Optional[pd.DataFrame]) -> None:
    lines = []
    lines.append(f"# ML-1M Rerank Report (K={k})")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    def best_of(col: str) -> Tuple[str, float]:
        idx = int(np.argmax(eval_df[col].values.astype(float)))
        return str(eval_df.loc[idx, "Model"]), float(eval_df.loc[idx, col])

    for col in [f"Recall@{k}", f"NDCG@{k}", f"Coverage@{k}", f"LongTailShare@{k}", f"Novelty@{k}"]:
        if col in eval_df.columns:
            m, v = best_of(col)
            lines.append(f"- Best {col}: **{m}** = {v:.4f}")
    lines.append("")

    # 相关性（模型层面的粗粒度解释）
    cols = [c for c in [f"Recall@{k}", f"NDCG@{k}", f"Coverage@{k}", f"LongTailShare@{k}", f"Novelty@{k}"] if c in eval_df.columns]
    if len(cols) >= 2:
        corr = eval_df[cols].corr()
        lines.append("## Metric Correlations (across models)")
        lines.append("")
        lines.append(corr.to_markdown())
        lines.append("")

    if merged is not None:
        lines.append("## Improvement vs Base (if base csv provided)")
        lines.append("")
        for col in [f"Recall@{k}", f"NDCG@{k}"]:
            dcol = f"Delta_{col}"
            if dcol in merged.columns:
                idx = int(np.argmax(merged[dcol].values.astype(float)))
                lines.append(f"- Largest Δ {col}: **{merged.loc[idx,'Model']}** = {float(merged.loc[idx,dcol]):.4f}")
        lines.append("")

    if tune_df is not None and len(tune_df) > 0:
        src = tune_df.attrs.get("source_path", None)
        if src:
            lines.append("## Tune Source")
            lines.append("")
            lines.append(f"- Latest tune csv used: `{src}`")
            lines.append("")

    write_md(os.path.join(out_dir, f"analysis_report_k{k}.md"), lines)


def main():
    out_dir = ensure_dir(os.path.join(PROJECT_ROOT, "artifacts", "ml1m", "plot_v3_ml1m", _ts()))

    eval_csv = os.path.join(PROJECT_ROOT, "results_v3_ml1m_rerank.csv")
    base_csv = None  # 如果你有 base 结果（未 rerank），把路径填上；或后面你再加自动生成 base 的脚本
    tune_df = load_tune_table_auto()

    eval_df, inferred_k = load_eval_table(eval_csv)
    if inferred_k is None:
        inferred_k = 10
    k = inferred_k

    # 可选：读取 base 结果
    base_df = safe_read_csv(base_csv) if (base_csv and os.path.exists(base_csv)) else None

    # 出图
    plot_metric_bars(eval_df, k, out_dir)
    plot_radar(eval_df, k, out_dir)
    plot_tradeoffs(eval_df, k, out_dir)
    merged = plot_improvements(eval_df, base_df, k, out_dir) if base_df is not None else None
    if tune_df is not None:
        plot_tune_diagnostics(tune_df, out_dir, k)

    # 报告 + 汇总导出
    generate_report(eval_df, k, out_dir, merged, tune_df)
    eval_df.to_excel(os.path.join(out_dir, f"eval_summary_k{k}.xlsx"), index=False)

    print(f"[OK] Plots & report saved to: {out_dir}")


if __name__ == "__main__":
    main()
