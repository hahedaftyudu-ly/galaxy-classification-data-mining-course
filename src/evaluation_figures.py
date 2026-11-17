#!/usr/bin/env python3
"""Evaluation figures and tables from the saved OOF predictions and governance/clustering results."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib import font_manager
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_curve, auc)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

BASE = config.BASE
FIG = BASE / "results" / "figures"
EVAL = BASE / "results" / "evaluation"
CLS = BASE / "results" / "classification"
CLASS_NAMES = list(config.CLASSES)
FEATURES = config.CATWISE_NUMERIC + ["w1_w2_color"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


def source_note(fig, text="数据来源：SDSS×CatWISE 融合表（本机只读复采集，2026-11-11）"):
    fig.text(0.01, 0.005, text, fontsize=7, color="#555555")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load_oof():
    df = pq.read_table(CLS / "oof_predictions.parquet").to_pandas()
    return df


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    result = json.loads((CLS / "model_comparison.json").read_text())
    models = list(result["cv"].keys())
    best = result["tests"]["best_model"]
    oof = load_oof()
    y = oof["y"].to_numpy()
    prob = oof[[f"prob_{best}_{c}" for c in CLASS_NAMES]].to_numpy()
    pred = oof[f"pred_{best}"].to_numpy()

    # 1 class distribution
    counts = np.bincount(y, minlength=3)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar(CLASS_NAMES, counts, color=[COLORS[0], COLORS[1], COLORS[2]])
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("天体类别")
    ax.set_ylabel("源数（个）")
    ax.set_title("三类天体样本量（固定 5 折全集）")
    source_note(fig)
    save(fig, "fig01_class_distribution.png")

    # 2 color distribution by class
    sub = pq.read_table(config.FUSED, columns=[config.LABEL, "w1mpro", "w2mpro"]).to_pandas()
    sub["color"] = sub["w1mpro"] - sub["w2mpro"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(-0.6, 2.2, 80)
    for i, c in enumerate(CLASS_NAMES):
        ax.hist(sub.loc[sub[config.LABEL] == c, "color"], bins=bins, density=True, alpha=0.55,
                label=c, color=COLORS[i])
    ax.set_xlabel("W1 − W2 颜色（mag）")
    ax.set_ylabel("概率密度（每mag）")
    ax.set_title("三类天体的 W1−W2 颜色分布")
    ax.legend()
    source_note(fig)
    save(fig, "fig02_color_by_class.png")

    # 3 missing-value curves
    gov = json.loads((BASE / "results/governance/governance_summary.json").read_text())
    rows = pd.DataFrame(gov["missing"]["rows"])
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 8.2), sharex=True, sharey=True)
    for ax, mech in zip(axes, ("MCAR", "MAR", "MNAR")):
        part = rows[rows["mechanism"] == mech]
        for strategy, color, marker in (("deletion", COLORS[0], "o"), ("median", COLORS[1], "s"),
                                        ("iterative", COLORS[2], "^")):
            p = part[part["strategy"] == strategy].sort_values("rate")
            ax.plot(p["rate"] * 100, p["macro_f1"], marker=marker, color=color, label={"deletion":"删除", "median":"中位数", "iterative":"迭代填补"}[strategy])
        ax.set_title(mech)
        ax.set_xlabel("整波段缺失率（%）")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Macro-F1（5 折均值）")
    axes[0].legend(fontsize=8)
    fig.suptitle("整波段缺失下的策略对比")
    source_note(fig)
    save(fig, "fig03_missing_strategies.png")

    # 4 outlier counts
    out = json.loads((BASE / "results/governance/outliers.json").read_text())
    feats = [f["feature"] for f in out["features"]]
    s3 = [f["sigma3_pct"] for f in out["features"]]
    iqr = [f["iqr_pct"] for f in out["features"]]
    order = np.argsort(iqr)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    yy = np.arange(len(feats))
    ax.barh(yy - 0.2, [s3[i] for i in order], height=0.4, label="3σ", color=COLORS[0])
    ax.barh(yy + 0.2, [iqr[i] for i in order], height=0.4, label="IQR", color=COLORS[1])
    ax.set_ylabel("测光特征")
    ax.set_yticks(yy)
    ax.set_yticklabels([feats[i] for i in order], fontsize=8)
    ax.set_xlabel("被判为异常的比例（%）")
    ax.set_title("两种异常值规则的比例对比")
    ax.legend()
    source_note(fig)
    save(fig, "fig04_outlier_rules.png")

    # 5 model comparison
    fig, ax = plt.subplots(figsize=(6, 3.4))
    names = models
    means = [result["cv"][m]["aggregate"]["macro_f1"]["mean"] for m in names]
    stds = [result["cv"][m]["aggregate"]["macro_f1"]["std"] for m in names]
    bars = ax.bar(names, means, yerr=stds, capsize=4,
                  color=[COLORS[i % len(COLORS)] for i in range(len(names))])
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(min(means) - 0.05, max(means) + 0.04)
    ax.set_xlabel("分类模型")
    ax.set_ylabel("Macro-F1（5 折均值±标准差，无量纲）")
    ax.set_title("分类模型固定 5 折对比")
    source_note(fig)
    save(fig, "fig05_model_comparison.png")

    # 6 confusion matrix
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"混淆矩阵（{best}，OOF）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    source_note(fig)
    save(fig, "fig06_confusion_matrix.png")

    # 7 ROC and 8 PR
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ap = {}
    for i, c in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y == i, prob[:, i])
        ax.plot(fpr, tpr, color=COLORS[i], label=f"{c} (AUC={auc(fpr, tpr):.3f})")
        ap[c] = float(average_precision_score(y == i, prob[:, i]))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    ax.set_xlabel("假阳性率")
    ax.set_ylabel("真阳性率")
    ax.set_title(f"ROC 曲线（{best}，OOF）")
    ax.legend(fontsize=8)
    source_note(fig)
    save(fig, "fig07_roc.png")

    fig, ax = plt.subplots(figsize=(4.8, 4))
    for i, c in enumerate(CLASS_NAMES):
        p, r, _ = precision_recall_curve(y == i, prob[:, i])
        ax.plot(r, p, color=COLORS[i], label=f"{c} (AP={ap[c]:.3f})")
    ax.set_xlabel("召回率")
    ax.set_ylabel("精确率")
    ax.set_title(f"PR 曲线（{best}，OOF）")
    ax.legend(fontsize=8)
    source_note(fig)
    save(fig, "fig08_pr.png")

    # 9 comes from supplement_evaluation.py using the exact saved outer-fold model.
    importance = json.loads((EVAL / "shap_importance.json").read_text())
    assert (FIG / "fig09_shap_importance.png").exists()
    result["shap_importance"] = importance

    # Render local explanations vertically so the labels remain legible in A4.
    extra = json.loads((EVAL / "supplement.json").read_text())
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 9.2))
    for ax, item in zip(axes, extra["local_samples"]):
        top = item["contributions"][:7][::-1]
        values = [x["shap"] for x in top]
        ax.barh([x["feature"] for x in top], values,
                color=[COLORS[0] if v >= 0 else COLORS[1] for v in values])
        ax.axvline(0, color="black", lw=.6)
        ax.set_title(f"已有标签 {item['true']} → 预测 {item['pred']}，概率 {item['probability']:.3f}", fontsize=10)
        ax.set_xlabel("对预测类概率的贡献（无量纲）")
    fig.suptitle("三个留出源的局部 SHAP 解释（RF，第0折）")
    source_note(fig, "数据来源：第0折留出源；蓝色提高、橙色降低概率；显示前7项。")
    save(fig, "fig13_local_shap.png")

    # 10 cluster PCA + 11 k selection
    if (BASE / "results/clustering/cluster_sample.parquet").exists():
        cs = pq.read_table(BASE / "results/clustering/cluster_sample.parquet").to_pandas()
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        idx = np.random.default_rng(config.SEED).choice(len(cs), min(20000, len(cs)), replace=False)
        sc = ax.scatter(cs["pca1"].to_numpy()[idx], cs["pca2"].to_numpy()[idx],
                        c=cs["cluster"].to_numpy()[idx], cmap=matplotlib.colors.ListedColormap(COLORS[:4]), s=2, alpha=0.6)
        ax.set_xlabel("主成分1（标准化空间，无量纲）")
        ax.set_ylabel("主成分2（标准化空间，无量纲）")
        ax.set_title("K-means 簇在 PCA 平面上的分布")
        fig.colorbar(sc, ax=ax, label="簇编号（类别）", fraction=0.046)
        source_note(fig, "数据来源：分层抽样 20 万源，15 个标准化特征")
        save(fig, "fig10_cluster_pca.png")

        clu = json.loads((BASE / "results/clustering/clustering.json").read_text())
        ks = [r["k"] for r in clu["search"]]
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.8))
        axes[0].plot(ks, [r["inertia"] for r in clu["search"]], marker="o", color=COLORS[0])
        axes[0].set_xlabel("簇数k（个）")
        axes[0].set_ylabel("簇内平方和（标准化，无量纲）")
        axes[0].set_title("肘部法")
        axes[1].plot(ks, [r["silhouette"] for r in clu["search"]], marker="s", color=COLORS[1])
        axes[1].set_xlabel("簇数k（个）")
        axes[1].set_ylabel("轮廓系数（无量纲）")
        axes[1].set_title("轮廓系数")
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.suptitle("聚类 k 值选择")
        source_note(fig, "数据来源：分层抽样 20 万源")
        save(fig, "fig11_k_selection.png")

    # 12 association rules
    if (BASE / "results/association/association_rules.json").exists():
        assoc = json.loads((BASE / "results/association/association_rules.json").read_text())
        rules = pd.DataFrame(assoc["top_rules"])
        if len(rules):
            fig, ax = plt.subplots(figsize=(5.6, 3.8))
            sc = ax.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis",
                            s=40, edgecolor="black", linewidth=0.3)
            ax.set_xlabel("支持度")
            ax.set_ylabel("置信度")
            ax.set_title("关联规则的支持度—置信度（颜色为提升度）")
            fig.colorbar(sc, ax=ax, label="提升度", fraction=0.046)
            source_note(fig, "数据来源：分层抽样 15 万源，离散化测光项集")
            save(fig, "fig12_association_rules.png")

    # error analysis table
    fused = pq.read_table(config.FUSED, columns=[config.GROUP_KEY, config.LABEL, "ra", "dec"] + config.CATWISE_NUMERIC).to_pandas()
    fused["w1_w2_color"] = fused["w1mpro"] - fused["w2mpro"]
    oof2 = oof[[config.GROUP_KEY, "y", f"pred_{best}", f"prob_{best}_GALAXY",
                f"prob_{best}_QSO", f"prob_{best}_STAR"]].copy()
    m = fused.merge(oof2, on=config.GROUP_KEY, how="inner")
    m["wrong"] = m["y"] != m[f"pred_{best}"]
    m["confidence"] = m[[f"prob_{best}_{c}" for c in CLASS_NAMES]].max(axis=1)
    errors = m[m["wrong"]].sort_values("confidence", ascending=False)
    picks = []
    for i, c in enumerate(CLASS_NAMES):
        sub_e = errors[errors["y"] == i].head(2)
        for _, r in sub_e.iterrows():
            picks.append({"source_id": r[config.GROUP_KEY], "true": CLASS_NAMES[int(r["y"])],
                          "pred": CLASS_NAMES[int(r[f"pred_{best}"])], "confidence": float(r["confidence"]),
                          "w1_w2_color": float(r["w1_w2_color"]), "w1mpro": float(r["w1mpro"]),
                          "w2mpro": float(r["w2mpro"]), "w1snr": float(r["w1snr"]),
                          "ra": float(r["ra"]), "dec": float(r["dec"])})
    error_summary = {
        "error_rate": float(m["wrong"].mean()),
        "by_class": {c: {"n": int(((m["y"] == i) & m["wrong"]).sum()),
                         "rate": float(m.loc[m["y"] == i, "wrong"].mean())} for i, c in enumerate(CLASS_NAMES)},
        "samples": picks[:6],
        "roc_auc": {}, "average_precision": ap,
    }
    for i, c in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y == i, prob[:, i])
        error_summary["roc_auc"][c] = float(auc(fpr, tpr))
    (EVAL / "evaluation_summary.json").write_text(json.dumps(error_summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figures": len(list(FIG.glob("*.png"))), "error_rate": error_summary["error_rate"],
                      "best": best}, ensure_ascii=False))


if __name__ == "__main__":
    main()
