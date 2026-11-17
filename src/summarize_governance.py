#!/usr/bin/env python3
"""Turn the governance result JSONs into a summary JSON and the course report section."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "governance"


def pct(x):
    return f"{x:.3f}%"


def recon_mean(missing, key, strategy):
    values = [f[strategy]["reconstruction"]["_all_masked_cells_normalized"]["mae"]
              for f in missing["scenarios"][key]["folds"]]
    return float(sum(values) / len(values))


def main():
    missing = json.loads((OUT / "missing_experiment.json").read_text())
    outliers = json.loads((OUT / "outliers.json").read_text())
    consistency = json.loads((OUT / "consistency.json").read_text())

    baseline = missing["baseline"]["aggregate"]
    rows = []
    for key, block in missing["scenarios"].items():
        for strategy in ("deletion", "median", "iterative"):
            a = block["aggregate"][strategy]
            rows.append({
                "scenario": key, "mechanism": block["mechanism"], "rate": block["rate"], "strategy": strategy,
                "coverage": a["coverage_mean"], "accuracy": a["accuracy"]["mean"],
                "macro_f1": a["macro_f1"]["mean"], "macro_f1_std": a["macro_f1"]["std"],
                "macro_pr_auc": a["macro_pr_auc"]["mean"], "log_loss": a["log_loss"]["mean"],
                "common_macro_f1": a.get("common_macro_f1", {}).get("mean"),
                "recall": a["recall_mean"],
            })

    selection = {}
    for key, block in missing["scenarios"].items():
        agg = block["aggregate"]
        selection[key] = {
            "best_common_macro_f1": max(("deletion", "median", "iterative"),
                                        key=lambda s: agg[s].get("common_macro_f1", {}).get("mean", -1)),
            "best_full_coverage": max(("median", "iterative"), key=lambda s: agg[s]["macro_f1"]["mean"]),
            "coverage": {s: agg[s]["coverage_mean"] for s in ("deletion", "median", "iterative")},
            "note": ("deletion evaluates only complete cases; its metric is not comparable across mechanisms or rates"
                     if block["mechanism"] != "MCAR" else "deletion keeps complete cases only; imputers cover all rows"),
        }

    summary = {
        "baseline": baseline,
        "missing": {"rows": rows, "selection": selection,
                    "seed": missing["config"]["seed"], "elapsed_s": missing["elapsed_s"]},
        "outliers": {"sensitivity": outliers["sensitivity_summary"],
                     "features": outliers["features"], "decision": outliers["decision"],
                     "reason": outliers["reason"]},
        "consistency": {"rows": consistency["rows"], "checks": consistency["checks"],
                        "before_after": consistency["before_after"], "conclusion": consistency["conclusion"]},
    }
    (OUT / "governance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines = ["# 数据治理对照实验报告", "",
             "日期：2026-11-16。只使用本机固定划分与本地表；服务器全程只读，未训练、未写入。",
             "", "## 0. 范围与数据", "",
             "主表 1,314,678 行、59 字段；治理实验使用 14 个 CatWISE 数值特征与 `class_01` 标签。"
             "划分固定为 seed=20261112 的分层 5 折（`data/processed/folds.parquet`），所有变换只在训练折拟合。"
             "24 个星表字段未检出缺失；59 字段表中头字段约 99.24% 未采集，属于未采集而非观测缺失，不参与填补实验。",
             "", f"无缺失基线（LogisticRegression，5 折均值）：Accuracy {baseline['accuracy']['mean']:.4f}，"
             f"Macro-F1 {baseline['macro_f1']['mean']:.4f}±{baseline['macro_f1']['std']:.4f}，"
             f"Macro PR-AUC {baseline['macro_pr_auc']['mean']:.4f}，Log loss {baseline['log_loss']['mean']:.4f}。",
             "", "## 1. 缺失值", "",
             "### 1.1 机制判定", "",
             "原始 24 字段未检出 NaN、空值或无穷值，因此**无法从观测数据判定真实缺失机制**；"
             "把未检出缺失当成 MCAR 证据是不成立的。为评价预处理鲁棒性，在保持原始表不变的条件下做受控遮蔽：",
             "",
             "- MCAR：按源独立随机选择 5%/10%/20% 的源，随机遮蔽 W1 或 W2 整组 7 个特征。",
             "- MAR：遮蔽 W2，条件是**可见的** W1 信噪比不高于训练折分位数（阈值只用训练折估计）。",
             "- MNAR：遮蔽 W2，条件是**被隐藏的** W2 星等不低于训练折分位数；该值不提供给填补器。",
             "",
             "掩码在划分之后生成，三种策略共享同一掩码；每源最多遮蔽一个波段组。"
             "本轮使用单一种子 20261112；多种子重复留作后续稳健性检查。",
             "", "### 1.2 三种策略", "",
             "1. **删除**：训练与验证都只保留完整行；验证覆盖率随缺失率下降到 95%/90%/80%。",
             "2. **中位数**：每个训练折单独估计中位数，验证折只做变换。",
             "3. **迭代填补**：`IterativeImputer(LinearRegression(), max_iter=3)`，只在训练折拟合。",
             "",
             "下游统一用 `LogisticRegression(C=1.0, max_iter=300)` + 训练折 StandardScaler；"
             "指标为 5 折均值（每折在留出折上计算）。",
             "", "### 1.3 结果", "",
             "| 场景 | 策略 | 覆盖率 | Accuracy | Macro-F1 | Macro PR-AUC | Log loss | 共同完整子集 F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        common = "—" if r["common_macro_f1"] is None else f"{r['common_macro_f1']:.4f}"
        lines.append(f"| {r['scenario']} | {r['strategy']} | {r['coverage']:.3f} | {r['accuracy']:.4f} | "
                     f"{r['macro_f1']:.4f}±{r['macro_f1_std']:.4f} | {r['macro_pr_auc']:.4f} | "
                     f"{r['log_loss']:.4f} | {common} |")

    lines += ["", "### 1.4 选择依据与结论", "",
              "- **MCAR**：删除法在完整子集上保持约 0.727，但覆盖率从 95% 降到 80%；"
              "迭代填补从 0.711 降到 0.669，中位数从 0.691 降到 0.619。MCAR三个缺失率下中位数均低于迭代；这不是所有机制下的排序。",
              "- **MAR**：删除法 F1 从 0.732 升到 0.738，这是**选择偏差**——缺失源集中在低信噪比端，"
              "删掉后验证集变成更容易的子集，不能把升高解释为模型变好。迭代/中位数在全体验证集上下降。",
              "- **MNAR**：删除法在 5%/10% 时降到 0.706/0.687，20% 时又升到 0.763，"
              "因为保留下来的是亮源子集；同一原因，跨缺失率的删除法指标不可直接比较。"
              "迭代填补在 20% MNAR 时与中位数接近（0.552 vs 0.557），两者都明显退化。",
              "- **共同完整子集**：在共同完整验证行上比较才有相同评价人群；各策略在MNAR 20%等场景的排序可变化，详见表中common Macro-F1；"
              "但该子集在 MAR/MNAR 下是有偏子集，只适合同机制内的策略比较，不能外推到全体。",
              f"- **重建误差**（训练折标准化单位，5 折均值；MCAR 10%）：迭代归一化 MAE {recon_mean(missing, 'mcar_0.1', 'iterative'):.3f}，"
              f"中位数 {recon_mean(missing, 'mcar_0.1', 'median'):.3f}；MNAR 10% 时两者接近"
              f"（{recon_mean(missing, 'mnar_0.1', 'iterative'):.3f} vs {recon_mean(missing, 'mnar_0.1', 'median'):.3f}），"
              "说明隐藏波段主导缺失时，单靠可见波段的线性填补也失效。",
              "", "**决策**：主表没有真实缺失，主建模流程不做填补；如后续出现整波段缺失，"
              "在MCAR作为参考且需近全覆盖时可优先用迭代填补；机制变化时重新验证，删除法只在“完整案例可接受且样本量充足”时使用；"
              "中位数填补不作为默认策略；任何删除法结果都必须同时报告覆盖率和被删子集的类别/信噪比构成。"
              "若以覆盖率≥99%为约束，迭代填补在9个场景中的8个优于中位数；MNAR 20%中位数略高（0.5565 vs 0.5523）；删除法只作为完整案例可接受时的对照。",
              "", "本轮的主要统计量受下游探针模型限制（LR），换用树模型后的排序可能变化；"
              "结论用于数据治理决策，不替代分类主实验。", ""]

    lines += ["## 2. 异常值", "",
              "阈值在四个训练折上估计、应用到留出折；下表的百分比是 OOF 统计。",
              "", "| 特征 | 3σ 比例 | IQR 比例 | 两者重叠数 |", "|---|---:|---:|---:|"]
    for f in outliers["features"]:
        lines.append(f"| {f['feature']} | {pct(f['sigma3_pct'])} | {pct(f['iqr_pct'])} | {f['overlap_count']} |")
    sens = outliers["sensitivity_summary"]
    lines += ["", f"3σ 与 IQR 的重叠很小：IQR 把通量/误差等重尾分布的大量观测判为异常（通量约 9.6%–12.8%），"
              f"3σ 比例低得多。把源直接删除会损失物理上有意义的亮源/暗源。",
              "",
              f"**敏感性检验**：训练折 1%/99% Winsor 化后，Macro-F1 从 {sens['baseline_mean']:.4f} "
              f"升到 {sens['winsorized_mean']:.4f}（+{sens['delta_mean']:.4f}，5 折标准差 {sens['delta_std']:.4f}）。"
              "**决策**：" + outliers["decision"] + "验证折复用训练折边界；原始值保留用于审计与误差分析。",
              "", "## 3. 一致性", "",
              "| 检查 | 结果 |", "|---|---|"]
    for c in consistency["checks"]:
        if "violations" in c:
            lines.append(f"| {c['check']} | {c['violations']} 条违规 |")
        elif c["check"] == "fold join cardinality":
            lines.append(f"| {c['check']} | {c['rows']:,} 行，未匹配 {c['unmatched']} |")
        elif c["check"] == "r_mjd_obs coverage":
            lines.append(f"| {c['check']} | {c['present']:,} 条（{c['pct']}%） |")
        elif c["check"] == "metadata scope":
            v = c["value"]
            lines.append(f"| {c['check']} | file_only {v.get('file_only', 0):,}；header_and_file {v.get('header_and_file', 0):,} |")
        elif c["check"] == "metadata_collected_utc range (UTC)":
            lines.append(f"| {c['check']} | {c['value']} |")
        elif c["check"] == "fold distribution":
            v = c["value"]
            lines.append("| " + c["check"] + " | " + "；".join(f"fold{k}={int(v[k]):,}" for k in sorted(v)) + " |")
    ba = consistency["before_after"]
    lines += ["", f"处理前 {ba['before']['rows']:,} 行、删除重复主键 {ba['removed_duplicates']} 条；"
              f"处理后 {ba['after']['rows']:,} 行，类别计数不变。数值字段缺失 0，主键 1:1，"
              f"`metadata_collected_utc` 全部为 UTC，`r_mjd_obs` 只覆盖 {next(c for c in consistency['checks'] if c['check'] == 'r_mjd_obs coverage')['present']:,} 条（file_only 无头信息）。",
              "", "量纲：通量 DN、星等 mag、坐标 deg、匹配距离 arcsec、SNR 无量纲；"
              "采集时间是本轮导出时间（2026-11-11 UTC），不是观测时间。",
              "", "## 4. 防泄漏", "",
              "固定分层 5 折在任何依赖标签的变换之前生成；填补器、标准化、类别权重、异常值截尾均只在训练折拟合；"
              "治理掩码在划分之后生成，三种策略共享；目标编码未使用。详细规则见 `docs/防泄漏与划分说明.md`。",
              "", "## 5. 产物与复现", "",
              "```bash",
              "python3 src/governance_missing.py",
              "python3 src/governance_outliers.py",
              "python3 src/governance_consistency.py",
              "python3 src/summarize_governance.py",
              "```",
              "",
              "- `results/governance/missing_experiment.json`：逐折结果、掩码、重建误差、选择。",
              "- `results/governance/missing_summary.csv`：策略对照表。",
              "- `results/governance/outliers.json`、`consistency.json`。",
              "- `results/governance/governance_summary.json`：本报告使用的汇总。",
              f"- 缺失实验耗时 {missing['elapsed_s']:.0f} 秒；单机 10 核 CPU、16GB 内存。",
              "", "## 6. 对后续建模的影响", "",
              "- 主建模流程：训练折 Winsor 化 + 标准化；主表无缺失，不做填补。",
              "- 类别不均衡：基线 STAR 召回率约 0.29，后续分类实验必须使用类别权重或重采样并报告前后变化。",
              "- 若做缺失鲁棒性分析，报告覆盖率与子集构成，不能只报删除法的高分。",
              "- 本轮只验证了 LR 探针；树模型的填补/截尾敏感度需在分类实验中复核。", ""]
    (BASE / "docs" / "数据治理报告.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"summary": str(OUT / "governance_summary.json"),
                      "report": str(BASE / "docs" / "数据治理报告.md"),
                      "scenarios": len(missing["scenarios"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
