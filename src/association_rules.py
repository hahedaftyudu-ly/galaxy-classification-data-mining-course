#!/usr/bin/env python3
"""Association-rule mining on discretised photometry: Apriori vs FP-Growth, sensitivity, pruning."""
import datetime
import json
import resource
import sys
import time
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

OUT = config.BASE / "results" / "association"
ITEM_TEXT = {
    "color_blue": "W1−W2 蓝端（<0.05）", "color_mid": "W1−W2 中间色（0.05–0.5）", "color_red": "W1−W2 红端（>0.5）",
    "mag_bright": "W1 亮源（<15 mag）", "mag_mid": "W1 中等亮度（15–16 mag）", "mag_faint": "W1 暗源（>16 mag）",
    "snr_low": "W1 信噪比低（<25）", "snr_mid": "W1 信噪比中（25–50）", "snr_high": "W1 信噪比高（>50）",
    "class_GALAXY": "星系", "class_QSO": "类星体", "class_STAR": "恒星",
}
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def transactions(size=150000):
    df = pq.read_table(config.FUSED, columns=[config.LABEL, "w1mpro", "w2mpro", "w1snr"]).to_pandas()
    df["w1_w2_color"] = df["w1mpro"] - df["w2mpro"]
    sample, _ = train_test_split(df, train_size=size, stratify=df[config.LABEL], random_state=config.SEED)
    sample = sample.reset_index(drop=True)
    frame = pd.DataFrame({
        "color": pd.cut(sample["w1_w2_color"], [-np.inf, 0.05, 0.5, np.inf],
                         labels=["color_blue", "color_mid", "color_red"]).astype(str),
        "mag": pd.cut(sample["w1mpro"], [-np.inf, 15.0, 16.0, np.inf],
                       labels=["mag_bright", "mag_mid", "mag_faint"]).astype(str),
        "snr": pd.cut(sample["w1snr"], [-np.inf, 25.0, 50.0, np.inf],
                       labels=["snr_low", "snr_mid", "snr_high"]).astype(str),
        "cls": "class_" + sample[config.LABEL].astype(str),
    })
    onehot = pd.get_dummies(frame, prefix_sep=":", dtype=bool)
    onehot.columns = [c.split(":")[-1] for c in onehot.columns]
    return sample, onehot


def mine(onehot, support, confidence, lift):
    t0 = time.time()
    ap = apriori(onehot, min_support=support, use_colnames=True, low_memory=True)
    t_ap = time.time() - t0
    t0 = time.time()
    fp = fpgrowth(onehot, min_support=support, use_colnames=True)
    t_fp = time.time() - t0
    rules = association_rules(ap, metric="confidence", min_threshold=confidence)
    rules = rules[rules["lift"] > lift].copy()
    return ap, fp, t_ap, t_fp, rules


def to_text(items):
    return " + ".join(sorted(ITEM_TEXT.get(str(i), str(i)) for i in items))


def prune(rules):
    """Drop empty antecedents, exact duplicates and rules whose subset antecedent is at least as confident."""
    rules = rules[rules["antecedents"].apply(len) > 0].copy()
    rules["ant_key"] = rules["antecedents"].apply(lambda x: tuple(sorted(map(str, x))))
    rules["con_key"] = rules["consequents"].apply(lambda x: tuple(sorted(map(str, x))))
    rules = rules.drop_duplicates(subset=["ant_key", "con_key"])
    keep = []
    for con, group in rules.groupby("con_key"):
        group = group.sort_values(["confidence", "lift"], ascending=False)
        chosen = []
        for _, row in group.iterrows():
            ant = set(row["ant_key"])
            if any(ant >= prev_ant and row["confidence"] <= prev_conf for prev_ant, prev_conf in chosen):
                continue
            chosen.append((ant, row["confidence"]))
            keep.append(row)
    if keep:
        rules = pd.DataFrame(keep)
    return rules


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sample, onehot = transactions()
    print("transactions", onehot.shape, flush=True)

    support, confidence, lift = 0.05, 0.7, 1.1
    ap, fp, t_ap, t_fp, rules = mine(onehot, support, confidence, lift)
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print("itemsets", len(ap), len(fp), "apriori", round(t_ap, 1), "fpgrowth", round(t_fp, 1), flush=True)

    # Matching counts alone cannot establish itemset equality.
    norm = lambda f: {tuple(sorted(r.itemsets)): round(float(r.support), 12) for r in f.itertuples()}
    assert norm(ap) == norm(fp), "Apriori/FP-Growth differ"
    sensitivity = []
    for s in (0.02, 0.05, 0.10):
        for c, lift_threshold in ((0.5, 1.0), (0.7, 1.1), (0.8, 1.5)):
            t0 = time.time()
            items = apriori(onehot, min_support=s, use_colnames=True, low_memory=True)
            r = association_rules(items, metric="confidence", min_threshold=c)
            r = r[r["lift"] > lift_threshold]
            sensitivity.append({"support": s, "confidence": c, "lift_min_exclusive": lift_threshold, "itemsets": len(items), "rules": len(r),
                                "max_lift": float(r["lift"].max()) if len(r) else None,
                                "elapsed_s": round(time.time() - t0, 1)})
            print("sens", s, c, len(r), flush=True)

    # Retain actionable rules with a single class consequent and no class in antecedent.
    # Exclude label-conditioned reverse descriptions and trivial low-lift rules.
    candidates = rules[rules["consequents"].apply(lambda z: len(z)==1 and all(str(i).startswith("class_") for i in z))
                       & rules["antecedents"].apply(lambda z: all(not str(i).startswith("class_") for i in z))].copy()
    pruned = prune(candidates)
    pruned = pruned.sort_values(["lift", "confidence"], ascending=False)
    top = []
    for _, r in pruned.head(20).iterrows():
        top.append({"antecedent": to_text(r["antecedents"]), "consequent": to_text(r["consequents"]),
                    "support": float(r["support"]), "confidence": float(r["confidence"]),
                    "lift": float(r["lift"]), "leverage": float(r["leverage"]),
                    "conviction": float(r["conviction"]) if np.isfinite(r["conviction"]) else None,
                    "antecedent_items": sorted(map(str, r["antecedents"])),
                    "consequent_items": sorted(map(str, r["consequents"]))})

    assert len(pruned) >= 5, "fewer than five actionable rules"
    result = {
        "finished_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "sample_rows": int(len(sample)), "items": ITEM_TEXT,
        "item_definitions": {"color": "W1−W2 <0.05 / 0.05–0.5 / >0.5",
                             "mag": "W1 <15 / 15–16 / >16 mag",
                             "snr": "W1 SNR <25 / 25–50 / >50",
                             "class": "GALAXY / QSO / STAR"},
        "main_thresholds": {"support": support, "confidence": confidence, "lift": lift},
        "efficiency": {"apriori_s": round(t_ap, 2), "fpgrowth_s": round(t_fp, 2),
                       "apriori_itemsets": int(len(ap)), "fpgrowth_itemsets": int(len(fp)),
                       "max_rss_mb": round(mem, 1),
                       "scan_note": "both run on the same one-hot matrix; timing is wall clock"},
        "sensitivity": sensitivity,
        "rules_before_prune": int(len(rules)), "actionable_candidates": int(len(candidates)), "rules_after_prune": int(len(pruned)),
        "top_rules": top,
        "elapsed_s": round(time.time() - started, 1),
    }
    (OUT / "association_rules.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    pruned.assign(
        antecedent=pruned["antecedents"].apply(to_text),
        consequent=pruned["consequents"].apply(to_text),
    )[["antecedent", "consequent", "support", "confidence", "lift", "leverage", "conviction"]].to_csv(
        OUT / "rules.csv", index=False)
    print(json.dumps({"rules": len(pruned), "apriori_s": result["efficiency"]["apriori_s"],
                      "fpgrowth_s": result["efficiency"]["fpgrowth_s"],
                      "elapsed_s": result["elapsed_s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
