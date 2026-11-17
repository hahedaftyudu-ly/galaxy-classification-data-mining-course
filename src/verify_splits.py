#!/usr/bin/env python3
"""Independent checks of the fixed split. Fails loudly instead of reporting a bad split."""
import datetime
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import make_splits


def main():
    fused = pq.read_table(config.FUSED, columns=[config.GROUP_KEY, config.LABEL]).to_pydict()
    folds = pq.read_table(config.FOLDS).to_pydict()
    summary = json.loads((config.SPLIT_DIR / "fold_summary.json").read_text())

    source_ids, labels = fused[config.GROUP_KEY], fused[config.LABEL]
    f_ids, f_labels, f_fold = folds[config.GROUP_KEY], folds[config.LABEL], np.asarray(folds["fold"])

    checks = []

    def check(name, condition, detail=""):
        if not condition:
            raise AssertionError("%s failed: %s" % (name, detail))
        checks.append(name)

    s_ids, s_labels = np.asarray(source_ids), np.asarray(labels)
    g_ids, g_labels = np.asarray(f_ids), np.asarray(f_labels)
    s_order = np.argsort(s_ids, kind="stable")
    g_order = np.argsort(g_ids, kind="stable")

    check("row count matches fused table", len(f_ids) == len(source_ids) == summary["rows"],
          "%d vs %d" % (len(f_ids), len(source_ids)))
    check("source_id unique", len(set(f_ids)) == len(f_ids))
    check("source_id set identical to fused table", set(f_ids) == set(source_ids))
    check("labels carried over unchanged",
          np.array_equal(s_ids[s_order], g_ids[g_order]) and np.array_equal(s_labels[s_order], g_labels[g_order]))
    check("all folds present", sorted(set(f_fold.tolist())) == list(range(config.N_FOLDS)))
    check("labels inside declared domain", set(f_labels) <= set(config.CLASSES))

    sizes = [int((f_fold == k).sum()) for k in range(config.N_FOLDS)]
    check("fold sizes within 3 rows", max(sizes) - min(sizes) <= 3, str(sizes))

    overall = {c: sum(1 for x in labels if x == c) / len(labels) for c in config.CLASSES}
    deviation = 0.0
    for k in range(config.N_FOLDS):
        mask = f_fold == k
        for c in config.CLASSES:
            share = float(np.sum(mask & (np.asarray(f_labels) == c))) / int(mask.sum())
            deviation = max(deviation, abs(share - overall[c]))
    check("class fractions within 1e-5 of overall", deviation < 1e-5, "%.2e" % deviation)

    order = np.argsort(np.asarray(f_ids, dtype=str), kind="stable")
    recomputed = make_splits.assign_folds(
        [f_ids[i] for i in order], [f_labels[i] for i in order], config.SEED, config.N_FOLDS)
    check("split reproducible from fixed seed",
          np.array_equal(recomputed, f_fold[order].astype(np.int8)))

    digest = make_splits.logical_sha256(f_ids, f_fold)
    check("logical hash matches recorded split", digest == summary["logical_sha256"],
          digest + " vs " + summary["logical_sha256"])
    check("source dataset hash matches acquisition record",
          summary["source_dataset_sha256"] == json.loads(
              (config.BASE / "results/acquisition/fusion_summary.json").read_text())["sha256"])

    result = {
        "verified_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "passed": True,
        "rows": len(f_ids),
        "fold_sizes": sizes,
        "max_class_fraction_deviation": deviation,
        "logical_sha256": digest,
        "checks": checks,
    }
    (config.SPLIT_DIR / "fold_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
