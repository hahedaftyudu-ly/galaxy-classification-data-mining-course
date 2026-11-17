#!/usr/bin/env python3
"""Build the fixed stratified 5-fold split used by every later stage.

Design: one row per source_id, stratified by class_01, deterministic given
config.SEED. The split is written once and never re-drawn, so data governance
masks, model tuning and final metrics share the same folds.
"""
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def assign_folds(source_ids, labels, seed=config.SEED, n_folds=config.N_FOLDS):
    """Return an int8 fold vector. Rows must already be sorted by source_id."""
    labels = np.asarray(labels)
    fold = np.full(len(labels), -1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    for cls in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == cls)
        order = idx[rng.permutation(len(idx))]
        fold[order] = (np.arange(len(order)) % n_folds).astype(np.int8)
    assert (fold >= 0).all(), "every row must receive a fold"
    return fold


def logical_sha256(source_ids, fold):
    digest = hashlib.sha256()
    for sid, f in zip(source_ids, fold):
        digest.update(str(sid).encode())
        digest.update(b"\t")
        digest.update(str(int(f)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def main():
    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(config.FUSED, columns=[config.GROUP_KEY, config.LABEL])
    source_ids = table.column(config.GROUP_KEY).to_pylist()
    labels = table.column(config.LABEL).to_pylist()
    assert len(source_ids) == len(labels) == len(set(source_ids)), "source_id must be unique"
    order = np.argsort(np.asarray(source_ids, dtype=str), kind="stable")
    source_ids = [source_ids[i] for i in order]
    labels = [labels[i] for i in order]

    fold = assign_folds(source_ids, labels)
    unknown = set(labels) - set(config.CLASSES)
    assert not unknown, "unexpected labels: %s" % unknown

    out = pa.table({
        config.GROUP_KEY: pa.array(source_ids, type=pa.string()),
        config.LABEL: pa.array(labels, type=pa.string()),
        "fold": pa.array(fold, type=pa.int8()),
    })
    tmp = config.FOLDS.with_suffix(".parquet.tmp")
    pq.write_table(out, tmp, compression="zstd")
    tmp.replace(config.FOLDS)

    classes = list(config.CLASSES)
    overall = {c: labels.count(c) for c in classes}
    per_fold = []
    for k in range(config.N_FOLDS):
        mask = fold == k
        counts = {c: int(np.sum(mask & (np.asarray(labels) == c))) for c in classes}
        per_fold.append({
            "fold": k,
            "rows": int(mask.sum()),
            "classes": counts,
            "class_fraction": {c: round(counts[c] / int(mask.sum()), 8) for c in classes},
        })
    proportions = {c: overall[c] / len(labels) for c in classes}
    max_dev = max(
        abs(entry["class_fraction"][c] - proportions[c])
        for entry in per_fold for c in classes
    )
    summary = {
        "created_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "seed": config.SEED,
        "n_folds": config.N_FOLDS,
        "rows": len(labels),
        "method": "per-class permutation, then position modulo n_folds",
        "classes": overall,
        "overall_fraction": {c: round(proportions[c], 8) for c in classes},
        "folds": per_fold,
        "max_class_fraction_deviation": max_dev,
        "logical_sha256": logical_sha256(source_ids, fold),
        "file": str(config.FOLDS),
        "source_dataset_sha256": json.loads(
            (config.BASE / "results/acquisition/fusion_summary.json").read_text()
        )["sha256"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pyarrow": pa.__version__},
    }
    (config.SPLIT_DIR / "fold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("rows", "n_folds", "seed", "max_class_fraction_deviation", "logical_sha256")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
