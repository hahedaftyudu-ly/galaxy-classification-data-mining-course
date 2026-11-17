#!/usr/bin/env python3
"""Controlled missing-value experiment for the course project.

The raw catalogue has no missing values, so the experiment masks whole WISE
band groups on the fixed folds and compares three strategies: complete-case
deletion, median imputation and iterative (linear) imputation. Everything that
depends on the data is fitted on the training folds only; the mask is shared by
all three strategies.
"""
import argparse
import datetime
import json
import sys
import time
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             log_loss, recall_score)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

W1_COLUMNS = ["w1sky", "w1sigsk", "w1snr", "w1flux", "w1sigflux", "w1mpro", "w1sigmpro"]
W2_COLUMNS = ["w2sky", "w2sigsk", "w2snr", "w2flux", "w2sigflux", "w2mpro", "w2sigmpro"]
CLASS_NAMES = list(config.CLASSES)  # GALAXY, QSO, STAR
CLASS_CODE = {name: i for i, name in enumerate(CLASS_NAMES)}
OUT = config.BASE / "results" / "governance"

warnings.filterwarnings("ignore", message=".*encountered in matmul")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def load_data():
    df = pq.read_table(config.FUSED, columns=[config.GROUP_KEY, config.LABEL] + config.CATWISE_NUMERIC).to_pandas()
    folds = pq.read_table(config.FOLDS, columns=[config.GROUP_KEY, "fold"]).to_pandas()
    df = df.merge(folds, on=config.GROUP_KEY, how="left", validate="1:1")
    assert df["fold"].notna().all(), "every source must have a fold"
    X = df[config.CATWISE_NUMERIC].to_numpy(np.float64)
    y = df[config.LABEL].map(CLASS_CODE).to_numpy(np.int8)
    return X, y, df["fold"].to_numpy(np.int8), df[config.LABEL].to_numpy()


def mcar_mask(n, rate, seed):
    rng = np.random.default_rng(seed + int(round(rate * 100)))
    return rng.random(n) < rate, rng.integers(0, 2, n)


def band_mask(column_values, threshold, upper=False):
    """Deterministic MAR/MNAR selection on one observed or hidden feature."""
    return column_values >= threshold if upper else column_values <= threshold


def apply_mask(X, row_mask, band):
    Xm = X.copy()
    w1 = np.flatnonzero(band == 0)
    w2 = np.flatnonzero(band == 1)
    w1 = w1[row_mask[w1]]
    w2 = w2[row_mask[w2]]
    Xm[np.ix_(w1, [config.CATWISE_NUMERIC.index(c) for c in W1_COLUMNS])] = np.nan
    Xm[np.ix_(w2, [config.CATWISE_NUMERIC.index(c) for c in W2_COLUMNS])] = np.nan
    return Xm


def apply_fixed_band_mask(X, row_mask, columns):
    Xm = X.copy()
    idx = [config.CATWISE_NUMERIC.index(c) for c in columns]
    Xm[np.ix_(np.flatnonzero(row_mask), idx)] = np.nan
    return Xm


def metrics(y_true, prob):
    pred = prob.argmax(axis=1)
    onehot = np.eye(len(CLASS_NAMES))[y_true]
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=list(range(len(CLASS_NAMES))))),
        "macro_pr_auc": float(average_precision_score(onehot, prob, average="macro")),
        "log_loss": float(log_loss(y_true, prob, labels=list(range(len(CLASS_NAMES))))),
        "recall": {name: float(v) for name, v in zip(CLASS_NAMES, recall_score(
            y_true, pred, average=None, labels=list(range(len(CLASS_NAMES)))))},
    }


def fit_lr(Xtr, ytr, Xva):
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(C=1.0, max_iter=300)
    model.fit(scaler.transform(Xtr), ytr)
    return model.predict_proba(scaler.transform(Xva)), int(model.n_iter_[0])


def reconstruction(imputed, truth, rows, band, columns, scale):
    """Per-feature reconstruction error on cells that were actually masked.

    Original-unit errors are reported per field; the aggregate uses errors
    divided by the training-fold standard deviation because flux and magnitude
    fields have different units and scales.
    """
    if rows.sum() == 0:
        return {}
    result, normalized = {}, []
    for name in columns:
        j = config.CATWISE_NUMERIC.index(name)
        cell_rows = rows & (band == (0 if name in W1_COLUMNS else 1))
        if not cell_rows.any():
            continue
        e = imputed[cell_rows, j] - truth[cell_rows, j]
        result[name] = {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))),
                        "normalized_mae": float(np.mean(np.abs(e)) / scale[j]),
                        "normalized_rmse": float(np.sqrt(np.mean(e * e)) / scale[j]),
                        "n": int(cell_rows.sum())}
        normalized.append(e / scale[j])
    e = np.concatenate(normalized) if normalized else np.array([])
    result["_all_masked_cells_normalized"] = {
        "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))), "n": int(e.size)}
    return result


def evaluate_scenario(X, y, fold, row_mask, band, tr, va, reconstruction_columns):
    Xm = apply_mask(X, row_mask, band)
    out = {}
    train_ok = tr & ~row_mask
    valid_ok = va & ~row_mask
    scale = np.std(X[tr], axis=0)

    # complete-case deletion: fit and evaluate only on unmasked rows
    prob_del, it_del = fit_lr(Xm[train_ok], y[train_ok], Xm[valid_ok])
    out["deletion"] = {"metrics": metrics(y[valid_ok], prob_del), "coverage": float(valid_ok.sum() / va.sum()),
                       "train_rows": int(train_ok.sum()), "iterations": it_del,
                       "common_subset": None}

    # median imputation fitted on the training folds only
    med = np.nanmedian(Xm[tr], axis=0)
    Xtr_med = np.where(np.isnan(Xm[tr]), med, Xm[tr])
    Xva_med = np.where(np.isnan(Xm[va]), med, Xm[va])
    prob_med, it_med = fit_lr(Xtr_med, y[tr], Xva_med)
    out["median"] = {"metrics": metrics(y[va], prob_med), "coverage": 1.0, "train_rows": int(tr.sum()),
                     "iterations": it_med,
                     "reconstruction": reconstruction(Xva_med, X[va], row_mask[va], band[va], reconstruction_columns, scale)}

    # iterative imputation fitted on the training folds only
    imputer = IterativeImputer(estimator=LinearRegression(), max_iter=3, random_state=config.SEED)
    Xtr_iter = imputer.fit_transform(Xm[tr])
    Xva_iter = imputer.transform(Xm[va])
    prob_iter, it_iter = fit_lr(Xtr_iter, y[tr], Xva_iter)
    out["iterative"] = {"metrics": metrics(y[va], prob_iter), "coverage": 1.0, "train_rows": int(tr.sum()),
                        "iterations": it_iter,
                        "reconstruction": reconstruction(Xva_iter, X[va], row_mask[va], band[va], reconstruction_columns, scale)}

    # fair comparison on the same complete-case validation rows
    if valid_ok.any():
        common_y = y[valid_ok]
        complete_va = ~row_mask[va]
        out["deletion"]["common_subset"] = metrics(common_y, prob_del)
        out["median"]["common_subset"] = metrics(common_y, prob_med[complete_va])
        out["iterative"]["common_subset"] = metrics(common_y, prob_iter[complete_va])
    return out


def aggregate(rows):
    keys = ["accuracy", "macro_f1", "macro_pr_auc", "log_loss"]
    result = {}
    for strategy in ("deletion", "median", "iterative"):
        block = {"coverage_mean": float(np.mean([r[strategy]["coverage"] for r in rows]))}
        for k in keys:
            values = [r[strategy]["metrics"][k] for r in rows]
            block[k] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "folds": values}
        for k in keys:
            values = [r[strategy]["common_subset"][k] for r in rows if r[strategy]["common_subset"]]
            if values:
                block["common_" + k] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
        block["recall_mean"] = {
            c: float(np.mean([r[strategy]["metrics"]["recall"][c] for r in rows])) for c in CLASS_NAMES}
        result[strategy] = block
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.05,0.10,0.20")
    ap.add_argument("--scenarios", default="mcar,mar,mnar")
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--out", default="missing_experiment.json")
    args = ap.parse_args()
    rates = [float(x) for x in args.rates.split(",")]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    started = time.time()
    X, y, fold, labels = load_data()
    n = len(y)
    print("loaded", n, "rows,", X.shape[1], "features", flush=True)
    folds = sorted(set(fold.tolist()))
    if args.max_folds:
        folds = folds[: args.max_folds]

    # baseline: no missing values, same folds
    baseline = {}
    baseline_folds = []
    t0 = time.time()
    for k in folds:
        tr, va = fold != k, fold == k
        prob, it = fit_lr(X[tr], y[tr], X[va])
        baseline_folds.append({"fold": k, "metrics": metrics(y[va], prob), "iterations": it})
        print("baseline fold", k, "n_iter", it, "macro_f1", baseline_folds[-1]["metrics"]["macro_f1"], flush=True)
    baseline["folds"] = baseline_folds
    baseline["aggregate"] = {
        k: {"mean": float(np.mean([f["metrics"][k] for f in baseline_folds])),
            "std": float(np.std([f["metrics"][k] for f in baseline_folds], ddof=1))}
        for k in ("accuracy", "macro_f1", "macro_pr_auc", "log_loss")}
    baseline["elapsed_s"] = round(time.time() - t0, 1)

    results = {"config": {"seed": config.SEED, "rates": rates, "scenarios": scenarios,
                          "folds": folds, "rows": n, "features": config.CATWISE_NUMERIC,
                          "baseline_model": "LogisticRegression(C=1.0, max_iter=300), StandardScaler",
                          "imputers": {"median": "train-fold nanmedian",
                                       "iterative": "IterativeImputer(LinearRegression, max_iter=3)"},
                          "mask_definition": "whole W1 or W2 band group per masked source"},
               "baseline": baseline, "scenarios": {}}

    for scenario in scenarios:
        for rate in rates:
            key = scenario + "_" + str(rate)
            scenario_folds = []
            t = time.time()
            for k in folds:
                tr, va = fold != k, fold == k
                if scenario == "mcar":
                    row_mask, band = mcar_mask(n, rate, config.SEED)
                    band_used = "random W1/W2 per source"
                    reconstruction_columns = W1_COLUMNS + W2_COLUMNS
                elif scenario == "mar":
                    threshold = float(np.quantile(X[tr, config.CATWISE_NUMERIC.index("w1snr")], rate))
                    row_mask = band_mask(X[:, config.CATWISE_NUMERIC.index("w1snr")], threshold, upper=False)
                    band = np.ones(n, dtype=np.int8)
                    band_used = "W2 masked when observed W1 SNR <= train quantile %.6f" % threshold
                    reconstruction_columns = W2_COLUMNS
                elif scenario == "mnar":
                    threshold = float(np.quantile(X[tr, config.CATWISE_NUMERIC.index("w2mpro")], 1 - rate))
                    row_mask = band_mask(X[:, config.CATWISE_NUMERIC.index("w2mpro")], threshold, upper=True)
                    band = np.ones(n, dtype=np.int8)
                    band_used = "W2 masked when hidden W2 magnitude >= train quantile %.6f" % threshold
                    reconstruction_columns = W2_COLUMNS
                else:
                    raise SystemExit("unknown scenario " + scenario)
                out = evaluate_scenario(X, y, fold, row_mask, band, tr, va, reconstruction_columns)
                scenario_folds.append({"fold": k, "masked_fraction_train": float(row_mask[tr].mean()),
                                       "masked_fraction_valid": float(row_mask[va].mean()),
                                       **out})
                print(key, "fold", k, "mask", round(float(row_mask.mean()), 4),
                      "f1 del/med/iter", round(out["deletion"]["metrics"]["macro_f1"], 4),
                      round(out["median"]["metrics"]["macro_f1"], 4),
                      round(out["iterative"]["metrics"]["macro_f1"], 4), flush=True)
            results["scenarios"][key] = {
                "mechanism": scenario.upper(),
                "rate": rate,
                "band_rule": band_used,
                "reconstruction_columns": reconstruction_columns,
                "folds": scenario_folds,
                "aggregate": aggregate(scenario_folds),
                "elapsed_s": round(time.time() - t, 1),
            }
            print(key, "done", round(time.time() - t, 1), "s", flush=True)

    # selection: primary on the shared complete-case subset, secondary full-valid
    selection = {}
    for key, block in results["scenarios"].items():
        agg = block["aggregate"]
        common = {s: agg[s].get("common_macro_f1", {}).get("mean") for s in ("deletion", "median", "iterative")}
        selection[key] = {"by_common_macro_f1": max(common, key=lambda s: common[s]) if all(v is not None for v in common.values()) else None,
                          "common_macro_f1": common,
                          "by_full_macro_f1_imputers": max(("median", "iterative"),
                                                           key=lambda s: agg[s]["macro_f1"]["mean"])}
    results["selection"] = selection
    results["elapsed_s"] = round(time.time() - started, 1)
    results["finished_at"] = datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    path = OUT / args.out
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")

    rows = []
    for key, block in results["scenarios"].items():
        for strategy in ("deletion", "median", "iterative"):
            a = block["aggregate"][strategy]
            rows.append({"scenario": key, "strategy": strategy, "coverage": round(a["coverage_mean"], 4),
                         "accuracy": round(a["accuracy"]["mean"], 4), "accuracy_std": round(a["accuracy"]["std"], 4),
                         "macro_f1": round(a["macro_f1"]["mean"], 4), "macro_f1_std": round(a["macro_f1"]["std"], 4),
                         "macro_pr_auc": round(a["macro_pr_auc"]["mean"], 4),
                         "log_loss": round(a["log_loss"]["mean"], 4),
                         "common_macro_f1": round(a.get("common_macro_f1", {}).get("mean", float("nan")), 4)})
    pd.DataFrame(rows).to_csv(OUT / "missing_summary.csv", index=False)
    print(json.dumps({"out": str(path), "elapsed_s": results["elapsed_s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
