#!/usr/bin/env python3
"""Classification model comparison on the fixed stratified 5-fold split.

Baselines: logistic regression and Gaussian naive Bayes.
Additional models: random forest, extremely randomised trees, histogram gradient
boosting (all ensembles). Hyper-parameters are searched inside each outer training fold with 3-fold CV; the selected configurations are then evaluated on
the fixed 5 folds. All preprocessing (winsorisation, scaling) is fitted inside
each training fold only.
"""
import argparse
import hashlib
import joblib
import datetime
import json
import sys
import time
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import chi2, friedmanchisquare, rankdata
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

CLASS_NAMES = list(config.CLASSES)
CLASS_CODE = {name: i for i, name in enumerate(CLASS_NAMES)}
OUT = config.BASE / "results" / "classification"
FEATURES = config.CATWISE_NUMERIC + ["w1_w2_color"]
MODELS = ["LR", "NB", "RF", "ET", "HGB"]
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load():
    df = pq.read_table(config.FUSED, columns=[config.GROUP_KEY, config.LABEL] + config.CATWISE_NUMERIC).to_pandas()
    folds = pq.read_table(config.FOLDS, columns=[config.GROUP_KEY, "fold"]).to_pandas()
    df = df.merge(folds, on=config.GROUP_KEY, how="left", validate="1:1")
    df["w1_w2_color"] = df["w1mpro"] - df["w2mpro"]
    X = df[FEATURES].to_numpy(np.float64)
    y = df[config.LABEL].map(CLASS_CODE).to_numpy(np.int8)
    return df[config.GROUP_KEY].to_numpy(), X, y, df["fold"].to_numpy(np.int8)


def prep(Xtr, Xva, scale):
    lo = np.quantile(Xtr, 0.01, axis=0)
    hi = np.quantile(Xtr, 0.99, axis=0)
    Xtr = np.clip(Xtr, lo, hi)
    Xva = np.clip(Xva, lo, hi)
    if scale:
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xva = scaler.transform(Xva)
    return Xtr, Xva


def build(name, params, final=False):
    if name == "LR":
        return LogisticRegression(max_iter=500, **params)
    if name == "NB":
        return GaussianNB(**params)
    if name == "RF":
        return RandomForestClassifier(n_jobs=4, random_state=config.SEED,
                                      n_estimators=100 if final else 100, **params)
    if name == "ET":
        return ExtraTreesClassifier(n_jobs=4, random_state=config.SEED,
                                    n_estimators=100 if final else 100, **params)
    if name == "HGB":
        return HistGradientBoostingClassifier(random_state=config.SEED,
                                              max_iter=120 if final else 150,
                                              early_stopping=True, n_iter_no_change=20,
                                              validation_fraction=0.1, **params)
    raise ValueError(name)


GRIDS = {
    "LR": [{"C": c, "class_weight": w} for c in (0.1, 1.0, 10.0) for w in (None, "balanced")],
    "NB": [{"var_smoothing": v, "priors": p} for v in (1e-9, 1e-7, 1e-5)
           for p in (None, [1 / 3, 1 / 3, 1 / 3])],
    "RF": [{"max_depth": d, "min_samples_leaf": l, "class_weight": w}
           for d in (None, 30) for l in (1, 5) for w in (None, "balanced_subsample")],
    "ET": [{"max_depth": d, "min_samples_leaf": l, "class_weight": w}
           for d in (None, 30) for l in (1, 5) for w in (None, "balanced_subsample")],
    "HGB": [{"learning_rate": r, "max_leaf_nodes": m, "class_weight": w}
            for r in (0.05, 0.1) for m in (31, 63) for w in (None, "balanced")],
}
SCALE = {"LR": True, "NB": True, "RF": False, "ET": False, "HGB": False}


def score_metrics(y, prob):
    pred = prob.argmax(axis=1)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2])),
        "macro_pr_auc": float(average_precision_score(np.eye(3)[y], prob, average="macro")),
        "log_loss": float(log_loss(y, prob, labels=[0, 1, 2])),
        "recall": {c: float(v) for c, v in zip(CLASS_NAMES, recall_score(y, pred, average=None, labels=[0, 1, 2]))},
        "precision": {c: float(v) for c, v in zip(CLASS_NAMES, precision_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0))},
        "f1": {c: float(v) for c, v in zip(CLASS_NAMES, f1_score(y, pred, average=None, labels=[0, 1, 2]))},
        "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "n_iter": None,
    }


def tune_model(name, Xdev, ydev):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
    splits = list(skf.split(Xdev, ydev))
    results = []
    for params in GRIDS[name]:
        scores, t0 = [], time.time()
        for tr, va in splits:
            Xtr, Xva = prep(Xdev[tr], Xdev[va], SCALE[name])
            model = build(name, params).fit(Xtr, ydev[tr])
            scores.append(float(f1_score(ydev[va], model.predict(Xva), average="macro", labels=[0, 1, 2])))
        results.append({"params": params, "mean_macro_f1": float(np.mean(scores)),
                        "std_macro_f1": float(np.std(scores, ddof=1)), "folds": scores,
                        "elapsed_s": round(time.time() - t0, 1)})
        print("tune", name, params, round(results[-1]["mean_macro_f1"], 4), flush=True)
    best = max(results, key=lambda r: r["mean_macro_f1"])
    return best, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-size", type=int, default=30000)
    ap.add_argument("--skip-tune", action="store_true", help="reuse signature-checked nested checkpoints only")
    ap.add_argument("--fresh", action="store_true", help="ignore all cached fits")
    ap.add_argument("--imbalance", action="store_true")
    ap.add_argument("--models", default="LR,NB,RF,HGB")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    assert models and all(m in MODELS for m in models), models
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS = OUT / "parts"
    PARTS.mkdir(exist_ok=True)
    started = time.time()
    source_ids, X, y, fold = load()
    n = len(y)
    print("loaded", n, "rows", X.shape[1], "features", "models", models, flush=True)
    folds = sorted(set(fold.tolist()))

    # Every outer validation fold is excluded from inner hyperparameter search.
    # Cache identity includes the full data/split digests, grids and executable code.
    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(chunk)
        return h.hexdigest()
    identity = {"data": sha(config.FUSED), "folds": sha(config.FOLDS),
                "code": sha(Path(__file__)), "seed": config.SEED,
                "tune_size": args.tune_size, "grids": GRIDS,
                "sklearn": __import__("sklearn").__version__}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    PARTS = OUT / "nested_parts" / signature[:16]
    PARTS.mkdir(parents=True, exist_ok=True)
    tuning = {}
    outer_tuning = {m: {} for m in models}
    oof_prob = {name: np.zeros((n, 3), dtype=np.float32) for name in models}
    oof_pred = {name: np.zeros(n, dtype=np.int8) for name in models}
    cv = {name: {"folds": []} for name in models}
    for name in models:
        t = time.time()
        for k in folds:
            tr, va = fold != k, fold == k
            tune_file = PARTS / (name + "_f" + str(k) + "_tuning.json")
            tr_idx, va_idx = np.flatnonzero(tr), np.flatnonzero(va)
            if tune_file.exists() and not args.fresh:
                tuning_k = json.loads(tune_file.read_text())
            else:
                chosen, _ = train_test_split(tr_idx, train_size=min(args.tune_size, len(tr_idx)-3),
                                             stratify=y[tr_idx], random_state=config.SEED + int(k))
                assert not np.intersect1d(chosen, va_idx).size, "inner/outer leakage"
                best, all_results = tune_model(name, X[chosen], y[chosen])
                tuning_k = {"best": best, "all": all_results,
                            "outer_fold": int(k), "tune_rows": len(chosen),
                            "outer_train_rows": len(tr_idx), "outer_validation_rows": len(va_idx),
                            "inner_outer_overlap": 0,
                            "selected_id_sha256": hashlib.sha256("\n".join(source_ids[chosen]).encode()).hexdigest()}
                tune_file.write_text(json.dumps(tuning_k, ensure_ascii=False, indent=2) + "\n")
            outer_tuning[name][str(k)] = tuning_k
            tuning[name] = outer_tuning[name].get("0", tuning_k)
            part = PARTS / (name + "_f" + str(k) + ".npz")
            if part.exists() and not args.fresh:
                z = np.load(part, allow_pickle=False)
                prob = z["prob"]
                metrics = json.loads(str(z["metrics"]))
                assert np.array_equal(z["source_ids"], source_ids[va].astype(str))
                print("cv", name, "fold", k, "signature-verified cache", flush=True)
            else:
                Xtr, Xva = prep(X[tr], X[va], SCALE[name])
                model = build(name, tuning_k["best"]["params"], final=True).fit(Xtr, y[tr])
                prob = model.predict_proba(Xva)
                metrics = score_metrics(y[va], prob)
                metrics["fold"] = int(k)
                if hasattr(model, "n_iter_"):
                    metrics["n_iter"] = int(np.max(model.n_iter_))
                np.savez_compressed(part, prob=prob.astype(np.float32),
                                    source_ids=source_ids[va].astype(str),
                                    metrics=np.array(json.dumps(metrics, ensure_ascii=False)))
                if name == "RF" and k == 0:
                    # Exact fold-0 predictor for local explanations on unseen fold 0.
                    joblib.dump({"model": model, "lo": np.quantile(X[tr], .01, axis=0),
                                 "hi": np.quantile(X[tr], .99, axis=0),
                                 "features": FEATURES, "heldout_fold": 0,
                                 "signature": signature}, OUT / "rf_fold0.joblib", compress=3)
                print("cv", name, "fold", k, "macro_f1", round(metrics["macro_f1"], 4), flush=True)
            assert prob.shape == (int(va.sum()), 3)
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-5)
            oof_prob[name][va] = prob.astype(np.float32)
            oof_pred[name][va] = prob.argmax(axis=1).astype(np.int8)
            cv[name]["folds"].append(metrics)
        agg = {}
        for key in ("accuracy", "macro_f1", "macro_pr_auc", "log_loss"):
            values = [f[key] for f in cv[name]["folds"]]
            agg[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
        agg["recall_mean"] = {c: float(np.mean([f["recall"][c] for f in cv[name]["folds"]])) for c in CLASS_NAMES}
        agg["precision_mean"] = {c: float(np.mean([f["precision"][c] for f in cv[name]["folds"]])) for c in CLASS_NAMES}
        agg["f1_mean"] = {c: float(np.mean([f["f1"][c] for f in cv[name]["folds"]])) for c in CLASS_NAMES}
        cv[name]["aggregate"] = agg
        cv[name]["elapsed_s"] = round(time.time() - t, 1)
        (OUT / "cv_progress.json").write_text(json.dumps({"cv": cv}, ensure_ascii=False, indent=2) + "\n")
        print("cv", name, "done", cv[name]["elapsed_s"], "s", flush=True)

    # OOF metrics and paired tests
    oof_metrics = {name: score_metrics(y, oof_prob[name]) for name in models}
    best_model = max(models, key=lambda m: oof_metrics[m]["macro_f1"])

    def mcnemar(a, b):
        ca = oof_pred[a] == y
        cb = oof_pred[b] == y
        n01 = int(np.sum(~ca & cb))
        n10 = int(np.sum(ca & ~cb))
        stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) else 0.0
        return {"a": a, "b": b, "a_wrong_b_right": n01, "a_right_b_wrong": n10,
                "statistic": float(stat), "p_value": float(chi2.sf(stat, 1)),
                "df": 1, "method": "McNemar with continuity correction, OOF predictions"}

    mcnemar_tests = [mcnemar(best_model, base) for base in models if base != best_model]

    fold_scores = {m: [f["macro_f1"] for f in cv[m]["folds"]] for m in models}
    friedman = friedmanchisquare(*[fold_scores[m] for m in models])
    ranks = pd.DataFrame(fold_scores).rank(axis=1, ascending=False)
    avg_ranks = {m: float(ranks[m].mean()) for m in models}
    k, nblocks = len(models), len(folds)
    q_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}
    q_alpha = q_table.get(k, 2.728)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * nblocks))
    pairs = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            pairs.append({"a": a, "b": b, "rank_diff": abs(avg_ranks[a] - avg_ranks[b]),
                          "significant": bool(abs(avg_ranks[a] - avg_ranks[b]) > cd)})
    tests = {
        "best_model": best_model,
        "mcnemar": mcnemar_tests,
        "friedman": {"statistic": float(friedman.statistic), "p_value": float(friedman.pvalue),
                     "df": k - 1, "scores_by_fold": fold_scores},
        "nemenyi": {"average_ranks": avg_ranks, "critical_difference": float(cd), "pairs": pairs,
                    "note": "Nemenyi post-hoc with q=%.3f (k=%d, alpha=0.05); 5 folds is a small block count." % (q_alpha, k)},
    }

    # optional class-imbalance sensitivity (five extra fits, can be run separately)
    imbalance = None
    if args.imbalance:
        base = best_model if best_model != "LR" else "HGB"
        if base == "NB":
            base = "HGB"
        params = dict(outer_tuning[base]["0"]["best"]["params"])
        params["class_weight"] = "balanced"
        t = time.time()
        folds_metrics = []
        for k in folds:
            tr, va = fold != k, fold == k
            Xtr, Xva = prep(X[tr], X[va], SCALE[base])
            model = build(base, params, final=True).fit(Xtr, y[tr])
            prob = model.predict_proba(Xva)
            metric = score_metrics(y[va], prob)
            metric["fold"] = k
            folds_metrics.append(metric)
        imbalance = {"model": base, "params": params, "folds": folds_metrics,
                     "aggregate": {key: {"mean": float(np.mean([f[key] for f in folds_metrics])),
                                         "std": float(np.std([f[key] for f in folds_metrics], ddof=1))}
                                   for key in ("accuracy", "macro_f1", "macro_pr_auc", "log_loss")},
                     "recall_mean": {c: float(np.mean([f["recall"][c] for f in folds_metrics])) for c in CLASS_NAMES},
                     "elapsed_s": round(time.time() - t, 1)}

    result = {
        "finished_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "config": {"features": FEATURES, "seed": config.SEED, "folds": folds, "rows": n, "models": models,
                   "tuning": "nested: inner stratified 3-fold search on up to %d rows from each outer training fold; final refit on full outer training fold" % args.tune_size,
                   "preprocessing": "train-fold 1%/99% winsorisation; StandardScaler only for LR/NB",
                   "class_weights": "searched per model; imbalance sensitivity reported separately",
                   "lightgbm": "not available (missing libomp on this host)"},
        "tuning": tuning, "outer_tuning": outer_tuning,
        "cache_identity": identity, "cache_signature": signature,
        "cv": cv,
        "oof_metrics": oof_metrics,
        "tests": tests,
        "imbalance": imbalance,
        "elapsed_s": round(time.time() - started, 1),
    }
    (OUT / "tuning.json").write_text(json.dumps({"outer_tuning": outer_tuning}, ensure_ascii=False, indent=2) + "\n")
    (OUT / "model_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    table = pa.table({
        "source_id": pa.array(source_ids),
        "fold": pa.array(fold, type=pa.int8()),
        "y": pa.array(y, type=pa.int8()),
        **{f"pred_{m}": pa.array(oof_pred[m], type=pa.int8()) for m in models},
        **{f"prob_{m}_{c}": pa.array(oof_prob[m][:, i], type=pa.float32())
           for m in models for i, c in enumerate(CLASS_NAMES)},
    })
    pq.write_table(table, OUT / "oof_predictions.parquet", compression="zstd")
    print(json.dumps({"out": str(OUT / "model_comparison.json"), "best": best_model,
                      "best_macro_f1": oof_metrics[best_model]["macro_f1"],
                      "elapsed_s": result["elapsed_s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
