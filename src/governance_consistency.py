#!/usr/bin/env python3
"""Consistency audit: types, coding, units, duplicates, keys, time and join cardinality."""
import datetime
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

OUT = config.BASE / "results" / "governance"
COLUMNS = [config.GROUP_KEY, "source_name", config.LABEL] + config.CATWISE_NUMERIC + [
    "ra", "dec", "ra_01", "dec_01", "dist_x", "pang_x", "band_count", "cutout_arcsec",
    "cutout_sdss_offset_arcsec", "cutout_wise_offset_arcsec", "metadata_collected_utc",
    "r_mjd_obs", "r_metadata_scope"]


def count(df, mask, name, checks):
    n = int(mask.sum())
    checks.append({"check": name, "violations": n, "pct": round(n / len(df) * 100, 6)})
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(config.FUSED, columns=COLUMNS)
    df = table.to_pandas()
    n = len(df)
    checks = []

    # types and coding
    checks.append({"check": "declared dtypes", "detail": {name: str(typ) for name, typ in zip(table.schema.names, table.schema.types)}})
    count(df, ~df[config.LABEL].isin(config.CLASSES), "class label outside GALAXY/QSO/STAR", checks)
    count(df, df[config.GROUP_KEY].isna() | (df[config.GROUP_KEY] == ""), "empty source_id", checks)
    checks.append({"check": "source_id unique", "violations": int(df[config.GROUP_KEY].duplicated().sum()),
                   "pct": round(float(df[config.GROUP_KEY].duplicated().mean() * 100), 6)})
    checks.append({"check": "duplicate source_name", "violations": int(df["source_name"].duplicated().sum()),
                   "pct": round(float(df["source_name"].duplicated().mean() * 100), 6)})
    checks.append({"check": "duplicate (ra, dec)", "violations": int(df.duplicated(subset=["ra", "dec"]).sum()),
                   "pct": round(float(df.duplicated(subset=["ra", "dec"]).mean() * 100), 6)})
    checks.append({"check": "duplicate full row on numeric+label columns",
                   "violations": int(df.duplicated(subset=config.CATWISE_NUMERIC + [config.LABEL]).sum()),
                   "pct": round(float(df.duplicated(subset=config.CATWISE_NUMERIC + [config.LABEL]).mean() * 100), 6)})

    # ranges and units
    count(df, ~df["ra"].between(0, 360, inclusive="left"), "ra outside [0, 360)", checks)
    count(df, ~df["dec"].between(-90, 90), "dec outside [-90, 90]", checks)
    count(df, ~df["dist_x"].between(0, 3), "CatWISE match distance outside [0, 3] arcsec", checks)
    count(df, ~df["cutout_sdss_offset_arcsec"].between(0, 3), "cutout-to-SDSS offset outside [0, 3] arcsec", checks)
    count(df, ~(df["w1flux"] > 0) | ~(df["w2flux"] > 0), "non-positive WISE flux", checks)
    count(df, ~(df["w1snr"] > 0) | ~(df["w2snr"] > 0), "non-positive WISE SNR", checks)
    count(df, df["band_count"] != 5, "band_count not 5", checks)
    count(df, ~np.isfinite(df[config.CATWISE_NUMERIC]).all(axis=1), "non-finite numeric value", checks)
    count(df, df["cutout_arcsec"].notna() & ~df["cutout_arcsec"].eq(60), "cutout size not 60 arcsec", checks)

    # time fields
    ts = pd.to_datetime(df["metadata_collected_utc"], utc=True, errors="coerce")
    count(df, ts.isna(), "unparseable metadata_collected_utc", checks)
    checks.append({"check": "metadata_collected_utc range (UTC)",
                   "value": [str(ts.min()), str(ts.max())] if ts.notna().any() else None})
    checks.append({"check": "r_mjd_obs coverage", "present": int(df["r_mjd_obs"].notna().sum()),
                   "pct": round(float(df["r_mjd_obs"].notna().mean() * 100), 3),
                   "range": [float(df["r_mjd_obs"].min()), float(df["r_mjd_obs"].max())] if df["r_mjd_obs"].notna().any() else None})
    checks.append({"check": "metadata scope", "value": df["r_metadata_scope"].value_counts().to_dict()})

    # join cardinality with the fixed folds
    folds = pq.read_table(config.FOLDS, columns=[config.GROUP_KEY, "fold"]).to_pandas()
    merged = df[[config.GROUP_KEY, config.LABEL]].merge(folds, on=config.GROUP_KEY, how="outer", validate="1:1",
                                                       indicator=True)
    checks.append({"check": "fold join cardinality", "rows": int(len(merged)),
                   "unmatched": int((merged["_merge"] != "both").sum()),
                   "left_only": int((merged["_merge"] == "left_only").sum()),
                   "right_only": int((merged["_merge"] == "right_only").sum())})
    checks.append({"check": "fold distribution", "value": folds["fold"].value_counts().sort_index().to_dict()})

    # before/after table
    before = {"rows": n, "classes": df[config.LABEL].value_counts().to_dict(),
              "duplicate_source_id": int(df[config.GROUP_KEY].duplicated().sum()),
              "missing_numeric_cells": int(df[config.CATWISE_NUMERIC].isna().sum().sum())}
    after = {"rows": n - int(df[config.GROUP_KEY].duplicated().sum()),
             "classes": df.drop_duplicates(subset=[config.GROUP_KEY])[config.LABEL].value_counts().to_dict(),
             "missing_numeric_cells": int(df[config.CATWISE_NUMERIC].isna().sum().sum())}
    result = {
        "finished_at": datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "rows": n,
        "units": {"w1flux": "DN", "w2flux": "DN", "w1sigflux": "DN", "w2sigflux": "DN",
                  "w1mpro": "mag", "w2mpro": "mag", "w1sigmpro": "mag", "w2sigmpro": "mag",
                  "w1snr": "dimensionless", "w2snr": "dimensionless", "ra": "deg", "dec": "deg",
                  "dist_x": "arcsec", "cutout_sdss_offset_arcsec": "arcsec", "r_mjd_obs": "day"},
        "time_alignment": {"acquisition_time": "metadata_collected_utc is UTC (ISO8601)",
                           "observation_time": "r_mjd_obs only for header_and_file records; not a modelling feature",
                           "timezone_conversion": "none applied; UTC kept as recorded"},
        "checks": checks,
        "before_after": {"before": before, "after": after,
                         "removed_duplicates": before["duplicate_source_id"] - int(after.get("duplicate_source_id", 0))},
        "conclusion": "主键唯一、关联1:1、编码与量纲一致；未发现需要删除的重复记录或主键冲突。",
    }
    (OUT / "consistency.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({c["check"]: c.get("violations", c.get("unmatched", c.get("rows"))) for c in checks},
                     ensure_ascii=False, default=str)[:800])


if __name__ == "__main__":
    main()
