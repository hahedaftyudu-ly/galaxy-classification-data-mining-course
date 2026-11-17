#!/usr/bin/env python3
"""Shared constants for the galaxy-classification course project (星系分类网络).

The split seed and feature scopes live here so that every later stage
(governance, models, clustering, figures) uses exactly the same definitions.
"""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

# Fixed seed for every stochastic step. Documented in docs/防泄漏与划分说明.md.
SEED = 20261112
N_FOLDS = 5

LABEL = "class_01"
GROUP_KEY = "source_id"          # one row per source; five bands are not five samples
CLASSES = ("GALAXY", "QSO", "STAR")

# Default predictors: CatWISE/SDSS photometry only. Identifiers, paths, file
# timestamps and partially collected header fields are engineering metadata and
# must not be used as default predictors (see data/processed/model_feature_scope.json).
CATWISE_NUMERIC = [
    "w1sky", "w1sigsk", "w2sky", "w2sigsk",
    "w1snr", "w2snr",
    "w1flux", "w1sigflux", "w2flux", "w2sigflux",
    "w1mpro", "w1sigmpro", "w2mpro", "w2sigmpro",
]

# Domain-standard derived features (computed inside the modelling pipeline,
# never from the validation/test labels). They are not the optional "feature
# engineering" rubric item; see the design note for why no time-window feature
# is possible with single-epoch photometry.
DERIVED_FEATURES = {
    "w1_w2_color": "w1mpro - w2mpro",
    "w1_w2_flux_ratio": "log10(w1flux) - log10(w2flux)",
    "w1_snr_over_sigmpro": "w1snr / w1sigmpro",
    "w2_snr_over_sigmpro": "w2snr / w2sigmpro",
    "w1_flux_over_sigflux": "w1flux / w1sigflux",
    "w2_flux_over_sigflux": "w2flux / w2sigflux",
}

# Coordinates are kept for diagnostics/error analysis and for optional spatial
# robustness checks, but are not default predictors: sky position is a proxy
# for survey selection and neighbouring sources may share sky regions.
DIAGNOSTIC_COLUMNS = ["ra", "dec", "cutout_sdss_offset_arcsec", "cutout_wise_offset_arcsec"]

FUSED = BASE / "data" / "processed" / "fused_sources.parquet"
FOLDS = BASE / "data" / "processed" / "folds.parquet"
SPLIT_DIR = BASE / "results" / "splits"
