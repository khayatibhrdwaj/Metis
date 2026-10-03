"""
Metis Backend - Multi-Disease Model Wrapper

Loads and serves the 4 additional disease-risk models (Metabolic
Syndrome, NAFLD, CKD, Hypertension) trained by
training/train_multi_disease.py. Each is a single monotonic XGBoost
model with its own non-circular predictor set (see that file's docstring
for exactly what's excluded and why).

The existing diabetes model (the stacked ensemble) stays in model.py,
untouched - this module is purely additive. predict_all() combines all 5
into one call for the unified "Metabolic Health" dashboard.

CALIBRATION: training/calibration_analysis.py found that the raw
Metabolic Syndrome and CKD models were meaningfully OVERCONFIDENT at
higher predicted probabilities (CKD's raw output said "80% risk" for
people where only ~37% actually had the condition) - a known failure
mode for tree-based models on lower-prevalence outcomes. Isotonic
recalibration fixed this (CKD's calibration error dropped from 0.190 to
0.020), with the fitted calibrators saved to
training/validation_results/calibration/*_calibrator.pkl. NAFLD and
Hypertension's raw models were already well-calibrated and get no
transform. This is loaded and applied automatically below if a
calibrator file exists for a given disease - no code changes needed if
you re-run calibration_analysis.py and get a different verdict.
Isotonic regression is itself a monotonic non-decreasing function, so
composing it with an already-monotonic base model preserves the
simulator's causal-sanity guarantee - verified empirically, not just
assumed (see the regression test re-run after this was wired in).
"""

import json
import pickle
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"
CALIBRATION_DIR = Path(__file__).resolve().parent / "training" / "validation_results" / "calibration"

DISEASE_NAMES = ["metabolic_syndrome", "nafld", "ckd", "hypertension"]

DISEASE_LABELS = {
    "diabetes": "Type 2 Diabetes",
    "metabolic_syndrome": "Metabolic Syndrome",
    "nafld": "Fatty Liver Disease (NAFLD)",
    "ckd": "Chronic Kidney Disease",
    "hypertension": "Hypertension",
}

_cache = {}  # name -> (model, explainer, medians, features, metrics, calibrator_or_None)


def _load(name: str):
    if name not in _cache:
        with open(MODEL_DIR / f"{name}_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(MODEL_DIR / f"{name}_explainer.pkl", "rb") as f:
            explainer = pickle.load(f)
        with open(MODEL_DIR / f"{name}_medians.pkl", "rb") as f:
            medians = pickle.load(f)
        with open(MODEL_DIR / f"{name}_metrics.json") as f:
            metrics = json.load(f)

        calibrator = None
        calibrator_path = CALIBRATION_DIR / f"{name}_calibrator.pkl"
        if calibrator_path.exists():
            with open(calibrator_path, "rb") as f:
                calibrator = pickle.load(f)

        _cache[name] = (model, explainer, medians, metrics["features"], metrics, calibrator)
    return _cache[name]


def _row(feature_dict: dict, medians, features) -> pd.DataFrame:
    return pd.DataFrame([{f: feature_dict.get(f, medians[f]) for f in features}])


# No statistical model trained on finite data should ever output an
# estimate indistinguishable from mathematical certainty - caught in
# practice when the Metabolic Syndrome calibrator's isotonic fit produced
# an exact 1.0 for high raw inputs (a small-sample tail artifact, not a
# real finding; see calibration_analysis.py's reliability diagram).
# Applied to every disease, not just the recalibrated ones, as a general
# safeguard consistent with the Intended Use statement's "risk estimate,
# not a certainty" framing.
_MIN_RISK = 0.01
_MAX_RISK = 0.99


def predict_disease_risk(name: str, feature_dict: dict) -> float:
    if name not in DISEASE_NAMES:
        raise ValueError(f"Unknown disease: {name}. Must be one of {DISEASE_NAMES}")
    model, _, medians, features, _, calibrator = _load(name)
    row = _row(feature_dict, medians, features)
    raw_proba = float(model.predict_proba(row)[0, 1])
    proba = float(calibrator.predict([raw_proba])[0]) if calibrator is not None else raw_proba
    return max(_MIN_RISK, min(_MAX_RISK, proba))


def disease_shap_breakdown(name: str, feature_dict: dict) -> list:
    if name not in DISEASE_NAMES:
        raise ValueError(f"Unknown disease: {name}. Must be one of {DISEASE_NAMES}")
    model, explainer, medians, features, _, _ = _load(name)
    row = _row(feature_dict, medians, features)
    sv = explainer(row)
    out = [
        {"feature": f, "value": float(val), "shap": float(shap_val)}
        for f, val, shap_val in zip(features, row.iloc[0].values, sv.values[0])
    ]
    out.sort(key=lambda d: abs(d["shap"]), reverse=True)
    return out


def get_disease_metrics(name: str) -> dict:
    _, _, _, _, metrics, _ = _load(name)
    return metrics


def predict_all(feature_dict: dict, diabetes_risk: float) -> dict:
    """Combines the existing diabetes prediction (passed in, computed by
    model.py's own ensemble) with all 4 new disease models into one
    unified result for the Metabolic Health dashboard."""
    out = {"diabetes": {"label": DISEASE_LABELS["diabetes"], "risk": diabetes_risk}}
    for name in DISEASE_NAMES:
        out[name] = {"label": DISEASE_LABELS[name], "risk": predict_disease_risk(name, feature_dict)}
    return out
