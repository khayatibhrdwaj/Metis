"""
Metis Backend - Model wrapper

Loads the trained STACKED ENSEMBLE (XGBoost + HistGradientBoosting +
Logistic Regression, combined by a meta-learner) and exposes
predict_risk() / shap_breakdown() for the API layer (main.py).

predict_risk() uses the FULL ensemble (all 3 base models -> meta-learner).
shap_breakdown() explains using ONLY the XGBoost component's exact SHAP
values - see training/train_model.py's module docstring for why this
simplification was made and what it means in practice.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"

# Must exactly match the feature list the model was trained on
# (training/train_model.py).
#
# NOTE: fasting_glucose is deliberately EXCLUDED as a predictor. Along with
# the diagnosed-diabetes questionnaire, it's what the label is defined from,
# so including it would let the model "cheat" by looking up the diagnosis
# rather than predicting risk.
#
# hba1c is EXCLUDED for the identical reason - it's the other standard ADA
# diagnostic criterion for diabetes (>=6.5%) and is also used to help
# define the label (see data_prep_2021_2023.py). Both fasting_glucose and
# hba1c are tracked on the dashboard and forecast by the What-If simulator
# as OUTCOMES (see simulator.py) - never as inputs to this risk model.
#
# sugar_g and drinks_per_day are ALSO excluded as predictors, for a
# different but related reason: marginal-effect testing showed both have a
# REVERSED relationship with risk in this cross-sectional data (lower
# self-reported sugar/alcohol intake predicts HIGHER modeled risk). This is
# reverse causation, not a real protective effect of eating more sugar -
# people already diagnosed with diabetes disproportionately report having
# already cut sugar/alcohol as part of managing the diagnosis. They remain
# as simulator-only levers (see simulator.py) whose effects flow through
# legitimate downstream biomarkers (triglycerides, cholesterol, liver fat)
# instead of through the confounded raw self-report field.
#
# insulin, triglycerides, ldl, homa_ir, and the 6 biomarkers below come from
# partial subsamples (~10-50% missing by design, not data quality). The
# tree-based components handle this natively; the Logistic Regression
# component gets a median-imputed version internally.
FEATURES = [
    "age", "bmi", "waist_cm", "waist_height_ratio",
    "bp_systolic", "bp_diastolic", "pulse_pressure",
    "hdl", "total_cholesterol", "ldl", "triglycerides",
    "crp", "insulin", "homa_ir",
    "liver_fat_cap", "liver_stiffness", "smoker_current",
    "sleep_hours", "activity_min_per_week", "sedentary_min_per_day",
    "kcal", "fiber_g",
    "occ_activity_level", "phq9_score",
    "poverty_ratio", "gender_male",
    "alt", "ast", "ggt", "uric_acid", "albumin", "wbc",
]

_xgb_model = None
_hgb_model = None
_lr_model = None
_lr_scaler = None
_meta_model = None
_explainer = None
_medians = None
_metrics = None


def _load():
    global _xgb_model, _hgb_model, _lr_model, _lr_scaler, _meta_model, _explainer, _medians, _metrics
    if _xgb_model is None:
        with open(MODEL_DIR / "risk_model.pkl", "rb") as f:
            _xgb_model = pickle.load(f)
        with open(MODEL_DIR / "hgb_model.pkl", "rb") as f:
            _hgb_model = pickle.load(f)
        with open(MODEL_DIR / "lr_model.pkl", "rb") as f:
            _lr_model = pickle.load(f)
        with open(MODEL_DIR / "lr_scaler.pkl", "rb") as f:
            _lr_scaler = pickle.load(f)
        with open(MODEL_DIR / "meta_model.pkl", "rb") as f:
            _meta_model = pickle.load(f)
        with open(MODEL_DIR / "explainer.pkl", "rb") as f:
            _explainer = pickle.load(f)
        with open(MODEL_DIR / "medians.pkl", "rb") as f:
            _medians = pickle.load(f)
        with open(MODEL_DIR / "metrics.json") as f:
            _metrics = json.load(f)
    return _xgb_model, _hgb_model, _lr_model, _lr_scaler, _meta_model, _explainer, _medians, _metrics


def _row(feature_dict: dict, medians) -> pd.DataFrame:
    return pd.DataFrame([{f: feature_dict.get(f, medians[f]) for f in FEATURES}])


# No statistical model trained on finite data should ever output an
# estimate indistinguishable from mathematical certainty - same safety
# clip applied in diseases.py (see that file's comment for how this was
# actually caught: an isotonic calibration tail producing an exact 1.0
# for one of the other 4 diseases). The ensemble's sigmoid output rarely
# hits exactly 0/1, but this makes that guarantee explicit rather than
# incidental.
_MIN_RISK = 0.01
_MAX_RISK = 0.99


def predict_risk(feature_dict: dict) -> float:
    """Full stacked-ensemble prediction: all 3 base models -> meta-learner."""
    xgb_model, hgb_model, lr_model, lr_scaler, meta_model, _, medians, _ = _load()
    row = _row(feature_dict, medians)

    p_xgb = xgb_model.predict_proba(row)[0, 1]
    p_hgb = hgb_model.predict_proba(row)[0, 1]
    row_imputed = row.fillna(medians)  # LR can't handle NaN natively
    p_lr = lr_model.predict_proba(lr_scaler.transform(row_imputed))[0, 1]

    meta_input = np.array([[p_xgb, p_hgb, p_lr]])
    proba = float(meta_model.predict_proba(meta_input)[0, 1])
    return max(_MIN_RISK, min(_MAX_RISK, proba))


def shap_breakdown(feature_dict: dict) -> list:
    """Explains the risk using the XGBoost component's exact SHAP values
    only - see training/train_model.py's docstring for why."""
    _, _, _, _, _, explainer, medians, _ = _load()
    row = _row(feature_dict, medians)
    sv = explainer(row)
    out = [
        {"feature": f, "value": float(val), "shap": float(shap_val)}
        for f, val, shap_val in zip(FEATURES, row.iloc[0].values, sv.values[0])
    ]
    out.sort(key=lambda d: abs(d["shap"]), reverse=True)
    return out


def get_metrics() -> dict:
    _, _, _, _, _, _, _, metrics = _load()
    return metrics


def get_medians() -> dict:
    _, _, _, _, _, _, medians, _ = _load()
    return {k: float(v) for k, v in medians.items()}
