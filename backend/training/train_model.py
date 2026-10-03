"""
MetaTwin - Risk Model Training (NHANES 2021-2023)

Trains a STACKED ENSEMBLE for diabetes risk: three base learners
(monotonic XGBoost, monotonic HistGradientBoosting, Logistic Regression)
combined by a meta-learner (Logistic Regression on the three base
probabilities), fit via proper out-of-fold stacking to avoid leakage.

Why stacking: shipped after testing repeatedly showed no single model or
naive-averaging ensemble legitimately clears ~0.87 AUC within the
monotonicity constraint (see the ceiling note below) - out-of-fold
stacking with a learned meta-learner gave a small but real additional
gain (0.857 solo XGBoost -> 0.860 stacked) at the cost of real serving
complexity (three models instead of one). Shipped anyway per explicit
request; the tradeoff is documented here rather than hidden.

SHAP explainability: the "Risk & Drivers" / "AI Explanation" tabs still
explain the risk score using ONLY the XGBoost component's exact SHAP
values (TreeExplainer), not a blended cross-model explanation. This is a
deliberate simplification - a mathematically exact Shapley decomposition
of a 3-model stack combined through a sigmoid meta-learner is a much
harder problem than explaining one tree model, and XGBoost is one of the
two co-dominant components in the meta-learner's weights (see printed
weights at training time). The displayed risk PERCENTAGE comes from the
full ensemble; the displayed EXPLANATION comes from XGBoost's view of it.
These are very likely to point in the same direction (all three base
models are trained on the same monotonic-consistent features) but are not
mathematically guaranteed to agree in every edge case. Documented here so
this simplification is a visible decision, not a silent one.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "nhanes_2021_2023_merged.csv"
MODEL_DIR = BASE_DIR.parent / "models"  # backend/models - consumed by the live API
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# NOTE: fasting_glucose and hba1c are deliberately EXCLUDED as predictors.
# Along with the diagnosed-diabetes questionnaire, both are what the label
# is defined from (see data_prep_2021_2023.py), so including either would
# let the model "cheat" by looking up the diagnosis rather than predicting
# risk. This mirrors real risk tools (ADA risk test, FINDRISC): predict
# from anthropometric/BP/lipid/inflammation/lifestyle risk factors; glucose
# and HbA1c are instead OUTCOMES the What-If simulator forecasts.
#
# sugar_g and drinks_per_day are ALSO excluded, for a related but distinct
# reason found via marginal-effect sweeps (holding all else fixed and
# varying one feature across its range): both showed a REVERSED
# relationship with predicted risk - lower self-reported sugar/alcohol
# intake predicted HIGHER risk. That's reverse causation from NHANES being
# cross-sectional: people already diagnosed with diabetes disproportionately
# report having already cut sugar/alcohol as part of managing the
# diagnosis. Both remain as simulator-only levers (simulator.py) whose
# effects flow through legitimate downstream biomarkers instead.
#
# total_cholesterol shows a milder version of the same issue at unusually
# low values (likely reflecting statin use in already-diagnosed patients).
# Kept in - real, mostly-sane, high-signal biomarker outside that low tail.
#
# insulin, triglycerides, ldl, homa_ir, and the 6 biomarkers below come
# from partial subsamples (~10-50% missing by design, not data quality).
# XGBoost/HistGB handle this natively; Logistic Regression gets a median-
# imputed version (see NATIVE_MISSING_OK below).
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
TARGET = "diabetes_label"

# Model ceiling note (full history): hyperparameter tuning, engineered
# interaction features, and naive-averaging ensembles all plateaued around
# 0.85-0.87 AUC within the monotonicity constraint. Adding 6 legitimately
# unused biomarkers (liver enzymes, uric acid, albumin, WBC - checked the
# biochemistry/CBC panels for real independent signal, not diagnostic
# criteria) gave a genuine gain to 0.857. Proper out-of-fold stacking
# (this file) adds another small real gain to ~0.860. 0.9+ is NOT reachable
# from non-diagnostic risk factors on this dataset without either
# reintroducing the fasting_glucose/hba1c circularity or overfitting - see
# README.md's "Model notes" for the full experimental history. What WOULD
# close the gap: genetic/family-history data (not in this NHANES extract)
# or longitudinal repeat-measures data (NHANES is cross-sectional) - both
# are data-availability problems, not modeling problems.
_MONOTONE = {
    "age": 1, "bmi": 1, "waist_cm": 1, "waist_height_ratio": 1,
    "bp_systolic": 1, "bp_diastolic": 1, "pulse_pressure": 1,
    "hdl": -1, "total_cholesterol": 0, "ldl": 1, "triglycerides": 1,
    "crp": 1, "insulin": 1, "homa_ir": 1,
    "liver_fat_cap": 1, "liver_stiffness": 1, "smoker_current": 1,
    "sleep_hours": 0, "activity_min_per_week": -1, "sedentary_min_per_day": 1,
    "kcal": 0, "fiber_g": -1,
    "occ_activity_level": -1, "phq9_score": 1,
    "poverty_ratio": -1, "gender_male": 0,
    "alt": 1, "ast": 1, "ggt": 1, "uric_acid": 1, "albumin": -1, "wbc": 1,
}

NATIVE_MISSING_OK = {"insulin", "homa_ir", "ldl", "triglycerides",
                      "liver_fat_cap", "liver_stiffness",
                      "occ_activity_level", "phq9_score",
                      "alt", "ast", "ggt", "uric_acid", "albumin", "wbc"}

N_FOLDS = 5
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["gender_male"] = (df["gender"] == "Male").astype(int)
    df = df.dropna(subset=[TARGET])
    return df


def _sample_weight(y):
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    return y.map({0: len(y) / (2 * n0), 1: len(y) / (2 * n1)})


def _fit_xgb(X, y, w):
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        missing=np.nan, monotone_constraints=tuple(_MONOTONE[f] for f in FEATURES),
        random_state=RANDOM_STATE,
    )
    m.fit(X, y, sample_weight=w)
    return m


def _fit_hgb(X, y, w):
    m = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        monotonic_cst=[_MONOTONE[f] for f in FEATURES], random_state=RANDOM_STATE,
    )
    m.fit(X, y, sample_weight=w)
    return m


def _fit_lr(X, y, w, medians):
    X_imp = X.fillna(medians)
    scaler = StandardScaler().fit(X_imp)
    m = LogisticRegression(max_iter=2000)
    m.fit(scaler.transform(X_imp), y, sample_weight=w)
    return m, scaler


def train():
    df = load_data()

    impute_cols = [c for c in FEATURES if c not in NATIVE_MISSING_OK]
    medians = df[FEATURES].median()
    df[impute_cols] = df[impute_cols].fillna(medians[impute_cols])

    X = df[FEATURES].reset_index(drop=True)
    y = df[TARGET].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # --- Stage 1: out-of-fold predictions on the TRAIN split only, to fit
    # the meta-learner without leakage (each fold's OOF prediction comes
    # from a model that never saw that fold during training) ---
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_xgb = np.zeros(len(X_train))
    oof_hgb = np.zeros(len(X_train))
    oof_lr = np.zeros(len(X_train))

    print(f"Fitting {N_FOLDS}-fold out-of-fold stack...")
    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        wtr = _sample_weight(ytr)

        m_xgb = _fit_xgb(Xtr, ytr, wtr)
        oof_xgb[val_idx] = m_xgb.predict_proba(Xval)[:, 1]

        m_hgb = _fit_hgb(Xtr, ytr, wtr)
        oof_hgb[val_idx] = m_hgb.predict_proba(Xval)[:, 1]

        fold_medians = Xtr.median()
        m_lr, scaler = _fit_lr(Xtr, ytr, wtr, fold_medians)
        oof_lr[val_idx] = m_lr.predict_proba(scaler.transform(Xval.fillna(fold_medians)))[:, 1]
        print(f"  fold {fold_i}/{N_FOLDS} done")

    # --- Stage 2: fit the meta-learner on OOF predictions (no leakage) ---
    meta_X_train = np.column_stack([oof_xgb, oof_hgb, oof_lr])
    meta_model = LogisticRegression()
    meta_model.fit(meta_X_train, y_train)
    print("Meta-learner weights (xgb, hgb, lr):", meta_model.coef_[0], "intercept:", meta_model.intercept_)

    # --- Stage 3: refit each base learner on the FULL train split (standard
    # stacking practice - OOF models were only for meta-learner training
    # data; the deployed base models should use all available train data) ---
    w_train_full = _sample_weight(y_train)
    final_xgb = _fit_xgb(X_train, y_train, w_train_full)
    final_hgb = _fit_hgb(X_train, y_train, w_train_full)
    final_lr, final_scaler = _fit_lr(X_train, y_train, w_train_full, medians)

    # --- Evaluate the full stacked ensemble on the untouched held-out test split ---
    test_p_xgb = final_xgb.predict_proba(X_test)[:, 1]
    test_p_hgb = final_hgb.predict_proba(X_test)[:, 1]
    test_p_lr = final_lr.predict_proba(final_scaler.transform(X_test.fillna(medians)))[:, 1]
    meta_X_test = np.column_stack([test_p_xgb, test_p_hgb, test_p_lr])
    test_proba = meta_model.predict_proba(meta_X_test)[:, 1]

    auc = roc_auc_score(y_test, test_proba)
    ap = average_precision_score(y_test, test_proba)
    report = classification_report(y_test, (test_proba >= 0.5).astype(int), output_dict=True)

    print(f"\nStacked ensemble test ROC-AUC: {auc:.3f}")
    print(f"Stacked ensemble test PR-AUC:  {ap:.3f}")
    print(classification_report(y_test, (test_proba >= 0.5).astype(int)))
    print(f"(solo XGBoost alone: {roc_auc_score(y_test, test_p_xgb):.3f} for comparison)")

    # SHAP explainer stays on the XGBoost component only - see module
    # docstring for why. Exact, fast, and already fully tested.
    explainer = shap.TreeExplainer(final_xgb)

    with open(MODEL_DIR / "risk_model.pkl", "wb") as f:
        pickle.dump(final_xgb, f)
    with open(MODEL_DIR / "hgb_model.pkl", "wb") as f:
        pickle.dump(final_hgb, f)
    with open(MODEL_DIR / "lr_model.pkl", "wb") as f:
        pickle.dump(final_lr, f)
    with open(MODEL_DIR / "lr_scaler.pkl", "wb") as f:
        pickle.dump(final_scaler, f)
    with open(MODEL_DIR / "meta_model.pkl", "wb") as f:
        pickle.dump(meta_model, f)
    with open(MODEL_DIR / "explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    with open(MODEL_DIR / "medians.pkl", "wb") as f:
        pickle.dump(medians, f)

    metrics = {
        "roc_auc": auc,
        "pr_auc": ap,
        "solo_xgb_roc_auc": float(roc_auc_score(y_test, test_p_xgb)),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": FEATURES,
        "native_missing_ok": sorted(NATIVE_MISSING_OK),
        "classification_report": report,
        "data_cycle": "NHANES August 2021-August 2023",
        "architecture": "stacked ensemble (XGBoost + HistGradientBoosting + Logistic Regression, "
                         "combined by an out-of-fold-trained meta-learner). SHAP explanations use "
                         "only the XGBoost component - see train_model.py docstring.",
        "meta_learner_weights": {"xgb": float(meta_model.coef_[0][0]),
                                  "hgb": float(meta_model.coef_[0][1]),
                                  "lr": float(meta_model.coef_[0][2])},
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved all artifacts to {MODEL_DIR}")
    return final_xgb, final_hgb, final_lr, final_scaler, meta_model, explainer, metrics


if __name__ == "__main__":
    train()
