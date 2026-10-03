"""
Metis - Multi-Disease Model Training

Trains FOUR additional disease-risk models (Metabolic Syndrome, NAFLD,
Chronic Kidney Disease, Hypertension), alongside the existing diabetes
ensemble (train_model.py). Each disease gets its OWN non-circular
predictor set - the same discipline applied to diabetes throughout this
project, just applied four more times:

  - Metabolic Syndrome: excludes waist/waist-height-ratio, triglycerides,
    HDL, BP, pulse pressure, fasting glucose, insulin/HOMA-IR - all of
    these are literally the label's own defining criteria or tightly
    glucose-coupled.
  - NAFLD: excludes liver_fat_cap and liver_stiffness (the label itself).
    ALT/AST/GGT are KEPT as predictors - they're correlated liver-stress
    markers, not diagnostic criteria (many NAFLD cases have normal
    enzymes; NAFLD is defined by imaging/elastography finding fat in the
    liver, not by enzyme levels).
  - CKD: excludes creatinine, BUN, eGFR (the label itself).
  - Hypertension: excludes bp_systolic, bp_diastolic, pulse_pressure (the
    label itself).

Uses single monotonic XGBoost models (not the stacked ensemble the
diabetes model uses) - the earlier validation study found ensembling
gives minimal uplift for this kind of tabular clinical data, so a single
well-tuned monotonic model is the pragmatic choice for four new models at
once. fasting_glucose, hba1c, sugar_g, and drinks_per_day stay excluded
from EVERY disease's predictor set, for the same reasons documented in
model.py (diagnostic circularity for glucose/HbA1c; reverse-causation
confounding for sugar/alcohol self-report, found via marginal-effect
sweeps on the diabetes model and assumed to generalize).
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "nhanes_2021_2023_merged.csv"
MODEL_DIR = BASE_DIR.parent / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Full available feature pool. fasting_glucose/hba1c/sugar_g/drinks_per_day
# are never in this pool - excluded project-wide, not per-disease.
# ---------------------------------------------------------------------------
_ALL_FEATURES = [
    "age", "bmi", "waist_cm", "waist_height_ratio",
    "bp_systolic", "bp_diastolic", "pulse_pressure",
    "hdl", "total_cholesterol", "ldl", "triglycerides",
    "crp", "insulin", "homa_ir",
    "liver_fat_cap", "liver_stiffness", "smoker_current",
    "sleep_hours", "activity_min_per_week", "sedentary_min_per_day",
    "kcal", "fiber_g", "occ_activity_level", "phq9_score",
    "poverty_ratio", "gender_male",
    "alt", "ast", "ggt", "uric_acid", "albumin", "wbc",
    "creatinine", "bun", "egfr",
]

_BASE_MONOTONE = {
    "age": 1, "bmi": 1, "waist_cm": 1, "waist_height_ratio": 1,
    "bp_systolic": 1, "bp_diastolic": 1, "pulse_pressure": 1,
    "hdl": -1, "total_cholesterol": 0, "ldl": 1, "triglycerides": 1,
    "crp": 1, "insulin": 1, "homa_ir": 1,
    "liver_fat_cap": 1, "liver_stiffness": 1, "smoker_current": 1,
    "sleep_hours": 0, "activity_min_per_week": -1, "sedentary_min_per_day": 1,
    "kcal": 0, "fiber_g": -1, "occ_activity_level": -1, "phq9_score": 1,
    "poverty_ratio": -1, "gender_male": 0,
    "alt": 1, "ast": 1, "ggt": 1, "uric_acid": 1, "albumin": -1, "wbc": 1,
    "creatinine": 1, "bun": 1, "egfr": -1,  # higher eGFR = better kidney function = lower risk
}

DISEASES = {
    "metabolic_syndrome": {
        "label": "metabolic_syndrome_label",
        "exclude": {"waist_cm", "waist_height_ratio", "triglycerides", "hdl",
                    "bp_systolic", "bp_diastolic", "pulse_pressure",
                    "insulin", "homa_ir"},
    },
    "nafld": {
        "label": "nafld_label",
        "exclude": {"liver_fat_cap", "liver_stiffness"},
    },
    "ckd": {
        "label": "ckd_label",
        "exclude": {"creatinine", "bun", "egfr"},
        # ALT/AST/GGT have no clear established directional relationship
        # with kidney function specifically - left unconstrained here even
        # though they're constrained for other diseases.
        # sleep_hours: CKD's small positive class (~9% prevalence) made the
        # unconstrained sleep dimension pick up noise during simulator
        # testing - a sleep IMPROVEMENT showed CKD risk increasing for a
        # test patient, the same class of bug found and fixed for
        # diabetes earlier in this project. Published literature does
        # support a genuine U-shape for sleep and CKD risk (like diabetes),
        # but the causal-sanity guarantee matters more than modeling that
        # nuance correctly here - constrained to -1 (more sleep never
        # increases predicted risk) rather than risk a backwards lever.
        "monotone_overrides": {"alt": 0, "ast": 0, "ggt": 0, "sleep_hours": -1},
    },
    "hypertension": {
        "label": "hypertension_label",
        "exclude": {"bp_systolic", "bp_diastolic", "pulse_pressure"},
        "monotone_overrides": {"alt": 0, "ast": 0, "ggt": 0},
    },
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["gender_male"] = (df["gender"] == "Male").astype(int)
    return df


def sample_weight(y):
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    return y.map({0: len(y) / (2 * n0), 1: len(y) / (2 * n1)})


def train_disease(name, config, df):
    label_col = config["label"]
    features = [f for f in _ALL_FEATURES if f not in config["exclude"]]
    monotone = dict(_BASE_MONOTONE)
    monotone.update(config.get("monotone_overrides", {}))

    data = df.dropna(subset=[label_col]).reset_index(drop=True)
    X = data[features]
    y = data[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    w_train = sample_weight(y_train)
    medians = X_train.median()

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        missing=np.nan, monotone_constraints=tuple(monotone[f] for f in features),
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    report = classification_report(y_test, (proba >= 0.5).astype(int), output_dict=True)

    print(f"\n=== {name} ===")
    print(f"  n={len(data)} (train={len(X_train)}, test={len(X_test)}), "
          f"prevalence={y.mean():.3f}, features={len(features)}")
    print(f"  Test ROC-AUC: {auc:.3f}   PR-AUC: {ap:.3f}")

    explainer = shap.TreeExplainer(model)

    with open(MODEL_DIR / f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / f"{name}_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    with open(MODEL_DIR / f"{name}_medians.pkl", "wb") as f:
        pickle.dump(medians, f)

    metrics = {
        "roc_auc": auc, "pr_auc": ap,
        "n_train": len(X_train), "n_test": len(X_test),
        "prevalence": float(y.mean()),
        "features": features,
        "classification_report": report,
    }
    with open(MODEL_DIR / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, features, medians


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows")

    results = {}
    for name, config in DISEASES.items():
        results[name] = train_disease(name, config, df)

    print(f"\nSaved all 4 disease models to {MODEL_DIR}")
    return results


if __name__ == "__main__":
    main()
