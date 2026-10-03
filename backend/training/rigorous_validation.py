"""
Metis - Rigorous Validation for Journal Readiness

Addresses two specific gaps identified when assessing publication
readiness against TRIPOD+AI (BMJ 2024) reporting standards:

1. SINGLE TRAIN/TEST SPLIT -> 5-FOLD CROSS-VALIDATION. Every validation
   study so far (validate_models.py, validate_multi_disease.py,
   calibration_analysis.py, fairness_analysis.py) used one 80/20 split.
   A single split's AUC is one point estimate with unknown variance - a
   reviewer will reasonably ask "how much would this number move with a
   different random split?" This script answers that directly: 5-fold
   stratified CV, same model config each fold, reporting mean +/- std
   AUC per disease.

2. UNWEIGHTED -> SURVEY-WEIGHTED ESTIMATES. NHANES is a complex survey
   design (deliberately oversamples some demographic groups) - treating
   it as a simple random sample, as every analysis in this project has
   done so far, is a known simplification that a clinical/epi journal
   reviewer will flag. This script reports BOTH the unweighted AUC
   (comparable to everything already in this project) and a
   weight-adjusted prevalence + weighted AUC using WTSAF2YR (the fasting
   subsample 2-year weight - the correct NHANES-recommended weight for
   any analysis using fasting-subsample variables, which all 5 models
   here include at least one of). Weighted estimates can only be computed
   on rows that actually have a survey weight (the fasting subsample
   itself, ~56% of the full sample) - this is a real constraint, not an
   oversight, and is reported explicitly rather than silently applying
   weights to rows that were never sampled with that weight in mind.

This does NOT change any production model or its predictions - it
re-trains disposable copies of each model under cross-validation purely
to characterize uncertainty, exactly like validate_models.py and
validate_multi_disease.py already do for their own comparison purposes.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import model as diabetes_model  # noqa: E402
import train_multi_disease as tmd  # noqa: E402

OUT_DIR = BASE_DIR / "validation_results" / "rigorous"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_FOLDS = 5


def sample_weight_balance(y):
    """Class-balance weighting (for training), NOT the NHANES survey
    weight - these are two unrelated kinds of 'weight' and this script
    needs both."""
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    return y.map({0: len(y) / (2 * n0), 1: len(y) / (2 * n1)})


def fit_production_config(name, X_train, y_train, features, monotone):
    """Replicates each disease's exact production hyperparameters/
    monotonicity so the CV estimate reflects the actual shipped config,
    not a different model."""
    w = sample_weight_balance(y_train)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
        missing=np.nan, monotone_constraints=tuple(monotone[f] for f in features), random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=w)
    return model


def cross_validate_disease(name, X, y, features, monotone, display_label):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = fit_production_config(name, X_train, y_train, features, monotone)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        fold_aucs.append(auc)
        print(f"    fold {fold_i}/{N_FOLDS}: AUC={auc:.3f}")

    fold_df = pd.DataFrame({"fold": range(1, N_FOLDS + 1), "AUC": fold_aucs})
    fold_df.to_csv(OUT_DIR / f"{name}_cv_folds.csv", index=False)

    return {
        "Disease": display_label,
        "Mean AUC (5-fold CV)": round(np.mean(fold_aucs), 3),
        "Std AUC": round(np.std(fold_aucs), 3),
        "Min AUC": round(np.min(fold_aucs), 3),
        "Max AUC": round(np.max(fold_aucs), 3),
    }


def weighted_analysis(name, df_with_weight, label_col, features, monotone, display_label):
    """Trains on an 80/20 split (same as production), then evaluates
    BOTH unweighted and survey-weighted AUC/prevalence on the held-out
    test portion that actually has a survey weight."""
    data = df_with_weight.dropna(subset=[label_col]).reset_index(drop=True)
    X = data[features]
    y = data[label_col].astype(int)
    w_survey = data["survey_weight"]

    X_train, X_test, y_train, y_test, w_survey_train, w_survey_test = train_test_split(
        X, y, w_survey, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = fit_production_config(name, X_train, y_train, features, monotone)
    proba_test = model.predict_proba(X_test)[:, 1]

    unweighted_prevalence = y_test.mean()
    unweighted_auc = roc_auc_score(y_test, proba_test)

    has_weight = w_survey_test.notna()
    n_weighted = has_weight.sum()
    if n_weighted >= 30 and y_test[has_weight].nunique() == 2:
        weighted_prevalence = np.average(y_test[has_weight], weights=w_survey_test[has_weight])
        weighted_auc = roc_auc_score(y_test[has_weight], proba_test[has_weight], sample_weight=w_survey_test[has_weight])
    else:
        weighted_prevalence, weighted_auc = None, None

    return {
        "Disease": display_label,
        "n_test_total": len(y_test), "n_test_with_weight": int(n_weighted),
        "Unweighted prevalence": round(unweighted_prevalence, 3),
        "Weighted prevalence": round(weighted_prevalence, 3) if weighted_prevalence is not None else "insufficient weighted n",
        "Unweighted AUC": round(unweighted_auc, 3),
        "Weighted AUC": round(weighted_auc, 3) if weighted_auc is not None else "insufficient weighted n",
    }


def main():
    print("Loading data...")
    df = tmd.load_data()

    cv_results, weighted_results = [], []

    print("\n=== Type 2 Diabetes ===")
    features = diabetes_model.FEATURES
    monotone = {f: tmd._BASE_MONOTONE.get(f, 0) for f in features}
    data = df.dropna(subset=["diabetes_label"]).reset_index(drop=True)
    X, y = data[features], data["diabetes_label"].astype(int)
    cv_results.append(cross_validate_disease("diabetes", X, y, features, monotone, "Type 2 Diabetes"))
    weighted_results.append(weighted_analysis("diabetes", df, "diabetes_label", features, monotone, "Type 2 Diabetes"))

    disease_labels = {
        "metabolic_syndrome": "Metabolic Syndrome", "nafld": "NAFLD (Fatty Liver)",
        "ckd": "Chronic Kidney Disease", "hypertension": "Hypertension",
    }
    for name, config in tmd.DISEASES.items():
        print(f"\n=== {disease_labels[name]} ===")
        label_col = config["label"]
        feats = [f for f in tmd._ALL_FEATURES if f not in config["exclude"]]
        monotone = dict(tmd._BASE_MONOTONE)
        monotone.update(config.get("monotone_overrides", {}))
        data = df.dropna(subset=[label_col]).reset_index(drop=True)
        X, y = data[feats], data[label_col].astype(int)
        cv_results.append(cross_validate_disease(name, X, y, feats, monotone, disease_labels[name]))
        weighted_results.append(weighted_analysis(name, df, label_col, feats, monotone, disease_labels[name]))

    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(OUT_DIR / "cv_summary.csv", index=False)
    print("\n=== 5-FOLD CROSS-VALIDATION SUMMARY ===")
    print(cv_df.to_string(index=False))

    weighted_df = pd.DataFrame(weighted_results)
    weighted_df.to_csv(OUT_DIR / "weighted_summary.csv", index=False)
    print("\n=== SURVEY-WEIGHTED VS UNWEIGHTED SUMMARY ===")
    print(weighted_df.to_string(index=False))

    print(f"\nSaved all outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
