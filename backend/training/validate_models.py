"""
Metis - Model Validation & Comparison Study

Trains 6 different model families on the EXACT same train/test split and
feature set as the production model (train_model.py), to sanity-check
that XGBoost's performance is genuinely competitive rather than a fluke
of one algorithm, and to visualize where the models agree/disagree.

Models compared:
  1. Logistic Regression   - linear baseline, most interpretable
  2. Random Forest          - bagged trees, robust to outliers
  3. Gradient Boosting      - sklearn's boosted trees (no monotonicity)
  4. HistGradientBoosting   - modern boosted trees, native NaN handling
  4b. HistGB (monotonic)     - same as above, but with the SAME
                              monotonicity constraints as production
                              XGBoost - tests whether HistGB's edge in raw
                              discrimination survives once it's held to the
                              same causal-sanity requirement
  5. Support Vector Machine - kernel method, needs scaled features
  6. XGBoost (production)   - same config as train_model.py, incl. the
                              monotonicity constraints that make the
                              What-If simulator causally sane

Fairness notes:
  - All 6 models see the identical train/test split (random_state=42,
    stratify=y, test_size=0.2) - same rows in train, same rows in test.
  - Class imbalance is handled the same way for all 6 (sample_weight
    computed from the training class balance), not left to each model's
    own default.
  - Models that can't handle missing values natively (Logistic Regression,
    Random Forest, Gradient Boosting, SVM) get the SAME median-imputed
    features used elsewhere in this app. XGBoost and HistGradientBoosting
    use the raw NaN-preserving features, since handling that missingness
    natively is one of the reasons those two algorithms were chosen for
    production in the first place - this comparison shows whether that
    choice actually paid off, not just how each model does on a
    complete-case subset.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "nhanes_2021_2023_merged.csv"
OUT_DIR = BASE_DIR / "validation_results"
OUT_DIR.mkdir(exist_ok=True)

# Same feature set and exclusions as train_model.py - see that file's
# docstring for why fasting_glucose/sugar_g/drinks_per_day are excluded.
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
]
TARGET = "diabetes_label"

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
}

PALETTE = {
    "Logistic Regression": "#7440B0",
    "Random Forest": "#2E8B7D",
    "Gradient Boosting": "#E8A33D",
    "HistGradientBoosting": "#4C2478",
    "HistGB (monotonic)": "#3A8FB7",
    "SVM": "#C4577B",
    "XGBoost (production)": "#C9A227",
}
MODEL_ORDER = list(PALETTE.keys())


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["gender_male"] = (df["gender"] == "Male").astype(int)
    df = df.dropna(subset=[TARGET])
    return df


def sample_weights(y):
    """Balanced sample weights: same imbalance handling for every model,
    rather than leaving it to each model's own (inconsistent) defaults."""
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = len(y) / (len(classes) * counts)
    weight_map = dict(zip(classes, weight_per_class))
    return y.map(weight_map).values


def build_models_and_predictions(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    w_train = sample_weights(y_train)

    # Median-imputed version (from TRAIN medians only, no leakage) for
    # models that can't handle NaN natively.
    medians = X_train.median()
    X_train_imp = X_train.fillna(medians)
    X_test_imp = X_test.fillna(medians)

    scaler = StandardScaler().fit(X_train_imp)
    X_train_scaled = scaler.transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    results = {}  # name -> {"proba": array, "model": fitted estimator, "uses": "imputed"/"native"/"scaled"}

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train_scaled, y_train, sample_weight=w_train)
    results["Logistic Regression"] = {
        "proba": lr.predict_proba(X_test_scaled)[:, 1], "model": lr, "kind": "linear",
    }

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train_imp, y_train, sample_weight=w_train)
    results["Random Forest"] = {
        "proba": rf.predict_proba(X_test_imp)[:, 1], "model": rf, "kind": "tree",
    }

    # 3. Gradient Boosting (sklearn's, no monotonicity constraints)
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    gb.fit(X_train_imp, y_train, sample_weight=w_train)
    results["Gradient Boosting"] = {
        "proba": gb.predict_proba(X_test_imp)[:, 1], "model": gb, "kind": "tree",
    }

    # 4. HistGradientBoosting (modern, native NaN handling - no imputation)
    hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42)
    hgb.fit(X_train, y_train, sample_weight=w_train)
    results["HistGradientBoosting"] = {
        "proba": hgb.predict_proba(X_test)[:, 1], "model": hgb, "kind": "tree_native",
    }

    # 4b. HistGradientBoosting WITH the same monotonicity constraints as
    # production XGBoost - tests whether HistGB's edge in raw discrimination
    # (see plain HistGradientBoosting above) survives once it's held to the
    # same causal-sanity requirement that drove XGBoost's monotone_constraints
    # in the first place (see train_model.py's docstring). sklearn >=1.4
    # supports monotonic_cst on this estimator.
    mono_cst = [_MONOTONE[f] for f in FEATURES]
    hgb_mono = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05, random_state=42, monotonic_cst=mono_cst,
    )
    hgb_mono.fit(X_train, y_train, sample_weight=w_train)
    results["HistGB (monotonic)"] = {
        "proba": hgb_mono.predict_proba(X_test)[:, 1], "model": hgb_mono, "kind": "tree_native",
    }

    # 5. SVM (kernel method, needs scaled features; probability=True for ROC/PR)
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train, sample_weight=w_train)
    results["SVM"] = {
        "proba": svm.predict_proba(X_test_scaled)[:, 1], "model": svm, "kind": "kernel",
    }

    # 6. XGBoost - the actual production model (monotonicity constraints included)
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        missing=np.nan, monotone_constraints=tuple(_MONOTONE[f] for f in FEATURES),
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, sample_weight=w_train)
    results["XGBoost (production)"] = {
        "proba": xgb_model.predict_proba(X_test)[:, 1], "model": xgb_model, "kind": "tree_native",
    }

    return results, y_test, X_test, X_test_imp


def compute_metrics(results, y_test):
    rows = []
    for name, r in results.items():
        proba = r["proba"]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, proba),
            "PR-AUC": average_precision_score(y_test, proba),
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "Brier Score (lower=better)": brier_score_loss(y_test, proba),
        })
    return pd.DataFrame(rows).set_index("Model").loc[MODEL_ORDER]


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for name in MODEL_ORDER:
        proba = results[name]["proba"]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=PALETTE[name], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Random chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — 7-Model Comparison")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_pr_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    prevalence = y_test.mean()
    for name in MODEL_ORDER:
        proba = results[name]["proba"]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=PALETTE[name], linewidth=2)
    ax.axhline(prevalence, linestyle="--", color="#999999", label=f"Baseline (prevalence={prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — 7-Model Comparison")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for ax, name in zip(axes.flat, MODEL_ORDER):
        proba = results[name]["proba"]
        pred = (proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Purples", ax=ax, cbar=False,
            xticklabels=["No diabetes", "Diabetes"], yticklabels=["No diabetes", "Diabetes"],
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes.flat[len(MODEL_ORDER):]:
        ax.axis("off")
    fig.suptitle("Confusion Matrices at 0.5 threshold — 7-Model Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_metrics_heatmap(metrics_df):
    display_df = metrics_df.copy()
    # Invert Brier score for color purposes (lower is better) by using a
    # separate color scale direction, handled via two heatmaps side by side
    # would be complex - simplest clear approach: annotate real values,
    # color by min-max normalized "goodness" per column.
    goodness = display_df.copy()
    for col in goodness.columns:
        if "Brier" in col:
            goodness[col] = 1 - (goodness[col] - goodness[col].min()) / (goodness[col].max() - goodness[col].min() + 1e-9)
        else:
            goodness[col] = (goodness[col] - goodness[col].min()) / (goodness[col].max() - goodness[col].min() + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(
        goodness, annot=display_df.round(3), fmt="", cmap="RdYlGn", ax=ax,
        cbar_kws={"label": "Relative performance (green=better)"}, linewidths=0.5, linecolor="white",
    )
    ax.set_title("Metrics Comparison Heatmap — 7 Models")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "metrics_heatmap.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(results):
    # Tree models expose feature_importances_; Logistic Regression exposes
    # coefficients (absolute value, standardized scale so comparable).
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    tree_models = ["Random Forest", "Gradient Boosting", "XGBoost (production)"]
    for ax, name in zip(axes.flat[:3], tree_models):
        importances = results[name]["model"].feature_importances_
        order = np.argsort(importances)[-12:]
        ax.barh(np.array(FEATURES)[order], importances[order], color=PALETTE[name])
        ax.set_title(f"{name} — top 12 features")
        ax.set_xlabel("Importance")

    lr_model = results["Logistic Regression"]["model"]
    coefs = np.abs(lr_model.coef_[0])
    order = np.argsort(coefs)[-12:]
    axes.flat[3].barh(np.array(FEATURES)[order], coefs[order], color=PALETTE["Logistic Regression"])
    axes.flat[3].set_title("Logistic Regression — top 12 |coefficients| (standardized)")
    axes.flat[3].set_xlabel("|Coefficient|")

    fig.suptitle("Feature Importance Comparison Across Model Families", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_calibration_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for name in MODEL_ORDER:
        proba = results[name]["proba"]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name, color=PALETTE[name], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of diabetes")
    ax.set_title("Calibration (Reliability) Curves — 7-Model Comparison")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_curves.png", dpi=150)
    plt.close(fig)


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows, {df[TARGET].mean()*100:.1f}% positive class")

    print("Training 7 models on the identical train/test split...")
    results, y_test, X_test, X_test_imp = build_models_and_predictions(df)

    print("Computing metrics...")
    metrics_df = compute_metrics(results, y_test)
    metrics_df.to_csv(OUT_DIR / "metrics_comparison.csv")
    print(metrics_df.round(3).to_string())

    print("Generating plots...")
    plot_roc_curves(results, y_test)
    plot_pr_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    plot_metrics_heatmap(metrics_df)
    plot_feature_importance(results)
    plot_calibration_curves(results, y_test)

    summary = {
        "n_test": len(y_test),
        "test_prevalence": float(y_test.mean()),
        "best_roc_auc_model": metrics_df["ROC-AUC"].idxmax(),
        "best_roc_auc": float(metrics_df["ROC-AUC"].max()),
        "metrics": metrics_df.round(4).to_dict(orient="index"),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved all outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
