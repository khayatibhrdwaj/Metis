"""
Metis - Multi-Disease Model Validation Study

Extends validate_models.py's methodology to the 4 newer disease models
(Metabolic Syndrome, NAFLD, CKD, Hypertension): trains 5 model families
per disease on the IDENTICAL train/test split and feature set, to check
whether each disease's production XGBoost model is genuinely competitive
with alternatives, the same question validate_models.py answered for
diabetes.

5 models per disease (not 7 - SVM and the "monotonic HistGB vs
unconstrained" bonus comparison from the diabetes study are skipped here
for runtime/scope reasons, not because they're uninteresting):
  Logistic Regression, Random Forest, Gradient Boosting,
  HistGradientBoosting, XGBoost (monotonic, production config)

Reuses the exact feature sets and monotonicity constraints defined in
train_multi_disease.py (DISEASES dict) - if those change, this stays
in sync automatically.
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import train_multi_disease as tmd  # reuses DISEASES, _ALL_FEATURES, _BASE_MONOTONE

BASE_DIR = Path(__file__).resolve().parent
OUT_ROOT = BASE_DIR / "validation_results"

PALETTE = {
    "Logistic Regression": "#7440B0",
    "Random Forest": "#2E8B7D",
    "Gradient Boosting": "#E8A33D",
    "HistGradientBoosting": "#4C2478",
    "XGBoost (production)": "#C9A227",
}
MODEL_ORDER = list(PALETTE.keys())
RANDOM_STATE = 42


def sample_weight(y):
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    return y.map({0: len(y) / (2 * n0), 1: len(y) / (2 * n1)})


def build_and_evaluate(disease_name, config, df):
    label_col = config["label"]
    features = [f for f in tmd._ALL_FEATURES if f not in config["exclude"]]
    monotone = dict(tmd._BASE_MONOTONE)
    monotone.update(config.get("monotone_overrides", {}))

    data = df.dropna(subset=[label_col]).reset_index(drop=True)
    X = data[features]
    y = data[label_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    w_train = sample_weight(y_train)
    medians = X_train.median()
    X_train_imp = X_train.fillna(medians)
    X_test_imp = X_test.fillna(medians)
    scaler = StandardScaler().fit(X_train_imp)
    X_train_scaled = scaler.transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    results = {}

    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train_scaled, y_train, sample_weight=w_train)
    results["Logistic Regression"] = {"proba": lr.predict_proba(X_test_scaled)[:, 1], "model": lr}

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_imp, y_train, sample_weight=w_train)
    results["Random Forest"] = {"proba": rf.predict_proba(X_test_imp)[:, 1], "model": rf}

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE)
    gb.fit(X_train_imp, y_train, sample_weight=w_train)
    results["Gradient Boosting"] = {"proba": gb.predict_proba(X_test_imp)[:, 1], "model": gb}

    hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE)
    hgb.fit(X_train, y_train, sample_weight=w_train)
    results["HistGradientBoosting"] = {"proba": hgb.predict_proba(X_test)[:, 1], "model": hgb}

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
        missing=np.nan, monotone_constraints=tuple(monotone[f] for f in features), random_state=RANDOM_STATE,
    )
    xgb_model.fit(X_train, y_train, sample_weight=w_train)
    results["XGBoost (production)"] = {"proba": xgb_model.predict_proba(X_test)[:, 1], "model": xgb_model}

    return results, y_test, features


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
            "Brier (lower=better)": brier_score_loss(y_test, proba),
        })
    return pd.DataFrame(rows).set_index("Model").loc[MODEL_ORDER]


def plot_roc(results, y_test, out_dir, disease_label):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name in MODEL_ORDER:
        proba = results[name]["proba"]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=PALETTE[name], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Random chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {disease_label}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "roc_curves.png", dpi=150); plt.close(fig)


def plot_metrics_heatmap(metrics_df, out_dir, disease_label):
    goodness = metrics_df.copy()
    for col in goodness.columns:
        rng = goodness[col].max() - goodness[col].min() + 1e-9
        if "Brier" in col:
            goodness[col] = 1 - (goodness[col] - goodness[col].min()) / rng
        else:
            goodness[col] = (goodness[col] - goodness[col].min()) / rng
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(goodness, annot=metrics_df.round(3), fmt="", cmap="RdYlGn", ax=ax,
                cbar_kws={"label": "Relative performance"}, linewidths=0.5, linecolor="white")
    ax.set_title(f"Metrics Comparison — {disease_label}")
    fig.tight_layout(); fig.savefig(out_dir / "metrics_heatmap.png", dpi=150); plt.close(fig)


def plot_confusion_matrices(results, y_test, out_dir, disease_label):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, name in zip(axes.flat, MODEL_ORDER):
        pred = (results[name]["proba"] >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax, cbar=False,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_title(name, fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for ax in axes.flat[len(MODEL_ORDER):]:
        ax.axis("off")
    fig.suptitle(f"Confusion Matrices (0.5 threshold) — {disease_label}", fontsize=13)
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrices.png", dpi=150); plt.close(fig)


def plot_feature_importance(results, features, out_dir, disease_label):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    tree_models = ["Random Forest", "Gradient Boosting", "XGBoost (production)"]
    for ax, name in zip(axes.flat[:3], tree_models):
        importances = results[name]["model"].feature_importances_
        order = np.argsort(importances)[-12:]
        ax.barh(np.array(features)[order], importances[order], color=PALETTE[name])
        ax.set_title(f"{name} — top 12", fontsize=10)
    lr_model = results["Logistic Regression"]["model"]
    coefs = np.abs(lr_model.coef_[0])
    order = np.argsort(coefs)[-12:]
    axes.flat[3].barh(np.array(features)[order], coefs[order], color=PALETTE["Logistic Regression"])
    axes.flat[3].set_title("Logistic Regression — top 12 |coef| (standardized)", fontsize=10)
    fig.suptitle(f"Feature Importance — {disease_label}", fontsize=13)
    fig.tight_layout(); fig.savefig(out_dir / "feature_importance.png", dpi=150); plt.close(fig)


def plot_calibration(results, y_test, out_dir, disease_label):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name in MODEL_ORDER:
        proba = results[name]["proba"]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name, color=PALETTE[name], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration — {disease_label}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "calibration_curves.png", dpi=150); plt.close(fig)


def main():
    df = tmd.load_data()
    print(f"Loaded {len(df)} rows\n")

    disease_labels = {
        "metabolic_syndrome": "Metabolic Syndrome",
        "nafld": "NAFLD (Fatty Liver)",
        "ckd": "Chronic Kidney Disease",
        "hypertension": "Hypertension",
    }

    summary_rows = []
    for name, config in tmd.DISEASES.items():
        label = disease_labels[name]
        out_dir = OUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== {label} ===")
        results, y_test, features = build_and_evaluate(name, config, df)
        metrics_df = compute_metrics(results, y_test)
        metrics_df.to_csv(out_dir / "metrics_comparison.csv")
        print(metrics_df.round(3).to_string())
        print()

        plot_roc(results, y_test, out_dir, label)
        plot_metrics_heatmap(metrics_df, out_dir, label)
        plot_confusion_matrices(results, y_test, out_dir, label)
        plot_feature_importance(results, features, out_dir, label)
        plot_calibration(results, y_test, out_dir, label)

        prod_auc = metrics_df.loc["XGBoost (production)", "ROC-AUC"]
        best_model = metrics_df["ROC-AUC"].idxmax()
        best_auc = metrics_df["ROC-AUC"].max()
        summary_rows.append({
            "Disease": label, "Production XGBoost AUC": round(prod_auc, 3),
            "Best model": best_model, "Best AUC": round(best_auc, 3),
            "Gap": round(best_auc - prod_auc, 3),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_ROOT / "multi_disease_validation_summary.csv", index=False)
    print("\n=== SUMMARY ACROSS ALL 4 DISEASES ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved all outputs under {OUT_ROOT}/<disease_name>/")


if __name__ == "__main__":
    main()
