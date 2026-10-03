"""
Metis - Subgroup Fairness Analysis (all 5 production models)

Checks whether each model performs equally well across age bands, sex,
and income level (poverty_ratio, already a feature) - standard, expected
practice for any clinical risk tool, not optional polish. Uses the SAME
held-out test splits as calibration_analysis.py (same random_state=42),
scored through the real production prediction functions (model.py /
diseases.py), so this measures what's actually deployed.

Reports ROC-AUC per subgroup where the subgroup has enough positive
cases to compute one reliably (n>=10 positives; smaller subgroups are
noted but not scored, since an AUC from a handful of cases is not a
meaningful estimate of anything).
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import model as diabetes_model  # noqa: E402
import diseases  # noqa: E402
import train_multi_disease as tmd  # noqa: E402

OUT_DIR = BASE_DIR / "validation_results" / "fairness"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
MIN_POSITIVES = 10


def predict_rowwise(predict_fn, X_df):
    return np.array([predict_fn(row.to_dict()) for _, row in X_df.iterrows()])


def age_band(age):
    if age < 40:
        return "18-39"
    elif age < 60:
        return "40-59"
    else:
        return "60+"


def income_band(ratio):
    if pd.isna(ratio):
        return None
    if ratio < 1.3:
        return "Low income (<1.3x poverty line)"
    elif ratio < 3.0:
        return "Middle income (1.3-3x)"
    else:
        return "Higher income (3x+)"


def analyze_disease(name, X_test, y_test, meta_test, predict_fn, display_label):
    proba = predict_rowwise(predict_fn, X_test)
    y_arr = y_test.to_numpy()
    overall_auc = roc_auc_score(y_arr, proba)

    rows = [{"Subgroup": "Overall", "n": len(y_arr), "n_positive": int(y_arr.sum()), "ROC-AUC": round(overall_auc, 3)}]

    groupings = {
        "Sex": meta_test["gender_male"].map({1: "Male", 0: "Female"}),
        "Age band": meta_test["age"].apply(age_band),
        "Income band": meta_test["poverty_ratio"].apply(income_band),
    }

    for group_name, group_series in groupings.items():
        for level in sorted(group_series.dropna().unique()):
            mask = (group_series == level).to_numpy()
            n = mask.sum()
            n_pos = int(y_arr[mask].sum())
            if n_pos < MIN_POSITIVES or (n - n_pos) < MIN_POSITIVES:
                rows.append({"Subgroup": f"{group_name}: {level}", "n": int(n), "n_positive": n_pos, "ROC-AUC": None})
                continue
            auc = roc_auc_score(y_arr[mask], proba[mask])
            rows.append({"Subgroup": f"{group_name}: {level}", "n": int(n), "n_positive": n_pos, "ROC-AUC": round(auc, 3)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"{name}_fairness.csv", index=False)

    # Bar chart of subgroup AUCs vs overall
    plot_df = df.dropna(subset=["ROC-AUC"])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(plot_df))))
    colors = ["#4C2478" if s == "Overall" else "#7440B0" for s in plot_df["Subgroup"]]
    ax.barh(plot_df["Subgroup"], plot_df["ROC-AUC"], color=colors)
    ax.axvline(overall_auc, color="#C4577B", linestyle="--", linewidth=1, label=f"Overall AUC ({overall_auc:.3f})")
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.5, 1.0)
    ax.set_title(f"Subgroup Performance — {display_label}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_fairness.png", dpi=150)
    plt.close(fig)

    max_gap = (plot_df[plot_df["Subgroup"] != "Overall"]["ROC-AUC"] - overall_auc).abs().max() if len(plot_df) > 1 else 0.0
    return {"Disease": display_label, "Overall AUC": round(overall_auc, 3), "Max subgroup gap": round(max_gap, 3), "table": df}


def main():
    print("Loading data...")
    df_multi = tmd.load_data()
    summary = []

    print("\n=== Type 2 Diabetes ===")
    features = diabetes_model.FEATURES
    data = df_multi.dropna(subset=["diabetes_label"]).reset_index(drop=True)
    X = data[features]
    y = data["diabetes_label"].astype(int)
    meta = data[["age", "gender_male", "poverty_ratio"]]
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    result = analyze_disease("diabetes", X_test, y_test, meta_test, diabetes_model.predict_risk, "Type 2 Diabetes")
    print(result["table"].to_string(index=False))
    summary.append({k: v for k, v in result.items() if k != "table"})

    disease_labels = {
        "metabolic_syndrome": "Metabolic Syndrome", "nafld": "NAFLD (Fatty Liver)",
        "ckd": "Chronic Kidney Disease", "hypertension": "Hypertension",
    }
    for name, config in tmd.DISEASES.items():
        print(f"\n=== {disease_labels[name]} ===")
        label_col = config["label"]
        feats = [f for f in tmd._ALL_FEATURES if f not in config["exclude"]]
        data = df_multi.dropna(subset=[label_col]).reset_index(drop=True)
        X = data[feats]
        y = data[label_col].astype(int)
        meta = data[["age", "gender_male", "poverty_ratio"]]
        X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
            X, y, meta, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        predict_fn = lambda fd, n=name: diseases.predict_disease_risk(n, fd)
        result = analyze_disease(name, X_test, y_test, meta_test, predict_fn, disease_labels[name])
        print(result["table"].to_string(index=False))
        summary.append({k: v for k, v in result.items() if k != "table"})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "fairness_summary.csv", index=False)
    print("\n=== FAIRNESS SUMMARY — ALL 5 MODELS ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved per-disease tables/charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
