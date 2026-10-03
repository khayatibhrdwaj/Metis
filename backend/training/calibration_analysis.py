"""
Metis - Calibration Analysis & Recalibration (all 5 production models)

Validation studies so far (validate_models.py, validate_multi_disease.py)
measured DISCRIMINATION (can the model rank higher-risk people above
lower-risk people - ROC-AUC). This script measures something different
and arguably more clinically important: CALIBRATION - if the model says
"40% risk", do roughly 40% of people at that score actually have the
condition? A model can have excellent discrimination (great ranking) and
still be poorly calibrated (the actual numbers shown to users are
systematically too high or too low) - these are genuinely different
properties, and a decision-support tool showing a raw percentage to a
user needs the SECOND one to be trustworthy, not just the first.

Methodology: this runs against the ACTUAL PRODUCTION MODELS loaded via
model.py/diseases.py (the exact code path the real API uses, not a
retrained stand-in), so this measures what's really deployed. To avoid
calibration leakage (fitting a calibrator on the same data used to
evaluate it), each disease's original held-out test split (same
random_state=42 split used in training) is further split in half:
one half fits an isotonic calibrator, the other half evaluates both raw
and calibrated probabilities. This means the original trained model
files are never touched or retrained - only a small calibration
transform is layered on top, saved separately.

Metrics reported: Brier score (lower=better) and Expected Calibration
Error (ECE, 10-bin, lower=better), for both raw and calibrated output.
"""

import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import model as diabetes_model  # noqa: E402
import diseases  # noqa: E402
import train_multi_disease as tmd  # noqa: E402

OUT_DIR = BASE_DIR / "validation_results" / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Mean absolute gap between predicted probability and observed
    frequency, weighted by how many predictions fall in each probability
    bin. 0 = perfectly calibrated."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(bin_acc - bin_conf)
    return ece


def predict_rowwise(predict_fn, X_df):
    """Runs the real production predict function row-by-row - slower than
    a vectorized call, but guarantees this measures the exact code path
    the live API uses, not a reimplementation of it."""
    return np.array([predict_fn(row.to_dict()) for _, row in X_df.iterrows()])


def analyze_disease(name, X_test_full, y_test_full, predict_fn, display_label):
    # Split the ORIGINAL held-out test set in half: one half calibrates,
    # the other half evaluates - neither half was used in training.
    X_calib, X_eval, y_calib, y_eval = train_test_split(
        X_test_full, y_test_full, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test_full
    )

    print(f"  Scoring {name} on calibration half (n={len(X_calib)}) and eval half (n={len(X_eval)})...")
    proba_calib = predict_rowwise(predict_fn, X_calib)
    proba_eval = predict_rowwise(predict_fn, X_eval)

    y_calib_arr = y_calib.to_numpy()
    y_eval_arr = y_eval.to_numpy()

    brier_raw = brier_score_loss(y_eval_arr, proba_eval)
    ece_raw = expected_calibration_error(y_eval_arr, proba_eval)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(proba_calib, y_calib_arr)
    proba_eval_calibrated = calibrator.predict(proba_eval)

    brier_cal = brier_score_loss(y_eval_arr, proba_eval_calibrated)
    ece_cal = expected_calibration_error(y_eval_arr, proba_eval_calibrated)

    # Reliability diagram: raw vs calibrated, both on the SAME eval half
    fig, ax = plt.subplots(figsize=(6.5, 6))
    frac_raw, mean_raw = calibration_curve(y_eval_arr, proba_eval, n_bins=8, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_eval_arr, proba_eval_calibrated, n_bins=8, strategy="quantile")
    ax.plot(mean_raw, frac_raw, marker="o", label=f"Raw (Brier={brier_raw:.3f}, ECE={ece_raw:.3f})", color="#C4577B", linewidth=2)
    ax.plot(mean_cal, frac_cal, marker="s", label=f"Calibrated (Brier={brier_cal:.3f}, ECE={ece_cal:.3f})", color="#2E8B7D", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration — {display_label}")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_calibration.png", dpi=150)
    plt.close(fig)

    improved = brier_cal < brier_raw
    if improved:
        with open(OUT_DIR / f"{name}_calibrator.pkl", "wb") as f:
            pickle.dump(calibrator, f)

    return {
        "Disease": display_label, "Brier (raw)": round(brier_raw, 4), "Brier (calibrated)": round(brier_cal, 4),
        "ECE (raw)": round(ece_raw, 4), "ECE (calibrated)": round(ece_cal, 4),
        "Calibration improves Brier?": improved,
    }


def main():
    print("Loading data...")
    df_multi = tmd.load_data()

    results = []

    # --- Diabetes (stacked ensemble via model.py's real predict_risk) ---
    print("\n=== Type 2 Diabetes ===")
    diabetes_features = diabetes_model.FEATURES
    df_diabetes = df_multi.dropna(subset=["diabetes_label"]).reset_index(drop=True)
    X = df_diabetes[diabetes_features]
    y = df_diabetes["diabetes_label"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    results.append(analyze_disease("diabetes", X_test, y_test, diabetes_model.predict_risk, "Type 2 Diabetes"))

    # --- The 4 newer diseases (single monotonic XGBoost via diseases.py's real predict_disease_risk) ---
    disease_labels = {
        "metabolic_syndrome": "Metabolic Syndrome",
        "nafld": "NAFLD (Fatty Liver)",
        "ckd": "Chronic Kidney Disease",
        "hypertension": "Hypertension",
    }
    for name, config in tmd.DISEASES.items():
        print(f"\n=== {disease_labels[name]} ===")
        label_col = config["label"]
        features = [f for f in tmd._ALL_FEATURES if f not in config["exclude"]]
        data = df_multi.dropna(subset=[label_col]).reset_index(drop=True)
        X = data[features]
        y = data[label_col].astype(int)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        predict_fn = lambda fd, n=name: diseases.predict_disease_risk(n, fd)
        results.append(analyze_disease(name, X_test, y_test, predict_fn, disease_labels[name]))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUT_DIR / "calibration_summary.csv", index=False)
    print("\n=== CALIBRATION SUMMARY — ALL 5 MODELS ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved reliability diagrams and summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
