"""
Metis - External Validation Against HRS (Health and Retirement Study)

RUN THIS ENTIRELY ON YOUR OWN MACHINE. Per HRS's AI and LLM Use Policy
(hrsdata.isr.umich.edu/data-products/ai-llm-use-policy, checked
2026-10), HRS data may not be processed by any LLM, including this one -
this script was written blind to the actual HRS files (never uploaded,
never will be) and is designed so you run it locally, and share back
ONLY the printed aggregate metrics at the end (AUC, calibration error,
etc.) - never the underlying row-level data, which this script never
writes to disk or prints.

WHAT THIS DOES: loads Metis's already-trained, already-saved production
models (no retraining - external validation means testing the EXISTING
model against data it has never seen, exactly like the internal
NHANES-holdout validation already done, but now on a genuinely different
cohort), maps HRS variables to Metis's feature names, and reports how
well each model's predictions line up with HRS's own recorded outcomes.

BEFORE RUNNING - YOU MUST FILL IN THE SECTIONS MARKED "TODO":
This script does not know HRS's exact column names with certainty - I
have general knowledge of what the RAND HRS Longitudinal File typically
contains, but column names (and their exact wave-numbering convention)
should be confirmed against YOUR actual codebook before trusting this
script's output, not assumed from my training data. Open your
`randhrs1992_2022v1.pdf` documentation alongside this file and fill in
the exact variable names for your HRS wave of interest.

Recommended wave: pick the HRS biomarker wave closest to NHANES
2021-2023 in time if comparing contemporaneously, OR (better for a true
prospective test) pick an EARLIER biomarker wave (e.g. 2016) as
"baseline" and a LATER wave's diagnosis variables (e.g. 2020 or later)
as "outcome" - this tests whether Metis's baseline risk predicts WHO
LATER got diagnosed, the actual claim internal validation cannot make.

Usage (RAND HRS Longitudinal File alone - demographics/BP/smoking/
diagnosis fields only, no HbA1c/cholesterol/CRP/cystatin-C):
  python3 external_validation_hrs.py --hrs-file /path/to/randhrs_file.dta --wave 13

Usage (RAND HRS + a separate Biomarker Data file - recommended, needed
for HbA1c/cholesterol/HDL/CRP/cystatin-C, which are NOT in the RAND HRS
Longitudinal File itself):
  python3 external_validation_hrs.py --hrs-file /path/to/randhrs_file.dta \
      --biomarker-file /path/to/biomarker_2016.dta --wave 13

The two files are merged internally on the shared person ID (HHIDPN) -
you do not need to merge them yourself first.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import model as diabetes_model  # noqa: E402
import diseases  # noqa: E402


def load_hrs(path: str) -> pd.DataFrame:
    """Loads an HRS file - .dta (Stata) is recommended, read natively by
    pandas with no extra dependency."""
    path = Path(path)
    if path.suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    elif path.suffix in (".csv",):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .dta (recommended) or .csv.")


def merge_biomarker_file(main_df: pd.DataFrame, biomarker_path: str) -> pd.DataFrame:
    """Merges a separate HRS Biomarker Data file onto the main RAND HRS
    file using the shared person ID. HRS's person-ID column is HHIDPN in
    both the RAND Longitudinal File and the Biomarker Data releases -
    confirm this against your own codebooks if the merge below produces
    an unexpectedly small or empty result (printed below either way, so
    you'll see it immediately rather than silently getting zero matches)."""
    bio = load_hrs(biomarker_path)
    id_col = "hhidpn" if "hhidpn" in bio.columns else "HHIDPN"
    merged = main_df.merge(bio, on=id_col, how="left", suffixes=("", "_bio"))
    n_matched = merged[id_col].isin(bio[id_col]).sum()
    print(f"  Biomarker merge: {n_matched} of {len(main_df)} respondents matched to a biomarker record.")
    return merged


def map_hrs_to_metis_features(hrs: pd.DataFrame, wave_suffix: str) -> pd.DataFrame:
    """
    Maps HRS variables (RAND HRS Longitudinal File naming convention -
    typically a letter prefix for the wave number, e.g. 'r13' = wave 13)
    to Metis's feature names. FILL IN THE EXACT COLUMN NAMES FOR YOUR
    WAVE below - these are illustrative based on RAND HRS's documented
    naming conventions, not confirmed against your actual file.

    wave_suffix: e.g. "13" for wave 13 (2016), "r" prefix is RAND's
    convention for respondent-level variables.
    """
    r = wave_suffix
    out = pd.DataFrame()
    out["hhidpn"] = hrs["hhidpn"]  # unique person ID - confirm this exists in your file

    # --- TODO: confirm each of these against your codebook ---
    out["age"] = hrs.get(f"r{r}agey_e")  # age at interview, years
    out["gender_male"] = (hrs.get("ragender") == 1).astype(int)  # 1=male, 2=female in RAND HRS convention - CONFIRM
    out["bmi"] = hrs.get(f"r{r}bmi")  # RAND HRS pre-computed BMI
    out["waist_cm"] = None  # RAND HRS longitudinal file typically does NOT include waist - check the separate Biomarker Data release instead
    out["bp_systolic"] = hrs.get(f"r{r}systo")  # if present in your biomarker wave
    out["bp_diastolic"] = hrs.get(f"r{r}diasto")
    out["smoker_current"] = (hrs.get(f"r{r}smoken") == 1).astype(int)  # currently smokes

    # Biomarker file fields (only present if you passed --biomarker-file -
    # these come from the HRS Biomarker Data releases, e.g. 2016 Biomarker
    # Data, NOT the RAND HRS Longitudinal File itself). CONFIRM each exact
    # column name against your biomarker file's own codebook - the names
    # below are my best general knowledge of HRS biomarker field naming,
    # not confirmed against your actual file. If a column below raises a
    # KeyError or silently stays all-None, check the codebook and fix the
    # name here.
    out["hba1c"] = hrs.get("A1C")  # TODO: confirm - HRS biomarker glycated hemoglobin field
    out["total_cholesterol"] = hrs.get("TC")  # TODO: confirm
    out["hdl"] = hrs.get("HDL")  # TODO: confirm
    out["crp"] = hrs.get("CRP")  # TODO: confirm
    out["creatinine"] = None  # HRS biomarker releases typically report cystatin-C, not creatinine directly
    out["cystatin_c"] = hrs.get("CYSC")  # TODO: confirm - used below for an independent eGFR cross-check,
                                           # NOT treated as identical to Metis's creatinine-based eGFR

    # --- Outcomes (what actually happened - for comparing against Metis's prediction) ---
    out["diabetes_diagnosed"] = (hrs.get(f"r{r}diabe") == 1).astype(int)  # ever told has diabetes
    out["hibp_diagnosed"] = (hrs.get(f"r{r}hibpe") == 1).astype(int)  # ever told high blood pressure

    return out


def egfr_from_cystatin_c(cystatin_c, age, is_male):
    """CKD-EPI 2012 cystatin-C-based equation (race-free) - HRS's
    cystatin-C is NOT directly comparable to Metis's creatinine-based
    eGFR, so this is reported as an independent cross-check, not treated
    as identical to the Metis CKD label's own eGFR definition."""
    kappa = 0.8
    alpha = -0.499 if is_male else -0.499
    min_ratio = np.minimum(cystatin_c / kappa, 1.0)
    max_ratio = np.maximum(cystatin_c / kappa, 1.0)
    sex_term = 0.932 if not is_male else 1.0
    return 133 * (min_ratio ** alpha) * (max_ratio ** -1.328) * (0.996 ** age) * sex_term


def run_validation(mapped: pd.DataFrame):
    results = []

    # --- Diabetes ---
    has_diabetes_data = mapped["diabetes_diagnosed"].notna() & mapped["age"].notna()
    if has_diabetes_data.sum() > 30:
        sub = mapped[has_diabetes_data].copy()
        patients = sub.to_dict("records")
        proba = np.array([diabetes_model.predict_risk(p) for p in patients])
        y = sub["diabetes_diagnosed"].to_numpy()
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y, proba)
            brier = brier_score_loss(y, proba)
            results.append({"Disease": "Type 2 Diabetes", "n": len(y), "AUC": round(auc, 3), "Brier": round(brier, 3)})

    # --- Hypertension ---
    has_htn_data = mapped["hibp_diagnosed"].notna() & mapped["age"].notna()
    if has_htn_data.sum() > 30:
        sub = mapped[has_htn_data].copy()
        patients = sub.to_dict("records")
        proba = np.array([diseases.predict_disease_risk("hypertension", p) for p in patients])
        y = sub["hibp_diagnosed"].to_numpy()
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y, proba)
            brier = brier_score_loss(y, proba)
            results.append({"Disease": "Hypertension", "n": len(y), "AUC": round(auc, 3), "Brier": round(brier, 3)})

    # --- Chronic Kidney Disease (cystatin-C cross-check) ---
    # HRS's biomarker releases measure cystatin-C, not creatinine - Metis's
    # CKD label is creatinine-based eGFR, so this is a genuinely
    # independent kidney-function marker, not the identical measurement.
    # Treat agreement here as supportive evidence, not a like-for-like
    # replication of the internal NHANES validation.
    has_ckd_data = mapped["cystatin_c"].notna() & mapped["age"].notna()
    if has_ckd_data.sum() > 30:
        sub = mapped[has_ckd_data].copy()
        sub["egfr_cystatin"] = egfr_from_cystatin_c(sub["cystatin_c"], sub["age"], sub["gender_male"] == 1)
        y_ckd = (sub["egfr_cystatin"] < 60).astype(int).to_numpy()
        patients = sub.to_dict("records")
        proba = np.array([diseases.predict_disease_risk("ckd", p) for p in patients])
        if len(np.unique(y_ckd)) == 2:
            auc = roc_auc_score(y_ckd, proba)
            brier = brier_score_loss(y_ckd, proba)
            results.append({"Disease": "Chronic Kidney Disease (cystatin-C cross-check)", "n": len(y_ckd), "AUC": round(auc, 3), "Brier": round(brier, 3)})

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hrs-file", required=True, help="Path to your downloaded RAND HRS Longitudinal file (.dta recommended)")
    parser.add_argument("--biomarker-file", default=None, help="Path to a separate HRS Biomarker Data file (.dta) - needed for HbA1c/cholesterol/HDL/CRP/cystatin-C. Optional, but strongly recommended.")
    parser.add_argument("--wave", default="13", help="HRS wave suffix, e.g. '13' for wave 13 (2016)")
    args = parser.parse_args()

    print("Loading HRS file(s) locally (never leaves your machine)...")
    hrs = load_hrs(args.hrs_file)
    print(f"Loaded {len(hrs)} HRS respondents from the main file.")

    if args.biomarker_file:
        hrs = merge_biomarker_file(hrs, args.biomarker_file)

    print("Mapping HRS variables to Metis features - CHECK THE TODO SECTIONS ABOVE FIRST if this looks wrong...")
    mapped = map_hrs_to_metis_features(hrs, args.wave)

    print("\nRunning Metis's production models against HRS outcomes...")
    results = run_validation(mapped)

    print("\n=== EXTERNAL VALIDATION RESULTS (HRS) ===")
    print("These are the ONLY numbers you should share back - never the underlying data.")
    print(results.to_string(index=False) if len(results) else "No results - check the TODO mappings above; likely a required field wasn't found.")


if __name__ == "__main__":
    main()
