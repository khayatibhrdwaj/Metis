"""
Metis - External Validation Against CHNS (China Health and Nutrition Survey)

RUN THIS ENTIRELY ON YOUR OWN MACHINE. CHNS data is publicly downloadable
after registration (cpc.unc.edu/projects/china/data) with no senior-
researcher restriction or IRB requirement for using already-public,
de-identified data - a materially lower barrier than HRS/LASI-DAD/
BioLINCC, all of which required institutional backing this project
couldn't get. Still good practice: this script never writes CHNS
row-level data to disk, and the only thing to share back is the final
printed metrics table.

IMPORTANT - MY CONFIDENCE IN CHNS'S EXACT FILE/VARIABLE NAMES IS LOWER
THAN IT WAS FOR HRS. I have detailed general knowledge of HRS's RAND
Longitudinal File naming convention from extensive documentation; CHNS
is less consistently documented in what I was trained on, and its data
is split across several separate per-module files (demographics,
physical exam, biomarkers, health/illness) that must be merged on an
individual ID, rather than one master file. EVERY column name and file
name below is a placeholder based on general knowledge of CHNS's
typical structure, not a confirmed mapping - treat this entire script as
a well-organized skeleton to fill in against your actual codebook, more
so than the HRS version was.

WHAT THIS DOES: loads Metis's already-trained production models (no
retraining), merges CHNS's separate per-module files, maps CHNS
variables to Metis's feature names, and compares Metis's predictions
against CHNS's own recorded outcomes - both cross-sectionally (one
wave) and, if you have two waves downloaded, prospectively (2009
baseline predicting 2011 incident diagnosis - the stronger claim,
matching what internal NHANES validation structurally cannot test).

NAFLD uses a PROXY, not a direct match: CHNS does not appear to include
liver elastography (CAP score) the way NHANES does, so this script
computes the Fatty Liver Index (FLI; Bedogni et al. 2006, Gastroenterology)
from BMI, waist circumference, triglycerides, and GGT - a validated,
widely-used proxy, but a genuinely different measurement than the
elastography-based label Metis's NAFLD model was trained to predict.
Report this distinction in any resulting manuscript section, not just in
this comment. GGT itself is mapped from an unlabeled CHNS column (Y48_2)
on quantitative evidence (plausible clinical range + 0.848 correlation
with confirmed ALT, far higher than other unlabeled candidates) rather
than a codebook-confirmed label - report that basis explicitly too; this
is two layers of approximation stacked on each other (proxy formula +
evidence-based column identity), not one.

CONFIRMED: creatinine IS in CHNS's biomarker release (CRE_MG) - resolved,
contrary to earlier uncertainty when this script was first drafted.

The CKD validation block still degrades gracefully (reports zero usable
rows rather than crashing) if creatinine is ever absent in a different
file release than the one this script was built against.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import model as diabetes_model  # noqa: E402
import diseases  # noqa: E402

# IMPORTANT, CONFIRMED BY ACTUALLY INSPECTING TWO REAL CHNS FILES: the ID
# column's CASE is NOT consistent across CHNS's own files. The Master ID
# File codebook shows "IDIND" (all caps); the actual 2009 Biomarker file
# (biomarker_09.sas7bdat, inspected directly - CC0 licensed, so this was
# safe to open, unlike HRS) uses "IDind" (mixed case). A naive merge on
# column name would silently produce ZERO matches between these two real
# files due to this exact mismatch - this is precisely the kind of bug
# find_id_col()/merge_chns_modules() were built to catch, but the
# original version only checked a fixed candidate list; it's now
# case-insensitive so it catches this regardless of which file you add
# next.
ID_COL_CANDIDATES = ["IDIND", "IDind", "idind"]


def find_id_col_case_insensitive(df: pd.DataFrame) -> str:
    """More robust than a fixed candidate list - matches any column whose
    name is 'idind' case-insensitively, since CHNS's own files don't agree
    on casing."""
    for col in df.columns:
        if col.strip().lower() == "idind":
            return col
    raise ValueError(f"No ID column found (case-insensitive 'idind' search). Columns: {list(df.columns)[:20]}...")

# CONFIRMED: GENDER (not "gender"), coded 1=Male, 2=Female - exactly as
# originally guessed for the coding, but the column name and case were
# both wrong before this was checked against the real codebook.
GENDER_COL = "GENDER"
GENDER_MALE_VALUE = 1


def load_chns_file(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".sas7bdat":
        return pd.read_sas(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def find_id_col(df: pd.DataFrame) -> str:
    """Case-insensitive - see the comment above ID_COL_CANDIDATES for why
    this matters: CHNS's own files don't agree on IDIND vs IDind casing."""
    return find_id_col_case_insensitive(df)


def _find_wave_col(df: pd.DataFrame):
    for col in df.columns:
        if col.strip().lower() == "wave":
            return col
    return None


def merge_chns_modules(demo, exam, biomarker, health) -> pd.DataFrame:
    """Merges CHNS's separate per-module files on the individual ID.

    IMPORTANT, CONFIRMED BY DIRECTLY CHECKING 4 REAL CHNS FILES: the
    Master ID/demographic file (mast_pub_12-style) is one row per person,
    EVER - no wave column, since birth date/gender/death don't change by
    year. But surveys_pub_12, pexam_00, and hlth_12 are each genuinely
    multi-wave - one row per person PER SURVEY YEAR they participated in
    (confirmed: pexam_00 has 110,449 rows for only 33,360 unique people,
    roughly 3.3 rows per person on average). Merging these multi-wave
    files together on ID ALONE (ignoring wave) would match every wave of
    one file against every wave of another for the same person - e.g.
    pairing someone's 2009 exam data with their 1991 health-module row.
    This function merges the one-time demo file on ID alone (correct,
    since those facts don't vary by wave), but merges every other,
    genuinely multi-wave file on BOTH ID and WAVE - required for a
    correct merge, not an optional refinement.
    """
    id_col = find_id_col(demo)
    merged = demo.copy()
    for name, df in [("exam", exam), ("biomarker", biomarker), ("health", health)]:
        if df is None:
            continue
        this_id = find_id_col(df)
        this_wave = _find_wave_col(df)
        before = len(merged)
        merge_keys_left = [id_col]
        merge_keys_right = [this_id]
        existing_wave = _find_wave_col(merged)
        if this_wave is not None and existing_wave is not None:
            merge_keys_left.append(existing_wave)
            merge_keys_right.append(this_wave)
        elif this_wave is not None and existing_wave is None:
            # First multi-wave file being merged onto the one-time demo
            # file - nothing to match wave against yet, so this merge is
            # still ID-only (correct for the FIRST multi-wave file added;
            # its own wave column becomes the anchor for every subsequent
            # multi-wave file's merge).
            pass
        merged = merged.merge(df, left_on=merge_keys_left, right_on=merge_keys_right, how="left", suffixes=("", f"_{name}"))
        print(f"  Merged {name} file ({'ID+WAVE' if len(merge_keys_left) > 1 else 'ID only'}): "
              f"{before} rows before -> {len(merged)} rows after")
    return merged


def fatty_liver_index(bmi, waist_cm, triglycerides_mgdl, ggt):
    """Bedogni et al. 2006 (BMC Gastroenterology) Fatty Liver Index -
    a validated proxy for hepatic steatosis using routine measurements,
    since CHNS doesn't have elastography. FLI >= 60 is the paper's own
    threshold for 'likely fatty liver' - used here as the positive label
    for this proxy comparison."""
    tg_mgdl = triglycerides_mgdl
    L = (
        0.953 * np.log(tg_mgdl)
        + 0.139 * bmi
        + 0.718 * np.log(np.maximum(ggt, 1))
        + 0.053 * waist_cm
        - 15.745
    )
    return (np.exp(L) / (1 + np.exp(L))) * 100


def egfr_ckd_epi(creatinine_mgdl, age, is_male):
    """Same CKD-EPI 2021 race-free equation used throughout this project
    (see data_prep_2021_2023.py) - applied here to CHNS's creatinine for
    a like-for-like CKD label, unlike the HRS script's cystatin-C version
    which was necessarily a different marker."""
    kappa = np.where(is_male, 0.9, 0.7)
    alpha = np.where(is_male, -0.302, -0.241)
    sex_term = np.where(is_male, 1.0, 1.012)
    min_ratio = np.minimum(creatinine_mgdl / kappa, 1.0)
    max_ratio = np.maximum(creatinine_mgdl / kappa, 1.0)
    return 142 * (min_ratio ** alpha) * (max_ratio ** -1.200) * (0.9938 ** age) * sex_term


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Returns the column if it exists, otherwise an all-NaN Series of the
    right length - prevents a crash (found the hard way: merged.get(col)
    returns plain None for a missing column, and None == 1 evaluates to a
    bare Python bool with no .astype() method) when a field genuinely
    hasn't been located yet, like smoker_current below. A missing field
    should show up as 0 usable rows downstream, not an exception."""
    if col in df.columns:
        return df[col]
    return pd.Series([float("nan")] * len(df), index=df.index)


def map_chns_to_metis_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Fields marked CONFIRMED were checked directly against real CHNS
    files you uploaded (surveys_pub_12, pexam_00, hlth_12, biomarker_09)
    - not guessed, not inferred from a diagram, actually read from each
    file's own SAS-generated codebook PDF via pdftotext. Only
    smoker_current remains unconfirmed - not found in pexam_00; check
    subi_12 (likely "substance use/intake") next."""
    out = pd.DataFrame()

    # CONFIRMED directly in surveys_pub_12.sas7bdat - age is NOT computed
    # from birth year/interview date as I originally (incorrectly) told
    # you; it's a direct field. This corrects that earlier guidance.
    out["age"] = _safe_col(merged, "age")
    out["gender_male"] = (merged[GENDER_COL] == GENDER_MALE_VALUE).astype(int)  # CONFIRMED column name + coding (mast file)

    # CONFIRMED in pexam_00: no direct BMI column exists (checked - only
    # data-quality-flag comments mention BMI, not an actual variable) -
    # computed here from HEIGHT (cm) and WEIGHT (kg), both confirmed
    # present and correctly named/unit-labeled in the file itself.
    out["bmi"] = _safe_col(merged, "WEIGHT") / ((_safe_col(merged, "HEIGHT") / 100) ** 2)
    out["waist_cm"] = _safe_col(merged, "U10")  # CONFIRMED: "WAIST CIRCUMFERENCE (CM)"

    # CONFIRMED: pexam_00 takes 3 separate BP readings per person per
    # wave (SYSTOL1/2/3, DIASTOL1/2/3) rather than one value - averaged
    # here, the standard approach for repeated clinical BP measurements.
    out["bp_systolic"] = merged[["SYSTOL1", "SYSTOL2", "SYSTOL3"]].mean(axis=1)
    out["bp_diastolic"] = merged[["DIASTOL1", "DIASTOL2", "DIASTOL3"]].mean(axis=1)

    smoke_raw = _safe_col(merged, "SMOKE")  # STILL UNCONFIRMED - check subi_12.sas7bdat next
    out["smoker_current"] = smoke_raw.map({1: 1, 0: 0})  # preserves NaN when unconfirmed/missing, rather than defaulting everyone to "non-smoker"

    # --- Biomarkers: CONFIRMED by directly inspecting the real 2009
    # biomarker file (biomarker_09.sas7bdat, CC0-licensed - safe to open
    # directly, unlike HRS). CHNS provides both SI and US-conventional
    # ("_MG" suffix) units for most analytes; the _MG versions are used
    # directly below since they already match Metis's expected units -
    # value ranges were checked against known clinical reference ranges
    # to confirm each ("UA_MG" mean ~5.2, consistent with mg/dL uric
    # acid; "CRE_MG" mean ~0.97, consistent with mg/dL creatinine; etc.)
    out["triglycerides"] = _safe_col(merged, "TG_MG")  # CONFIRMED mg/dL (mean ~185 in the real file - plausible population mean)
    out["total_cholesterol"] = _safe_col(merged, "TC_MG")  # CONFIRMED mg/dL (mean ~185, matches expected range)
    out["hdl"] = _safe_col(merged, "HDL_C_MG")  # CONFIRMED present with _MG conversion
    out["ldl"] = _safe_col(merged, "LDL_C_MG")  # CONFIRMED present with _MG conversion
    out["crp"] = _safe_col(merged, "HS_CRP")  # CONFIRMED column name is HS_CRP not CRP; already mg/L, no _MG variant needed (enzymes/CRP don't have a molar/mass unit split the way e.g. uric acid does)
    out["insulin"] = _safe_col(merged, "INS")  # CONFIRMED column name is INS not INSULIN; range (mean ~14.6) consistent with uIU/mL, no conversion needed
    out["uric_acid"] = _safe_col(merged, "UA_MG")  # CONFIRMED mg/dL
    out["albumin"] = _safe_col(merged, "ALB") / 10  # CONFIRMED present as ALB, but in g/L (mean ~47.6) - Metis expects g/dL, so /10 (gives ~4.76 g/dL, matching the expected 3.5-5.0 reference range)
    out["wbc"] = _safe_col(merged, "WBC")  # CONFIRMED present, range (2.1-46.0, mean 6.35) already matches Metis's expected units directly
    out["creatinine"] = _safe_col(merged, "CRE_MG")  # CONFIRMED present (contrary to earlier uncertainty) as CRE_MG, mg/dL

    # --- GGT: used with documented quantitative evidence, NOT a codebook-
    # confirmed label - disclose this exact basis in any manuscript using
    # it, don't quietly present it as confirmed. Y48_2 was checked two
    # ways: (1) its value range (1-480, mean 23.8) is plausible for GGT
    # (normal ~8-61 U/L, severe elevation into the hundreds in liver
    # disease/heavy alcohol use), and (2) it correlates with ALT (a
    # confirmed liver enzyme) at r=0.848 - far higher than the other
    # unlabeled candidates checked (Y48_3: r=0.079, Y48_4: r=0.058,
    # Y46_1DL: r=0.156), consistent with GGT and ALT's well-documented
    # tendency to rise together as co-elevated liver-stress markers. Both
    # checks point the same direction; neither is a label confirmation -
    # if the real codebook later says otherwise, defer to that, not this.
    out["ggt"] = _safe_col(merged, "Y48_2")  # EVIDENCE-BASED (range + 0.848 ALT correlation), NOT codebook-confirmed
    out["alt"] = _safe_col(merged, "ALT")  # CONFIRMED present and named directly
    # AST: RULED OUT, not just unconfirmed. Checked every remaining
    # unlabeled column's correlation with ALT (AST and ALT are typically
    # even MORE correlated with each other than GGT/ALT are, often
    # r=0.6-0.8+ in general populations) - the highest of the 7 remaining
    # candidates was r=0.209 (Y48_6), nowhere close to a real AST
    # signature. This is a confirmed absence from this biomarker release,
    # not an unresolved lead - stop looking for it here.
    out["ast"] = float("nan")

    # --- Outcomes: CONFIRMED in pexam_00's own codebook (not hlth_12 -
    # that file turned out to be about illness symptoms/healthcare
    # utilization, not chronic disease diagnosis; these questions are
    # bundled into the physical exam file instead, a real finding worth
    # knowing if this ever needs re-deriving). Both use identical 0=No,
    # 1=Yes, 9=Don't know coding - "don't know" rows are EXCLUDED (set to
    # NaN, not silently treated as "No"), matching the same ambiguous-
    # response-exclusion discipline used throughout this entire project
    # for NHANES's own self-report fields.
    diabetes_raw = _safe_col(merged, "U24A")  # "DIAGNOSED WITH DIABETES?"
    out["diabetes_diagnosed"] = diabetes_raw.where(diabetes_raw != 9).map({0: 0, 1: 1})

    hibp_raw = _safe_col(merged, "U22")  # "DIAGNOSED WITH HIGH BLOOD PRESSURE?"
    out["hibp_diagnosed"] = hibp_raw.where(hibp_raw != 9).map({0: 0, 1: 1})

    return out


def predict_risk_batch(df_features: pd.DataFrame) -> np.ndarray:
    """Vectorized version of model.py's predict_risk() - same ensemble,
    same medians fallback, same safety clip, just computed for the whole
    batch in a handful of calls instead of one Python loop iteration (and
    4 model calls) per row. This re-uses the exact trained objects
    model.py already loads (via its own _load()), not a re-implementation
    of the model itself - only the batching is new."""
    xgb_model, hgb_model, lr_model, lr_scaler, meta_model, _, medians, _ = diabetes_model._load()
    row_batch = pd.DataFrame({
        f: df_features[f] if f in df_features.columns else pd.Series(medians[f], index=df_features.index)
        for f in diabetes_model.FEATURES
    })
    p_xgb = xgb_model.predict_proba(row_batch)[:, 1]
    p_hgb = hgb_model.predict_proba(row_batch)[:, 1]
    row_imputed = row_batch.fillna(medians)
    p_lr = lr_model.predict_proba(lr_scaler.transform(row_imputed))[:, 1]
    meta_input = np.column_stack([p_xgb, p_hgb, p_lr])
    proba = meta_model.predict_proba(meta_input)[:, 1]
    return np.clip(proba, diabetes_model._MIN_RISK, diabetes_model._MAX_RISK)


def predict_disease_risk_batch(name: str, df_features: pd.DataFrame) -> np.ndarray:
    """Vectorized version of diseases.py's predict_disease_risk() for one
    of the 4 single-model diseases - same reasoning as predict_risk_batch
    above."""
    model, _, medians, features, _, calibrator = diseases._load(name)
    row_batch = pd.DataFrame({
        f: df_features[f] if f in df_features.columns else pd.Series(medians[f], index=df_features.index)
        for f in features
    })
    raw_proba = model.predict_proba(row_batch)[:, 1]
    proba = calibrator.predict(raw_proba) if calibrator is not None else raw_proba
    return np.clip(proba, diseases._MIN_RISK, diseases._MAX_RISK)


def _report_skip(disease_name: str, n_usable: int, reason: str):
    """Every disease now reports why it was or wasn't computed, with the
    exact row count - no disease goes silent the way Diabetes/Hypertension
    did in an earlier run (only CKD had an explicit message; this fixes
    that gap for all 4)."""
    print(f"  {disease_name}: {n_usable} usable rows (need >30) - {reason}")


def run_cross_sectional_validation(mapped: pd.DataFrame):
    results = []

    has_diabetes = mapped["diabetes_diagnosed"].notna() & mapped["age"].notna()
    n = has_diabetes.sum()
    if n > 30:
        sub = mapped[has_diabetes].copy()
        proba = predict_risk_batch(sub)
        y = sub["diabetes_diagnosed"].to_numpy()
        if len(np.unique(y)) == 2:
            results.append({
                "Disease": "Type 2 Diabetes", "n": len(y),
                "AUC": round(roc_auc_score(y, proba), 3), "Brier": round(brier_score_loss(y, proba), 3),
            })
            _report_skip("Type 2 Diabetes", n, "computed, see table below")
        else:
            _report_skip("Type 2 Diabetes", n, "SKIPPED - outcome has only one class in this subset, cannot compute AUC")
    else:
        _report_skip("Type 2 Diabetes", n, "SKIPPED - not enough rows with both diabetes_diagnosed and age present")

    has_htn = mapped["hibp_diagnosed"].notna() & mapped["age"].notna()
    n = has_htn.sum()
    if n > 30:
        sub = mapped[has_htn].copy()
        proba = predict_disease_risk_batch("hypertension", sub)
        y = sub["hibp_diagnosed"].to_numpy()
        if len(np.unique(y)) == 2:
            results.append({
                "Disease": "Hypertension", "n": len(y),
                "AUC": round(roc_auc_score(y, proba), 3), "Brier": round(brier_score_loss(y, proba), 3),
            })
            _report_skip("Hypertension", n, "computed, see table below")
        else:
            _report_skip("Hypertension", n, "SKIPPED - outcome has only one class in this subset, cannot compute AUC")
    else:
        _report_skip("Hypertension", n, "SKIPPED - not enough rows with both hibp_diagnosed and age present")

    # NAFLD via Fatty Liver Index proxy (not a direct label match - see module docstring)
    has_fli_inputs = mapped[["bmi", "waist_cm", "triglycerides", "ggt"]].notna().all(axis=1)
    n = has_fli_inputs.sum()
    if n > 30:
        sub = mapped[has_fli_inputs].copy()
        fli = fatty_liver_index(sub["bmi"], sub["waist_cm"], sub["triglycerides"], sub["ggt"])
        y_proxy = (fli >= 60).astype(int).to_numpy()
        proba = predict_disease_risk_batch("nafld", sub)
        if len(np.unique(y_proxy)) == 2:
            results.append({
                "Disease": "NAFLD (Fatty Liver Index proxy, NOT elastography - see docstring)", "n": len(y_proxy),
                "AUC": round(roc_auc_score(y_proxy, proba), 3), "Brier": round(brier_score_loss(y_proxy, proba), 3),
            })
            _report_skip("NAFLD", n, "computed, see table below")

            # SENSITIVITY CHECK: the Fatty Liver Index label is built from
            # BMI + waist + triglycerides + GGT - two of which (BMI,
            # waist) are ALSO real Metis NAFLD predictors. A high AUC here
            # could just mean Metis is rediscovering its own proxy label's
            # ingredients, not learning anything about liver fat. Trivial
            # test: how well do BMI+waist ALONE (a 2-variable toy model,
            # no Metis involved) predict the SAME label? If this toy
            # model's AUC is also very high, that's direct, quantified
            # evidence the 0.975 is mostly shared math, not real signal.
            toy_X = sub[["bmi", "waist_cm"]].to_numpy()
            toy_y = y_proxy
            toy_scaler = StandardScaler().fit(toy_X)
            toy_model = LogisticRegression().fit(toy_scaler.transform(toy_X), toy_y)
            toy_proba = toy_model.predict_proba(toy_scaler.transform(toy_X))[:, 1]
            toy_auc = roc_auc_score(toy_y, toy_proba)
            print(f"\n  *** NAFLD SENSITIVITY CHECK ***")
            print(f"  Metis's full model AUC:               {round(roc_auc_score(y_proxy, proba), 3)}")
            print(f"  Toy model (BMI+waist ONLY) AUC:        {round(toy_auc, 3)}")
            gap = roc_auc_score(y_proxy, proba) - toy_auc
            print(f"  Gap (Metis's real added value):        {round(gap, 3)}")
            if toy_auc > 0.85:
                print(f"  INTERPRETATION: the toy model alone gets AUC={round(toy_auc,3)} using just 2 of the proxy "
                      f"label's own 4 ingredients. This confirms the circularity concern - most of Metis's 0.975 is "
                      f"very likely attributable to shared math with the label's own construction, not genuine "
                      f"predictive signal about liver fat. Do NOT report 0.975 as a clean external validation result.")
            else:
                print(f"  INTERPRETATION: the toy model alone only gets AUC={round(toy_auc,3)} - Metis's full model "
                      f"adds real discrimination beyond what BMI+waist alone explain. Still disclose the shared-"
                      f"ingredient relationship, but this is less concerning than initially flagged.")
            print()
        else:
            _report_skip("NAFLD", n, "SKIPPED - proxy outcome has only one class in this subset")
    else:
        _report_skip("NAFLD", n, "SKIPPED - not enough rows with bmi+waist+triglycerides+ggt all present")

    # CKD via creatinine-based eGFR, only if creatinine is actually present
    has_ckd_inputs = mapped["creatinine"].notna() & mapped["age"].notna() if "creatinine" in mapped else pd.Series(dtype=bool)
    n = has_ckd_inputs.sum()
    if n > 30:
        sub = mapped[has_ckd_inputs].copy()
        egfr = egfr_ckd_epi(sub["creatinine"], sub["age"], sub["gender_male"] == 1)
        y_ckd = (egfr < 60).astype(int).to_numpy()
        proba = predict_disease_risk_batch("ckd", sub)
        if len(np.unique(y_ckd)) == 2:
            results.append({
                "Disease": "Chronic Kidney Disease", "n": len(y_ckd),
                "AUC": round(roc_auc_score(y_ckd, proba), 3), "Brier": round(brier_score_loss(y_ckd, proba), 3),
            })
            _report_skip("CKD", n, "computed, see table below")
        else:
            _report_skip("CKD", n, "SKIPPED - outcome has only one class in this subset")
    else:
        _report_skip("CKD", n, "SKIPPED - no creatinine+age rows found (see the DIAGNOSTIC block above "
                                "for whether CRE_MG survived the merge at all)")

    return pd.DataFrame(results)


def run_prospective_validation(mapped_baseline: pd.DataFrame, followup_health: pd.DataFrame, wave: str, followup_wave: str):
    """The stronger claim: does Metis's BASELINE (wave 1) prediction
    forecast who got diagnosed BY the follow-up wave, among people who
    did NOT already have the condition at baseline. This directly tests
    incident prediction, which internal NHANES validation structurally
    cannot - NHANES is cross-sectional, one measurement per person."""
    id_col = find_id_col(mapped_baseline)
    followup_id_col = find_id_col(followup_health)
    fu = followup_health[[followup_id_col, "diabetes", "hypertension"]].rename(
        columns={"diabetes": "diabetes_followup", "hypertension": "hibp_followup"}
    )
    merged = mapped_baseline.merge(fu, left_on=id_col, right_on=followup_id_col, how="inner")
    print(f"  {len(merged)} people matched between baseline ({wave}) and follow-up ({followup_wave}).")

    results = []
    # Incident diabetes: baseline NOT diagnosed, follow-up diagnosed
    incident_eligible = (merged["diabetes_diagnosed"] == 0) & merged["age"].notna()
    if incident_eligible.sum() > 30:
        sub = merged[incident_eligible].copy()
        proba = predict_risk_batch(sub)
        y = (sub["diabetes_followup"] == 1).astype(int).to_numpy()
        if len(np.unique(y)) == 2:
            results.append({
                "Disease": f"Type 2 Diabetes (INCIDENT, {wave}->{followup_wave})", "n": len(y),
                "AUC": round(roc_auc_score(y, proba), 3), "Brier": round(brier_score_loss(y, proba), 3),
            })

    incident_htn_eligible = (merged["hibp_diagnosed"] == 0) & merged["age"].notna()
    if incident_htn_eligible.sum() > 30:
        sub = merged[incident_htn_eligible].copy()
        proba = predict_disease_risk_batch("hypertension", sub)
        y = (sub["hibp_followup"] == 1).astype(int).to_numpy()
        if len(np.unique(y)) == 2:
            results.append({
                "Disease": f"Hypertension (INCIDENT, {wave}->{followup_wave})", "n": len(y),
                "AUC": round(roc_auc_score(y, proba), 3), "Brier": round(brier_score_loss(y, proba), 3),
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-file", required=True)
    parser.add_argument("--exam-file", required=True)
    parser.add_argument("--biomarker-file", required=True)
    parser.add_argument("--health-file", required=True)
    parser.add_argument("--surveys-file", required=True,
                         help="surveys_pub_12.sas7bdat or equivalent - REQUIRED, this is the only file with a direct 'age' "
                              "column (confirmed: age is not computed from birth year/interview date, it's a real field "
                              "here). Without this, every age-gated disease check (diabetes, hypertension, CKD) silently "
                              "finds 0 usable rows - this exact bug was caught and fixed after a real run showed only "
                              "NAFLD (the one disease check that doesn't require age) producing results.")
    parser.add_argument("--wave", default="2009")
    parser.add_argument("--followup-health-file", default=None, help="Optional - a later wave's health module, for prospective/incident validation")
    parser.add_argument("--followup-wave", default=None)
    args = parser.parse_args()

    print("Loading CHNS files locally (never leaves your machine)...")
    demo = load_chns_file(args.demo_file)
    exam = load_chns_file(args.exam_file)
    biomarker = load_chns_file(args.biomarker_file)
    health = load_chns_file(args.health_file)
    surveys = load_chns_file(args.surveys_file)

    print("Merging per-module files on individual ID...")
    merged = merge_chns_modules(demo, exam, biomarker, health)

    print("Merging in age from the surveys file (ID+WAVE)...")
    surveys_id = find_id_col(surveys)
    surveys_wave = _find_wave_col(surveys)
    merged_id = find_id_col(merged)
    merged_wave = _find_wave_col(merged)
    if surveys_wave is None or merged_wave is None:
        raise ValueError("Could not find a wave column in the surveys file or the already-merged data - "
                          "age cannot be safely merged without matching on wave too (same ID+WAVE discipline "
                          "used everywhere else in this script).")
    before_age_merge = len(merged)
    merged = merged.merge(
        surveys[[surveys_id, surveys_wave, "age"]],
        left_on=[merged_id, merged_wave], right_on=[surveys_id, surveys_wave],
        how="left", suffixes=("", "_surveys"),
    )
    print(f"  Merged surveys file (ID+WAVE) for age: {before_age_merge} rows before -> {len(merged)} rows after "
          f"(should be unchanged - if not, something matched more than once per row, investigate before trusting results)")
    print(f"  age: {merged['age'].notna().sum()} non-null values out of {len(merged)} rows after this merge")

    # CRITICAL FIX: restrict to the target wave BEFORE any validation runs.
    # pexam_00 (and hlth_12, surveys_pub_12) span multiple survey years
    # (pexam_00: 1989-2011), not just the target wave. The diabetes/
    # hypertension checks only require age + an outcome field, NOT
    # biomarker data - so without this filter, they silently pull in
    # every year from 1989-2011, not the 2009 cross-section the
    # methodology describes. Caught after a real run showed an
    # implausibly young mean age (36.8, SD 20.9 - consistent with
    # children being included) and implausibly low diabetes prevalence
    # (2.1% vs NHANES's 16.4%) for what was meant to be a same-year
    # comparison. Only ~9,463 of the ~67,000-84,000 "usable rows" in that
    # run actually had real 2009 biomarker values - the rest were relying
    # on median-imputed lab predictors for a population that was never
    # actually measured in 2009 at all.
    wave_col = _find_wave_col(merged)
    target_wave = float(args.wave)
    before_wave_filter = len(merged)
    merged = merged[merged[wave_col] == target_wave].copy()
    print(f"\n  WAVE FILTER: restricting to wave={args.wave} only - {before_wave_filter} rows before -> "
          f"{len(merged)} rows after (this should now be close to the biomarker file's own row count, "
          f"not a much larger multi-year total)")

    # DIAGNOSTIC: confirms whether each key column actually survived the
    # merge intact, or got silently renamed (e.g. a name collision during
    # merge forcing "CRE_MG" to become "CRE_MG_biomarker") - added after a
    # real run showed CKD being skipped despite CRE_MG being independently
    # confirmed present in the raw biomarker file. Prints exact non-null
    # counts, not just whether the column exists, and flags any column
    # whose name merely CONTAINS one of these strings (catching a rename)
    # even if the exact expected name is gone.
    print("\n=== DIAGNOSTIC: checking key columns survived the merge ===")
    for expected_col in ["CRE_MG", "UA_MG", "U24A", "U22", "HEIGHT", "WEIGHT", "U10", "ALT", "age"]:
        if expected_col in merged.columns:
            n_nonnull = merged[expected_col].notna().sum()
            print(f"  {expected_col}: FOUND, {n_nonnull} non-null values out of {len(merged)} rows")
        else:
            similar = [c for c in merged.columns if expected_col in c]
            if similar:
                print(f"  {expected_col}: NOT found under this exact name - but found similar column(s) {similar} "
                      f"(likely renamed during merge due to a name collision)")
            else:
                print(f"  {expected_col}: NOT FOUND anywhere in the merged data, not even under a renamed variant")
    print("=== END DIAGNOSTIC ===\n")

    print("Mapping CHNS variables to Metis features - CHECK EVERY TODO ABOVE if anything looks wrong...")
    mapped = map_chns_to_metis_features(merged)

    print("\n=== CROSS-SECTIONAL VALIDATION (one wave) ===")
    cross_results = run_cross_sectional_validation(mapped)
    print("These are the ONLY numbers you should share back - never the underlying data.")
    print(cross_results.to_string(index=False) if len(cross_results) else "No results - check TODO mappings above.")

    # COHORT SUMMARY STATISTICS - for COHORT_COMPARISON.md (TRIPOD+AI item
    # 20c, the development-vs-evaluation population comparison). All
    # aggregate (mean/SD/%), never row-level - same "only share these
    # numbers back" discipline as the disease results above.
    print("\n=== COHORT SUMMARY STATISTICS (for COHORT_COMPARISON.md) ===")
    print("These are aggregate statistics only - safe to share back, same as the disease results above.\n")
    has_age = mapped["age"].notna()
    print(f"n (has age): {has_age.sum()}")
    print(f"Age, mean (SD): {mapped['age'].mean():.1f} ({mapped['age'].std():.1f})")
    print(f"BMI, mean (SD): {mapped['bmi'].mean():.1f} ({mapped['bmi'].std():.1f})")
    print(f"Male, %: {mapped['gender_male'].mean()*100:.1f}")
    print(f"Waist circumference (cm), mean (SD): {mapped['waist_cm'].mean():.1f} ({mapped['waist_cm'].std():.1f})")
    print(f"Systolic BP, mean (SD): {mapped['bp_systolic'].mean():.1f} ({mapped['bp_systolic'].std():.1f})")
    print(f"Diastolic BP, mean (SD): {mapped['bp_diastolic'].mean():.1f} ({mapped['bp_diastolic'].std():.1f})")
    diabetes_valid = mapped["diabetes_diagnosed"].notna()
    print(f"Diabetes prevalence, % (n valid): {mapped.loc[diabetes_valid, 'diabetes_diagnosed'].mean()*100:.1f} ({diabetes_valid.sum()})")
    hibp_valid = mapped["hibp_diagnosed"].notna()
    print(f"Hypertension prevalence, % (n valid): {mapped.loc[hibp_valid, 'hibp_diagnosed'].mean()*100:.1f} ({hibp_valid.sum()})")
    print("=== END COHORT SUMMARY ===\n")

    if args.followup_health_file:
        print(f"\n=== PROSPECTIVE / INCIDENT VALIDATION ({args.wave} -> {args.followup_wave}) ===")
        followup_health = load_chns_file(args.followup_health_file)
        prospective_results = run_prospective_validation(mapped, followup_health, args.wave, args.followup_wave)
        print(prospective_results.to_string(index=False) if len(prospective_results) else "No results - check TODO mappings above.")


if __name__ == "__main__":
    main()
