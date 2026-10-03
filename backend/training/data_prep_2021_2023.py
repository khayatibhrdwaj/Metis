"""
MetaTwin - Data Preparation v2 (NHANES August 2021-August 2023, cycle "L")

Real, current-cycle NHANES data (actual respondent SEQN values, cycle L =
Aug 2021-Aug 2023). Replaces the earlier 2007-2008 pull.

Source files (all real NHANES XPT extracts, merged on SEQN):
  DEMO_L    - demographics (age, gender, poverty ratio)
  BMX_L     - body measures (weight, height, BMI, waist)
  BPXO_L    - oscillometric blood pressure (3 readings, averaged)
  HDL_L     - HDL cholesterol
  TCHOL_L   - total cholesterol
  TRIGLY_L  - triglycerides + LDL (fasting subsample only)
  INS_L     - fasting insulin (fasting subsample only)
  GLU_L     - fasting plasma glucose (fasting subsample only) - LABEL input +
              tracked/simulator-outcome metric, NOT a predictor
  GHB_L     - glycohemoglobin / HbA1c (full sample) - LABEL input +
              tracked/simulator-outcome metric, NOT a predictor
  HSCRP_L   - high-sensitivity CRP (inflammation marker)
  SLQ_L     - sleep duration (weekday/weekend hours)
  PAQ_L     - physical activity (moderate/vigorous minutes, unit-converted to /week)
  ALQ_L     - alcohol consumption
  DR1TOT_L  - 24-hour dietary recall (kcal, added sugar, fiber)
  DIQ_L     - diagnosed diabetes questionnaire - PRIMARY LABEL input
  FASTQX_L  - fasting time (needed to validate fasting-subsample labs)
  BIOPRO_L  - comprehensive metabolic panel: ALT/AST/GGT (liver enzymes -
              independent NAFLD/insulin-resistance markers, distinct from
              the liver elastography CAP/stiffness already used), uric
              acid (established metabolic syndrome marker), albumin,
              creatinine + BUN (kidney function - CKD label input)
  CBC_L     - complete blood count: white blood cell count (inflammation
              marker)
  KIQ_U_L   - kidney questionnaire: self-reported "ever told weak/failing
              kidneys" (KIQ022) - CKD label input, same dual-criteria
              pattern as diabetes (self-report OR lab-confirmed)
  BPQ_L     - blood pressure questionnaire: self-reported "ever told high
              blood pressure" (BPQ020) - hypertension label input, same
              dual-criteria pattern

NHANES uses special placeholder codes for refused/don't-know (commonly
7777/9999 or a tiny denormalized float from SAS missing-value encoding).
Cleaning below drops those to NaN.

============================================================================
MULTI-DISEASE LABELS (added alongside the original diabetes_label)
============================================================================
This file now defines FIVE disease labels from the same merged dataset:

  diabetes_label            - self-report OR fasting glucose>=126 OR HbA1c>=6.5%
  metabolic_syndrome_label  - NCEP ATP III: 3+ of {abdominal obesity (sex-
                               specific waist threshold), triglycerides>=150,
                               low HDL (sex-specific), BP>=130/85, fasting
                               glucose>=100}
  nafld_label                - liver elastography CAP score >=248 dB/m
                               (standard steatosis threshold; self-reported
                               "fatty liver" was checked and rejected as a
                               label input - only 15 positive responses in
                               the whole dataset, too sparse to combine
                               reliably with the elastography measurement)
  ckd_label                   - eGFR<60 (CKD-EPI 2021 race-free equation,
                               from creatinine) OR self-reported diagnosis
                               (KIQ022)
  hypertension_label           - measured BP>=130/80 (2017 ACC/AHA Stage 1+)
                               OR self-reported diagnosis (BPQ020)

Each disease gets its OWN predictor set in train_model.py, excluding
whatever features define THAT disease's own label (e.g. waist/BP/lipids/
glucose are excluded from metabolic syndrome's predictors since they ARE
the label; BP is excluded from hypertension's predictors; creatinine/BUN
are excluded from CKD's predictors) - the same non-circularity discipline
applied to diabetes throughout this project, just applied five times.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw_2021_2023"
PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MISSING_CODES = {7777, 9999, 77777, 99999, 777, 999}


def _clean_missing(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # SAS XPT encodes missing as a tiny denormalized float; treat near-zero
    # subnormal values and NHANES refused/don't-know placeholder codes as NaN
    s = s.mask(s.abs() < 1e-70)
    s = s.mask(s.isin(MISSING_CODES))
    return s


def _read(name):
    return pd.read_sas(RAW_DIR / f"{name}.xpt", format="xport")


def build_dataset() -> pd.DataFrame:
    demo = _read("DEMO_L")[["SEQN", "RIDAGEYR", "RIAGENDR"]].rename(
        columns={"RIDAGEYR": "age", "RIAGENDR": "gender_code"}
    )
    # poverty ratio isn't in this DEMO_L pull's core columns in all releases;
    # guard for its absence
    demo_full = _read("DEMO_L")
    if "INDFMPIR" in demo_full.columns:
        demo["poverty_ratio"] = demo_full["INDFMPIR"]

    bmx = _read("BMX_L")[["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"]].rename(
        columns={"BMXWT": "weight_kg", "BMXHT": "height_cm", "BMXBMI": "bmi", "BMXWAIST": "waist_cm"}
    )

    bpxo = _read("BPXO_L")
    bpxo["bp_systolic"] = bpxo[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    bpxo["bp_diastolic"] = bpxo[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    bpxo = bpxo[["SEQN", "bp_systolic", "bp_diastolic"]]

    hdl = _read("HDL_L")[["SEQN", "LBDHDD"]].rename(columns={"LBDHDD": "hdl"})
    tchol = _read("TCHOL_L")[["SEQN", "LBXTC"]].rename(columns={"LBXTC": "total_cholesterol"})
    trigly = _read("TRIGLY_L")[["SEQN", "LBXTLG", "LBDLDL"]].rename(
        columns={"LBXTLG": "triglycerides", "LBDLDL": "ldl"}
    )
    ins = _read("INS_L")[["SEQN", "LBXIN"]].rename(columns={"LBXIN": "insulin"})
    # WTSAF2YR: NHANES Fasting Subsample 2-Year Weight - the correct survey
    # weight for any analysis using fasting-subsample variables (glucose,
    # insulin, triglycerides, HOMA-IR are all fasting-subsample-only), per
    # NHANES's own analytic guidelines. Used for weighted
    # prevalence/AUC estimates in the journal-readiness validation work -
    # see benchmark_and_weighted_validation.py.
    glu = _read("GLU_L")[["SEQN", "LBXGLU", "WTSAF2YR"]].rename(
        columns={"LBXGLU": "fasting_glucose", "WTSAF2YR": "survey_weight"}
    )
    ghb = _read("GHB_L")[["SEQN", "LBXGH"]].rename(columns={"LBXGH": "hba1c"})
    crp = _read("HSCRP_L")[["SEQN", "LBXHSCRP"]].rename(columns={"LBXHSCRP": "crp"})

    fastqx = _read("FASTQX_L")[["SEQN", "PHAFSTHR"]].rename(columns={"PHAFSTHR": "fasting_hours"})
    fastqx["fasting_hours"] = _clean_missing(fastqx["fasting_hours"])

    # --- Sleep (NHANES already derives hours; average weekday*5 + weekend*2) ---
    slq = _read("SLQ_L")[["SEQN", "SLD012", "SLD013"]]
    slq["sleep_hours"] = (slq["SLD012"] * 5 + slq["SLD013"] * 2) / 7

    # --- Physical activity: unit-normalize frequency to per-week, weight
    # vigorous activity 2x per CDC guidance, sum moderate+vigorous minutes/week ---
    paq = _read("PAQ_L")[["SEQN", "PAD790Q", "PAD790U", "PAD800", "PAD810Q", "PAD810U", "PAD820"]].copy()
    unit_to_weekly = {b"D": 7, b"W": 1, b"M": 12 / 52, b"Y": 1 / 52}

    def weekly_minutes(freq_col, unit_col, dur_col, df):
        freq = _clean_missing(df[freq_col])
        dur = _clean_missing(df[dur_col])
        factor = df[unit_col].map(unit_to_weekly)
        sessions_per_week = freq * factor
        return (sessions_per_week * dur).clip(upper=1680)  # cap at 4h/day sanity bound

    mod_min = weekly_minutes("PAD790Q", "PAD790U", "PAD800", paq)
    vig_min = weekly_minutes("PAD810Q", "PAD810U", "PAD820", paq)
    paq["activity_min_per_week"] = mod_min.fillna(0) + 2 * vig_min.fillna(0)
    paq.loc[mod_min.isna() & vig_min.isna(), "activity_min_per_week"] = np.nan
    paq = paq[["SEQN", "activity_min_per_week"]]

    alq = _read("ALQ_L")[["SEQN", "ALQ130"]].rename(columns={"ALQ130": "drinks_per_day"})
    alq["drinks_per_day"] = _clean_missing(alq["drinks_per_day"])

    diet = _read("DR1TOT_L")[["SEQN", "DR1TKCAL", "DR1TSUGR", "DR1TFIBE"]].rename(
        columns={"DR1TKCAL": "kcal", "DR1TSUGR": "sugar_g", "DR1TFIBE": "fiber_g"}
    )

    diq = _read("DIQ_L")[["SEQN", "DIQ010"]]

    # --- Additional metabolic biomarkers (legitimate, non-diagnostic-circular
    # independent risk factors that weren't in the original feature set) ---
    bio = _read("BIOPRO_L")[["SEQN", "LBXSATSI", "LBXSASSI", "LBXSGTSI", "LBXSUA", "LBXSAL",
                              "LBXSCR", "LBXSBU"]].rename(
        columns={"LBXSATSI": "alt", "LBXSASSI": "ast", "LBXSGTSI": "ggt",
                 "LBXSUA": "uric_acid", "LBXSAL": "albumin",
                 "LBXSCR": "creatinine", "LBXSBU": "bun"}
    )
    for c in ["alt", "ast", "ggt", "uric_acid", "albumin", "creatinine", "bun"]:
        bio[c] = _clean_missing(bio[c])

    cbc = _read("CBC_L")[["SEQN", "LBXWBCSI"]].rename(columns={"LBXWBCSI": "wbc"})
    cbc["wbc"] = _clean_missing(cbc["wbc"])

    # --- Kidney self-report (KIQ022: "ever told weak/failing kidneys") ---
    kiq = _read("KIQ_U_L")[["SEQN", "KIQ022"]]

    # --- Blood pressure self-report (BPQ020: "ever told high blood pressure") ---
    bpq = _read("BPQ_L")[["SEQN", "BPQ020"]]

    # --- Smoking: current smoker flag (independent risk factor, not a
    # downstream consequence of diabetes, so safe to use as a predictor) ---
    smq = _read("SMQ_L")[["SEQN", "SMQ040"]]
    smq["smoker_current"] = smq["SMQ040"].isin([1, 2]).astype(int)
    smq = smq[["SEQN", "smoker_current"]]

    # --- Liver elastography: CAP (steatosis/fatty liver) and stiffness
    # (fibrosis). NAFLD is mechanistically upstream of insulin resistance,
    # so this is a legitimate independent risk factor, not a diagnostic
    # circularity like using glucose/HbA1c would be. ---
    lux = _read("LUX_L")[["SEQN", "LUXCAPM", "LUXSMED"]].rename(
        columns={"LUXCAPM": "liver_fat_cap", "LUXSMED": "liver_stiffness"}
    )

    # --- Occupation: physical demand of usual work (OCQ215, ordinal 1-5:
    # roughly sedentary -> heavy labor). Independent lifestyle-adjacent risk
    # factor, not a diagnosis-derived one. ---
    ocq = _read("OCQ_L")[["SEQN", "OCQ215"]].rename(columns={"OCQ215": "occ_activity_level"})
    ocq["occ_activity_level"] = _clean_missing(ocq["occ_activity_level"])

    # --- Stress/mood screening (PHQ-9 sum, standard 0-27 scale). This is a
    # validated screening instrument, not a diagnosis - kept as a plain
    # numeric covariate. Values 7/9 (refused/don't know) per item cleaned
    # to NaN before summing. ---
    dpq = _read("DPQ_L")
    dpq_items = [c for c in dpq.columns if c.startswith("DPQ0")]
    for c in dpq_items:
        dpq[c] = _clean_missing(dpq[c]).where(lambda s: s <= 3)
    dpq["phq9_score"] = dpq[dpq_items].sum(axis=1, min_count=1)
    dpq = dpq[["SEQN", "phq9_score"]]

    # --- Sedentary time (min/day), from the same PAQ_L file already loaded above ---
    paq_raw = _read("PAQ_L")[["SEQN", "PAD680"]].rename(columns={"PAD680": "sedentary_min_per_day"})
    paq_raw["sedentary_min_per_day"] = _clean_missing(paq_raw["sedentary_min_per_day"]).clip(upper=1440)

    # --- Merge: core (large N) tables first ---
    df = demo.merge(bmx, on="SEQN", how="inner") \
              .merge(bpxo, on="SEQN", how="inner") \
              .merge(hdl, on="SEQN", how="left") \
              .merge(tchol, on="SEQN", how="left") \
              .merge(crp, on="SEQN", how="left") \
              .merge(slq, on="SEQN", how="left")[
                  ["SEQN", "age", "gender_code", "poverty_ratio", "weight_kg", "height_cm",
                   "bmi", "waist_cm", "bp_systolic", "bp_diastolic", "hdl",
                   "total_cholesterol", "crp", "sleep_hours"]
              ] \
              .merge(paq, on="SEQN", how="left") \
              .merge(alq, on="SEQN", how="left") \
              .merge(diet, on="SEQN", how="left") \
              .merge(diq, on="SEQN", how="left") \
              .merge(trigly, on="SEQN", how="left") \
              .merge(ins, on="SEQN", how="left") \
              .merge(glu, on="SEQN", how="left") \
              .merge(ghb, on="SEQN", how="left") \
              .merge(fastqx, on="SEQN", how="left") \
              .merge(smq, on="SEQN", how="left") \
              .merge(lux, on="SEQN", how="left") \
              .merge(ocq, on="SEQN", how="left") \
              .merge(dpq, on="SEQN", how="left") \
              .merge(paq_raw, on="SEQN", how="left") \
              .merge(bio, on="SEQN", how="left") \
              .merge(cbc, on="SEQN", how="left") \
              .merge(kiq, on="SEQN", how="left") \
              .merge(bpq, on="SEQN", how="left")

    # --- Filters ---
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    df = df[(df["bmi"] > 12) & (df["bmi"] < 80)]
    df = df.dropna(subset=["bmi", "waist_cm", "bp_systolic", "age"])

    # --- Label: diagnosed diabetes (DIQ010==1) OR fasting-confirmed via
    # glucose (>=126 mg/dL with >=8h fast) OR lab-confirmed via HbA1c
    # (>=6.5%, the standard ADA diagnostic threshold - HbA1c doesn't
    # require fasting, so no fasting_hours gate needed for it). Using both
    # glucose and HbA1c to DEFINE the label is legitimate (same treatment
    # NHANES's own diabetes prevalence estimates use); the circularity risk
    # this project has been careful about is specifically using them as
    # model PREDICTORS, which they are not - see train_model.py/model.py's
    # FEATURES list, which excludes both. Drop ambiguous "borderline"(3)/
    # "don't know"(9) self-report rows so the label stays clean.
    df = df[~df["DIQ010"].isin([3, 9])]
    lab_confirmed_glucose = (df["fasting_glucose"] >= 126) & (df["fasting_hours"] >= 8)
    lab_confirmed_hba1c = df["hba1c"] >= 6.5
    df["diabetes_label"] = (
        (df["DIQ010"] == 1) | lab_confirmed_glucose | lab_confirmed_hba1c
    ).astype(int)

    df["gender"] = df["gender_code"].map({1: "Male", 2: "Female"})
    df["waist_height_ratio"] = df["waist_cm"] / df["height_cm"]
    df["pulse_pressure"] = df["bp_systolic"] - df["bp_diastolic"]
    df["homa_ir"] = np.where(
        (df["fasting_hours"] >= 8) & df["insulin"].notna() & df["fasting_glucose"].notna(),
        (df["insulin"] * df["fasting_glucose"]) / 405,
        np.nan,
    )

    # =========================================================================
    # METABOLIC SYNDROME LABEL (NCEP ATP III: 3+ of 5 criteria)
    # =========================================================================
    is_male = df["gender"] == "Male"
    crit_waist = np.where(is_male, df["waist_cm"] >= 102, df["waist_cm"] >= 88)
    crit_trig = df["triglycerides"] >= 150
    crit_hdl = np.where(is_male, df["hdl"] < 40, df["hdl"] < 50)
    crit_bp = (df["bp_systolic"] >= 130) | (df["bp_diastolic"] >= 85)
    crit_glucose = df["fasting_glucose"] >= 100
    # Count only criteria with non-missing data; require at least 3 of the
    # 5 to have been evaluable at all, else the label itself is unreliable
    crit_matrix = np.vstack([
        np.where(df["waist_cm"].notna(), crit_waist, np.nan).astype(float),
        np.where(df["triglycerides"].notna(), crit_trig, np.nan).astype(float),
        np.where(df["hdl"].notna(), crit_hdl, np.nan).astype(float),
        np.where(df["bp_systolic"].notna(), crit_bp, np.nan).astype(float),
        np.where(df["fasting_glucose"].notna(), crit_glucose, np.nan).astype(float),
    ])
    n_evaluable = (~np.isnan(crit_matrix)).sum(axis=0)
    n_met = np.nansum(crit_matrix, axis=0)
    df["metabolic_syndrome_label"] = np.where(n_evaluable >= 3, (n_met >= 3).astype(int), np.nan)

    # =========================================================================
    # NAFLD LABEL (liver elastography CAP score >= 248 dB/m - standard
    # steatosis threshold; self-reported "fatty liver" checked and rejected,
    # see module docstring)
    # =========================================================================
    df["nafld_label"] = np.where(df["liver_fat_cap"].notna(), (df["liver_fat_cap"] >= 248).astype(int), np.nan)

    # =========================================================================
    # CKD LABEL (eGFR<60 via CKD-EPI 2021 race-free equation, OR self-report)
    # =========================================================================
    scr = df["creatinine"]
    kappa = np.where(is_male, 0.9, 0.7)
    alpha = np.where(is_male, -0.302, -0.241)
    female_bonus = np.where(is_male, 1.0, 1.012)
    min_ratio = np.minimum(scr / kappa, 1.0)
    max_ratio = np.maximum(scr / kappa, 1.0)
    egfr = 142 * (min_ratio ** alpha) * (max_ratio ** -1.200) * (0.9938 ** df["age"]) * female_bonus
    df["egfr"] = np.where(scr.notna(), egfr, np.nan)
    # Filter ambiguous self-report rows FIRST, then compute the label from
    # the already-filtered df - computing these Series before the filter
    # and using them after would silently create a length/index mismatch.
    df = df[~df["KIQ022"].isin([7, 9])]  # drop refused/don't-know self-report rows
    ckd_lab_confirmed = df["egfr"] < 60
    ckd_self_report = df["KIQ022"] == 1
    df["ckd_label"] = np.where(
        df["egfr"].notna() | df["KIQ022"].notna(),
        ((ckd_lab_confirmed.fillna(False)) | (ckd_self_report.fillna(False))).astype(int),
        np.nan,
    )

    # =========================================================================
    # HYPERTENSION LABEL (measured BP>=130/80, 2017 ACC/AHA, OR self-report)
    # =========================================================================
    df = df[~df["BPQ020"].isin([7, 9])]
    htn_measured = (df["bp_systolic"] >= 130) | (df["bp_diastolic"] >= 80)
    htn_self_report = df["BPQ020"] == 1
    df["hypertension_label"] = (htn_measured | htn_self_report.fillna(False)).astype(int)

    df = df.drop(columns=["DIQ010", "gender_code", "KIQ022", "BPQ020"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_dataset()
    out_path = PROCESSED_DIR / "nhanes_2021_2023_merged.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} cols -> {out_path}")
    print(df["diabetes_label"].value_counts(normalize=True))
    print("\nMissingness by column:")
    print((df.isna().mean() * 100).round(1).sort_values(ascending=False))
