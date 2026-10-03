# Metis — Model Cards

Standardized documentation for all 5 production disease-risk models, consolidating the training (`train_model.py`, `train_multi_disease.py`), cross-model validation (`validate_models.py`, `validate_multi_disease.py`), calibration (`calibration_analysis.py`), subgroup fairness (`fairness_analysis.py`), and journal-readiness rigor work (`rigorous_validation.py`, `benchmark_score.py`) into one citable reference per model.

**A note on numbers that move slightly between documents in this project's history:** rebuilding the merged NHANES dataset (`data_prep_2021_2023.py`) — done repeatedly while adding new data sources over the course of this project — can shift row order in the underlying file, and `train_test_split`'s reproducibility depends on row order as well as row content. This was found and fixed by retraining all 5 production models fresh against the final, current data file; the numbers below are from that retrain and are internally consistent with the current `metrics.json` files. A small (~0.01 AUC) residual difference between the production training script's own reported number and other validation scripts' independently-reconstructed test splits remains (a minor implementation difference between two loader functions, not a data integrity issue) and is reported explicitly where relevant rather than hidden.

---

## Intended Use (applies to all 5 models)

**Metis is a clinical decision-support tool, not a diagnostic device.** Each model estimates risk for one metabolic condition from routinely available anthropometric, laboratory, and lifestyle data, to help identify people who may benefit from further clinical evaluation. A Metis risk estimate does not confirm or rule out any condition and is not intended to be the sole basis for a clinical decision. Results should be interpreted by a qualified healthcare professional alongside clinical judgment and confirmatory diagnostic testing.

**Validation status: internal only.** All performance figures below come from a held-out test split of the same NHANES (August 2021–August 2023) cross-sectional survey data used for training. This confirms the models generalize to unseen rows from the same data collection wave — it does **not** confirm that a person flagged high-risk today will go on to develop that condition. That claim requires prospective longitudinal validation, which has not yet been done (see main README's "External longitudinal validation" section).

---

## 1. Type 2 Diabetes

- **Architecture:** Stacked ensemble (monotonic XGBoost + monotonic HistGradientBoosting + Logistic Regression, combined by a meta-learner via 5-fold out-of-fold stacking)
- **Label:** Self-reported diagnosis OR fasting glucose ≥126 mg/dL OR HbA1c ≥6.5%
- **Excluded predictors:** `fasting_glucose`, `hba1c` (define the label), `sugar_g`, `drinks_per_day` (reverse-causation confound found via marginal-effect testing)
- **Test performance:** ROC-AUC 0.871 (production training script's own held-out split), 16.4% prevalence. **5-fold CV: 0.860 ± 0.010** (range 0.850–0.878) — confirms the single-split number is stable and reproducible, not a lucky draw.
- **Survey-weighted estimate:** unweighted prevalence 16.4% vs. **weighted (population-representative) prevalence 14.1%** — the sample modestly over-represents higher-risk people relative to the general population. Weighted AUC 0.880 vs. unweighted 0.862 on the same test rows with a survey weight (n=659 of 1,137) — performance holds up, arguably slightly better, once correctly weighted.
- **Benchmark:** head-to-head against an adapted FINDRISC-style score (6 of 8 original components available in this dataset — family history and fruit/vegetable intake are not available in this NHANES cycle and are omitted, not approximated) on the identical test cohort: **Metis 0.859 vs. adapted FINDRISC 0.751** — a substantial, same-population win even against a deliberately handicapped comparator. See `benchmark_score.py` for exactly which components are included/excluded and why.
- **Calibration:** Already well-calibrated as-is (Brier 0.109, ECE 0.047) — recalibration tested and found **not** to help; no calibrator applied.
- **Subgroup fairness:** Max gap 0.088 AUC. **Known limitation:** meaningfully worse for ages 60+ and for lower-income respondents — disclose this when using the model for older or lower-income patients specifically.
- **Cross-model validation:** 7 model families tested (see `VALIDATION_REPORT.md`); production AUC competitive with all of them.

## 2. Metabolic Syndrome

- **Architecture:** Single monotonic XGBoost
- **Label:** NCEP ATP III — 3+ of: abdominal obesity (sex-specific waist threshold), triglycerides ≥150, low HDL (sex-specific), BP ≥130/85, fasting glucose ≥100
- **Excluded predictors:** `waist_cm`, `waist_height_ratio`, `triglycerides`, `hdl`, `bp_systolic`, `bp_diastolic`, `pulse_pressure`, `insulin`, `homa_ir` (all are the label's own defining criteria or tightly glucose-coupled)
- **Test performance:** ROC-AUC 0.853, n=1,057 test rows, 20.8% prevalence. **5-fold CV: 0.849 ± 0.004** (range 0.844–0.854) — the most stable of all 5 models across folds.
- **Survey-weighted estimate:** unweighted prevalence 20.8% vs. **weighted prevalence 29.3%** — the single largest prevalence correction of any of the 5 diseases; the true population rate may be substantially higher than the naive sample suggests. Weighted AUC 0.823 vs. unweighted 0.853 (n=633 of 1,057 with a survey weight) — a real but modest discrimination cost once correctly weighted, still solid.
- **Calibration:** **Meaningfully miscalibrated as trained** (Brier 0.155, ECE 0.144) — isotonic recalibration applied in production, improving to Brier 0.123, ECE 0.044.
- **Subgroup fairness:** Max gap only 0.028 AUC — the most consistent of all 5 models across age, sex, and income.
- **Cross-model validation:** Best of 5 models tested was HistGradientBoosting (0.855) — production XGBoost (0.853) within 0.001, not a meaningful gap.

## 3. Fatty Liver Disease (NAFLD)

- **Architecture:** Single monotonic XGBoost
- **Label:** Liver elastography CAP score ≥248 dB/m (standard steatosis threshold). Self-reported "fatty liver" diagnosis was checked as an alternative label input and rejected — only 15 positive responses in the whole dataset, too sparse to use.
- **Excluded predictors:** `liver_fat_cap`, `liver_stiffness` (the label itself). ALT/AST/GGT are **kept** as predictors — they're correlated liver-stress markers, not diagnostic criteria (NAFLD is defined by imaging finding fat in the liver, not by enzyme levels; many real NAFLD cases have normal enzymes).
- **Test performance:** ROC-AUC 0.831, n=1,075 test rows, 54.5% prevalence (elevated vs. commonly-cited 25-30% NAFLD prevalence figures — expected, since CAP-based elastography detects steatosis more sensitively than the ultrasound-based diagnoses those commonly-cited figures come from). **5-fold CV: 0.842 ± 0.005** (range 0.837–0.851).
- **Survey-weighted estimate:** unweighted prevalence 54.5% vs. **weighted prevalence 51.9%** — close, a modest correction. Weighted AUC 0.862 vs. unweighted 0.831 (n=592 of 1,075 with a survey weight) — performance holds up well, arguably better, once weighted.
- **Calibration:** Already well-calibrated (Brier 0.175, ECE 0.069); recalibration tested, doesn't help, not applied.
- **Subgroup fairness:** Max gap 0.029 AUC — consistent across subgroups, though slightly weaker for ages 60+ (0.802 vs. 0.831 overall).
- **Cross-model validation:** Production XGBoost is the **best of all 5 models tested** (0.831) — not just competitive, outright the top performer.

## 4. Chronic Kidney Disease (CKD)

- **Architecture:** Single monotonic XGBoost
- **Label:** eGFR <60 (CKD-EPI 2021 race-free equation, from serum creatinine) OR self-reported diagnosis (`KIQ022`)
- **Excluded predictors:** `creatinine`, `bun`, `egfr` (the label itself). `sleep_hours` constrained to monotonic-decreasing (not left unconstrained like other diseases) after a real regression was caught during simulator testing — see `train_multi_disease.py`'s comments.
- **Test performance:** ROC-AUC 0.845, n=1,127 test rows, 8.8% prevalence. **5-fold CV: 0.839 ± 0.021** (range 0.811–0.869) — the widest fold-to-fold spread of any of the 5 models, expected given CKD's low prevalence means each fold's positive-case count is small and more variable.
- **Survey-weighted estimate:** unweighted prevalence 8.8% vs. **weighted prevalence 5.2%** — the sample notably over-represents CKD cases relative to the general population. Weighted AUC 0.904 vs. unweighted 0.845 (n=648 of 1,127 with a survey weight) — **the single biggest weighted-AUC improvement of any of the 5 diseases**, a genuinely encouraging finding once the analysis is done correctly.
- **Calibration:** **Severely miscalibrated as trained** — the worst of all 5 models. Raw output claimed up to 80% risk for a group where only ~37% actually had the condition (Brier 0.127, ECE 0.190). Isotonic recalibration applied in production, a dramatic fix: Brier 0.064, ECE 0.020. **This was the single most important finding across all of the refinement work** — low-prevalence outcomes are a well-documented failure mode for tree-based model calibration, and this confirms it happened here.
- **Subgroup fairness:** **Largest gap of all 5 models** — 0.114 AUC. Notably worse for ages 40-59 (AUC 0.740 vs. 0.854 overall). Age 18-39 band has only 1 positive case in the test split — too few to score at all, not just a gap finding.
- **Cross-model validation:** Production XGBoost is the **best of all 5 models tested** (0.845).

## 5. Hypertension

- **Architecture:** Single monotonic XGBoost
- **Label:** Measured BP ≥130/80 (2017 ACC/AHA Stage 1+) OR self-reported diagnosis (`BPQ020`)
- **Excluded predictors:** `bp_systolic`, `bp_diastolic`, `pulse_pressure` (the label itself)
- **Test performance:** ROC-AUC 0.787, n=1,137 test rows, 54.6% prevalence — **the lowest-performing of the 5 models**, consistent with it being flagged as the hardest target when these diseases were first scoped. **5-fold CV: 0.787 ± 0.015** (range 0.768–0.800) — exactly matches the single-split estimate.
- **Survey-weighted estimate:** unweighted prevalence 54.6% vs. **weighted prevalence 47.6%**. Weighted AUC 0.774 vs. unweighted 0.787 (n=622 of 1,137 with a survey weight) — close, a small and expected amount of movement.
- **Calibration:** Already reasonably calibrated (Brier 0.172, ECE 0.044); recalibration tested, doesn't help, not applied.
- **Subgroup fairness:** **Second-largest gap** — 0.111 AUC. Notably worse for ages 40-59 (0.683) and 60+ (0.676) than for ages 18-39 (0.775) — a real concern given older adults are a primary hypertension-screening population in practice.
- **Cross-model validation:** Best of 5 models tested was Gradient Boosting (0.792) — production XGBoost (0.787) within 0.005, not a meaningful gap.

---

## Cross-cutting findings worth flagging regardless of which model is in use

1. **Calibration failures were real and disease-specific, not predictable from discrimination (AUC) alone.** CKD had strong discrimination (0.845 AUC) but was badly miscalibrated; NAFLD had good discrimination and was already well-calibrated. Checking AUC alone would have missed the CKD finding entirely — calibration must be checked independently, not inferred from AUC.
2. **Age 40-59 and 60+ show the weakest, most consistent subgroup performance gaps** across CKD and Hypertension specifically (the two lowest-overall-AUC models) — worth investigating further if this tool is extended, since older adults are a primary real-world screening population for both conditions.
3. **A silent-certainty bug was caught and fixed during this work**: the Metabolic Syndrome calibrator's isotonic fit produced an exact 1.0 (100% risk) for some high-input values — a small-sample tail artifact, not a real finding. All 5 models now hard-clip output to [0.01, 0.99] — no model may ever claim mathematical certainty. See `diseases.py`/`model.py` comments.
4. **5-fold cross-validation confirms every single-split AUC estimate is stable, not a lucky draw** — all 5 models' CV means land within 0.01-0.02 of their single-split number, with CKD showing the widest (still modest) fold-to-fold spread, expected given its low prevalence.
5. **Survey-weighted re-analysis moved several numbers meaningfully, in both directions** — Metabolic Syndrome's true population prevalence may be ~8.5 percentage points higher than the naive sample suggests; CKD's may be ~3.6 points lower, with a correspondingly large weighted-AUC improvement (0.845→0.904). This is a real, substantive correction a clinical/epi audience would expect to see done, not a formality.
6. **Benchmarked against an adapted FINDRISC-style score on the identical NHANES test cohort** (not citing FINDRISC's own published AUC from a different population, which would not be a fair comparison): Metis's diabetes model (0.859) beats the adapted score (0.751) by a wide margin, even though the adapted score is missing 2 of 8 original components and is therefore a deliberately handicapped comparator.
7. **None of this is prospective validation.** Every number in this document, including the weighted and cross-validated ones, comes from the same cross-sectional NHANES data collection wave. See the main README's "External longitudinal validation" section for what prospective validation requires and which datasets are realistic candidates.

## External validation (CHNS) — FINAL RESULTS

Genuine external validation — a different country's population, not another split of the same NHANES data — using the China Health and Nutrition Survey, 2009 wave. Full methodology, exact variable mappings (and which are codebook-confirmed vs. evidence-based), and known limitations are documented in `external_validation_chns.py`'s module docstring. Two real bugs were caught and fixed during this process before these numbers can be trusted: a multi-wave merge bug (would have cross-matched different years' data for the same person) and a missing age-merge step (silently zeroed out every age-gated disease check until fixed) — both verified fixed via an explicit diagnostic block now built into the script.

| Disease | Internal (NHANES) | External (CHNS, 2009 only) | n | Direction |
|---|---|---|---|---|
| Type 2 Diabetes | ~0.86-0.87 | **0.739** | 10,624 | Lower, as expected for cross-cohort validation — still well above chance on a different country's population |
| Hypertension | 0.787 | **0.776** | 10,622 | Essentially comparable to internal, very slightly lower — the expected pattern for honest external validation |
| Chronic Kidney Disease | 0.845 | **0.858** | 9,463 | Higher than internal, using real creatinine-based eGFR (not a proxy) — clean, credible |
| NAFLD | 0.831 | 0.975 raw / 0.946 toy-baseline (n=9,142) | — | **Flagged, do not report as-is** — see below |

**A real data-integrity bug was caught and fixed during this process, worth keeping in the methods section.** `pexam_00` spans multiple survey years (1989-2011), not just 2009. The diabetes/hypertension validity checks only required age + outcome fields to be present — not biomarker data — so an earlier run silently pulled in people from every wave in that range rather than the intended 2009 cross-section (reported then: Diabetes 0.761 n=66,839, Hypertension 0.811 n=84,679 — both incorrect, now superseded by the correctly wave-restricted numbers above). This was caught when a cohort summary showed an implausible mean age (36.8, SD 20.9) and implausible diabetes prevalence (2.1%) for what was meant to be a same-year comparison with NHANES. Fixed with an explicit wave filter applied immediately after merging, before any validation logic runs - verified by rerunning and confirming the sample size dropped to a plausible ~12,000 (close to the biomarker file's own 9,549 rows) with a correspondingly more plausible age distribution. CKD and NAFLD were never affected, since their own validity gates already implicitly required 2009-biomarker-only fields (confirmed: both produced byte-for-byte identical numbers before and after the fix).

**NAFLD — confirmed confounded, with a quantified sensitivity check, not just a flagged number.** CHNS has no liver elastography, so this uses the Fatty Liver Index (FLI) proxy, built from BMI + waist + triglycerides + GGT — two of which (BMI, waist) are *also* real Metis NAFLD predictors. The sensitivity check: a trivial 2-variable (BMI+waist only) logistic regression, no Metis model involved at all, scores **AUC 0.946** on the exact same label. Metis's full model (26+ features, trained ensemble) only reaches **0.975** — a gap of just **0.03**. A 2-variable toy model getting within 0.03 of the full model is about as clean a demonstration of label-circularity as this kind of check produces.

**Reporting recommendation**: present both numbers together as a methods contribution (showing this was actively checked and quantified, not glossed over), not as a clean external-validation AUC. Suggested framing: *"NAFLD's apparent external AUC (0.975) was substantially attributable to shared predictors between the Fatty Liver Index proxy label and the model (a 2-variable BMI+waist baseline alone achieved AUC 0.946 on the same label), limiting interpretability of this result as true external validation of liver-fat prediction."* This turns a real limitation into demonstrated rigor rather than a quietly-dropped result.

**AST: confirmed absent from this CHNS release, not just unconfirmed.** Checked every remaining unlabeled biomarker column's correlation with ALT (AST and ALT are typically strongly correlated, often r=0.6-0.8+, in general populations — this is how GGT was identified via `Y48_2`'s r=0.848). The highest among the 7 remaining candidates was r=0.209 — nowhere close. Report as a genuine data limitation (AST simply isn't in this biomarker release), not an open question.

**Smoking status**: could not be identified and the search was closed out, not left as an open TODO. Checked `pexam_00` (physical exam), `subi_12` (turned out to be individual subsidy income, not substance use — a reasonable name-based guess that didn't pan out), and `media_00`/`pact_12`/`pstress_12`/`timea_12`/`wages_12` (media use, physical activity, psychological stress, time allocation, wages — none relevant by name or content). Every remaining CHNS file uses opaque alphanumeric question codes (`U551`, `K7B`, `B2D`, etc.) rather than readable names, making further guessing unproductive without the full original-language questionnaire instrument. This does not affect the diabetes, hypertension, or CKD external validation results, nor the NAFLD sensitivity check — none of those required smoking status.

See `COHORT_COMPARISON.md` for the development-vs-evaluation population comparison TRIPOD+AI requires.

## Reproducing this analysis

```bash
cd backend/training
pip install -r requirements-validation.txt
python3 calibration_analysis.py
python3 fairness_analysis.py
python3 rigorous_validation.py      # 5-fold CV + survey-weighted estimates
python3 benchmark_score.py           # head-to-head vs. adapted FINDRISC
```

Outputs: `validation_results/calibration/`, `validation_results/fairness/`, and `validation_results/rigorous/` — per-disease reliability diagrams, subgroup bar charts, CV fold detail, and summary CSVs.
