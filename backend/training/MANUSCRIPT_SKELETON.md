# Manuscript Skeleton — Metis

Structured against the TRIPOD+AI section order (`TRIPOD_AI_CHECKLIST.md`). Each section names which existing document supplies its content and what's genuinely left to write versus just reformat. `[PENDING: ...]` marks anything waiting on the full CHNS run — search for that tag when the numbers come in.

---

## Title

Needs to identify this as a development study (TRIPOD+AI item 1 requirement). Draft options:

- *"Development and Internal-External Validation of an Explainable Multi-Disease Metabolic Risk Screening Model: A Non-Circular Feature Engineering Approach"*
- *"Metis: A Multi-Disease Clinical Decision-Support Model for Metabolic Risk, Developed on NHANES and Externally Validated in a Chinese Cohort"*

Second option is more concrete and names the external validation up front — probably stronger. Avoid any phrase implying diagnostic or clinical-release readiness (see Intended Use statement, `dashboard.html`) — "screening model" / "decision-support model," never "diagnostic tool."

## Abstract

Write last, per usual practice, but structure now against TRIPOD+AI for Abstracts' 13 items: objective, data source (NHANES/CHNS), participants (n, both cohorts), outcome(s) (5 diseases, exact definitions in one sentence each), predictors (anthropometric/lab/lifestyle, explicitly non-diagnostic), model type (ensemble + monotonic XGBoost), internal validation (AUC range + CV), external validation (`[PENDING: final CHNS AUCs, all 4 diseases]`), limitations (1-2 sentences), conclusion.

## 1. Introduction

**Source material**: this entire project's chat history is the actual rationale — condense, don't re-derive.

- **Background**: 5 related metabolic conditions share risk factors; most existing tools (ADA Risk Test, FINDRISC, QRISK3) target one condition each
- **The circularity problem**: motivate with the concrete example already documented — glucose/HbA1c reaching ~1.0 AUC when used as diabetes predictors, correctly identified as circular and excluded project-wide
- **Objective** (TRIPOD+AI item 4, verified wording: *"specify the study objectives, including whether the study describes the development or validation of a prediction model (or both)"*): state plainly — this is a model **development** study with internal validation (5-fold CV, calibration, fairness) **and** external validation (CHNS)

## 2. Methods

### 2.1 Data sources (item 5)
NHANES August 2021-August 2023 (development); CHNS 2009 wave (external evaluation). Source: README's data section, `data_prep_2021_2023.py` docstring.

### 2.2 Participants (item 6)
NHANES: n=5,683 after label-availability filtering. CHNS: `[PENDING: final n after full merge/mapping]` — preliminary merge produced 116,763 person-wave rows; final analytic n depends on per-disease outcome availability (diabetes: 66,840 non-missing in the full file).

### 2.3 Outcomes (item 7)
Pull directly from `MODEL_CARDS.md`'s "Intended Use" and per-disease sections — this is the most thoroughly documented material in the whole project. Five outcomes, each with dual-criteria definitions (lab + self-report) where available.

### 2.4 Predictors (item 8)
Pull from `model.py`/`diseases.py` docstrings — explicit per-disease excluded-predictor tables already exist, basically copy-paste into prose form.

### 2.5 Sample size (item 9)
State honestly: no formal a priori power calculation; sample size determined by NHANES's existing cycle size. Note as a limitation, not hidden.

### 2.6 Missing data (item 10)
Native XGBoost/HistGB missing-value handling + median fallback for Logistic Regression component. Source: `NATIVE_MISSING_OK` sets and surrounding comments in training scripts.

### 2.7 Model-building (items 11-12)
Monotonic XGBoost per disease; diabetes additionally uses a stacked ensemble (XGBoost+HistGB+LR, meta-learner via 5-fold out-of-fold stacking). Source: `train_model.py`/`train_multi_disease.py` docstrings.

### 2.8 Internal validation (item 13)
5-fold stratified CV + single held-out split + calibration (isotonic recalibration where needed) + subgroup fairness (age/sex/income). Source: `rigorous_validation.py`, `calibration_analysis.py`, `fairness_analysis.py` — all three docstrings basically are this subsection already.

### 2.9 External validation (item 14)
CHNS methodology — cross-sectional comparison on the 2009 wave. Source: `external_validation_chns.py` docstring, in full. Explicitly name: NAFLD via Fatty Liver Index proxy (not elastography), GGT via evidence-based column identification (not codebook-confirmed) — both need disclosure here, not just in code comments.

### 2.10 Performance measures (item 15)
AUC (discrimination), Brier score + ECE (calibration), subgroup AUC (fairness). Justify each in 1-2 sentences — already justified in the respective script docstrings, just needs translating to manuscript register.

### 2.11 Differences between development and evaluation data (item 16)
Source: `COHORT_COMPARISON.md` directly — this table largely **is** this subsection, once complete.

### 2.12 Model updating (item 17)
State explicitly: model was **not** retrained or tuned based on CHNS results — a deliberate choice avoiding validation-data leakage, not an oversight.

## 3. Results

### 3.1 Participant flow (item 20)
`[PENDING: formal flow diagram/table]` — numbers exist (n at each filtering stage), presentation format doesn't yet.

### 3.2 Distribution comparison (item 20c)
`COHORT_COMPARISON.md` — paste the table directly once complete, `[PENDING: CHNS demographic rows]`.

### 3.3 Model development results (item 21)
Per-disease n/prevalence/feature-count — `MODEL_CARDS.md` table, ready now.

### 3.4 Model performance — internal (item 23)
All 5 diseases' AUC, CV mean±std, Brier/ECE, subgroup gaps — `MODEL_CARDS.md`, ready now. This is genuinely the strongest, most complete section of the whole paper already.

### 3.5 Model performance — external (item 23/24)
**DONE, all 4 diseases, final numbers, nothing pending.** Diabetes AUC 0.739 (n=10,624), Hypertension 0.776 (n=10,622), CKD 0.858 (n=9,463) — all three clean, all correctly restricted to the 2009 cross-section. Hypertension landing just below its internal NHANES number (0.787) and Diabetes dropping from its internal ~0.86-0.87 are both the expected, credible direction for honest external validation. NAFLD: raw AUC 0.975, but the built-in sensitivity check (2-variable BMI+waist toy model scores 0.946 on the same label) confirms this is substantially confounded by shared predictors between the proxy label and the model - report both numbers together per `MODEL_CARDS.md`'s suggested sentence, not the raw 0.975 alone. Worth a methods-section mention: a real multi-wave data leakage bug was caught and fixed during this analysis (see `MODEL_CARDS.md`'s integrity note) - this is a demonstrated-rigor point, not just a footnote.

## 4. Discussion

### 4.1 Limitations (items 25)
Source material is extensive — nearly every script's docstring documents a specific limitation found during development. Worth specifically naming: no survey weighting in the primary analysis (though `rigorous_validation.py` does report weighted estimates separately), no formal sample-size calculation, CHNS not nationally representative, GGT identification evidence-based not codebook-confirmed, NAFLD proxy is a different measurement than the elastography-based training label, no prospective/longitudinal validation (only cross-sectional internal and external data).

### 4.2 Interpretation (item 26)
**Genuine writing task** — synthesize what the numbers mean together: does external performance holding up (or not) change how the non-circular design should be understood? Does the calibration/fairness work change how "accurate" should be interpreted for this kind of tool?

### 4.3 Implications (item 27)
Translate the Intended Use statement's framing into manuscript register — decision-support positioning, QRISK3-class comparison (already researched and cited earlier in this project's development), explicit non-diagnostic framing.

---

## What to do with this file
Fill in every `[PENDING: ...]` tag once the full CHNS run is done, then this skeleton converts directly into a full draft — most sections are closer to "reformat existing documentation" than "write from scratch."
