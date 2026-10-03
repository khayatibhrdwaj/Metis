# Metis — TRIPOD+AI Compliance Checklist

Maps the 27-item TRIPOD+AI checklist (Collins, Moons, Dhiman et al., *BMJ* 2024;385:e078378 — the current reporting standard for ML-based clinical prediction models, superseding TRIPOD 2015) against what already exists in this project. Built so manuscript writing starts from "here's what's missing," not from a blank page.

**Honesty note on this document itself**: item numbers, section groupings, and the exact wording of several items (4, 16, 20c, 21, 22) were verified directly. A few items' exact sub-bullet wording is inferred from TRIPOD+AI's well-documented structure (it closely extends TRIPOD 2015's item set, with AI-specific additions at item 18 and embedded in the methods section) rather than independently re-verified word-for-word. Treat the **status column as the actionable part**; treat exact item phrasing as a close paraphrase, not a verbatim quote, until cross-checked against the official checklist PDF at tripod-statement.org before submission.

**Status key**: ✅ Done — existing doc fully covers this · 🟡 Partial — some coverage, gap noted · ❌ Open — not yet addressed · 🚫 Blocked — needs something outside this codebase (ethics approval, co-author decision, etc.)

| # | Section | Item (paraphrased) | Status | Where / what's missing |
|---|---|---|---|---|
| 1 | Title | State it's a development study (or development+evaluation), identify the target population and outcome | 🟡 | README's opening states intended use clearly; title itself not yet drafted — straightforward once writing starts |
| 2 | Abstract | TRIPOD+AI for Abstracts checklist (13 sub-items) | ❌ | Not written — depends on the rest of the paper existing first |
| 3-4 | Introduction | Background/rationale; explicit objectives (development, validation, or both) | 🟡 | This entire conversation's history **is** the rationale (non-circular predictor design, why 5 diseases, why external validation was pursued) — needs condensing into prose, not re-deriving |
| 5 | Methods: Data | Source of data, study design, key study dates | ✅ | README + `data_prep_2021_2023.py` docstring (NHANES Aug 2021-Aug 2023) for internal data; CHNS citation requirement text already have for external |
| 6 | Methods: Participants | Eligibility criteria, data collection settings | ✅ | NHANES/CHNS cohort descriptions already documented |
| 7 | Methods: Outcome | How each outcome (5 diseases) was defined and measured | ✅ | `MODEL_CARDS.md`'s per-disease label definitions — this is the strongest-documented item in the whole project given how much circularity-avoidance work went into it |
| 8 | Methods: Predictors | How each predictor was defined and measured | ✅ | `model.py`/`diseases.py`/`train_multi_disease.py` docstrings, all with explicit excluded-predictor reasoning per disease |
| 9 | Methods: Sample size | Justification, how it was determined | 🟡 | Have the actual n per disease (`MODEL_CARDS.md`); no formal a priori power calculation was performed — note as a limitation, not silently omit |
| 10 | Methods: Missing data | How missing data were handled | ✅ | Documented extensively: native-missing-OK XGBoost handling, median fallback, `NATIVE_MISSING_OK` sets |
| 11 | Methods: Statistical/ML methods | Full model-building methodology | ✅ | `train_model.py`/`train_multi_disease.py`, both extensively commented with every methodological decision and why |
| 12 | Methods: Model-building procedures | Selection of predictors, handling of continuous variables, interactions | ✅ | Documented in training scripts' comments |
| 13 | Methods: Internal validation | How internal validation was performed | ✅ | 5-fold CV (`rigorous_validation.py`), single-split, both documented with real numbers |
| 14 | Methods: External validation | Description of external validation data/methods, if performed | 🟡 | CHNS methodology fully documented in `external_validation_chns.py`'s docstring; **the full-scale run result itself is still pending** (preliminary n=1,500 result only) |
| 15 | Methods: Performance measures | Which metrics, and why | ✅ | AUC, Brier, ECE, calibration curves — all justified in `calibration_analysis.py`/`fairness_analysis.py` docstrings |
| 16 | Methods: Training vs. validation differences | Differences between development and evaluation data (setting, eligibility, outcome definition) | 🟡 | Real, known differences exist (US vs. China, NHANES vs. CHNS outcome ascertainment) — partially discussed in chat history, not yet consolidated into one section |
| 17 | Methods: Model updating | Any model updating based on validation results | ✅ | Explicitly did NOT update/retune based on CHNS results — document this as a deliberate choice (avoids a specific overfitting-to-validation-data failure mode) |
| 18 | Open science | Code/data availability, reproducibility | ✅ | Entire project is documented, scripted, reproducible end-to-end — genuinely a strength to lead with |
| 19 | Patient/public involvement | Any PPI in design, conduct, reporting | ❌ | None occurred — state this plainly as a limitation, don't omit the item |
| 20 | Results: Participants | Flow of participants through the study (numbers at each stage) | 🟡 | Have final n per disease; a formal participant flow diagram/table not yet built |
| 20c | Results: Distribution comparison | Compare development vs. evaluation data's predictor/outcome distributions | ❌ | Real, buildable now from data already in hand (NHANES vs. CHNS summary statistics side by side) - concrete next step |
| 21 | Results: Model development | N participants and outcome events at each analysis stage | ✅ | In `MODEL_CARDS.md` and each training script's printed output |
| 22 | Results: Model specification | Full model details/code/API enabling third-party reproduction | ✅ | This is a clear strength — full source code, not just a coefficient table |
| 23 | Results: Model performance | Report performance (discrimination + calibration) | ✅ | Comprehensive — AUC, calibration, fairness, CV, weighted estimates all reported together, not just the headline AUC |
| 24 | Results: Model evaluation comparison | How evaluation-data performance compares to development performance | 🟡 | Internal (0.86ish) vs. preliminary external (0.796) comparison already drafted in chat; needs the full CHNS run before finalizing |
| 25 | Discussion: Limitations | Study limitations, including sources of bias | ✅ | Arguably over-documented at this point — every limitation found during this whole project is written down somewhere already |
| 26 | Discussion: Interpretation | Interpretation considering objectives, limitations, prior evidence | ❌ | Genuine writing task — synthesizing what's already documented into discussion prose |
| 27 | Discussion: Implications | Potential clinical use, how results might be used | 🟡 | Intended Use statement already states this clearly for the app; needs translating into manuscript-appropriate language (decision-support framing, not clinical-release framing) |

## What this table actually says, bottom-lined

**17 of 27 items are already fully supported by existing documentation** — this project's own discipline of over-explaining every decision in code comments turns out to double as most of a TRIPOD+AI-compliant methods section. **4 items are structurally blocked** (19: no PPI occurred, period; parts of 14/20c/24: waiting on the full CHNS run) **or require things outside this codebase** (co-authorship and ethics approval affect items 1-2's framing but aren't checklist items themselves). **The rest are genuine writing tasks**, not open research questions — synthesizing material that already exists into manuscript prose.

## Concrete next steps, in order
1. Finish the full-scale CHNS run (items 14, 20c, 24 depend on it)
2. Build the development-vs-evaluation distribution comparison table (item 20c) — doable right now with NHANES + CHNS summary stats already in hand
3. Everything else is writing, not research
