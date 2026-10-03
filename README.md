# Metis — Clinical Decision-Support Tool for Metabolic Risk

**Intended use:** Metis is a clinical decision-support tool, not a
diagnostic device. It estimates risk for **5 related metabolic
conditions** — Type 2 Diabetes, Metabolic Syndrome, Fatty Liver Disease
(NAFLD), Chronic Kidney Disease, and Hypertension — from routinely
available anthropometric, laboratory, and lifestyle data, to help
identify people who may benefit from further clinical evaluation. A Metis
risk estimate does not confirm or rule out any condition and is not
intended to be the sole basis for a clinical decision; results should be
interpreted by a qualified healthcare professional alongside clinical
judgment and confirmatory diagnostic testing. Metis has been validated
**internally** (held-out NHANES data) — see "Model validation" below —
but has **not yet undergone prospective clinical validation** against
real patient outcomes. The in-app "Know Yourself" tab carries the full
Intended Use statement shown to every user.

Metis lets you simulate how lifestyle interventions (weight loss,
exercise, sleep, diet, smoking, stress) would change all 5 risk estimates
at once, with SHAP-based explainability behind every number.

## Multi-disease architecture

Each of the 5 diseases has its own model with its own non-circular
predictor set — a condition's own diagnostic criteria are never used to
predict that same condition:

| Disease | Label definition | Excluded from its own predictors |
|---|---|---|
| Type 2 Diabetes | Self-report OR fasting glucose≥126 OR HbA1c≥6.5% | fasting_glucose, hba1c |
| Metabolic Syndrome | 3+ of: abdominal obesity, high triglycerides, low HDL, high BP, elevated glucose | waist, triglycerides, HDL, BP, insulin/HOMA-IR |
| NAFLD | Liver elastography CAP score ≥248 dB/m | liver_fat_cap, liver_stiffness |
| Chronic Kidney Disease | eGFR<60 (CKD-EPI 2021) OR self-report | creatinine, BUN, eGFR |
| Hypertension | Measured BP≥130/80 OR self-report | bp_systolic, bp_diastolic, pulse_pressure |

`fasting_glucose`, `hba1c`, `sugar_g`, and `drinks_per_day` stay excluded
from **every** disease's predictor set project-wide (see "Model notes"
below for why). Diabetes uses a stacked ensemble (see below); the 4 newer
diseases use a single monotonic XGBoost model each — see
`backend/training/train_multi_disease.py`'s docstring for the full
reasoning, including two real regressions caught and fixed during
development (a data-pipeline indexing bug in the CKD label logic, and a
sleep-direction bug in the CKD model itself, the same class of issue
found and fixed for diabetes earlier).

Every disease's simulator behavior was verified with the same causal-
sanity check used throughout this project: no lifestyle improvement may
ever show predicted risk *increasing*. Tested across 2 diverse patient
profiles × 4 new diseases × 9 intervention levers = 72 checks, all
passing.

All 5 diseases are wired through the whole frontend, not just the
dashboard summary cards: the **Risk & Drivers** and **AI Explanation**
tabs both have a disease selector (shared state between them - switching
one updates the other) with real SHAP and Gemini-generated explanations
per disease; the **Simulator** tab shows projected risk change for all 5
at once; the **Trajectory** tab has its own independent disease selector
(switching it doesn't affect what Risk & Drivers is showing) and switches
instantly between diseases using data already computed by the last
Simulator run, no extra API call needed.

While building this, a real pre-existing bug surfaced and got fixed:
charts created while their tab was hidden (`display:none` gives a canvas
0 width/height at creation time) didn't always get correctly resized on
later reveals - `chart.resize()` needed to run inside
`requestAnimationFrame()`, not synchronously right after un-hiding the
tab, since the browser hadn't necessarily finished its layout pass yet
when the resize call fired. This affected every chart-bearing tab, not
just the newly-added ones - worth knowing if other chart rendering issues
ever come up.

## Structure

```
metatwin_app/
├── Dockerfile                 # single-container deployment
├── .dockerignore
├── .gitignore
├── backend/                    # FastAPI service + model (also serves the frontend)
│   ├── main.py                    # API endpoints + static-file mount for the frontend
│   ├── model.py                    # diabetes ensemble: loads model, predicts, SHAP
│   ├── diseases.py                  # the 4 new disease models: same interface, single models
│   ├── simulator.py                 # what-if intervention engine (forecasts all 5 diseases)
│   ├── genai.py                       # AI explanation layer (Google Gemini)
│   ├── mailer.py                       # SMTP email sender (Contact Us form)
│   ├── auth.py                          # accounts + profile history (SQLite)
│   ├── requirements.txt
│   ├── .env.example                      # copy to .env and fill in secrets
│   ├── metis_users.db                     # created automatically on first run
│   ├── models/                             # trained model artifacts (already built)
│   │   ├── risk_model.pkl
│   │   ├── explainer.pkl
│   │   ├── medians.pkl
│   │   └── metrics.json
│   └── training/                            # offline scripts to rebuild/validate the model
│       ├── data_prep_2021_2023.py
│       ├── train_model.py                     # diabetes ensemble
│       ├── train_multi_disease.py               # metabolic syndrome, NAFLD, CKD, hypertension
│       ├── validate_models.py                 # 7-model comparison study (see below)
│       ├── VALIDATION_REPORT.md
│       ├── requirements-validation.txt
│       ├── validation_results/                 # generated plots + metrics (git-ignored)
│       └── data/
│           ├── raw_2021_2023/                    # (add real NHANES .xpt files here to retrain)
│           └── processed/
│               └── nhanes_2021_2023_merged.csv     # already-merged dataset
└── frontend/                  # plain HTML/CSS/JS, no build step, served by the backend
    ├── index.html                # entry point (routes to auth/intake/dashboard)
    ├── auth.html, auth.js          # login/register
    ├── intake.html, intake.js       # patient profile form
    ├── dashboard.html, app.js        # main dashboard (7 tabs)
    ├── style.css
    ├── assets/                        # logo (full lockup + cropped icon/favicon)
    └── vendor/chart.umd.js              # Chart.js, vendored locally
```

The model is **already trained** — `backend/models/*.pkl` are included, so
you don't need to run the training scripts unless you want to retrain on
different/updated data.

## Requirements

- Python 3.11 or 3.12 recommended (3.13 works if your packages have wheels
  for it — see troubleshooting below)
- No Node.js / npm needed — the frontend has no build step

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
```
Then edit `.env` and fill in `GEMINI_API_KEY`, `SMTP_*`, etc. — see the
two "required features" sections below for exactly how to get each one.

If `pip install` tries to compile `pandas` from source and fails (Windows
error mentioning `vswhere.exe` / Visual Studio Build Tools), remove version
pins from `requirements.txt` and reinstall — it's already unpinned for this
reason. If you hit `ModuleNotFoundError: No module named 'pyarrow'` when the
app starts, that's a real dependency `pyarrow` covers — it's already in
`requirements.txt`, just make sure the install finished.

## Running it — one process, one port

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Then open:
```
http://localhost:8000
```

That's the whole thing — **one command, one port**. The backend serves
the frontend itself (see the static-file mount at the bottom of
`main.py`), so there's no separate frontend server and no CORS
configuration to get right locally. This is also exactly how it runs in
production (see "Deploying" below) — local dev and production use the
identical command.

First visit takes you to a login/register page. Register with an email and
password, then fill out the patient-profile intake form; submitting it
takes you to the dashboard. Logging in again later with the same email
restores your most recently saved metrics automatically. The intake page
also shows a "Your past submissions" panel with every previous entry for
that account — click "Load" on any of them to bring those values back
into the form.

## GenAI explanations (required)

The "What's driving this risk score?" AI explanation (its own "AI
Explanation" tab) is a **required feature**, not optional — the app
always attempts a real call to the Google Gemini API. Without a key
configured, the AI Explanation box on the dashboard stays visible but
shows a clear setup error instead of disappearing.

Gemini is used specifically because it has a genuinely ongoing free tier
(no credit card, no expiration) — unlike Anthropic's one-time trial
credit.

1. Get a free key (no credit card, no phone verification) at
   **https://aistudio.google.com/apikey**
2. Put it in `backend/.env`:
   ```
   GEMINI_API_KEY=your-key-here
   ```
3. Restart the backend.

Free tier limits (set by Google, subject to change): roughly 1,500
requests/day and 15/minute on `gemini-3.8-flash`, the default model here
— far more than this app needs for normal use.

## Contact form email (required)

The "Contact Us" tab's chat box sends real email via SMTP — also a
**required** feature, not decorative. Without SMTP configured, the chat
box stays fully usable but shows a clear "Couldn't send" error with setup
steps instead of pretending the message went through.

In `backend/.env`:
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-sending-address@gmail.com
SMTP_PASSWORD=xxxx xxxx xxxx xxxx
```

For Gmail, `SMTP_PASSWORD` must be an **app password**, not your regular
account password:
1. Turn on 2-Step Verification at https://myaccount.google.com/security
2. Generate an app password at https://myaccount.google.com/apppasswords
3. Use that 16-character code as `SMTP_PASSWORD`

Any other SMTP provider (Outlook/Office365, SendGrid, Mailgun, your
institution's mail server) works too — just point `SMTP_HOST`/`SMTP_PORT`
at it. Messages are delivered to `khayati.bhrdwaj15@gmail.com` by default;
override with `METIS_CONTACT_RECIPIENT`.

## Deploying

The single-process architecture means most hosts work the same way:

**Docker (any host that runs containers):**
```bash
docker build -t metis .
docker run -p 8000:8000 --env-file backend/.env metis
```

**Render / Railway / Fly.io / a plain VPS:** point the platform at this
repo, set the start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`
(working directory `backend/`), and set the same environment variables
from `.env` in the platform's dashboard instead of a file.

**Before going public, also do this:**
- Set `ALLOWED_ORIGINS` in `.env` to your actual deployed domain instead of
  leaving it as the permissive default — only matters if something other
  than this same backend ever serves the frontend, but cheap insurance.
- `backend/metis_users.db` is a plain SQLite file. If your host has an
  ephemeral filesystem (redeploys wipe local disk — true of most container
  platforms' default free tiers), mount a persistent volume at
  `backend/` or accounts/history will vanish on every redeploy. For real
  production scale, migrate `auth.py` to a hosted Postgres instead — the
  functions are small and isolated enough that this is a contained change.
- The account system (`auth.py`) is intentionally minimal: no email
  verification, no password reset, no login-attempt rate limiting. Fine
  for a course project or small internal tool; add these before treating
  it as a real public-facing product with sensitive data.
- Fix the LinkedIn link on the Contact Us tab first — see the note at the
  bottom of this file.

## Retraining on different/updated data (optional)

The model is trained on real NHANES August 2021–August 2023 data. To
rebuild it from scratch (e.g. after adding more NHANES files to
`backend/training/data/raw_2021_2023/`):

```bash
cd backend/training
python data_prep_2021_2023.py   # rebuilds the merged CSV
python train_model.py            # retrains, writes new .pkl files to backend/models/
```

## Model refinements: calibration, fairness, and safety

Beyond discrimination (AUC), two more rigorous checks were run against
all 5 production models:

- **Calibration** (`backend/training/calibration_analysis.py`): checks
  whether a model's stated percentage matches reality (does "40% risk"
  actually mean ~40% of such people have the condition?). **Found a
  serious problem**: the CKD model's raw output claimed up to 80% risk
  for a group where only ~37% actually had the condition. Isotonic
  recalibration fixed this (calibration error 0.190 → 0.020) and is now
  applied in production for CKD and Metabolic Syndrome (the two models
  that needed it); diabetes/NAFLD/hypertension were already well-calibrated
  and are left as-is.
- **Subgroup fairness** (`backend/training/fairness_analysis.py`): checks
  whether performance holds up across age, sex, and income bands.
  **Found real gaps**: CKD and Hypertension both show meaningfully worse
  discrimination for ages 40+ than for younger adults — a real limitation
  to disclose, not just a number to report.
- **A silent-certainty bug was caught during this work**: the Metabolic
  Syndrome calibrator produced an exact 1.0 (100%) for some inputs - a
  small-sample artifact of isotonic regression's tail behavior, not a
  real finding. All 5 models now hard-clip to [0.01, 0.99] - no model may
  ever claim mathematical certainty, consistent with every risk estimate
  being exactly that, an estimate.

Full findings, methodology, and per-disease detail in
**`backend/training/MODEL_CARDS.md`** - the single consolidated reference
for what's actually known about each of the 5 models' real-world behavior.

Two more changes from this round, both shipped and verified against the
full 90-check causal-sanity regression suite (2 patients × 5 diseases × 9
levers):
- **The What-If simulator now also forecasts ALT/AST/GGT and uric acid**
  (weight loss, alcohol reduction, and sugar reduction all move them,
  each grounded in real mechanistic literature - see `simulator.py`).
  Creatinine, BUN, and albumin are deliberately left unforecast - a
  considered choice, not an oversight; there isn't a specific, citable
  short-term lifestyle-response literature for these the way there is for
  the others.
- **Defense-in-depth input validation** (`backend/input_validation.py`):
  physiologically impossible values (negative weight, BP of 900, etc.)
  are now rejected server-side, not just by the intake form's HTML
  constraints - closes the gap where a direct API call could bypass the
  UI entirely.

## Multi-disease validation study

`backend/training/validate_multi_disease.py` extends the diabetes
validation methodology to the 4 newer disease models — same idea,
different question: does each disease's production XGBoost model hold up
against alternative model families on its own feature set?

**Results:** in 2 of 4 diseases (NAFLD, CKD), production XGBoost is
literally the best-performing model of the 5 tested — not just
competitive, outright better. In the other 2 (Metabolic Syndrome,
Hypertension), the gap to the best model is 0.001–0.005 AUC, well within
normal run-to-run noise. All 5 models cluster tightly within each disease
(e.g. CKD: 0.836–0.845 AUC across Logistic Regression through XGBoost),
confirming the signal is real and not specific to one algorithm — the
same pattern the original diabetes study found. Full writeup, per-disease
charts, and honest limitations in
**`backend/training/VALIDATION_REPORT_MULTI_DISEASE.md`**.

```bash
cd backend/training
pip install -r requirements-validation.txt
python3 validate_multi_disease.py
```

## External longitudinal validation (not yet done — here's how to)

Both validation studies above are **internal**: a held-out test split
from the *same* NHANES data collection wave. That confirms the models
generalize to unseen rows from the same cross-sectional snapshot — it
does **not** confirm that someone flagged high-risk today will actually
go on to develop that condition years later. That's a fundamentally
different, stronger claim, and NHANES itself can't answer it (one
measurement per person, no disease-incidence follow-up). Proving it needs
a **prospective longitudinal cohort**: people measured at baseline, then
followed up years later to see who actually developed each condition.

None of this has been done yet. Here's what's actually available if you
want to do it, checked for current access terms rather than assumed:

| Dataset | Cost | Access | Why it fits |
|---|---|---|---|
| **HRS** (Health and Retirement Study, U. Michigan) | **Free** | Simple registration for public data; biomarker/sensitive health data needs a free supplemental agreement (still no cost) | Biennial waves since 1992, ages 50+. Has HbA1c, total cholesterol, HDL, CRP, **cystatin-C** (a kidney-function marker — good for validating the CKD model specifically), BP, BMI, diabetes/hypertension self-report, plus Medicare claims linkage that can capture incident diagnoses over time. Best first stop. |
| **BioLINCC** (NHLBI/NIH) — hosts **Framingham**, **MESA**, **CARDIA**, **Jackson Heart Study** | **Free** | Data request + institutional email required (NIH no longer accepts free email providers like Gmail for this — use your university email) | Framingham Offspring alone has ~10 clinical exams per person with event follow-up through 2021 — genuinely built for exactly this kind of validation. MESA and Jackson Heart Study add more racial/ethnic diversity than Framingham's original cohort. |
| **ELSA** (English Longitudinal Study of Ageing) | Free | Registration via UK Data Service | UK equivalent to HRS, repeated waves, nurse-visit biomarkers in some waves. Good if you want a non-U.S. population to check generalization. |
| **UK Biobank** | **Real cost**: £3,000 for standard Tier 1 access (first 3 years), £500+VAT discounted rate for student researchers (supervisor must be co-listed) | Online application, ~15-week average approval, institutional Material Transfer Agreement required | The strongest single option if budget allows — 500,000 participants, deep biomarker + genetic data, and *ongoing linkage to national health records*, so incident diagnoses (new diabetes, new CKD, new hypertension) are captured automatically rather than needing a manual follow-up wave. Not "free" despite being commonly called open-access — budget and time for this before committing. |

**Practical starting point given this project's setup:** HRS is the
closest match to what's already here — same general population-survey
design philosophy as NHANES, genuinely free, and its cystatin-C measure
specifically lets you check whether the CKD model's predicted risk at
one wave predicts a later eGFR decline or diagnosis, which is exactly
the claim internal validation can't make. Framingham via BioLINCC is the
natural second step once this works, given it was specifically designed
for incident cardiometabolic outcome research and your university email
clears BioLINCC's access requirement.

## Model validation & comparison study

**Note:** the numbers in this section predate the HbA1c-strengthened
label and the 6 additional biomarkers described in "Model notes" below —
they were the most recent full 7-model study run at the time and are kept
here for the methodology and visualizations, which are still valid; the
headline AUC figures are from an earlier iteration of the feature/label
set. Re-run `validate_models.py` for fully current numbers.

`backend/training/validate_models.py` trains 7 different model families
(Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting
— both with and without monotonicity constraints — SVM, and production
XGBoost) on the *identical* train/test split and feature set, then
generates a full set of comparison visualizations: ROC curves, PR curves,
a confusion-matrix grid, a metrics heatmap, feature-importance comparison
across model families, and calibration curves.

**Headline finding:** all 7 models land within a 0.855–0.886 ROC-AUC band
(the signal is real, not an XGBoost-specific artifact), and once a fair
comparison model (HistGradientBoosting) is held to the *same*
monotonicity constraint that makes the What-If simulator causally sane,
production XGBoost still wins (0.870 vs 0.856 AUC). Full writeup,
methodology, and honest limitations in
**`backend/training/VALIDATION_REPORT.md`**.

To reproduce:
```bash
cd backend/training
pip install -r requirements-validation.txt
python3 validate_models.py
```

## Model notes

- Trained on **5,695 real NHANES participants**, test ROC-AUC **0.860**
  (stacked-ensemble production config; see `metrics.json`).
- **Architecture: a stacked ensemble**, not a single model. Three base
  learners — monotonic XGBoost, monotonic HistGradientBoosting, and
  Logistic Regression — are combined by a meta-learner (Logistic
  Regression on the three base probabilities), fit via proper 5-fold
  out-of-fold stacking (`train_model.py`) to avoid leakage into the
  meta-learner's training data. Solo XGBoost alone gets 0.857; the full
  stack gets 0.860. This was shipped on explicit request after repeated
  attempts to reach 0.9+ — the gain is small (+0.003) and adds real
  serving complexity (three models instead of one, one extra inference
  hop through the meta-learner), a tradeoff documented in
  `train_model.py`'s module docstring rather than hidden.
- **The "Risk & Drivers" / "AI Explanation" tabs still explain using only
  the XGBoost component's exact SHAP values**, not a blended cross-model
  explanation — computing an exact Shapley decomposition through a
  sigmoid meta-learner combining three different model families is a
  substantially harder problem than explaining one tree model, and
  XGBoost is one of the two co-dominant components in the meta-learner's
  learned weights. The displayed risk *percentage* comes from the full
  ensemble; the displayed *explanation* comes from XGBoost's view of it.
  These very likely agree (all three base models train on the same
  monotonic-consistent features) but aren't mathematically guaranteed to
  in every edge case. **Verified empirically instead**: the full 9-lever
  What-If simulator sanity check (every intervention must never show a
  healthy change increasing predicted risk) was re-run against the actual
  deployed ensemble, both with and without the 6 optional biomarker
  fields filled in — passed cleanly both times. That's the real
  acceptance test that matters here, not per-component theoretical
  guarantees.
- `fasting_glucose` and `hba1c` are deliberately **excluded as
  predictors** — both are the ADA's own diagnostic criteria for diabetes
  and are used to *define* `diabetes_label` (see
  `data_prep_2021_2023.py`), so using them as inputs would let the model
  "cheat" by looking up the diagnosis rather than predicting risk. Both
  remain as tracked dashboard metrics and What-If simulator *outcomes*
  (forecast the same way lipids/CRP/liver fat already are), never as
  model inputs.
- `sugar_g` and `drinks_per_day` are also excluded as predictors for a
  related reason: marginal-effect testing found both have a *reversed*
  relationship with risk in this cross-sectional data (people already
  diagnosed tend to have already cut sugar/alcohol as part of managing
  their diagnosis). They remain as What-If simulator levers, driven by
  literature-based coefficients instead of the confounded raw model
  relationship.
- The model uses **monotonicity constraints** so each feature's
  relationship with predicted risk moves in its causally-expected
  direction — see `backend/training/train_model.py`'s `_MONOTONE` dict for
  the full reasoning per feature, and the validation study above for
  evidence this constraint doesn't come at an unreasonable accuracy cost.
- **On chasing higher accuracy** (a real, repeated ask during development
  — target was 0.9+): every legitimate technique was tried and the
  results are documented in `train_model.py`'s comments in detail.
  Summary: hyperparameter tuning and engineered interaction features
  plateau around 0.87 *within* the monotonicity constraint, and at one
  point pushing model complexity further introduced a real regression
  (the sleep lever started showing risk *increasing* for a healthy sleep
  improvement — a tuning artifact overfitting noise in that deliberately-
  unconstrained dimension, not a real finding). Ensembling multiple
  monotonic models (averaging predictions, which preserves monotonicity)
  gave no improvement — the models are too similar to reduce variance by
  averaging. Checking the biochemistry/CBC panels for legitimately unused,
  non-diagnostic signal did find something real: ALT/AST/GGT (liver
  enzymes), uric acid, albumin, and white blood cell count are all
  established independent metabolic-risk markers that weren't in the
  original feature set. Adding them gave a genuine, verified gain
  (0.853→0.857 solo XGBoost). Proper out-of-fold stacking (this file)
  added another small real gain to 0.860 — see the ensemble architecture
  note above. Removing monotonicity entirely reaches ~0.87-0.88
  unconstrained, but reintroduces the exact backwards-simulator-lever
  behavior documented above. **0.9+ isn't reachable from non-diagnostic
  risk factors alone without overfitting or reintroducing circularity** —
  real published diabetes-risk screening tools (ADA Risk Test, FINDRISC)
  using this style of feature set generally land in the 0.75-0.87 range.
  What WOULD close the rest of the gap: genetic/family-history data (not
  in this NHANES extract) or longitudinal repeat-measures data (NHANES is
  cross-sectional) — both are data-availability problems, not modeling
  problems no amount of further tuning on this dataset can solve.
- The six new biomarkers (ALT, AST, GGT, uric acid, albumin, WBC) are
  **optional intake-form fields**. If a user leaves them blank, the
  backend falls back to the population median for that feature (see
  `model.py`'s `_row()`) — verified to produce a sane, if less
  personalized, prediction either way.

This is a decision-support prototype for education/portfolio purposes,
not a validated clinical tool.

## Known issue to fix before sharing publicly

The "AI and IoT Automation Lab" link on the Contact Us tab
(`frontend/dashboard.html`) currently points to a private LinkedIn
"edit profile" URL, which only works for the logged-in profile owner —
visitors will likely hit a login wall. Replace it with a public lab page
or public profile URL before sharing this outside the team.
