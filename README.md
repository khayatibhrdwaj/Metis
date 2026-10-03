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
