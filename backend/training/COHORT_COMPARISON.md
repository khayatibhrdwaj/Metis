# Development vs. Evaluation Cohort Comparison (TRIPOD+AI Item 20c)

Compares Metis's development cohort (NHANES) against the external evaluation cohort (CHNS, 2009 wave only), as TRIPOD+AI requires. Complete — no pending rows.

**A data-integrity bug was caught and fixed while building this table, worth keeping in mind when reading it.** The first full run showed an implausible mean age (36.8, SD 20.9) and implausible diabetes prevalence (2.1%) — caused by a merge bug letting multiple decades of CHNS data (1989-2011) into what was meant to be a 2009-only comparison. The numbers below are from the corrected, properly wave-restricted run (n dropped from 116,763 to the correct 11,929). See `MODEL_CARDS.md`'s integrity note for the full explanation.

| Characteristic | NHANES 2021-2023 (development) | CHNS 2009 (evaluation) |
|---|---|---|
| n | 5,683 | 11,929 (full 2009 wave); 9,463 with biomarker data specifically |
| Country | United States | China |
| Design | Nationally representative cross-sectional survey | 9-province survey, explicitly **not** nationally representative (provinces chosen for geographic/economic variation, per CHNS's own documentation) |
| Age, mean (SD) | 51.9 (18.2) | 43.3 (20.8) |
| BMI, mean (SD) | 29.4 (7.1) | 22.4 (4.1) |
| Male, % | 45.4% | 49.0% |
| Waist circumference (cm), mean (SD) | 100.1 (17.0) | 80.8 (11.9) |
| Systolic BP, mean (SD) | 122.1 (17.8) | 122.2 (19.8) |
| Diastolic BP, mean (SD) | 74.5 (10.9) | 79.1 (11.8) |
| Diabetes prevalence | 16.4% | 2.7% |
| Hypertension prevalence | 54.6% | 12.0% |
| CKD prevalence | 8.8% | *Not separately tabulated here — see MODEL_CARDS.md's CKD section* |
| Uric acid (mg/dL), mean | *(see MODEL_CARDS.md)* | 5.18 |
| Creatinine (mg/dL), mean | *(see MODEL_CARDS.md)* | 0.97 |
| CRP (mg/L), mean | *(see MODEL_CARDS.md)* | 2.50 |
| Total cholesterol (mg/dL), mean | *(see MODEL_CARDS.md)* | 184.8 |
| Fasting glucose (mg/dL), mean | *(see MODEL_CARDS.md)* | 96.4 |
| WBC, mean | *(see MODEL_CARDS.md)* | 6.35 |

## What this table shows — real, disclosable population differences, not errors

1. **CHNS is explicitly not nationally representative**, unlike NHANES — a genuine design difference worth naming plainly in the manuscript.
2. **CHNS's population is notably younger, leaner, and has much lower disease prevalence** across every measure — consistent with (a) CHNS surveying entire households including children and younger adults, unlike NHANES's design, and (b) genuine population-level differences between a 2009 Chinese cohort and a 2021-2023 US cohort. Systolic BP is a notable exception — nearly identical between the two cohorts (122.1 vs. 122.2) despite every other measure differing substantially, worth a specific mention in the discussion.
3. **This age/prevalence gap is the most likely explanation for the external validation AUCs landing somewhat lower than internal** (Diabetes 0.739 vs. ~0.86 internal) — a model trained on an older, sicker population being tested on a younger, healthier one is expected to show reduced discrimination partly because there's less severe disease to distinguish, not purely because the model fails to generalize. Worth stating explicitly in the discussion section rather than leaving the AUC drop unexplained.
