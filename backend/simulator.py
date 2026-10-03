"""
MetaTwin - What-If Metabolic Simulator (NHANES 2021-2023 feature set)

Translates lifestyle interventions (weight loss, exercise, sleep, diet)
into projected changes in a patient's real, measured clinical variables
(BMI, waist, BP, HDL/LDL/triglycerides, CRP, insulin/HOMA-IR, fasting
glucose, HbA1c), using approximate, evidence-informed coefficients
synthesized from published literature.

IMPORTANT: These coefficients are reasonable order-of-magnitude estimates
for a decision-support PROTOTYPE, not validated clinical dosing. Anyone
extending this toward publication should replace each coefficient with a
citation to a specific trial/meta-analysis, and ideally refit some of them
directly against NHANES's own weight-history (WHQ) self-reported change
data or a longitudinal cohort.

fasting_glucose and hba1c are forecast OUTCOMES here, exactly like
triglycerides/CRP/liver fat - never model predictors (see
backend/model.py's FEATURES list, which deliberately excludes both, for
the same reason they're excluded from the diabetes-risk model itself:
they're diagnostic criteria, not independent risk factors).

ALT/AST/GGT and uric_acid are ALSO forecast outcomes (weight loss,
alcohol reduction, and sugar reduction all move them, with real
mechanistic literature behind each pathway - see each intervention
block below). creatinine, BUN, and albumin are deliberately NOT
forecast - left as static inputs the simulator does not touch. This is
a considered choice, not an oversight: creatinine/BUN primarily reflect
kidney filtration rate and muscle mass, which don't reliably shift within
the kind of lifestyle-change timeframes this simulator otherwise models,
and albumin reflects longer-run nutritional/inflammatory status where the
available literature doesn't support a specific, citable short-term
lifestyle-response curve the way it does for the other markers here.
Better to leave these three unmodified than fabricate a relationship
without real evidence behind it.

Approximate literature basis:
  - Look AHEAD / Diabetes Prevention Program: weight loss & glycemic/lipid
    improvement (DPP: ~7% weight loss -> ~58% reduction in diabetes
    incidence; substantial fasting glucose/HbA1c improvement)
  - Neter et al. 2003 (Hypertension): weight loss & blood pressure
  - Kraus et al. / Kelley & Kelley meta-analyses: exercise & HDL/triglycerides
  - Umpierre et al. 2011 (JAMA): structured exercise & glycemic control
    (meta-analysis: supervised aerobic/resistance training -> HbA1c
    reduction of roughly 0.3-0.6 percentage points in at-risk/T2D populations)
  - CRP/inflammation: weight loss and exercise trials showing ~20-40% CRP
    reduction with meaningful weight loss (e.g. Selvin et al., meta-analyses)
  - Sleep-restriction crossover studies: insulin sensitivity effects
  - Nathan et al. 2008 (ADAG study): eAG-HbA1c relationship used to keep
    the fasting-glucose and HbA1c forecasts numerically consistent with
    each other rather than drifting independently
"""

from dataclasses import dataclass, field
from typing import Optional
import copy


@dataclass
class Interventions:
    weight_loss_kg: float = 0.0
    exercise_min_per_week: float = 0.0
    sleep_hours: Optional[float] = None
    current_sleep_hours: Optional[float] = None
    sugar_reduction_pct: float = 0.0
    quit_smoking: bool = False
    sedentary_reduction_min: float = 0.0
    fiber_increase_g: float = 0.0
    alcohol_reduction_drinks: float = 0.0
    stress_reduction_points: float = 0.0


@dataclass
class SimulationResult:
    before: dict
    after: dict
    deltas: dict
    narrative_points: list = field(default_factory=list)


def _pct_weight_loss(weight_kg, loss_kg):
    if weight_kg <= 0:
        return 0.0
    return 100 * loss_kg / weight_kg


def apply_interventions(patient: dict, interventions: Interventions) -> SimulationResult:
    before = copy.deepcopy(patient)
    after = copy.deepcopy(patient)
    notes = []

    weight_kg = patient.get("weight_kg", 85.0)
    height_m = patient.get("height_cm", 170.0) / 100.0
    pct_loss = _pct_weight_loss(weight_kg, interventions.weight_loss_kg)

    # ---------------------------------------------------------------
    # 1. Weight loss effects
    # ---------------------------------------------------------------
    if interventions.weight_loss_kg > 0:
        new_weight = max(weight_kg - interventions.weight_loss_kg, 30)
        after["weight_kg"] = new_weight
        after["bmi"] = new_weight / (height_m ** 2)
        after["waist_cm"] = max(patient.get("waist_cm", 95) - 0.9 * interventions.weight_loss_kg, 50)

        after["bp_systolic"] = max(patient.get("bp_systolic", 120) - 1.0 * interventions.weight_loss_kg, 90)
        after["bp_diastolic"] = max(patient.get("bp_diastolic", 78) - 0.6 * interventions.weight_loss_kg, 60)

        # Lipids: ~1 mg/dL HDL increase per ~3kg lost; triglycerides drop
        # roughly 1.5% per kg (Look AHEAD-informed approximation)
        after["hdl"] = patient.get("hdl", 48) + 0.33 * interventions.weight_loss_kg
        tg_drop_frac = min(0.015 * interventions.weight_loss_kg, 0.4)
        after["triglycerides"] = patient.get("triglycerides", 130) * (1 - tg_drop_frac) \
            if patient.get("triglycerides") is not None else None
        after["ldl"] = (patient.get("ldl", 110) - 0.7 * interventions.weight_loss_kg) \
            if patient.get("ldl") is not None else None
        # NOTE: total_cholesterol is deliberately NOT modified here. Marginal-
        # effect testing on the trained model found its relationship with
        # predicted risk is ambiguous/non-monotonic at low values in this
        # cross-sectional data (likely reflecting statin use in already-
        # diagnosed patients rather than a true protective effect of low
        # cholesterol) - see training/train_model.py's _MONOTONE docstring.
        # LDL and triglycerides are the reliable, monotonic-constrained
        # lipid levers instead.

        # CRP (inflammation): meaningful weight loss produces disproportionate
        # CRP reduction, roughly 3% relative reduction per kg, capped at 50%
        crp_drop_frac = min(0.03 * interventions.weight_loss_kg, 0.5)
        after["crp"] = patient.get("crp", 2.0) * (1 - crp_drop_frac)

        # Insulin sensitivity
        homa_reduction_frac = min(pct_loss * 0.02, 0.5)
        if patient.get("homa_ir") is not None:
            after["homa_ir"] = patient.get("homa_ir", 2.5) * (1 - homa_reduction_frac)
        if patient.get("insulin") is not None:
            after["insulin"] = patient.get("insulin", 10) * (1 - homa_reduction_frac * 0.7)

        # Liver fat (CAP score): NAFLD responds strongly to weight loss -
        # roughly 7-10% body-weight loss is associated with substantial
        # steatosis reduction in imaging studies; approximate as ~3% relative
        # CAP reduction per kg, capped at 40% relative reduction.
        if patient.get("liver_fat_cap") is not None:
            cap_drop_frac = min(0.03 * interventions.weight_loss_kg, 0.4)
            after["liver_fat_cap"] = patient["liver_fat_cap"] * (1 - cap_drop_frac)

        # Liver enzymes (ALT/AST/GGT): weight-loss NAFLD remission trials
        # consistently show ALT/AST normalizing alongside steatosis
        # reduction (they're not diagnostic criteria for NAFLD, but are a
        # real downstream consequence of the same fat reduction) -
        # approximated at a similar relative rate to liver fat itself,
        # somewhat more conservative since enzyme normalization tends to
        # lag imaging improvement in these studies.
        enzyme_drop_frac = min(0.02 * interventions.weight_loss_kg, 0.35)
        for enzyme in ("alt", "ast", "ggt"):
            if patient.get(enzyme) is not None:
                after[enzyme] = patient[enzyme] * (1 - enzyme_drop_frac)

        # Uric acid: weight loss improves renal urate clearance and
        # reduces production - a modest, well-documented effect, smaller
        # in magnitude than the enzyme/liver-fat response.
        if patient.get("uric_acid") is not None:
            uric_drop_frac = min(0.015 * interventions.weight_loss_kg, 0.25)
            after["uric_acid"] = max(patient["uric_acid"] * (1 - uric_drop_frac), 2.0)

        # Glycemic outcomes (HbA1c, fasting glucose) - forecast OUTCOMES,
        # never model inputs. DPP/Look AHEAD-informed: ~0.045 percentage
        # points HbA1c reduction per 1% of body weight lost, capped at 1.6
        # points for very large losses. Fasting glucose is moved by the
        # same underlying change using the ADAG study's eAG-HbA1c slope
        # (28.7 mg/dL per 1 HbA1c point) so the two numbers stay physiologically
        # coupled to each other rather than drifting independently.
        hba1c_drop = min(pct_loss * 0.045, 1.6)
        if patient.get("hba1c") is not None:
            after["hba1c"] = max(patient["hba1c"] - hba1c_drop, 4.8)
        if patient.get("fasting_glucose") is not None:
            after["fasting_glucose"] = max(patient["fasting_glucose"] - hba1c_drop * 28.7, 70)

        notes.append(
            f"Losing {interventions.weight_loss_kg:.0f} kg (~{pct_loss:.1f}% of body weight) is "
            f"projected to raise HDL by roughly {0.33*interventions.weight_loss_kg:.1f} mg/dL, cut "
            f"triglycerides by about {tg_drop_frac*100:.0f}%, lower CRP (inflammation) by "
            f"roughly {crp_drop_frac*100:.0f}%, meaningfully reduce liver fat (steatosis), and lower "
            f"HbA1c by roughly {hba1c_drop:.2f} points."
        )

    # ---------------------------------------------------------------
    # 2. Exercise effects
    # ---------------------------------------------------------------
    if interventions.exercise_min_per_week > 0:
        dose_factor = min(interventions.exercise_min_per_week / 150.0, 2.0)

        after["bp_systolic"] = max(after.get("bp_systolic", patient.get("bp_systolic", 120)) - 3.5 * dose_factor, 90)
        after["bp_diastolic"] = max(after.get("bp_diastolic", patient.get("bp_diastolic", 78)) - 2.0 * dose_factor, 60)

        # HDL: aerobic exercise raises HDL ~1.5-3 mg/dL at 150 min/week doses
        hdl_gain = 2.0 * dose_factor
        after["hdl"] = after.get("hdl", patient.get("hdl", 48)) + hdl_gain

        # Triglycerides: exercise-specific reduction independent of weight loss
        if after.get("triglycerides") is not None:
            after["triglycerides"] = after["triglycerides"] * (1 - min(0.1 * dose_factor, 0.25))

        crp_ex_drop = min(0.1 * dose_factor, 0.3)
        after["crp"] = after.get("crp", patient.get("crp", 2.0)) * (1 - crp_ex_drop)

        homa_ex_reduction = min(0.18 * dose_factor, 0.35)
        if after.get("homa_ir") is not None:
            after["homa_ir"] = after["homa_ir"] * (1 - homa_ex_reduction)
        if after.get("insulin") is not None:
            after["insulin"] = after["insulin"] * (1 - homa_ex_reduction * 0.6)

        # Glycemic outcomes: Umpierre et al. 2011 (JAMA) meta-analysis of
        # structured exercise in at-risk/T2D populations found roughly
        # 0.3-0.6 percentage-point HbA1c reductions at full dose; fasting
        # glucose moved via the same ADAG slope used in the weight-loss
        # block above, for consistency between the two numbers.
        hba1c_drop_ex = min(0.35 * dose_factor, 0.7)
        if after.get("hba1c") is not None:
            after["hba1c"] = max(after["hba1c"] - hba1c_drop_ex, 4.8)
        if after.get("fasting_glucose") is not None:
            after["fasting_glucose"] = max(after["fasting_glucose"] - hba1c_drop_ex * 28.7, 70)

        if interventions.weight_loss_kg == 0:
            extra_loss = 2.5 * dose_factor
            after["weight_kg"] = max(after.get("weight_kg", weight_kg) - extra_loss, 30)
            after["bmi"] = after["weight_kg"] / (height_m ** 2)

        after["activity_min_per_week"] = patient.get("activity_min_per_week", 0) + interventions.exercise_min_per_week

        notes.append(
            f"{interventions.exercise_min_per_week:.0f} min/week of added moderate activity is "
            f"projected to raise HDL by about {hdl_gain:.1f} mg/dL, lower systolic BP by "
            f"~{3.5*dose_factor:.1f} mmHg, and lower HbA1c by roughly {hba1c_drop_ex:.2f} points, "
            f"mainly via improved insulin sensitivity and reduced inflammation."
        )

    # ---------------------------------------------------------------
    # 3. Sleep effects
    # ---------------------------------------------------------------
    if interventions.sleep_hours is not None and interventions.current_sleep_hours is not None:
        sleep_gain = max(interventions.sleep_hours - interventions.current_sleep_hours, 0)
        if sleep_gain > 0:
            sleep_gain_capped = min(sleep_gain, 3.0)
            homa_sleep_reduction = min(0.05 * sleep_gain_capped, 0.15)
            if after.get("homa_ir") is not None:
                after["homa_ir"] = after["homa_ir"] * (1 - homa_sleep_reduction)
            if after.get("insulin") is not None:
                after["insulin"] = after["insulin"] * (1 - homa_sleep_reduction * 0.6)
            after["bp_systolic"] = max(after.get("bp_systolic", patient.get("bp_systolic", 120)) - 1.5 * sleep_gain_capped, 90)
            after["crp"] = after.get("crp", patient.get("crp", 2.0)) * (1 - min(0.04 * sleep_gain_capped, 0.15))
            after["sleep_hours"] = interventions.sleep_hours

            notes.append(
                f"Increasing sleep from {interventions.current_sleep_hours:.1f}h to "
                f"{interventions.sleep_hours:.1f}h/night is projected to improve insulin "
                f"sensitivity and modestly lower blood pressure and CRP."
            )

    # ---------------------------------------------------------------
    # 4. Sugar / diet reduction
    # ---------------------------------------------------------------
    if interventions.sugar_reduction_pct > 0:
        frac = min(interventions.sugar_reduction_pct / 100.0, 1.0)
        base_sugar = patient.get("sugar_g", 90)
        after["sugar_g"] = base_sugar * (1 - frac)
        # Added sugar/fructose has strong, specific literature support for
        # raising triglycerides (e.g. Stanhope et al. fructose-feeding
        # trials) - that's the reliable causal pathway. total_cholesterol
        # is deliberately NOT modified here for the same reason noted in
        # the weight-loss block above (ambiguous/non-monotonic in this data).
        if after.get("triglycerides") is not None:
            after["triglycerides"] = after["triglycerides"] * (1 - min(0.15 * frac, 0.3))
        # Direct glycemic benefit of cutting added sugar (smaller, more
        # specific effect than the weight-loss/exercise pathways above)
        hba1c_drop_sugar = 0.25 * frac
        if after.get("hba1c") is not None:
            after["hba1c"] = max(after["hba1c"] - hba1c_drop_sugar, 4.8)
        if after.get("fasting_glucose") is not None:
            after["fasting_glucose"] = max(after["fasting_glucose"] - hba1c_drop_sugar * 28.7, 70)
        # Fructose specifically (the dominant sugar in most "added sugar"
        # sources - sucrose and high-fructose corn syrup are both ~50%
        # fructose) has well-documented, specific literature linking it to
        # raised uric acid via purine metabolism during fructose
        # breakdown - a distinct mechanism from the general weight-loss
        # effect on uric acid already modeled above.
        if after.get("uric_acid") is not None:
            after["uric_acid"] = max(after["uric_acid"] * (1 - min(0.08 * frac, 0.15)), 2.0)
        notes.append(
            f"Cutting added sugar intake by {interventions.sugar_reduction_pct:.0f}% "
            f"(from ~{base_sugar:.0f}g/day) is projected to reduce triglycerides, lower "
            f"HbA1c by roughly {hba1c_drop_sugar:.2f} points, and modestly lower uric acid."
        )

    # ---------------------------------------------------------------
    # 5. Smoking cessation
    # ---------------------------------------------------------------
    if interventions.quit_smoking and patient.get("smoker_current"):
        after["smoker_current"] = 0
        after["crp"] = after.get("crp", patient.get("crp", 2.0)) * 0.85
        after["hdl"] = after.get("hdl", patient.get("hdl", 45)) + 3.0
        notes.append(
            "Quitting smoking is projected to modestly raise HDL and lower CRP "
            "(inflammation), on top of removing smoking's own direct contribution to risk."
        )

    # ---------------------------------------------------------------
    # 6. Reduce sedentary time (independent of added structured exercise -
    # breaking up sitting time improves glycemic control even without a
    # formal workout, per interrupted-sitting trials)
    # ---------------------------------------------------------------
    if interventions.sedentary_reduction_min > 0:
        after["sedentary_min_per_day"] = max(
            patient.get("sedentary_min_per_day", 480) - interventions.sedentary_reduction_min, 0
        )
        sed_factor = min(interventions.sedentary_reduction_min / 120.0, 1.5)
        homa_sed_reduction = min(0.06 * sed_factor, 0.15)
        if after.get("homa_ir") is not None:
            after["homa_ir"] = after["homa_ir"] * (1 - homa_sed_reduction)
        if after.get("insulin") is not None:
            after["insulin"] = after["insulin"] * (1 - homa_sed_reduction * 0.6)
        after["bp_systolic"] = max(after.get("bp_systolic", patient.get("bp_systolic", 120)) - 1.0 * sed_factor, 90)
        notes.append(
            f"Cutting sedentary time by {interventions.sedentary_reduction_min:.0f} min/day "
            f"(breaking up sitting, independent of workouts) is projected to modestly improve "
            f"insulin sensitivity and blood pressure."
        )

    # ---------------------------------------------------------------
    # 7. Increase fiber intake (independent of overall sugar reduction -
    # higher fiber slows glucose absorption and improves lipids)
    # ---------------------------------------------------------------
    if interventions.fiber_increase_g > 0:
        after["fiber_g"] = patient.get("fiber_g", 15) + interventions.fiber_increase_g
        frac = min(interventions.fiber_increase_g / 10.0, 2.0)
        if after.get("triglycerides") is not None:
            after["triglycerides"] = after["triglycerides"] * (1 - min(0.05 * frac, 0.15))
        after["hdl"] = after.get("hdl", patient.get("hdl", 45)) + 0.5 * frac
        notes.append(
            f"Adding {interventions.fiber_increase_g:.0f}g/day of fiber is projected to modestly "
            f"improve triglycerides and HDL."
        )

    # ---------------------------------------------------------------
    # 8. Reduce alcohol intake
    # ---------------------------------------------------------------
    if interventions.alcohol_reduction_drinks > 0:
        base_drinks = patient.get("drinks_per_day", 1.0) or 0.0
        after["drinks_per_day"] = max(base_drinks - interventions.alcohol_reduction_drinks, 0)
        frac = min(interventions.alcohol_reduction_drinks / 2.0, 1.5)
        if after.get("triglycerides") is not None:
            after["triglycerides"] = after["triglycerides"] * (1 - min(0.08 * frac, 0.25))
        if patient.get("liver_fat_cap") is not None:
            after["liver_fat_cap"] = after.get("liver_fat_cap", patient["liver_fat_cap"]) * (1 - min(0.04 * frac, 0.2))
        after["bp_systolic"] = max(after.get("bp_systolic", patient.get("bp_systolic", 120)) - 1.5 * frac, 90)
        # GGT is the classic, specifically alcohol-sensitive liver enzyme
        # (more so than ALT/AST) - a well-established clinical marker used
        # to monitor drinking reduction specifically, given a distinctly
        # larger effect size here than the general weight-loss-linked
        # enzyme effect modeled above.
        if patient.get("ggt") is not None:
            after["ggt"] = after.get("ggt", patient["ggt"]) * (1 - min(0.12 * frac, 0.4))
        notes.append(
            f"Cutting {interventions.alcohol_reduction_drinks:.1f} drink(s)/day is projected to "
            f"lower triglycerides, liver fat, blood pressure, and GGT (a liver enzyme especially "
            f"sensitive to alcohol intake)."
        )

    # ---------------------------------------------------------------
    # 9. Stress reduction (approximate - PHQ-9 screening score improvement
    # is associated with modest reductions in cortisol-linked inflammation
    # and blood pressure in intervention trials, effect sizes are small
    # and heterogeneous compared to the other levers)
    # ---------------------------------------------------------------
    if interventions.stress_reduction_points > 0 and patient.get("phq9_score") is not None:
        after["phq9_score"] = max(patient["phq9_score"] - interventions.stress_reduction_points, 0)
        frac = min(interventions.stress_reduction_points / 5.0, 1.5)
        after["crp"] = after.get("crp", patient.get("crp", 2.0)) * (1 - min(0.03 * frac, 0.1))
        after["bp_systolic"] = max(after.get("bp_systolic", patient.get("bp_systolic", 120)) - 0.8 * frac, 90)
        notes.append(
            f"Reducing stress/mood-screening score by {interventions.stress_reduction_points:.0f} points "
            f"is projected to modestly lower CRP and blood pressure — the smallest-effect lever here, "
            f"included for completeness rather than as a primary driver."
        )

    for k, v in patient.items():
        after.setdefault(k, v)

    deltas = {
        k: (after[k] - before[k])
        for k in before
        if isinstance(before.get(k), (int, float)) and isinstance(after.get(k), (int, float))
    }

    return SimulationResult(before=before, after=after, deltas=deltas, narrative_points=notes)


def rank_interventions(patient: dict, candidate_interventions: dict, predict_fn) -> list:
    baseline_risk = predict_fn(patient)
    results = []
    for name, interv in candidate_interventions.items():
        sim = apply_interventions(patient, interv)
        new_risk = predict_fn(sim.after)
        results.append({
            "name": name,
            "baseline_risk": baseline_risk,
            "new_risk": new_risk,
            "risk_reduction_pp": (baseline_risk - new_risk) * 100,
        })
    results.sort(key=lambda r: r["risk_reduction_pp"], reverse=True)
    return results
