// Metis frontend — talks to the FastAPI backend over HTTP.
// A relative path works both in local dev and in production because the
// backend now serves this frontend itself (see backend/main.py's static
// mount) - so the page and the API are always same-origin.
const API_BASE = "/api";

const COLOR_GOOD = "#2E8B7D";
const COLOR_WARN = "#E8A33D";
const COLOR_BAD = "#C4577B";
const COLOR_BRAND = "#6E2FA0";
const COLOR_BRAND_LIGHT = "#C0A2E0";

const FEATURE_LABELS = {
  age: "Age", bmi: "BMI", waist_cm: "Waist circumference (cm)",
  waist_height_ratio: "Waist-to-height ratio", bp_systolic: "Systolic BP",
  bp_diastolic: "Diastolic BP", pulse_pressure: "Pulse pressure",
  hdl: "HDL cholesterol", total_cholesterol: "Total cholesterol",
  ldl: "LDL cholesterol", triglycerides: "Triglycerides",
  crp: "CRP (inflammation)", insulin: "Fasting insulin",
  homa_ir: "HOMA-IR (insulin resistance)", liver_fat_cap: "Liver fat (CAP score)",
  liver_stiffness: "Liver stiffness (fibrosis)", smoker_current: "Current smoker",
  sleep_hours: "Sleep (hrs/night)", activity_min_per_week: "Exercise (min/week)",
  sedentary_min_per_day: "Sedentary time (min/day)", drinks_per_day: "Alcohol (drinks/day)",
  sugar_g: "Added sugar (g/day)", kcal: "Calorie intake (kcal/day)",
  fiber_g: "Fiber intake (g/day)", occ_activity_level: "Occupational activity level",
  phq9_score: "Stress/mood screening score", poverty_ratio: "Income-to-poverty ratio",
  gender_male: "Male",
  alt: "ALT (liver enzyme)", ast: "AST (liver enzyme)", ggt: "GGT (liver enzyme)",
  uric_acid: "Uric acid", albumin: "Albumin", wbc: "White blood cell count",
  creatinine: "Creatinine", bun: "BUN (blood urea nitrogen)", egfr: "eGFR (kidney function)",
};

let medians = {};
let metrics = {};
let charts = {};
let debounceTimer = null;
let storedProfile = null;

// ---------------- helpers ----------------
function num(id) { return parseFloat(document.getElementById(id).value); }
function checked(id) { return document.getElementById(id).checked; }

function bindLiveLabel(id) {
  const el = document.getElementById(id);
  const out = document.getElementById(id + "Val");
  if (el && out) el.addEventListener("input", () => (out.textContent = el.value));
}

async function api(path, body) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : { method: "GET" };
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ---------------- build patient object from stored intake profile ----------------
function buildPatient() {
  const p = storedProfile;
  const bmi = p.weight_kg / Math.pow(p.height_cm / 100, 2);
  // fasting_glucose and hba1c are tracked/simulator-outcome metrics, never
  // model predictors (see backend/model.py's FEATURES list - both are
  // deliberately excluded to avoid the same circularity issue documented
  // extensively for the label definition). When the person didn't provide
  // a real lab value, fall back to a rough BMI-based proxy so HOMA-IR and
  // the dashboard tiles still have a reasonable illustrative number rather
  // than showing nothing.
  const fastingGlucoseProxy = 90 + (bmi - 25) * 1.2; // illustrative only
  const fasting_glucose = (p.fasting_glucose ?? null) !== null && !isNaN(p.fasting_glucose)
    ? p.fasting_glucose : fastingGlucoseProxy;
  // ADAG study formula (Nathan et al. 2008): eAG = 28.7 * HbA1c - 46.7,
  // inverted here to estimate HbA1c from glucose when HbA1c wasn't given.
  const hba1cProxy = (fasting_glucose + 46.7) / 28.7;
  const hba1c = (p.hba1c ?? null) !== null && !isNaN(p.hba1c) ? p.hba1c : hba1cProxy;
  const homa_ir = (p.insulin * fasting_glucose) / 405;

  const patient = {
    age: p.age, weight_kg: p.weight_kg, height_cm: p.height_cm, bmi,
    waist_cm: p.waist_cm, waist_height_ratio: p.waist_cm / p.height_cm,
    bp_systolic: p.bp_systolic, bp_diastolic: p.bp_diastolic,
    pulse_pressure: p.bp_systolic - p.bp_diastolic,
    hdl: p.hdl, total_cholesterol: p.total_cholesterol,
    ldl: p.ldl, triglycerides: p.triglycerides,
    fasting_glucose, hba1c,
    crp: p.crp, insulin: p.insulin, homa_ir,
    liver_fat_cap: p.liver_fat_cap, liver_stiffness: p.liver_stiffness,
    smoker_current: p.smoker_current,
    sleep_hours: p.sleep_hours, activity_min_per_week: p.activity_min_per_week,
    sedentary_min_per_day: p.sedentary_min_per_day,
    drinks_per_day: p.drinks_per_day, sugar_g: p.sugar_g,
    kcal: medians.kcal ?? 2000, fiber_g: p.fiber_g,
    occ_activity_level: p.occ_activity_level,
    phq9_score: p.phq9_score, poverty_ratio: p.poverty_ratio,
    gender_male: p.gender_male,
  };

  // alt/ast/ggt/uric_acid/albumin/wbc are genuinely optional model
  // predictors (unlike the fields above, nothing downstream depends on
  // them) - only include a key when the person actually provided a valid
  // number. Omitting the key entirely (rather than sending NaN) lets the
  // backend's own population-median fallback handle it cleanly - see
  // model.py's _row(), which does feature_dict.get(f, medians[f]).
  ["alt", "ast", "ggt", "uric_acid", "albumin", "wbc", "creatinine", "bun"].forEach((key) => {
    const v = p[key];
    if (v !== null && v !== undefined && !isNaN(v)) patient[key] = v;
  });

  return patient;
}

function buildInterventions() {
  return {
    weight_loss_kg: num("i_weight_loss_kg"),
    exercise_min_per_week: num("i_exercise_min_per_week"),
    sleep_hours: num("i_sleep_hours"),
    current_sleep_hours: storedProfile.sleep_hours,
    sugar_reduction_pct: num("i_sugar_reduction_pct"),
    quit_smoking: checked("i_quit_smoking"),
    sedentary_reduction_min: num("i_sedentary_reduction_min"),
    fiber_increase_g: num("i_fiber_increase_g"),
    alcohol_reduction_drinks: num("i_alcohol_reduction_drinks"),
    stress_reduction_points: num("i_stress_reduction_points"),
  };
}

// ---------------- gauge (half-doughnut) ----------------
function renderGauge(riskPct) {
  const color = riskPct < 15 ? "#5FE0C4" : riskPct < 30 ? "#F5C563" : "#F09BB4";
  const ctx = document.getElementById("gaugeChart");
  const data = {
    datasets: [{
      data: [riskPct, 60 - Math.min(riskPct, 60)],
      backgroundColor: [color, "rgba(255,255,255,0.25)"],
      borderWidth: 0,
    }],
  };
  if (charts.gauge) {
    charts.gauge.data = data;
    charts.gauge.update();
  } else {
    charts.gauge = new Chart(ctx, {
      type: "doughnut",
      data,
      options: {
        maintainAspectRatio: false,
        circumference: 180, rotation: -90, cutout: "62%",
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
      },
    });
  }
  document.getElementById("gaugeNumber").textContent = `${riskPct.toFixed(0)}%`;
}

// ---------------- Radar: key markers vs healthy reference ----------------
// Each marker mapped to a 0-100 "deviation from healthy" score (0 = healthy,
// 100 = far outside healthy range). Reference polygon is a flat mild
// baseline for visual contrast, not a claim that "20" is a precise cutoff.
function markerDeviationScore(name, value) {
  const clamp = (x) => Math.max(0, Math.min(100, x));
  switch (name) {
    case "BMI": return clamp(((value - 22) / (38 - 22)) * 100);
    case "Waist:Height": return clamp(((value - 0.45) / (0.68 - 0.45)) * 100);
    case "HDL (inv)": return clamp(((55 - value) / (55 - 30)) * 100);
    case "Triglycerides": return clamp(((value - 100) / (300 - 100)) * 100);
    case "CRP": return clamp(((value - 1) / (10 - 1)) * 100);
    case "HOMA-IR": return clamp(((value - 1.5) / (6 - 1.5)) * 100);
    default: return 0;
  }
}

function renderRadarChart(patient) {
  const labels = ["BMI", "Waist:Height", "HDL (inv)", "Triglycerides", "CRP", "HOMA-IR"];
  const values = [
    patient.bmi,
    patient.waist_height_ratio,
    patient.hdl,
    patient.triglycerides,
    patient.crp,
    patient.homa_ir,
  ];
  const patientScores = labels.map((l, i) => markerDeviationScore(l, values[i]));
  const healthyRef = labels.map(() => 15);

  const cfg = {
    type: "radar",
    data: {
      labels,
      datasets: [
        {
          label: "You",
          data: patientScores,
          backgroundColor: "rgba(110, 47, 160, 0.18)",
          borderColor: COLOR_BRAND,
          pointBackgroundColor: COLOR_BRAND,
        },
        {
          label: "Healthy reference",
          data: healthyRef,
          backgroundColor: "rgba(46, 139, 125, 0.08)",
          borderColor: COLOR_GOOD,
          borderDash: [4, 4],
          pointRadius: 0,
        },
      ],
    },
    options: {
      plugins: { legend: { position: "bottom", labels: { font: { family: "Clash Grotesk" } } } },
      scales: { r: { min: 0, max: 100, ticks: { display: false }, pointLabels: { font: { size: 11, family: "Clash Grotesk" } } } },
    },
  };
  if (charts.radar) charts.radar.destroy();
  charts.radar = new Chart(document.getElementById("radarChart"), cfg);
}

// ---------------- SHAP bar chart ----------------
function renderShapChart(shap) {
  const top = shap.slice(0, 15);
  const labels = top.map((d) => FEATURE_LABELS[d.feature] || d.feature).reverse();
  const values = top.map((d) => d.shap).reverse();
  const colors = values.map((v) => (v > 0 ? COLOR_BAD : COLOR_GOOD));

  const cfg = {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: colors }] },
    options: {
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: { legend: { display: false }, title: { display: true, text: "Rose = increases risk, teal = decreases", font: { family: "Clash Grotesk" } } },
      scales: { x: { title: { display: true, text: "Impact on risk (log-odds)" } } },
    },
  };
  if (charts.shap) charts.shap.destroy();
  charts.shap = new Chart(document.getElementById("shapChart"), cfg);
}

// ---------------- Rank bar chart ----------------
function renderRankChart(ranked) {
  const labels = ranked.map((r) => r.name).reverse();
  const values = ranked.map((r) => r.risk_reduction_pp).reverse();
  const cfg = {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: COLOR_BRAND }] },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: "Predicted risk reduction (percentage points)" } } },
    },
  };
  if (charts.rank) charts.rank.destroy();
  charts.rank = new Chart(document.getElementById("rankChart"), cfg);
}

// ---------------- Trajectory line chart ----------------
function trajectory(riskNow, riskTarget, months) {
  return months.map((m) => riskNow + (riskTarget - riskNow) * (1 - Math.exp(-m / 5)));
}

function renderTrajectoryChart(riskNow, lifestyleRisk, aggressiveRisk, disease) {
  const months = [0, 3, 6, 9, 12];
  const noChangeRisk = Math.min(riskNow * 1.15, 0.95);
  const label = DISEASE_DISPLAY_NAMES[disease] || "Diabetes";
  const cfg = {
    type: "line",
    data: {
      labels: months.map((m) => `${m}mo`),
      datasets: [
        { label: "No change", data: trajectory(riskNow, noChangeRisk, months).map((v) => v * 100), borderColor: COLOR_BAD, fill: false },
        { label: "Your simulated plan", data: trajectory(riskNow, lifestyleRisk, months).map((v) => v * 100), borderColor: COLOR_WARN, fill: false },
        { label: "Aggressive intervention", data: trajectory(riskNow, aggressiveRisk, months).map((v) => v * 100), borderColor: COLOR_GOOD, fill: false },
      ],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { position: "top" }, title: { display: true, text: `Predicted ${label} risk over time`, font: { family: "Clash Grotesk" } } },
      scales: { y: { title: { display: true, text: `Predicted ${label} risk (%)` } } },
    },
  };
  if (charts.trajectory) charts.trajectory.destroy();
  charts.trajectory = new Chart(document.getElementById("trajectoryChart"), cfg);
}

// ---------------- metric formatting ----------------
function setMetric(id, deltaId, value, delta, suffix = "") {
  document.getElementById(id).textContent = `${value}${suffix}`;
  if (deltaId) {
    const el = document.getElementById(deltaId);
    el.textContent = delta > 0 ? `up +${delta.toFixed(1)}` : delta < 0 ? `down ${delta.toFixed(1)}` : "";
    el.className = "metric-delta " + (delta > 0 ? "up" : delta < 0 ? "down" : "");
  }
}

// ---------------- GenAI explanation (required feature - always shown) ----------------
async function renderGenaiExplanation(patient, shap, risk, disease) {
  const box = document.getElementById("genaiBox");
  const textEl = document.getElementById("genaiText");
  const titleEl = document.getElementById("genaiDiseaseLabel");
  if (titleEl) titleEl.textContent = DISEASE_DISPLAY_NAMES[disease] || disease;
  box.className = "genai-box";
  textEl.textContent = "Loading AI explanation…";
  try {
    const res = await api("/explain", { patient, shap, risk, disease });
    if (res.available) {
      textEl.textContent = res.text;
    } else {
      box.className = "genai-box genai-error";
      textEl.innerHTML =
        `<strong>AI explanation isn't set up yet.</strong><br>${res.error || "Unknown error."}` +
        `<ol class="genai-setup-steps">` +
        `<li>Get a free key (no credit card) at <a href="https://aistudio.google.com/apikey" target="_blank">aistudio.google.com/apikey</a></li>` +
        `<li>Set it: <code>export GEMINI_API_KEY="..."</code> (or <code>$env:GEMINI_API_KEY</code> on Windows)</li>` +
        `<li>Restart the backend: <code>uvicorn main:app --reload --port 8000</code></li>` +
        `<li>Reload this page</li>` +
        `</ol>`;
    }
  } catch (e) {
    box.className = "genai-box genai-error";
    textEl.innerHTML = `<strong>Could not reach the AI explanation endpoint.</strong><br>${e.message || e}`;
  }
}

// ---------------- main render pipeline ----------------
const DISEASE_DISPLAY_NAMES = {
  diabetes: "Type 2 Diabetes",
  metabolic_syndrome: "Metabolic Syndrome",
  nafld: "Fatty Liver Disease (NAFLD)",
  ckd: "Chronic Kidney Disease",
  hypertension: "Hypertension",
};

let currentDisease = "diabetes";

function riskBadgeClass(riskPct) {
  if (riskPct < 15) return "risk-low";
  if (riskPct < 40) return "risk-moderate";
  return "risk-high";
}
// Shape symbols give a second, color-independent channel alongside the
// text label itself - color alone (even with text) can still be the
// fastest thing a sighted user's eye catches first, so for someone with
// a color vision deficiency these distinct shapes (circle/triangle/
// square) carry the same at-a-glance meaning the color would for
// everyone else. Plain Unicode, not an icon font or SVG, so it renders
// identically everywhere with no extra asset.
function riskBadgeLabel(riskPct) {
  if (riskPct < 15) return "\u25CF Low";
  if (riskPct < 40) return "\u25B2 Moderate";
  return "\u25A0 High";
}

async function renderDiseaseRiskRow(patient) {
  const result = await api("/predict-all", { patient });
  Object.entries(result).forEach(([disease, data]) => {
    const pct = data.risk * 100;
    const valueEl = document.getElementById(`risk_${disease}`);
    const badgeEl = document.getElementById(`badge_${disease}`);
    if (valueEl) valueEl.textContent = `${pct.toFixed(0)}%`;
    if (badgeEl) {
      badgeEl.textContent = riskBadgeLabel(pct);
      badgeEl.className = "disease-risk-badge " + riskBadgeClass(pct);
    }
  });
  return result;
}

// Fetches and renders the SHAP breakdown + explanation text for whichever
// disease is currently selected on the Risk & Drivers tab. Diabetes uses
// /api/predict (the full stacked ensemble + its own SHAP explainer); the
// 4 new diseases use /api/predict-disease (single monotonic XGBoost each,
// see backend/diseases.py). GenAI explanations stay diabetes-only for now.
async function loadDriversForDisease(disease, patient) {
  const displayName = DISEASE_DISPLAY_NAMES[disease] || disease;
  let risk, shap;
  if (disease === "diabetes") {
    const pred = await api("/predict", { patient });
    risk = pred.risk;
    shap = pred.shap;
    renderGauge(risk * 100);
  } else {
    const pred = await api("/predict-disease", { patient, disease });
    risk = pred.risk;
    shap = pred.shap;
  }
  renderShapChart(shap);
  renderGenaiExplanation(patient, shap, risk, disease);

  const topDrivers = shap.slice(0, 3).map((d) => FEATURE_LABELS[d.feature] || d.feature).join(", ");
  document.getElementById("driversExplanation").innerHTML =
    `<strong>Explanation:</strong> ${displayName} risk is primarily driven by <strong>${topDrivers}</strong>. ` +
    `The model estimates a ${(risk * 100).toFixed(0)}% probability of ${displayName.toLowerCase()}-risk classification ` +
    `based on ${shap.length} clinical, inflammatory, occupational, and lifestyle factors, ` +
    `calibrated on real 2021-2023 NHANES participant data.`;

  return { risk, shap };
}

function setActiveDiseasePill(disease) {
  // Two separate pill rows (Risk & Drivers tab, AI Explanation tab) share
  // the same currentDisease state - keep both in sync regardless of which
  // one was clicked, so switching tabs never shows a stale selection.
  // Trajectory's pills are deliberately excluded - see initDiseaseSelectors.
  document.querySelectorAll(".disease-pill:not(.disease-pill-traj)").forEach((p) => {
    p.classList.toggle("active", p.dataset.disease === disease);
  });
}

// Trajectory tab state: cached from the last Simulator run so switching
// disease here doesn't need a fresh API round-trip, and deliberately kept
// independent of currentDisease (viewing a trajectory for one disease
// shouldn't change what Risk & Drivers / AI Explanation are showing).
let lastTrajectoryData = null; // { diabetes: {before, after, aggressive}, ... }

function renderTrajectoryForDisease(disease) {
  if (!lastTrajectoryData || !lastTrajectoryData[disease]) return;
  const { before, after, aggressive } = lastTrajectoryData[disease];
  renderTrajectoryChart(before, after, aggressive, disease);
}

// Printable patient summary: fetches fresh risk + top-driver data for
// all 5 diseases (not reused from cached state, since the person may
// have just edited their profile), populates the hidden .print-summary
// section, then opens the browser's native print dialog - lets the
// person "print" to an actual printer or save as PDF without needing any
// backend PDF library.
async function buildAndPrintSummary() {
  const patient = buildPatient();
  document.getElementById("printDate").textContent = "Generated " + new Date().toLocaleString();

  const allRisks = await api("/predict-all", { patient });
  const tbody = document.querySelector("#printRiskTable tbody");
  tbody.innerHTML = "";

  for (const disease of ["diabetes", "metabolic_syndrome", "nafld", "ckd", "hypertension"]) {
    let risk, shap;
    if (disease === "diabetes") {
      const pred = await api("/predict", { patient });
      risk = pred.risk; shap = pred.shap;
    } else {
      const pred = await api("/predict-disease", { patient, disease });
      risk = pred.risk; shap = pred.shap;
    }
    const topDrivers = shap.slice(0, 3).map((d) => FEATURE_LABELS[d.feature] || d.feature).join(", ");
    const row = document.createElement("tr");
    row.innerHTML = `<td>${DISEASE_DISPLAY_NAMES[disease]}</td><td>${(risk * 100).toFixed(0)}%</td><td>${topDrivers}</td>`;
    tbody.appendChild(row);
  }

  window.print();
}

function initPrintSummary() {
  const btn = document.getElementById("printSummaryBtn");
  if (!btn) return;
  btn.addEventListener("click", () => {
    btn.disabled = true;
    btn.textContent = "Preparing summary…";
    buildAndPrintSummary()
      .catch((e) => console.error("Print summary error:", e))
      .finally(() => {
        btn.disabled = false;
        btn.textContent = "🖨️ Print / save summary for your doctor";
      });
  });
}

function initDiseaseSelectors() {
  // Pills on the Risk & Drivers tab and the AI Explanation tab (SHAP +
  // GenAI explanation selection - shared currentDisease state)
  document.querySelectorAll(".disease-pill:not(.disease-pill-traj)").forEach((pill) => {
    pill.addEventListener("click", async () => {
      currentDisease = pill.dataset.disease;
      setActiveDiseasePill(currentDisease);
      const patient = buildPatient();
      await loadDriversForDisease(currentDisease, patient);
    });
  });

  // Pills on the Trajectory tab - independent selection, no API call
  // needed since all 5 diseases' before/after/aggressive risk were
  // already computed by the last Simulator run (see renderSimulator).
  document.querySelectorAll(".disease-pill-traj").forEach((pill) => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".disease-pill-traj").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      renderTrajectoryForDisease(pill.dataset.disease);
    });
  });

  // Cards on the Home tab: clicking one jumps to Risk & Drivers with that disease selected
  document.querySelectorAll(".disease-risk-card").forEach((card) => {
    card.addEventListener("click", () => {
      const disease = card.dataset.disease;
      document.querySelector('.disease-pill[data-disease="' + disease + '"]')?.click();
      document.querySelector('.tab-btn[data-tab="drivers"]')?.click();
    });
  });
}

async function renderAll() {
  const patient = buildPatient();

  document.getElementById("t_bmi").textContent = patient.bmi.toFixed(1);
  document.getElementById("t_waist").textContent = patient.waist_cm.toFixed(0);
  document.getElementById("t_bp").textContent = `${patient.bp_systolic.toFixed(0)}/${patient.bp_diastolic.toFixed(0)}`;
  document.getElementById("t_lipids").textContent = `${patient.hdl.toFixed(0)} / ${patient.triglycerides.toFixed(0)}`;
  document.getElementById("t_glucose").textContent = patient.fasting_glucose.toFixed(0);
  document.getElementById("t_hba1c").textContent = patient.hba1c.toFixed(1);
  document.getElementById("t_crp").textContent = patient.crp.toFixed(1);
  document.getElementById("t_liver").textContent = patient.liver_fat_cap.toFixed(0);
  document.getElementById("t_sleep").textContent = patient.sleep_hours.toFixed(1);
  document.getElementById("t_activity").textContent = patient.activity_min_per_week.toFixed(0);
  document.getElementById("t_stress").textContent = patient.phq9_score.toFixed(0);
  document.getElementById("t_smoke").textContent = patient.smoker_current ? "Current smoker" : "Non-smoker";

  renderRadarChart(patient);

  const allRisks = await renderDiseaseRiskRow(patient);
  const diabetesRisk = allRisks.diabetes.risk;

  await loadDriversForDisease(currentDisease, patient);

  await renderSimulator(patient, diabetesRisk);
}

async function renderSimulator(patient, baselineRisk) {
  const interventions = buildInterventions();
  const sim = await api("/simulate", { patient, interventions });

  setMetric("outRisk", "outRiskDelta", (sim.new_risk * 100).toFixed(0) + "%", (sim.new_risk - baselineRisk) * 100);
  setMetric("outBmi", "outBmiDelta", sim.after.bmi.toFixed(1), sim.after.bmi - patient.bmi);
  setMetric("outHdl", "outHdlDelta", (sim.after.hdl ?? patient.hdl).toFixed(0) + " mg/dL", (sim.after.hdl ?? patient.hdl) - patient.hdl);
  setMetric("outTrig", "outTrigDelta", (sim.after.triglycerides ?? patient.triglycerides).toFixed(0) + " mg/dL", (sim.after.triglycerides ?? patient.triglycerides) - patient.triglycerides);
  setMetric("outGlucose", "outGlucoseDelta", (sim.after.fasting_glucose ?? patient.fasting_glucose).toFixed(0) + " mg/dL", (sim.after.fasting_glucose ?? patient.fasting_glucose) - patient.fasting_glucose);
  setMetric("outHba1c", "outHba1cDelta", (sim.after.hba1c ?? patient.hba1c).toFixed(1) + "%", (sim.after.hba1c ?? patient.hba1c) - patient.hba1c);

  // The 4 new disease models - backend already computes before/after for
  // all 5 in one /api/simulate call (see main.py), just display them here.
  if (sim.before_all_diseases && sim.after_all_diseases) {
    const diseaseOutMap = {
      metabolic_syndrome: "outMetSyn",
      nafld: "outNafld",
      ckd: "outCkd",
      hypertension: "outHtn",
    };
    Object.entries(diseaseOutMap).forEach(([disease, elId]) => {
      const before = sim.before_all_diseases[disease].risk;
      const after = sim.after_all_diseases[disease].risk;
      setMetric(elId, elId + "Delta", (after * 100).toFixed(0) + "%", (after - before) * 100);
    });
  }

  const list = document.getElementById("narrativeList");
  list.innerHTML = sim.narrative.length
    ? sim.narrative.map((n) => `<li>${n}</li>`).join("")
    : "<li class='hint'>Adjust the sliders above to simulate an intervention.</li>";

  const candidates = {
    "Lose 8 kg": { weight_loss_kg: 8 },
    "150 min/week exercise": { exercise_min_per_week: 150 },
    "Cut sedentary time 2h/day": { sedentary_reduction_min: 120 },
    "Improve sleep to 8h": { sleep_hours: 8.0, current_sleep_hours: patient.sleep_hours },
    "Cut added sugar 50%": { sugar_reduction_pct: 50 },
    "Add 15g fiber/day": { fiber_increase_g: 15 },
  };
  if (patient.drinks_per_day > 0) candidates["Cut alcohol in half"] = { alcohol_reduction_drinks: patient.drinks_per_day / 2 };
  if (patient.smoker_current) candidates["Quit smoking"] = { quit_smoking: true };

  const rankRes = await api("/rank", { patient, candidates });
  renderRankChart(rankRes.ranked);

  const aggressiveInterventions = {
    weight_loss_kg: Math.max(interventions.weight_loss_kg, 12),
    exercise_min_per_week: Math.max(interventions.exercise_min_per_week, 250),
    sleep_hours: 8.0, current_sleep_hours: patient.sleep_hours,
    sugar_reduction_pct: 70,
    sedentary_reduction_min: Math.max(interventions.sedentary_reduction_min, 120),
    fiber_increase_g: Math.max(interventions.fiber_increase_g, 15),
    alcohol_reduction_drinks: patient.drinks_per_day,
    quit_smoking: !!patient.smoker_current,
  };
  const aggSim = await api("/simulate", { patient, interventions: aggressiveInterventions });

  // Cache before/after/aggressive risk for all 5 diseases so the
  // Trajectory tab's disease pills can switch instantly without another
  // API round-trip. sim/aggSim already carry all 5 via /api/simulate's
  // before_all_diseases/after_all_diseases (see main.py).
  lastTrajectoryData = { diabetes: { before: baselineRisk, after: sim.new_risk, aggressive: aggSim.new_risk } };
  if (sim.before_all_diseases && sim.after_all_diseases && aggSim.after_all_diseases) {
    ["metabolic_syndrome", "nafld", "ckd", "hypertension"].forEach((disease) => {
      lastTrajectoryData[disease] = {
        before: sim.before_all_diseases[disease].risk,
        after: sim.after_all_diseases[disease].risk,
        aggressive: aggSim.after_all_diseases[disease].risk,
      };
    });
  }

  const activeTrajPill = document.querySelector(".disease-pill-traj.active");
  renderTrajectoryForDisease(activeTrajPill ? activeTrajPill.dataset.disease : "diabetes");
}

function scheduleRender() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    renderAll().catch((e) => console.error("Render error:", e));
  }, 250);
}

// ---------------- Contact Us chat box ----------------
function appendChatBubble(text, kind) {
  const container = document.getElementById("chatMessages");
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble chat-bubble-${kind}`;
  bubble.textContent = text;
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
}

async function sendContactMessage() {
  const name = document.getElementById("chatName").value.trim();
  const email = document.getElementById("chatEmail").value.trim();
  const messageEl = document.getElementById("chatMessage");
  const message = messageEl.value.trim();
  if (!message) return;

  appendChatBubble(message, "user");
  messageEl.value = "";

  const btn = document.getElementById("chatSendBtn");
  btn.disabled = true;
  btn.textContent = "Sending…";

  try {
    const res = await api("/contact", { name, email, message });
    if (res.success) {
      appendChatBubble("Message sent — thanks! The team will get back to you.", "success");
    } else {
      appendChatBubble(`Couldn't send: ${res.error || "Unknown error."}`, "error");
    }
  } catch (e) {
    appendChatBubble(`Couldn't reach the server: ${e.message || e}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Send";
  }
}

function initContactForm() {
  const btn = document.getElementById("chatSendBtn");
  if (!btn) return;
  btn.addEventListener("click", sendContactMessage);
  document.getElementById("chatMessage").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendContactMessage();
    }
  });
}

// ---------------- wire up events ----------------
function init() {
  const email = localStorage.getItem("metisUserEmail");
  if (!email) {
    window.location.href = "auth.html";
    return;
  }
  const raw = localStorage.getItem("metisPatientProfile");
  if (!raw) {
    window.location.href = "intake.html";
    return;
  }
  try {
    storedProfile = JSON.parse(raw);
  } catch (e) {
    window.location.href = "intake.html";
    return;
  }

  const logoutBtn = document.getElementById("logoutBtn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
      localStorage.removeItem("metisUserEmail");
      localStorage.removeItem("metisPatientProfile");
      window.location.href = "auth.html";
    });
  }

  const liveLabelIds = [
    "i_weight_loss_kg", "i_exercise_min_per_week", "i_sedentary_reduction_min",
    "i_sleep_hours", "i_sugar_reduction_pct", "i_fiber_increase_g",
    "i_alcohol_reduction_drinks", "i_stress_reduction_points",
  ];
  liveLabelIds.forEach(bindLiveLabel);

  document.querySelectorAll(".sim-grid input, .sim-grid select").forEach((el) => {
    el.addEventListener("input", scheduleRender);
    el.addEventListener("change", scheduleRender);
  });

  initContactForm();
  initDiseaseSelectors();
  initPrintSummary();

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
      // Charts created while their tab was hidden (display:none gives the
      // canvas 0 width/height at creation time) don't reliably pick up
      // their real size again on every subsequent reveal - the browser's
      // resize observer only reliably refires on an actual size CHANGE,
      // not a second identical hide/reveal cycle. Explicitly resizing
      // every live chart on each tab click is cheap and makes this robust
      // regardless of how many times a tab has been visited before.
      // Deferred to the next frame: resize() reads the container's
      // current layout size, but the "hidden" class was just removed in
      // this same synchronous tick, so the browser may not have finished
      // reflowing yet - calling resize() immediately can still see the
      // pre-reveal (0-height) layout. requestAnimationFrame guarantees a
      // layout pass has happened first.
      requestAnimationFrame(() => {
        Object.values(charts).forEach((chart) => {
          if (chart && typeof chart.resize === "function") chart.resize();
        });
      });
    });
  });

  api("/meta").then((meta) => {
    medians = meta.medians;
    metrics = meta.metrics;
    document.getElementById("subtitleText").textContent =
      `A clinical decision-support tool, not a diagnostic device — internally validated on real ` +
      `NHANES August 2021-August 2023 data (diabetes model ROC-AUC ${metrics.roc_auc.toFixed(2)} on ` +
      `held-out test set, n=${metrics.n_train + metrics.n_test}). See the Know Yourself tab's ` +
      `Intended Use statement before relying on any result.`;
    renderAll().catch((e) => console.error("Initial render error:", e));
    loadHistoryTab();
  }).catch((e) => {
    console.error("Could not reach backend at " + API_BASE, e);
    document.getElementById("subtitleText").textContent =
      "Could not reach the backend API. Make sure the server is running (uvicorn main:app --port 8000) and you're viewing this page at that same address.";
  });
}

// ---------------- History tab ----------------
function formatHistoryDate(iso) {
  try {
    return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  } catch (e) {
    return iso;
  }
}

async function loadHistoryTab() {
  const container = document.getElementById("historyListDash");
  if (!container) return;
  const email = localStorage.getItem("metisUserEmail");
  try {
    const res = await api(`/profile/history/${encodeURIComponent(email)}`);
    const history = res.history || [];
    if (history.length === 0) {
      container.innerHTML = "<p class='hint'>No saved submissions yet — fill out the intake form to create your first one.</p>";
      return;
    }
    container.innerHTML = "";
    history.forEach((entry, idx) => {
      const item = document.createElement("div");
      item.className = "history-item-dash";
      const isLatest = idx === 0;
      item.innerHTML = `
        <div>
          <div class="history-date">${formatHistoryDate(entry.submitted_at)}${isLatest ? " (current)" : ""}</div>
          <div class="history-meta">Age ${entry.profile.age ?? "--"} · BMI inputs: ${entry.profile.weight_kg ?? "--"}kg / ${entry.profile.height_cm ?? "--"}cm · Waist ${entry.profile.waist_cm ?? "--"}cm</div>
        </div>
      `;
      const btn = document.createElement("button");
      btn.className = "btn btn-ghost";
      btn.textContent = isLatest ? "Currently viewing" : "Restore this snapshot";
      btn.disabled = isLatest;
      btn.addEventListener("click", () => {
        localStorage.setItem("metisPatientProfile", JSON.stringify(entry.profile));
        storedProfile = entry.profile;
        renderAll().catch((e) => console.error(e));
        loadHistoryTab();
        document.querySelector('.tab-btn[data-tab="home"]').click();
      });
      item.appendChild(btn);
      container.appendChild(item);
    });
  } catch (e) {
    container.innerHTML = "<p class='hint'>Could not load history right now.</p>";
  }
}

document.addEventListener("DOMContentLoaded", init);
