// Metis — intake page logic.
// Requires a logged-in account (set by auth.html). Collects the patient
// profile form, stores it locally for the dashboard, and also saves it to
// that account's server-side history so logging in again restores it.
const API_BASE = "/api";

function num(id) { return parseFloat(document.getElementById(id).value); }
function checked(id) { return document.getElementById(id).checked; }
function val(id) { return document.getElementById(id).value; }

function bindLiveLabel(id) {
  const el = document.getElementById(id);
  const out = document.getElementById(id + "Val");
  if (el && out) el.addEventListener("input", () => (out.textContent = el.value));
}

function bindRadioPills() {
  document.querySelectorAll(".radio-pill").forEach((pill) => {
    const input = pill.querySelector("input[type=radio]");
    input.addEventListener("change", () => {
      document.querySelectorAll(`.radio-pill[data-for], .radio-pill`).forEach((p) => {
        if (p.querySelector("input").name === input.name) p.classList.remove("selected");
      });
      pill.classList.add("selected");
    });
  });
}

function buildProfile() {
  const weight_kg = num("weight_kg");
  const height_cm = num("height_cm");
  const waist_cm = num("waist_cm");
  const gender = document.querySelector('input[name="gender"]:checked').value;

  return {
    age: num("age"), weight_kg, height_cm, waist_cm,
    bp_systolic: num("bp_systolic"), bp_diastolic: num("bp_diastolic"),
    hdl: num("hdl"), total_cholesterol: num("total_cholesterol"),
    ldl: num("ldl"), triglycerides: num("triglycerides"), insulin: num("insulin"),
    fasting_glucose: num("fasting_glucose"), hba1c: num("hba1c"),
    uric_acid: num("uric_acid"), albumin: num("albumin"), wbc: num("wbc"),
    creatinine: num("creatinine"), bun: num("bun"),
    crp: num("crp"), liver_fat_cap: num("liver_fat_cap"), liver_stiffness: num("liver_stiffness"),
    alt: num("alt"), ast: num("ast"), ggt: num("ggt"),
    sleep_hours: num("sleep_hours"), activity_min_per_week: num("activity_min_per_week"),
    sedentary_min_per_day: num("sedentary_min_per_day"), drinks_per_day: num("drinks_per_day"),
    sugar_g: num("sugar_g"), fiber_g: num("fiber_g"),
    smoker_current: checked("smoker_current") ? 1 : 0,
    occ_activity_level: parseInt(val("occ_activity_level"), 10),
    phq9_score: num("phq9_score"), poverty_ratio: num("poverty_ratio"),
    gender_male: gender === "Male" ? 1 : 0,
  };
}

function fillFormFrom(profile) {
  Object.entries(profile).forEach(([key, value]) => {
    if (key === "gender_male") {
      const id = value ? "gender-male" : "gender-female";
      const input = document.getElementById(id);
      if (input) { input.checked = true; input.dispatchEvent(new Event("change")); }
      return;
    }
    const el = document.getElementById(key);
    if (!el) return;
    if (el.type === "checkbox") el.checked = !!value;
    else el.value = value;
    const labelSpan = document.getElementById(key + "Val");
    if (labelSpan) labelSpan.textContent = value;
  });
}

function formatDate(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  } catch (e) {
    return iso;
  }
}

async function loadHistoryPanel(email) {
  const panel = document.getElementById("historyPanel");
  const list = document.getElementById("historyList");
  if (!panel || !list) return;
  try {
    const res = await fetch(`${API_BASE}/profile/history/${encodeURIComponent(email)}`);
    const data = await res.json();
    const history = data.history || [];
    if (history.length === 0) return; // keep panel hidden
    list.innerHTML = "";
    history.forEach((entry) => {
      const item = document.createElement("div");
      item.className = "history-item";
      item.innerHTML = `<span>${formatDate(entry.submitted_at)}</span>`;
      const loadBtn = document.createElement("button");
      loadBtn.type = "button";
      loadBtn.className = "btn btn-ghost";
      loadBtn.textContent = "Load";
      loadBtn.addEventListener("click", () => fillFormFrom(entry.profile));
      item.appendChild(loadBtn);
      list.appendChild(item);
    });
    panel.classList.remove("hidden");
  } catch (e) {
    // Backend unreachable - just don't show history, not a blocking error
    // for the intake form itself.
  }
}

function init() {
  const email = localStorage.getItem("metisUserEmail");
  if (!email) {
    window.location.href = "auth.html";
    return;
  }

  const liveLabelIds = [
    "age", "sleep_hours", "activity_min_per_week", "sedentary_min_per_day",
    "drinks_per_day", "sugar_g", "fiber_g", "phq9_score", "poverty_ratio",
  ];
  liveLabelIds.forEach(bindLiveLabel);
  bindRadioPills();

  document.getElementById("intakeForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const profile = buildProfile();
    localStorage.setItem("metisPatientProfile", JSON.stringify(profile));

    // Best-effort save to this account's server-side history. The app
    // still works offline-first via localStorage even if this fails, but
    // we surface the failure rather than hiding it entirely.
    try {
      await fetch(`${API_BASE}/profile/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, profile }),
      });
    } catch (err) {
      console.warn("Could not save profile history to backend:", err);
    }

    window.location.href = "dashboard.html";
  });

  // If a profile already exists locally (returning user editing), prefill the form
  const existing = localStorage.getItem("metisPatientProfile");
  if (existing) {
    try { fillFormFrom(JSON.parse(existing)); } catch (e) { /* ignore malformed stored profile */ }
  }

  loadHistoryPanel(email);
}

document.addEventListener("DOMContentLoaded", init);
