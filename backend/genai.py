"""
Metis Backend - GenAI Explanation Layer

Turns a patient's SHAP breakdown into natural-language prose via the
Google Gemini API. Uses Gemini specifically because it has a genuinely
ongoing free tier (no credit card, no expiration - unlike Anthropic's
one-time trial credit) - see the setup steps below.

This is a REQUIRED feature of the app, not an optional extra:
/api/explain always attempts a real call, and if it can't (no key, bad
key, network/API error, rate limit), it returns a clear error message
instead of silently disappearing - the frontend surfaces that error to
the user rather than hiding the explanation box.
"""

import os

MODEL_NAME = os.environ.get("METIS_GENAI_MODEL", "gemini-3.8-flash")

FEATURE_LABELS = {
    "age": "age", "bmi": "BMI", "waist_cm": "waist circumference",
    "waist_height_ratio": "waist-to-height ratio", "bp_systolic": "systolic blood pressure",
    "bp_diastolic": "diastolic blood pressure", "pulse_pressure": "pulse pressure",
    "hdl": "HDL cholesterol", "total_cholesterol": "total cholesterol",
    "ldl": "LDL cholesterol", "triglycerides": "triglycerides",
    "crp": "CRP (inflammation marker)", "insulin": "fasting insulin",
    "homa_ir": "insulin resistance (HOMA-IR)", "liver_fat_cap": "liver fat",
    "liver_stiffness": "liver stiffness", "smoker_current": "smoking status",
    "sleep_hours": "sleep duration", "activity_min_per_week": "weekly exercise",
    "sedentary_min_per_day": "daily sedentary time", "kcal": "calorie intake",
    "fiber_g": "fiber intake", "occ_activity_level": "occupational activity level",
    "phq9_score": "stress/mood screening score", "poverty_ratio": "income-to-poverty ratio",
    "gender_male": "sex",
    "alt": "ALT (liver enzyme)", "ast": "AST (liver enzyme)", "ggt": "GGT (liver enzyme)",
    "uric_acid": "uric acid", "albumin": "albumin", "wbc": "white blood cell count",
    "creatinine": "creatinine", "bun": "BUN (blood urea nitrogen)", "egfr": "eGFR (kidney function)",
}


class GenAIConfigError(Exception):
    """Raised when the GenAI layer can't run because it isn't configured."""


def _client():
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise GenAIConfigError(
            "GEMINI_API_KEY is not set. This app requires it for the AI "
            "explanation feature - get a free key at "
            "https://aistudio.google.com/apikey (no credit card needed), "
            "set the environment variable, and restart the backend. See README.md."
        )
    try:
        from google import genai
    except ImportError:
        raise GenAIConfigError(
            "The 'google-genai' package isn't installed. Run: pip install google-genai"
        )
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        raise GenAIConfigError(f"Could not create Gemini client: {e}")


def generate_explanation(patient: dict, shap: list, risk: float, disease_name: str = "Type 2 Diabetes") -> dict:
    try:
        client = _client()
    except GenAIConfigError as e:
        return {"available": False, "text": "", "error": str(e)}

    top = sorted(shap, key=lambda d: abs(d["shap"]), reverse=True)[:5]
    driver_lines = []
    for d in top:
        label = FEATURE_LABELS.get(d["feature"], d["feature"])
        direction = "raising" if d["shap"] > 0 else "lowering"
        driver_lines.append(f"- {label} = {d['value']:.1f} ({direction} risk)")
    driver_text = "\n".join(driver_lines)

    prompt = (
        f"A clinical decision-support model (not a diagnostic device) estimates this person's "
        f"predicted {disease_name} risk at {risk*100:.0f}%. The factors contributing most to "
        f"that estimate (SHAP-based, direction relative to this specific prediction) are:\n"
        f"{driver_text}\n\n"
        f"Write a short (3-4 sentence), plain-language, encouraging explanation of what's "
        f"driving this {disease_name} risk estimate and what it means, suitable for someone "
        f"with no medical background. Do not diagnose or state/imply the person has or does not "
        f"have {disease_name} - describe this strictly as a risk estimate, never as a finding "
        f"about their actual health status. Do not give specific medical/dosing advice. End by "
        f"explicitly stating this is a decision-support estimate, not a diagnosis, and that it "
        f"does not replace professional medical evaluation - recommend discussing results with "
        f"a doctor. No headers, no bullet points, just prose."
    )

    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        text = (response.text or "").strip()
        if not text:
            return {"available": False, "text": "", "error": "The AI returned an empty response. Try again."}
        return {"available": True, "text": text, "error": None}
    except Exception as e:
        return {"available": False, "text": "", "error": f"Gemini API call failed: {e}"}
