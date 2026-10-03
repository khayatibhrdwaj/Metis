"""
Metis Backend - FastAPI service

Endpoints:
  GET  /api/health              - liveness check
  GET  /api/meta                - feature list, medians (defaults), model metrics
  POST /api/predict              - {patient: {...}} -> risk + SHAP breakdown (diabetes)
  POST /api/predict-all           - {patient: {...}} -> risk for all 5 diseases
                                     (diabetes, metabolic syndrome, NAFLD, CKD,
                                     hypertension) - powers the unified
                                     Metabolic Health dashboard
  POST /api/predict-disease        - {patient, disease} -> risk + SHAP breakdown
                                     for one of the 4 new disease models
                                     (disease in metabolic_syndrome/nafld/
                                     ckd/hypertension; see diseases.py)
  POST /api/simulate             - {patient, interventions} -> projected outcome
                                     for ALL 5 diseases
  POST /api/rank                 - {patient, candidates} -> ranked intervention list
  POST /api/explain               - {patient, shap, risk} -> GenAI explanation
                                     (required feature - returns a clear
                                     error message if GEMINI_API_KEY isn't
                                     configured; see genai.py)
  POST /api/contact                - {name, email, message} -> sends an email
                                     (required feature - returns a clear
                                     error message if SMTP isn't configured;
                                     see mailer.py)
  POST /api/auth/register          - {email, password} -> creates an account
  POST /api/auth/login             - {email, password} -> verifies login,
                                     returns the most recently saved profile
                                     (if any) so returning users skip intake
  POST /api/profile/save           - {email, profile} -> appends to that
                                     user's profile history
  GET  /api/profile/history/{email} - full list of past submissions
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load backend/.env (if present) before anything reads os.environ - this
# must happen before importing the modules below, since genai.py/mailer.py/
# auth.py read environment variables (some at import time, some lazily).
_env_path = Path(__file__).resolve().parent / ".env"
_env_loaded = load_dotenv(_env_path)

# Startup diagnostics: prints to the terminal so you can immediately see
# whether your .env was found and whether each required key was picked up,
# without ever printing the actual secret values. If GEMINI_API_KEY or
# SMTP_* still show as "NOT SET" after you've edited backend/.env, the
# most common causes are: you edited the file but didn't restart uvicorn
# (env vars are only read once, at process startup), the file isn't at
# backend/.env exactly, or a stray quote/typo in the variable name.
print(f"[Metis] .env file found at {_env_path}: {_env_loaded}")
_gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
print(f"[Metis] GEMINI_API_KEY: {'set (' + str(len(_gemini_key)) + ' chars)' if _gemini_key else 'NOT SET'}")
_smtp_user = os.environ.get("SMTP_USER") or ""
_smtp_pw = os.environ.get("SMTP_PASSWORD") or ""
print(f"[Metis] SMTP_USER: {'set' if _smtp_user else 'NOT SET'}, SMTP_PASSWORD: {'set' if _smtp_pw else 'NOT SET'}")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import predict_risk, shap_breakdown, get_metrics, get_medians, FEATURES
from input_validation import validate_patient, ValidationError
from simulator import Interventions, apply_interventions, rank_interventions
from genai import generate_explanation
from mailer import send_contact_message
import auth
import diseases

app = FastAPI(title="Metis API", version="1.0")

# CORS: needed for local dev when the frontend is opened from a different
# origin (e.g. a separate `http.server` on another port), and harmless to
# leave on for any external API-only consumers in production. In the
# single-process deployment (frontend served by this same app, see the
# static mount below) the browser never actually crosses origins, so this
# has no practical effect there - but a public deployment fronted by a
# CDN or separate static host should set ALLOWED_ORIGINS explicitly rather
# than relying on the "*" default.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allowed_origins == "*" else [o.strip() for o in _allowed_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    patient: Dict[str, Any]


class SimulateRequest(BaseModel):
    patient: Dict[str, Any]
    interventions: Dict[str, Any] = {}


class RankRequest(BaseModel):
    patient: Dict[str, Any]
    candidates: Dict[str, Dict[str, Any]]


class PredictDiseaseRequest(BaseModel):
    patient: Dict[str, Any]
    disease: str


class ExplainRequest(BaseModel):
    patient: Dict[str, Any]
    shap: List[Dict[str, Any]]
    risk: float
    disease: str = "diabetes"


class ContactRequest(BaseModel):
    name: str = ""
    email: str = ""
    message: str


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class SaveProfileRequest(BaseModel):
    email: str
    profile: Dict[str, Any]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/debug/config")
def debug_config():
    """Browser-checkable version of the startup diagnostics printed to the
    terminal - never returns actual secret values, only whether each is set."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    return {
        "env_file_found": _env_loaded,
        "env_file_path": str(_env_path),
        "gemini_api_key_set": bool(gemini_key),
        "smtp_user_set": bool(os.environ.get("SMTP_USER")),
        "smtp_password_set": bool(os.environ.get("SMTP_PASSWORD")),
    }


@app.get("/api/meta")
def meta():
    return {"features": FEATURES, "medians": get_medians(), "metrics": get_metrics()}


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        validate_patient(req.patient)
        risk = predict_risk(req.patient)
        shap = shap_breakdown(req.patient)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"risk": risk, "shap": shap}


@app.post("/api/predict-all")
def predict_all(req: PredictRequest):
    try:
        validate_patient(req.patient)
        diabetes_risk = predict_risk(req.patient)
        result = diseases.predict_all(req.patient, diabetes_risk)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post("/api/predict-disease")
def predict_disease(req: PredictDiseaseRequest):
    if req.disease not in diseases.DISEASE_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown disease: {req.disease}")
    try:
        validate_patient(req.patient)
        risk = diseases.predict_disease_risk(req.disease, req.patient)
        shap = diseases.disease_shap_breakdown(req.disease, req.patient)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"risk": risk, "shap": shap, "label": diseases.DISEASE_LABELS[req.disease]}


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    try:
        validate_patient(req.patient)
        interv = Interventions(**req.interventions)
        sim = apply_interventions(req.patient, interv)
        new_risk = predict_risk(sim.after)

        before_all = diseases.predict_all(req.patient, predict_risk(req.patient))
        after_all = diseases.predict_all(sim.after, new_risk)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Bad intervention field: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "before": sim.before,
        "after": sim.after,
        "deltas": sim.deltas,
        "narrative": sim.narrative_points,
        "new_risk": new_risk,
        "before_all_diseases": before_all,
        "after_all_diseases": after_all,
    }


@app.post("/api/rank")
def rank(req: RankRequest):
    try:
        candidates = {name: Interventions(**params) for name, params in req.candidates.items()}
        ranked = rank_interventions(req.patient, candidates, predict_risk)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ranked": ranked}


@app.post("/api/explain")
def explain(req: ExplainRequest):
    # Doesn't raise an HTTP error even when unconfigured - returns
    # {"available": false, "error": "..."} so the frontend can show the
    # user exactly what to do, rather than a generic 500.
    disease_name = diseases.DISEASE_LABELS.get(req.disease, "Type 2 Diabetes")
    return generate_explanation(req.patient, req.shap, req.risk, disease_name)


@app.post("/api/contact")
def contact(req: ContactRequest):
    return send_contact_message(req.name, req.email, req.message)


@app.post("/api/auth/register")
def register(req: RegisterRequest):
    return auth.register_user(req.email, req.password)


@app.post("/api/auth/login")
def login(req: LoginRequest):
    return auth.verify_login(req.email, req.password)


@app.post("/api/profile/save")
def save_profile(req: SaveProfileRequest):
    return auth.save_profile(req.email, req.profile)


@app.get("/api/profile/history/{email}")
def profile_history(email: str):
    return {"history": auth.get_profile_history(email)}


# ---------------------------------------------------------------------------
# Serve the frontend (everything not matched by an /api/* route above).
# html=True makes StaticFiles serve index.html for "/" and fall back to it
# for unknown sub-paths is NOT enabled here (check_dir default) - each page
# is served by its own filename (auth.html, intake.html, dashboard.html),
# matching how the frontend's own JS does its redirects.
# ---------------------------------------------------------------------------
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
