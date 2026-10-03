"""
Metis Backend - Input Validation

Defense-in-depth for a tool now explicitly framed as clinical decision
support: the intake form already enforces these bounds via HTML min/max
attributes, but that only protects the one intended entry point. Any
direct API call (a script, a future integration, a malformed request)
bypasses HTML validation entirely, so the backend enforces the same
physiological bounds independently. Values outside these ranges are not
just unusual - most are incompatible with life, so letting them silently
reach a model and produce a confident-looking risk percentage would be
actively misleading rather than just an edge case.
"""

PHYSIOLOGICAL_BOUNDS = {
    "weight_kg": (20, 300), "height_cm": (100, 250), "waist_cm": (40, 200),
    "bp_systolic": (60, 250), "bp_diastolic": (30, 150), "hdl": (10, 150),
    "total_cholesterol": (50, 500), "ldl": (10, 400), "triglycerides": (10, 2000),
    "insulin": (0, 300), "fasting_glucose": (40, 600), "hba1c": (3, 20),
    "uric_acid": (1, 20), "albumin": (1, 6), "wbc": (1, 50),
    "creatinine": (0.1, 20), "bun": (1, 150), "crp": (0, 50),
    "liver_fat_cap": (100, 400), "liver_stiffness": (1, 75),
    "alt": (1, 500), "ast": (1, 500), "ggt": (1, 500),
    "age": (0, 120), "sleep_hours": (0, 24), "phq9_score": (0, 27),
}


class ValidationError(Exception):
    """Raised when a patient dict has a physiologically impossible value."""


def validate_patient(patient: dict) -> None:
    """Raises ValidationError on the first out-of-bounds field found.
    Silently skips fields not in the patient dict (optional fields,
    simulator-only fields like sugar_g, etc. - this only guards the
    fields that have a known safe physiological range)."""
    for field, (lo, hi) in PHYSIOLOGICAL_BOUNDS.items():
        if field not in patient or patient[field] is None:
            continue
        value = patient[field]
        if not isinstance(value, (int, float)):
            raise ValidationError(f"'{field}' must be a number, got {type(value).__name__}")
        if value != value:  # NaN check (NaN != NaN is True)
            raise ValidationError(f"'{field}' is NaN, not a valid value")
        if not (lo <= value <= hi):
            raise ValidationError(
                f"'{field}'={value} is outside the physiologically plausible range "
                f"({lo}-{hi}) - this looks like a data entry error, not a real patient value."
            )
