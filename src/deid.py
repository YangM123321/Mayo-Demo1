# src/deid.py
import hashlib, hmac, os, datetime as dt
from copy import deepcopy

# Use an env var for secrecy; provide a default for demo
DEID_SALT = os.environ.get("DEID_SALT", "change-me-demo-salt").encode()

def _hash_id(raw: str) -> str:
    """Stable, non-reversible pseudonym using HMAC-SHA256."""
    return hmac.new(DEID_SALT, raw.encode(), hashlib.sha256).hexdigest()[:16]

def _patient_offset(pid: str) -> int:
    """Stable 0..180-day offset per patient."""
    return int(hashlib.sha1((pid + "dates").encode()).hexdigest(), 16) % 181

def _shift_date(iso_dt: str, days: int) -> str:
    # Accepts "YYYY-MM-DD" or "...T...Z"; returns full ISO date-time with Z
    base = iso_dt.replace("Z", "")
    if "T" in base:
        d = dt.datetime.fromisoformat(base)
    else:
        d = dt.datetime.fromisoformat(base + "T00:00:00")
    return (d + dt.timedelta(days=days)).isoformat(timespec="seconds") + "Z"

def deid_observation(obs: dict) -> dict:
    """
    De-identify a FHIR Observation:
      - Pseudonymize Patient id in subject.reference
      - Shift effectiveDateTime / issued by a patient-specific offset
      - Drop performer, identifier, text (common places for identifiers)
    """
    o = deepcopy(obs)

    # Subject like "Patient/P001"
    subj_ref = o.get("subject", {}).get("reference")
    pseudo = "unknown"
    if subj_ref and subj_ref.startswith("Patient/"):
        real_pid = subj_ref.split("/", 1)[1]
        pseudo = _hash_id(real_pid)
        o["subject"]["reference"] = f"Patient/{pseudo}"

    # Drop likely sensitive fields if present
    for k in ("performer", "identifier", "text"):
        o.pop(k, None)

    # Shift dates
    offset = _patient_offset(pseudo)
    if "effectiveDateTime" in o:
        o["effectiveDateTime"] = _shift_date(o["effectiveDateTime"], offset)
    if "issued" in o:
        o["issued"] = _shift_date(o["issued"], offset)

    # Example generalization: if this is an "Age" Observation, clamp >= 90
    code_text = (o.get("code", {}).get("text") or "").lower()
    if "age" in code_text:
        vq = o.get("valueQuantity", {})
        v = vq.get("value")
        if isinstance(v, (int, float)) and v >= 89:
            vq["value"] = 90
            vq["unit"] = "years (90+)"
            o["valueQuantity"] = vq

    return o


