"""
Slalom business@ PII Scanner
Pre-submission middleware that detects PII before any prompt reaches a model.

Blocks or warns on:
  - SSNs, credit card numbers, phone numbers
  - Email addresses, IP addresses
  - Names (via NER), medical record numbers
  - Client-specific identifiers (Salesforce IDs, engagement codes)

Deploy as a pre-processing layer between the chat UI and the router.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import re

app = FastAPI(title="business@ PII Scanner", version="1.0.0")


# ── Detection Patterns ──────────────────────────────────────────────────────

PII_PATTERNS = {
    "ssn": {
        "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        "severity": "BLOCK",
        "description": "Social Security Number",
    },
    "credit_card": {
        "pattern": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        "severity": "BLOCK",
        "description": "Credit Card Number",
    },
    "phone_us": {
        "pattern": r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
        "severity": "WARN",
        "description": "US Phone Number",
    },
    "email": {
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "severity": "WARN",
        "description": "Email Address",
    },
    "ip_address": {
        "pattern": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        "severity": "WARN",
        "description": "IP Address",
    },
    "aws_key": {
        "pattern": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        "severity": "BLOCK",
        "description": "AWS Access Key",
    },
    "api_key_generic": {
        "pattern": r"\b(?:sk-|pk_|rk_)[a-zA-Z0-9]{20,}\b",
        "severity": "BLOCK",
        "description": "API Key (generic pattern)",
    },
    "salesforce_id": {
        "pattern": r"\b[a-zA-Z0-9]{15}(?:[a-zA-Z0-9]{3})?\b",
        "severity": "INFO",
        "description": "Possible Salesforce ID (15 or 18 char)",
        "min_context": True,  # Only flag if near keywords like "engagement", "opportunity"
    },
    "medical_record": {
        "pattern": r"\b(?:MRN|Medical Record|Patient ID)[:\s#]*\d{6,}\b",
        "severity": "BLOCK",
        "description": "Medical Record Number",
    },
    "date_of_birth": {
        "pattern": r"\b(?:DOB|Date of Birth|born)[:\s]*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        "severity": "WARN",
        "description": "Date of Birth",
    },
}

# Models that should NEVER receive client data
RESTRICTED_MODELS = {
    "deepseek-r1": "Chinese infrastructure — data sovereignty risk",
    "llama-4-maverick": "Open-source — no enterprise data agreements",
    "llama-4-scout": "Open-source — no enterprise data agreements",
    "llama-3-8b": "Open-source — no enterprise data agreements",
    "llama-3.3-70b": "Open-source — no enterprise data agreements",
    "gpt-oss-120b": "Open-source — no enterprise data agreements",
    "gpt-oss-20b": "Open-source — no enterprise data agreements",
}

# Data classification tiers
DATA_TIERS = {
    "public": {
        "description": "Non-sensitive, publicly available information",
        "allowed_models": "all",
    },
    "internal": {
        "description": "Slalom internal — methodologies, templates, processes",
        "blocked_models": list(RESTRICTED_MODELS.keys()),
    },
    "confidential": {
        "description": "Client data, engagement details, financial information",
        "allowed_models": [
            "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini",
            "claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5",
            "gemini-3.0-pro", "gemini-2.5-pro",
        ],
    },
    "restricted": {
        "description": "PII, PHI, financial records, legal documents",
        "allowed_models": [
            "gpt-5.2", "gpt-5.1",  # Azure-hosted with BAA
            "claude-opus-4.5", "claude-sonnet-4.5",  # AWS-hosted with BAA
        ],
        "requires_approval": True,
    },
}


# ── Scanner ──────────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    text: str
    target_model: Optional[str] = None
    data_tier: Optional[str] = "internal"  # public, internal, confidential, restricted


class PiiMatch(BaseModel):
    type: str
    description: str
    severity: str  # BLOCK, WARN, INFO
    match: str
    position: tuple[int, int]
    redacted: str


class ScanResponse(BaseModel):
    allowed: bool
    pii_detected: list[PiiMatch]
    model_allowed: bool
    model_warning: Optional[str] = None
    redacted_text: Optional[str] = None
    data_tier: str
    action: str  # PASS, WARN, BLOCK


@app.post("/scan", response_model=ScanResponse)
async def scan_for_pii(req: ScanRequest):
    """Scan text for PII before sending to any model."""

    matches = []
    text = req.text

    for pii_type, config in PII_PATTERNS.items():
        # Skip Salesforce ID detection unless near context keywords
        if config.get("min_context"):
            context_keywords = ["engagement", "opportunity", "account", "contact", "salesforce"]
            if not any(kw in text.lower() for kw in context_keywords):
                continue

        for m in re.finditer(config["pattern"], text, re.IGNORECASE):
            matched_text = m.group()
            # Redact: show first 2 and last 2 chars
            if len(matched_text) > 6:
                redacted = matched_text[:2] + "*" * (len(matched_text) - 4) + matched_text[-2:]
            else:
                redacted = "*" * len(matched_text)

            matches.append(PiiMatch(
                type=pii_type,
                description=config["description"],
                severity=config["severity"],
                match=matched_text,
                position=(m.start(), m.end()),
                redacted=redacted,
            ))

    # ── Model restriction check ──
    model_allowed = True
    model_warning = None

    if req.target_model:
        # Check restricted models
        if req.target_model in RESTRICTED_MODELS:
            if req.data_tier in ("confidential", "restricted"):
                model_allowed = False
                model_warning = f"Model '{req.target_model}' blocked for {req.data_tier} data: {RESTRICTED_MODELS[req.target_model]}"
            elif req.data_tier == "internal":
                model_warning = f"Warning: '{req.target_model}' is not recommended for internal data: {RESTRICTED_MODELS[req.target_model]}"

        # Check data tier allowed models
        tier_config = DATA_TIERS.get(req.data_tier, {})
        allowed_models = tier_config.get("allowed_models")
        if allowed_models and allowed_models != "all":
            if req.target_model not in allowed_models:
                model_allowed = False
                model_warning = f"Model '{req.target_model}' not approved for '{req.data_tier}' data tier. Approved: {', '.join(allowed_models)}"

    # ── Build redacted text ──
    redacted_text = text
    for match in sorted(matches, key=lambda m: m.position[0], reverse=True):
        start, end = match.position
        redacted_text = redacted_text[:start] + match.redacted + redacted_text[end:]

    # ── Determine action ──
    has_blocks = any(m.severity == "BLOCK" for m in matches)
    has_warns = any(m.severity == "WARN" for m in matches)

    if has_blocks or not model_allowed:
        action = "BLOCK"
        allowed = False
    elif has_warns:
        action = "WARN"
        allowed = True  # Allow with warning
    else:
        action = "PASS"
        allowed = True

    return ScanResponse(
        allowed=allowed,
        pii_detected=matches,
        model_allowed=model_allowed,
        model_warning=model_warning,
        redacted_text=redacted_text if matches else None,
        data_tier=req.data_tier or "internal",
        action=action,
    )


@app.get("/data-tiers")
async def get_data_tiers():
    """Return data classification tiers and their model allowlists."""
    return DATA_TIERS


@app.get("/restricted-models")
async def get_restricted_models():
    """Return models that should never receive sensitive data."""
    return RESTRICTED_MODELS


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
