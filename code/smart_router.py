"""
Slalom business@ Smart Model Router
Replaces static Model Advisor with intelligent auto-routing.

Routes requests to the optimal model based on:
  - Task type (reasoning, coding, creative, classification, translation)
  - Input language
  - File size
  - Cost tier preference
  - Required capabilities (tool calling, multimodal)

Deploy as middleware between the business@ frontend and model APIs.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from typing import Optional
import re

app = FastAPI(title="business@ Smart Router", version="1.0.0")


# ── Model Registry ──────────────────────────────────────────────────────────

class CostTier(str, Enum):
    BUDGET = "budget"
    STANDARD = "standard"
    PREMIUM = "premium"
    FRONTIER = "frontier"


class ModelCapability(str, Enum):
    TOOL_CALLING = "tool_calling"
    MULTIMODAL_INPUT = "multimodal_input"
    IMAGE_GEN = "image_gen"
    REASONING = "reasoning"
    CODE = "code"
    CREATIVE = "creative"
    TRANSLATION = "translation"


MODEL_REGISTRY = {
    # ── OpenAI ──
    "gpt-5.2": {
        "provider": "openai", "cost_tier": "frontier",
        "max_file_mb": 3, "languages": 27, "tool_calling": True,
        "strengths": ["reasoning", "code", "creative", "multimodal_input"],
        "cost_per_1k_tokens": 0.06,
    },
    "gpt-5.1": {
        "provider": "openai", "cost_tier": "premium",
        "max_file_mb": 3, "languages": 27, "tool_calling": True,
        "strengths": ["reasoning", "code", "creative"],
        "cost_per_1k_tokens": 0.04,
    },
    "gpt-5": {
        "provider": "openai", "cost_tier": "premium",
        "max_file_mb": 3, "languages": 27, "tool_calling": True,
        "strengths": ["reasoning", "code"],
        "cost_per_1k_tokens": 0.03,
    },
    "gpt-5-mini": {
        "provider": "openai", "cost_tier": "standard",
        "max_file_mb": 3, "languages": 27, "tool_calling": True,
        "strengths": ["reasoning", "code"],
        "cost_per_1k_tokens": 0.01,
    },
    "gpt-5-nano": {
        "provider": "openai", "cost_tier": "budget",
        "max_file_mb": 3, "languages": 27, "tool_calling": True,
        "strengths": ["classification"],
        "cost_per_1k_tokens": 0.003,
    },
    "o3": {
        "provider": "openai", "cost_tier": "frontier",
        "max_file_mb": 3, "languages": 1, "tool_calling": True,
        "strengths": ["reasoning"],
        "cost_per_1k_tokens": 0.10,
    },
    "o4-mini": {
        "provider": "openai", "cost_tier": "standard",
        "max_file_mb": 3, "languages": 27, "tool_calling": True,
        "strengths": ["reasoning", "code"],
        "cost_per_1k_tokens": 0.015,
    },
    # ── Anthropic ──
    "claude-opus-4.5": {
        "provider": "anthropic", "cost_tier": "frontier",
        "max_file_mb": 4.5, "languages": 3, "tool_calling": True,
        "strengths": ["code", "reasoning", "creative"],
        "cost_per_1k_tokens": 0.075,
    },
    "claude-sonnet-4.5": {
        "provider": "anthropic", "cost_tier": "premium",
        "max_file_mb": 4.5, "languages": 3, "tool_calling": True,
        "strengths": ["code", "reasoning", "creative"],
        "cost_per_1k_tokens": 0.015,
    },
    "claude-haiku-4.5": {
        "provider": "anthropic", "cost_tier": "budget",
        "max_file_mb": 4.5, "languages": 3, "tool_calling": True,
        "strengths": ["code", "classification"],
        "cost_per_1k_tokens": 0.004,
    },
    # ── Google ──
    "gemini-3.0-pro": {
        "provider": "google", "cost_tier": "premium",
        "max_file_mb": 50, "languages": 13, "tool_calling": True,
        "strengths": ["reasoning", "multimodal_input", "creative"],
        "cost_per_1k_tokens": 0.025,
    },
    "gemini-3.0-flash": {
        "provider": "google", "cost_tier": "standard",
        "max_file_mb": 50, "languages": 13, "tool_calling": True,
        "strengths": ["code", "creative"],
        "cost_per_1k_tokens": 0.008,
    },
    # ── Meta (no tool calling) ──
    "llama-4-maverick": {
        "provider": "meta", "cost_tier": "budget",
        "max_file_mb": 3, "languages": 12, "tool_calling": False,
        "strengths": ["creative", "translation", "multimodal_input"],
        "cost_per_1k_tokens": 0.002,
    },
    # ── Amazon ──
    "nova-pro": {
        "provider": "amazon", "cost_tier": "standard",
        "max_file_mb": 3, "languages": 3, "tool_calling": True,
        "strengths": ["code", "classification"],
        "cost_per_1k_tokens": 0.008,
    },
    # ── DeepSeek (RESTRICTED — no client data) ──
    "deepseek-r1": {
        "provider": "deepseek", "cost_tier": "budget",
        "max_file_mb": 3, "languages": 2, "tool_calling": False,
        "strengths": ["reasoning"],
        "cost_per_1k_tokens": 0.002,
        "restricted": True,
        "restriction_reason": "Chinese infrastructure — no client-sensitive data",
    },
}

# Language detection (simplified — production would use langdetect)
ENGLISH_ONLY_MODELS = {"o3"}
LIMITED_LANGUAGE_MODELS = {
    "claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5",
    "nova-pro", "deepseek-r1"
}

# ── Routing Logic ────────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    task_type: str  # reasoning, code, creative, classification, translation, general
    message: str
    file_size_mb: Optional[float] = None
    language: Optional[str] = "en"
    cost_tier: Optional[str] = "standard"  # budget, standard, premium, frontier
    requires_tool_calling: Optional[bool] = False
    contains_client_data: Optional[bool] = False


class RouteResponse(BaseModel):
    recommended_model: str
    provider: str
    reason: str
    alternatives: list[str]
    warnings: list[str]
    estimated_cost_per_1k: float


TASK_STRENGTH_MAP = {
    "reasoning": "reasoning",
    "code": "code",
    "coding": "code",
    "creative": "creative",
    "writing": "creative",
    "classification": "classification",
    "translation": "translation",
    "general": "creative",
}

COST_TIER_ORDER = ["budget", "standard", "premium", "frontier"]


@app.post("/route", response_model=RouteResponse)
async def route_request(req: RouteRequest):
    """Route a request to the optimal model."""

    candidates = dict(MODEL_REGISTRY)
    warnings = []
    task_strength = TASK_STRENGTH_MAP.get(req.task_type.lower(), "creative")

    # ── Filter: client data restriction ──
    if req.contains_client_data:
        restricted = [k for k, v in candidates.items() if v.get("restricted")]
        for model_id in restricted:
            del candidates[model_id]
            warnings.append(f"Blocked {model_id}: {MODEL_REGISTRY[model_id].get('restriction_reason')}")

    # ── Filter: file size ──
    if req.file_size_mb:
        too_small = [k for k, v in candidates.items() if v["max_file_mb"] < req.file_size_mb]
        for model_id in too_small:
            del candidates[model_id]
        if not candidates:
            raise HTTPException(400, f"No model supports files of {req.file_size_mb}MB. Max available: 50MB (Gemini)")

    # ── Filter: tool calling ──
    if req.requires_tool_calling:
        no_tools = [k for k, v in candidates.items() if not v["tool_calling"]]
        for model_id in no_tools:
            del candidates[model_id]

    # ── Filter: language ──
    if req.language and req.language.lower() != "en":
        eng_only = [k for k, v in candidates.items() if v["languages"] <= 2]
        for model_id in eng_only:
            del candidates[model_id]
            warnings.append(f"Excluded {model_id}: English-only, your input is {req.language}")

    if not candidates:
        raise HTTPException(400, "No models match your requirements. Try relaxing constraints.")

    # ── Score candidates ──
    def score_model(model_id: str, model: dict) -> float:
        s = 0.0
        # Task fit (strongest signal)
        if task_strength in model["strengths"]:
            s += 40
        # Cost tier match
        req_tier_idx = COST_TIER_ORDER.index(req.cost_tier or "standard")
        model_tier_idx = COST_TIER_ORDER.index(model["cost_tier"])
        tier_diff = abs(req_tier_idx - model_tier_idx)
        s += max(0, 30 - (tier_diff * 10))
        # Language breadth bonus
        s += min(model["languages"] / 3, 10)
        # File size headroom
        if req.file_size_mb:
            headroom = model["max_file_mb"] / req.file_size_mb
            s += min(headroom * 2, 10)
        # Tool calling bonus when needed
        if req.requires_tool_calling and model["tool_calling"]:
            s += 10
        return s

    scored = [(model_id, score_model(model_id, model), model)
              for model_id, model in candidates.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_id, best_score, best_model = scored[0]
    alternatives = [s[0] for s in scored[1:4]]

    reason_parts = []
    if task_strength in best_model["strengths"]:
        reason_parts.append(f"strong at {req.task_type}")
    reason_parts.append(f"{best_model['cost_tier']} tier")
    if req.file_size_mb and best_model["max_file_mb"] >= req.file_size_mb:
        reason_parts.append(f"handles {req.file_size_mb}MB files")
    reason_parts.append(f"{best_model['languages']} languages")

    return RouteResponse(
        recommended_model=best_id,
        provider=best_model["provider"],
        reason=f"Selected {best_id}: {', '.join(reason_parts)}",
        alternatives=alternatives,
        warnings=warnings,
        estimated_cost_per_1k=best_model["cost_per_1k_tokens"],
    )


# ── Usage Analytics Endpoint ─────────────────────────────────────────────────

# In production, this writes to a database. Here we show the schema.
from datetime import datetime
from collections import defaultdict

usage_log: list[dict] = []

class UsageEvent(BaseModel):
    user_email: str
    model_id: str
    task_type: str
    input_tokens: int
    output_tokens: int
    file_size_mb: Optional[float] = None
    contains_client_data: bool = False
    team: Optional[str] = None


@app.post("/log-usage")
async def log_usage(event: UsageEvent):
    """Log every model interaction for analytics."""
    model = MODEL_REGISTRY.get(event.model_id)
    cost = 0.0
    if model:
        cost = (event.input_tokens + event.output_tokens) / 1000 * model["cost_per_1k_tokens"]

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": event.user_email,
        "model": event.model_id,
        "provider": model["provider"] if model else "unknown",
        "task_type": event.task_type,
        "input_tokens": event.input_tokens,
        "output_tokens": event.output_tokens,
        "total_tokens": event.input_tokens + event.output_tokens,
        "estimated_cost_usd": round(cost, 4),
        "file_size_mb": event.file_size_mb,
        "contains_client_data": event.contains_client_data,
        "team": event.team,
    }
    usage_log.append(record)
    return {"status": "logged", "estimated_cost_usd": cost}


@app.get("/analytics/summary")
async def analytics_summary():
    """Dashboard data: usage by model, team, cost."""
    by_model = defaultdict(lambda: {"count": 0, "tokens": 0, "cost": 0.0})
    by_team = defaultdict(lambda: {"count": 0, "tokens": 0, "cost": 0.0})
    by_provider = defaultdict(lambda: {"count": 0, "tokens": 0, "cost": 0.0})

    for r in usage_log:
        for key, bucket in [(r["model"], by_model), (r.get("team", "unknown"), by_team), (r["provider"], by_provider)]:
            bucket[key]["count"] += 1
            bucket[key]["tokens"] += r["total_tokens"]
            bucket[key]["cost"] += r["estimated_cost_usd"]

    return {
        "total_requests": len(usage_log),
        "by_model": dict(by_model),
        "by_team": dict(by_team),
        "by_provider": dict(by_provider),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
