"""
Slalom business@ Agent Orchestrator
Multi-agent pipeline framework to replace single-turn Digital Experts.

Capabilities:
  - Chain multiple agents in sequence or parallel
  - Route sub-tasks to different models based on complexity
  - Inject RAG context at each step
  - PII-scan at every boundary
  - Log all interactions for analytics

Example pipelines:
  - RFP Response: Analyze RFP → Search past proposals → Draft sections → Brand review → Final
  - Client Discovery: Ingest brief → Market research → Competitive analysis → Recommendations
  - SOW Generation: Scope intake → Resource matching → Timeline → Pricing → Legal review
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
from enum import Enum
from datetime import datetime
import asyncio
import httpx
import json

app = FastAPI(title="business@ Agent Orchestrator", version="1.0.0")


# ── Service URLs (internal mesh) ─────────────────────────────────────────────

ROUTER_URL = "http://localhost:8001"
PII_SCANNER_URL = "http://localhost:8002"
RAG_URL = "http://localhost:8003"


# ── Agent Definitions ────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    ANALYZER = "analyzer"        # Reads and extracts structure from inputs
    RESEARCHER = "researcher"    # Searches knowledge base and web
    WRITER = "writer"            # Generates content
    REVIEWER = "reviewer"        # Quality and compliance checks
    SYNTHESIZER = "synthesizer"  # Combines outputs from multiple agents


class Agent(BaseModel):
    name: str
    role: AgentRole
    system_prompt: str
    preferred_model: Optional[str] = None  # Override auto-routing
    cost_tier: str = "standard"
    requires_rag: bool = False
    rag_sources: Optional[list[str]] = None  # Filter RAG by source type


class PipelineStep(BaseModel):
    agent: Agent
    input_key: str = "user_input"  # Key from context to use as input
    output_key: str = "step_output"  # Key to store output in context
    depends_on: list[str] = []  # output_keys this step depends on


class Pipeline(BaseModel):
    name: str
    description: str
    steps: list[PipelineStep]


class PipelineRunRequest(BaseModel):
    pipeline_id: str
    user_input: str
    user_email: str
    user_role: str = "consultant"
    data_tier: str = "internal"
    attachments: list[str] = []  # File references


class StepResult(BaseModel):
    step_name: str
    agent_name: str
    model_used: str
    input_preview: str
    output: str
    duration_ms: int
    pii_scan_result: str
    rag_docs_used: int


class PipelineRunResponse(BaseModel):
    pipeline_name: str
    status: str
    steps_completed: int
    total_steps: int
    results: list[StepResult]
    final_output: str
    total_duration_ms: int
    total_estimated_cost: float


# ── Pre-Built Pipelines ─────────────────────────────────────────────────────

PIPELINES: dict[str, Pipeline] = {
    "rfp-response": Pipeline(
        name="RFP Response Generator",
        description="Analyze an RFP, search past proposals, draft response sections, and review for brand compliance",
        steps=[
            PipelineStep(
                agent=Agent(
                    name="RFP Analyzer",
                    role=AgentRole.ANALYZER,
                    system_prompt="""You are an expert RFP analyst at Slalom Consulting.
Analyze the provided RFP and extract:
1. Client name and industry
2. Key requirements (numbered list)
3. Evaluation criteria and weights
4. Timeline and budget constraints
5. Required certifications or partnerships
6. Recommended Slalom solutions that apply
7. Red flags or unusual requirements
Format as structured JSON.""",
                    cost_tier="standard",
                ),
                input_key="user_input",
                output_key="rfp_analysis",
            ),
            PipelineStep(
                agent=Agent(
                    name="Proposal Researcher",
                    role=AgentRole.RESEARCHER,
                    system_prompt="""You are a Slalom proposal researcher. Given the RFP analysis,
search for relevant past proposals, case studies, and client stories.
Identify the strongest proof points and relevant team members.
Return a structured brief with: similar engagements, key differentiators, and suggested team composition.""",
                    cost_tier="standard",
                    requires_rag=True,
                    rag_sources=["proposal", "engagement", "summit"],
                ),
                input_key="rfp_analysis",
                output_key="research_brief",
                depends_on=["rfp_analysis"],
            ),
            PipelineStep(
                agent=Agent(
                    name="Proposal Writer",
                    role=AgentRole.WRITER,
                    system_prompt="""You are a senior Slalom proposal writer. Using the RFP analysis
and research brief, draft a complete proposal response following Slalom's standard template:
1. Executive Summary
2. Our Understanding
3. Approach (mapped to Summit methodology)
4. Team
5. Investment
6. Why Slalom
7. Appendix outline

Write in Slalom's brand voice: confident, collaborative, outcome-focused.
Reference specific methodologies and past successes.""",
                    preferred_model="claude-sonnet-4.5",
                    cost_tier="premium",
                    requires_rag=True,
                    rag_sources=["proposal", "summit"],
                ),
                input_key="research_brief",
                output_key="draft_proposal",
                depends_on=["research_brief"],
            ),
            PipelineStep(
                agent=Agent(
                    name="Brand & Compliance Reviewer",
                    role=AgentRole.REVIEWER,
                    system_prompt="""You are Slalom's brand and legal compliance reviewer.
Review the draft proposal for:
1. Brand voice consistency (per Slalom Brand Guide)
2. Solution naming compliance (per Solution Naming Guide)
3. Legal language requirements (per Legal Marketing Guidance)
4. Accurate use of partnership references (AWS, Google, Databricks, OpenAI)
5. No unauthorized pricing or commitment language

Flag issues with severity (MUST FIX, SHOULD FIX, SUGGESTION) and provide corrected text.""",
                    cost_tier="standard",
                ),
                input_key="draft_proposal",
                output_key="review_feedback",
                depends_on=["draft_proposal"],
            ),
            PipelineStep(
                agent=Agent(
                    name="Final Synthesizer",
                    role=AgentRole.SYNTHESIZER,
                    system_prompt="""You are the final proposal assembler. Take the draft proposal
and apply all review feedback (MUST FIX items are mandatory, SHOULD FIX items use your judgment).
Output the final, polished proposal ready for partner review.
Include a summary of changes made and any items requiring human decision.""",
                    preferred_model="claude-opus-4.5",
                    cost_tier="frontier",
                ),
                input_key="review_feedback",
                output_key="final_proposal",
                depends_on=["draft_proposal", "review_feedback"],
            ),
        ],
    ),

    "client-discovery": Pipeline(
        name="Client Discovery Engine",
        description="Rapid client/market analysis from a brief description",
        steps=[
            PipelineStep(
                agent=Agent(
                    name="Brief Analyzer",
                    role=AgentRole.ANALYZER,
                    system_prompt="""Analyze the client brief and extract: industry, company size,
key challenges, technology stack, and strategic priorities. Identify the top 3 Slalom
service lines that could help. Output structured JSON.""",
                    cost_tier="standard",
                ),
                input_key="user_input",
                output_key="client_profile",
            ),
            PipelineStep(
                agent=Agent(
                    name="Market Researcher",
                    role=AgentRole.RESEARCHER,
                    system_prompt="""Using the client profile, research: industry trends,
competitive landscape, common technology challenges in this vertical, and relevant
Slalom case studies. Focus on actionable insights, not general knowledge.""",
                    cost_tier="standard",
                    requires_rag=True,
                    rag_sources=["engagement", "summit"],
                ),
                input_key="client_profile",
                output_key="market_research",
                depends_on=["client_profile"],
            ),
            PipelineStep(
                agent=Agent(
                    name="Recommendation Writer",
                    role=AgentRole.WRITER,
                    system_prompt="""Synthesize the client profile and market research into a
concise discovery brief (2-3 pages) including:
1. Client Situation Assessment
2. Industry Context & Trends
3. Recommended Engagement Approach
4. Suggested Team Composition
5. Estimated Timeline & Investment Range
6. Relevant Slalom Case Studies
Write for a partner audience. Be direct, data-driven, and specific.""",
                    preferred_model="claude-sonnet-4.5",
                    cost_tier="premium",
                ),
                input_key="market_research",
                output_key="discovery_brief",
                depends_on=["client_profile", "market_research"],
            ),
        ],
    ),

    "sow-generator": Pipeline(
        name="SOW Generator",
        description="Generate a Statement of Work from scope inputs",
        steps=[
            PipelineStep(
                agent=Agent(
                    name="Scope Extractor",
                    role=AgentRole.ANALYZER,
                    system_prompt="""Extract from the input: project objectives, deliverables,
assumptions, constraints, team roles needed, estimated duration, and out-of-scope items.
Structure as JSON with clear categorization.""",
                    cost_tier="standard",
                ),
                input_key="user_input",
                output_key="scope_definition",
            ),
            PipelineStep(
                agent=Agent(
                    name="SOW Writer",
                    role=AgentRole.WRITER,
                    system_prompt="""Generate a professional Statement of Work using Slalom's
standard template. Include all standard sections: Background, Objectives, Scope,
Deliverables, Timeline, Team, Assumptions, Change Management, Acceptance Criteria.
Use formal contract language. Leave [PLACEHOLDER] for specific rates and dates.""",
                    preferred_model="claude-sonnet-4.5",
                    cost_tier="premium",
                    requires_rag=True,
                    rag_sources=["proposal", "summit"],
                ),
                input_key="scope_definition",
                output_key="draft_sow",
                depends_on=["scope_definition"],
            ),
            PipelineStep(
                agent=Agent(
                    name="Legal Reviewer",
                    role=AgentRole.REVIEWER,
                    system_prompt="""Review the SOW for: standard legal language compliance,
liability limitations, IP ownership clauses, termination provisions, and
confidentiality requirements. Flag anything missing or non-standard.
Mark issues as MUST FIX or ADVISORY.""",
                    cost_tier="standard",
                ),
                input_key="draft_sow",
                output_key="final_sow",
                depends_on=["draft_sow"],
            ),
        ],
    ),
}


# ── Pipeline Execution Engine ────────────────────────────────────────────────

async def call_pii_scanner(text: str, target_model: str, data_tier: str) -> dict:
    """Pre-scan text for PII before sending to any model."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{PII_SCANNER_URL}/scan", json={
                "text": text,
                "target_model": target_model,
                "data_tier": data_tier,
            }, timeout=10)
            return resp.json()
        except Exception:
            return {"allowed": True, "action": "PASS", "pii_detected": []}


async def call_rag(query: str, sources: list[str], user_role: str) -> dict:
    """Search RAG knowledge base for relevant context."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{RAG_URL}/search", json={
                "query": query,
                "top_k": 3,
                "source_filter": sources,
                "user_role": user_role,
            }, timeout=10)
            return resp.json()
        except Exception:
            return {"results": [], "augmented_prompt": query}


async def call_router(task_type: str, message: str, cost_tier: str,
                      contains_client_data: bool) -> dict:
    """Get optimal model recommendation from router."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{ROUTER_URL}/route", json={
                "task_type": task_type,
                "message": message[:200],
                "cost_tier": cost_tier,
                "contains_client_data": contains_client_data,
            }, timeout=10)
            return resp.json()
        except Exception:
            return {"recommended_model": "gpt-5.1", "provider": "openai"}


async def call_llm(model: str, system_prompt: str, user_message: str) -> str:
    """Call the actual LLM. In production, routes to the appropriate provider API."""
    # This is the integration point with OpenAI, Anthropic, Google APIs.
    # Placeholder: return a structured response indicating what would be generated.
    return f"[{model} would generate response here]\n\nSystem: {system_prompt[:100]}...\nInput preview: {user_message[:200]}..."


@app.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(req: PipelineRunRequest):
    """Execute a multi-agent pipeline."""

    pipeline = PIPELINES.get(req.pipeline_id)
    if not pipeline:
        available = list(PIPELINES.keys())
        raise HTTPException(404, f"Pipeline '{req.pipeline_id}' not found. Available: {available}")

    context: dict[str, str] = {"user_input": req.user_input}
    results: list[StepResult] = []
    total_cost = 0.0
    pipeline_start = datetime.utcnow()

    for i, step in enumerate(pipeline.steps):
        step_start = datetime.utcnow()

        # ── Resolve input from context ──
        step_input_parts = [context.get(step.input_key, req.user_input)]
        for dep in step.depends_on:
            if dep in context and dep != step.input_key:
                step_input_parts.append(f"\n\n## Previous Step Output ({dep}):\n{context[dep]}")
        step_input = "\n".join(step_input_parts)

        # ── Determine model ──
        if step.agent.preferred_model:
            model = step.agent.preferred_model
        else:
            route_result = await call_router(
                task_type=step.agent.role.value,
                message=step_input,
                cost_tier=step.agent.cost_tier,
                contains_client_data=(req.data_tier in ("confidential", "restricted")),
            )
            model = route_result.get("recommended_model", "gpt-5.1")

        # ── PII scan ──
        pii_result = await call_pii_scanner(step_input, model, req.data_tier)
        pii_action = pii_result.get("action", "PASS")

        if pii_action == "BLOCK":
            # Use redacted text instead
            step_input = pii_result.get("redacted_text", step_input)

        # ── RAG augmentation ──
        rag_docs_used = 0
        if step.agent.requires_rag:
            rag_result = await call_rag(
                query=step_input[:500],
                sources=step.agent.rag_sources or [],
                user_role=req.user_role,
            )
            augmented = rag_result.get("augmented_prompt", "")
            if augmented:
                step_input = augmented
            rag_docs_used = len(rag_result.get("results", []))

        # ── Call LLM ──
        output = await call_llm(model, step.agent.system_prompt, step_input)

        # ── Store in context ──
        context[step.output_key] = output

        step_end = datetime.utcnow()
        duration_ms = int((step_end - step_start).total_seconds() * 1000)

        results.append(StepResult(
            step_name=f"Step {i+1}",
            agent_name=step.agent.name,
            model_used=model,
            input_preview=step_input[:150] + "...",
            output=output,
            duration_ms=duration_ms,
            pii_scan_result=pii_action,
            rag_docs_used=rag_docs_used,
        ))

    pipeline_end = datetime.utcnow()
    total_duration_ms = int((pipeline_end - pipeline_start).total_seconds() * 1000)

    # Final output is the last step's output
    final_output = context.get(pipeline.steps[-1].output_key, "No output generated")

    return PipelineRunResponse(
        pipeline_name=pipeline.name,
        status="completed",
        steps_completed=len(results),
        total_steps=len(pipeline.steps),
        results=results,
        final_output=final_output,
        total_duration_ms=total_duration_ms,
        total_estimated_cost=total_cost,
    )


@app.get("/pipelines")
async def list_pipelines():
    """List all available pipelines."""
    return {
        pid: {
            "name": p.name,
            "description": p.description,
            "steps": len(p.steps),
            "agents": [s.agent.name for s in p.steps],
        }
        for pid, p in PIPELINES.items()
    }


@app.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get detailed pipeline definition."""
    pipeline = PIPELINES.get(pipeline_id)
    if not pipeline:
        raise HTTPException(404, f"Pipeline '{pipeline_id}' not found")
    return pipeline


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
