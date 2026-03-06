"""
Slalom business@ RAG Layer
Retrieval-Augmented Generation for institutional knowledge.

Indexes:
  - Summit Methodology docs
  - Past proposals and SOWs (permission-trimmed)
  - Delivery frameworks and templates
  - Client engagement summaries
  - Digital Expert system prompts

Makes every model "Slalom-aware" without fine-tuning.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import hashlib
import json
from datetime import datetime

app = FastAPI(title="business@ RAG Layer", version="1.0.0")


# ── Vector Store Abstraction ─────────────────────────────────────────────────

# Production: Use Pinecone, Weaviate, or pgvector.
# This implementation shows the schema and API contract.

class Document(BaseModel):
    id: str
    title: str
    content: str
    source: str  # "summit", "proposal", "sow", "template", "engagement"
    permissions: list[str]  # Team/role-based access: ["all", "partners", "delivery-leads"]
    metadata: dict = {}
    created_at: str = ""


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    source_filter: Optional[list[str]] = None  # Filter by source type
    user_role: Optional[str] = "consultant"  # For permission trimming
    include_scores: bool = True


class SearchResult(BaseModel):
    document_id: str
    title: str
    content_snippet: str
    source: str
    relevance_score: float
    metadata: dict


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    total_indexed: int
    augmented_prompt: str  # Ready-to-use prompt with context injected


# ── In-Memory Store (replace with vector DB in production) ───────────────────

document_store: dict[str, Document] = {}


def generate_id(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


# Simulated embedding similarity (production: use text-embedding-3-large or Cohere embed)
def simple_relevance(query: str, content: str) -> float:
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    if not query_words:
        return 0.0
    overlap = len(query_words & content_words)
    return overlap / len(query_words)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/index")
async def index_document(doc: Document):
    """Add a document to the knowledge base."""
    if not doc.id:
        doc.id = generate_id(doc.content)
    doc.created_at = datetime.utcnow().isoformat()
    document_store[doc.id] = doc
    return {"status": "indexed", "id": doc.id, "total_documents": len(document_store)}


@app.post("/index/batch")
async def index_batch(documents: list[Document]):
    """Batch index multiple documents."""
    indexed = []
    for doc in documents:
        if not doc.id:
            doc.id = generate_id(doc.content)
        doc.created_at = datetime.utcnow().isoformat()
        document_store[doc.id] = doc
        indexed.append(doc.id)
    return {"status": "indexed", "count": len(indexed), "ids": indexed}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Search the knowledge base and return augmented prompt."""

    candidates = list(document_store.values())

    # ── Permission trimming ──
    role_hierarchy = {
        "consultant": ["all"],
        "senior-consultant": ["all", "senior"],
        "delivery-lead": ["all", "senior", "delivery-leads"],
        "partner": ["all", "senior", "delivery-leads", "partners"],
        "admin": ["all", "senior", "delivery-leads", "partners", "admin"],
    }
    allowed_perms = set(role_hierarchy.get(req.user_role, ["all"]))
    candidates = [d for d in candidates if any(p in allowed_perms for p in d.permissions)]

    # ── Source filtering ──
    if req.source_filter:
        candidates = [d for d in candidates if d.source in req.source_filter]

    # ── Score and rank ──
    scored = []
    for doc in candidates:
        score = simple_relevance(req.query, doc.content + " " + doc.title)
        if score > 0:
            scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_results = scored[:req.top_k]

    # ── Build augmented prompt ──
    context_blocks = []
    results = []
    for doc, score in top_results:
        snippet = doc.content[:500] + ("..." if len(doc.content) > 500 else "")
        context_blocks.append(f"[Source: {doc.source} | {doc.title}]\n{snippet}")
        results.append(SearchResult(
            document_id=doc.id,
            title=doc.title,
            content_snippet=snippet,
            source=doc.source,
            relevance_score=round(score, 4),
            metadata=doc.metadata,
        ))

    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No relevant documents found."

    augmented_prompt = f"""You are a Slalom consultant. Use the following internal knowledge to inform your response.
Do not fabricate information beyond what is provided. If the context doesn't fully answer the question, say so.

## Slalom Internal Context
{context_text}

## User Question
{req.query}

## Instructions
- Reference specific Slalom methodologies, frameworks, or past engagements when relevant
- Maintain Slalom's brand voice: confident, collaborative, outcome-focused
- If recommending an approach, tie it to Summit methodology phases where applicable
"""

    return SearchResponse(
        results=results,
        query=req.query,
        total_indexed=len(document_store),
        augmented_prompt=augmented_prompt,
    )


@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the index."""
    if doc_id in document_store:
        del document_store[doc_id]
        return {"status": "deleted", "id": doc_id}
    raise HTTPException(404, f"Document {doc_id} not found")


@app.get("/stats")
async def index_stats():
    """Return index statistics."""
    by_source = {}
    for doc in document_store.values():
        by_source[doc.source] = by_source.get(doc.source, 0) + 1
    return {
        "total_documents": len(document_store),
        "by_source": by_source,
    }


# ── Bootstrap: Pre-load Summit Methodology ──────────────────────────────────

SUMMIT_SEED_DOCS = [
    Document(
        id="summit-overview",
        title="Summit Methodology Overview",
        content="""Summit is Slalom's proprietary delivery methodology structured around five dimensions:
1. Business and Customer Value — defining strategy and experiences improved by AI
2. Strategy Alignment and Orchestration — planning frameworks and resource management
3. Security, Ethics, and Governance — responsible, ethical, standardized practices
4. Technology and Data — infrastructure, data management, and MLOps readiness
5. Organization and Workforce — cultural shifts and human-machine collaboration

Summit emphasizes agile delivery with local teams, combining strategy through delivery
in a single engagement model. Key phases: Discovery, Define, Design, Deliver, Drive.""",
        source="summit",
        permissions=["all"],
        metadata={"type": "methodology", "version": "2026"},
    ),
    Document(
        id="summit-ai-framework",
        title="Summit AI Transformation Framework",
        content="""Slalom's AI transformation approach follows four key shifts:
1. Turn Confidence into Capability — reskilling at scale, not just awareness
2. Modernize Without Compromise — unlock legacy constraints, don't just migrate
3. Redesign, Don't Retrofit — transform processes from ground up, not bolt-on AI
4. Go Beyond Efficiency to Excel — drive reinvention, not just optimization

Key metrics from 2026 research:
- 90% of companies increasing AI investment
- Only 39% have clear ROI metrics
- Only 21% report enterprise-wide AI use cases
- 93% report workforce barriers (skills gaps)
- Only 39% achieved multi-agent workflows""",
        source="summit",
        permissions=["all"],
        metadata={"type": "framework", "year": 2026},
    ),
    Document(
        id="proposal-template",
        title="Slalom Proposal Template — AI Engagement",
        content="""Standard AI engagement proposal structure:
1. Executive Summary — client challenge, Slalom's POV, expected outcomes
2. Our Understanding — restate the problem in client's language
3. Approach — Summit phases mapped to client timeline
4. Team — named resources with relevant experience
5. Investment — fixed-fee or T&M with not-to-exceed
6. Why Slalom — local presence, relevant case studies, partnership ecosystem
7. Appendix — team bios, case studies, certifications

Key differentiators to emphasize:
- Local teams (not fly-in consultants)
- Partnership ecosystem (AWS, Google, Databricks, OpenAI, Salesforce)
- Industry-specific AI accelerators
- Summit methodology with proven delivery track record""",
        source="proposal",
        permissions=["all"],
        metadata={"type": "template"},
    ),
]


@app.on_event("startup")
async def seed_knowledge_base():
    """Pre-load foundational documents on startup."""
    for doc in SUMMIT_SEED_DOCS:
        doc.created_at = datetime.utcnow().isoformat()
        document_store[doc.id] = doc
    print(f"Seeded {len(SUMMIT_SEED_DOCS)} foundational documents")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
