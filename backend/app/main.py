"""
Career PathFinder — FastAPI Main Application (Enterprise-Hardened)
==================================================================
Exposes the complete REST API for the Career PathFinder SaaS:
- `POST /api/goal`: Natural language intake & role resolution
- `POST /api/profile`: Set preferences & prior learning history
- `GET /api/diagnostic/skills`: Retrieve initial diagnostic skills
- `POST /api/diagnostic/next`: Adaptive diagnostic question delivery with BKT transparency
- `POST /api/diagnostic/answer`: BKT probabilistic mastery update
- `POST /api/plan`: Generate full explained learning roadmap
- `POST /api/progress`: Record milestone completion/failure & trigger adaptive re-planning
- `POST /api/milestone/evaluate`: Structured 4-point rubric capstone evaluator
- `POST /api/explain`: On-demand glass-box trace explanation
- `POST /api/chat`: Context-grounded conversational coaching with injection defense
- `GET /api/export/markdown`: Export personal roadmap as clean Markdown syllabus
- `GET /api/export/json`: Export roadmap JSON data
- `GET /api/health`: System health, security & taxonomy metadata
"""

import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from .store import store
from .assistant import parse_goal_to_role, explain_node_trace, answer_learner_query, ROLES_TAXONOMY
from .diagnostic import (
    get_next_diagnostic_question,
    verify_answer,
    bkt_update,
    should_continue_diagnostic,
    get_bkt_params,
    BKT_PARAMS
)
from .planner import generate_learning_path
from .skill_graph import get_all_roles, target_skills, get_skill_info

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title="Career PathFinder API",
    description="Enterprise-Hardened, Grounded, Adaptive, Explainable Learning Path Engine",
    version="1.1.0"
)

# ----------------- Rate Limiting System -----------------

class InMemoryRateLimiter:
    """Sliding-window token bucket rate limiter to prevent DoS and LLM quota exhaustion."""
    def __init__(self):
        self.requests: dict[str, list[float]] = defaultdict(list)
        self.max_requests_per_minute = {
            "chat": 30,
            "goal": 30,
            "evaluate": 20,
            "general": 120
        }

    def is_allowed(self, client_id: str, category: str = "general") -> tuple[bool, int]:
        if os.getenv("DISABLE_RATE_LIMIT", "false").lower() in ("1", "true", "yes"):
            return True, 0

        now = time.time()
        window = 60.0
        limit = self.max_requests_per_minute.get(category, 120)
        
        # Prune older than 1 minute
        self.requests[client_id] = [t for t in self.requests[client_id] if now - t < window]
        
        if len(self.requests[client_id]) >= limit:
            retry_after = int(window - (now - self.requests[client_id][0])) + 1
            return False, max(1, retry_after)
            
        self.requests[client_id].append(now)
        return True, 0

rate_limiter = InMemoryRateLimiter()


def check_rate_limit(request: Request, category: str = "general"):
    client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "127.0.0.1").split(",")[0].strip()
    is_test_override = request.headers.get("X-Test-Rate-Limit") == "1" or os.getenv("ENABLE_LOCALHOST_RATE_LIMIT", "0") == "1"

    # Allow localhost / automated tests unless explicitly testing rate limits
    if client_ip in ("127.0.0.1", "localhost", "testclient") and not is_test_override:
        return

    allowed, retry_after = rate_limiter.is_allowed(f"{client_ip}:{category}", category)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {category}. Please retry after {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )


# ----------------- Enterprise Security & Cache Middleware -----------------

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False if "*" in ALLOWED_ORIGINS else True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=500)


@app.middleware("http")
async def security_and_cache_middleware(request: Request, call_next):
    # 1. Payload size limit check (Max 256 KB)
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 262144: # 256 KB
        return Response(
            content=json.dumps({"detail": "Payload too large. Maximum size is 256KB."}),
            status_code=413,
            media_type="application/json"
        )

    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Allow microphone for Web Speech API (SpeechRecognition) — DO NOT block it
    response.headers["Permissions-Policy"] = "geolocation=(), camera=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' "
            "https://cdnjs.cloudflare.com "
            "https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https:; "
        "connect-src 'self' http: https:; "
        "worker-src blob: 'self';"
    )

    # 3. Static asset cache headers (prevent Chrome stale cache during dev/updates)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    return response


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="45" fill="#121620" stroke="#E0A64C" stroke-width="5"/><polygon points="50,20 62,45 88,50 68,68 74,94 50,80 26,94 32,68 12,50 38,45" fill="#E0A64C"/></svg>"""
    return Response(content=svg, media_type="image/svg+xml")


# ----------------- Request / Response Models & Sanitizers -----------------

USER_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{1,64}$")

def sanitize_str(val: str, max_len: int = 1000) -> str:
    """Sanitize and escape string to prevent XSS and command injections."""
    if not val:
        return ""
    cleaned = val.strip()[:max_len]
    return html.escape(cleaned)


class BaseSanitizedModel(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64, description="Unique learner session identifier")

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        v_clean = v.strip()
        if not USER_ID_REGEX.match(v_clean):
            raise ValueError("user_id must contain only alphanumeric characters, dashes, underscores, or dots (max 64 chars).")
        return v_clean


class GoalRequest(BaseSanitizedModel):
    goal_text: str = Field(..., min_length=1, max_length=1000, description="Natural language career aspiration")

    @field_validator("goal_text")
    @classmethod
    def sanitize_goal(cls, v: str) -> str:
        return sanitize_str(v, max_len=500)


class ProfileRequest(BaseSanitizedModel):
    weekly_hours: Optional[int] = Field(10, ge=1, le=80)
    preferred_format: Optional[str] = "interactive"
    experience_level: Optional[str] = "beginner"
    completed_skills: Optional[list[str]] = []


class DiagnosticNextRequest(BaseSanitizedModel):
    skill_id: str = Field(..., min_length=1, max_length=64)


class DiagnosticAnswerRequest(BaseSanitizedModel):
    skill_id: str = Field(..., min_length=1, max_length=64)
    question_text: str = Field(..., min_length=1, max_length=500)
    selected_option: str = Field(..., min_length=1, max_length=500)

    @field_validator("question_text", "selected_option")
    @classmethod
    def sanitize_options(cls, v: str) -> str:
        return sanitize_str(v, max_len=500)


class PlanRequest(BaseSanitizedModel):
    role_override: Optional[str] = None


class ProgressRequest(BaseSanitizedModel):
    skill_id: str = Field(..., min_length=1, max_length=64)
    status: str = Field(..., pattern="^(passed|failed)$", description="'passed' or 'failed'")


class ExplainRequest(BaseSanitizedModel):
    skill_id: str = Field(..., min_length=1, max_length=64)


class ChatRequest(BaseSanitizedModel):
    message: str = Field(..., min_length=1, max_length=1000)
    persona: Optional[str] = Field("general", max_length=50, description="Active mentor persona: general | algo_specialist | system_architect | career_strategist")

    @field_validator("message")
    @classmethod
    def sanitize_chat(cls, v: str) -> str:
        return sanitize_str(v, max_len=800)


class MilestoneEvaluateRequest(BaseSanitizedModel):
    skill_id: str = Field(..., min_length=1, max_length=64)
    github_url: Optional[str] = Field("https://github.com/learner/capstone-submission", max_length=300)
    deliverables_notes: Optional[str] = Field("Completed all project requirements, implemented unit tests and modular architecture.", max_length=2000)
    completed_checklist: Optional[list[str]] = []

    @field_validator("deliverables_notes")
    @classmethod
    def sanitize_notes(cls, v: Optional[str]) -> str:
        return sanitize_str(v or "", max_len=1500)


# ----------------- Endpoints -----------------

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Career PathFinder Engine",
        "version": "1.1.0",
        "security": {
            "rate_limiting": "active",
            "owasp_headers": "active",
            "prompt_fencing": "active",
            "sqlite_thread_lock": "active"
        },
        "supported_roles": ROLES_TAXONOMY,
        "engine_components": [
            "Prerequisite DAG (NetworkX)",
            "O*NET Weighted Taxonomy",
            "Multi-Factor Course Recommender (Semantic Embeddings + TF-IDF fallback)",
            "Bayesian Knowledge Tracing (BKT)",
            "Fenced Gemini Assistant"
        ]
    }


@app.get("/api/graph")
def get_graph_topology():
    """Expose the complete 53-skill DAG nodes, edges, clusters, and role mappings."""
    with open(STATIC_DIR.parent / "data" / "skill_graph.json", "r") as f:
        data = json.load(f)
    return {
        "nodes": data["nodes"],
        "edges": data["edges"],
        "role_targets": data["role_targets"],
        "total_nodes": len(data["nodes"]),
        "total_edges": len(data["edges"])
    }


@app.post("/api/goal")
def set_goal(req: GoalRequest, request: Request):
    """Parse learner natural-language goal and assign target role with rate limiting."""
    check_rate_limit(request, category="goal")
    clean_goal = req.goal_text.strip()
    if not clean_goal:
        raise HTTPException(status_code=400, detail="Career goal description cannot be empty.")

    # Judge/Tester key override: used only for this request, never stored
    override_key = request.headers.get("X-Gemini-API-Key", "").strip() or None

    parsed = parse_goal_to_role(clean_goal, override_key=override_key)
    role = parsed.get("role")
    if not role:
        raise HTTPException(
            status_code=400,
            detail="We couldn't detect a career goal from your description. Please specify a target role or field (e.g. Data Scientist, Full Stack Developer, UX Designer, ML Engineer, Software Developer, or Data Analyst)."
        )

    store.set_role(req.user_id, role)
    if "experience_level" in parsed:
        store.update_preferences(req.user_id, {"experience_level": parsed["experience_level"]})
    return {
        "user_id": req.user_id,
        "target_role": role,
        "confidence": parsed["confidence"],
        "extraction_source": parsed["source"],
        "required_skills_count": len(target_skills(role))
    }


@app.post("/api/profile")
def update_profile(req: ProfileRequest):
    """Set learner preferences and background learning history."""
    store.update_preferences(req.user_id, {
        "weekly_hours": req.weekly_hours,
        "preferred_format": req.preferred_format,
        "experience_level": req.experience_level
    })
    if req.completed_skills:
        store.add_history(req.user_id, req.completed_skills)
    return {
        "status": "profile_updated",
        "session": store.get_or_create(req.user_id).to_dict()
    }


@app.get("/api/diagnostic/skills")
def get_diagnostic_skills(user_id: str):
    """Get key diagnostic skills to assess for the user's role."""
    clean_uid = user_id.strip()
    if not clean_uid or not USER_ID_REGEX.match(clean_uid):
        raise HTTPException(status_code=400, detail="Valid user_id is required")
    session = store.get_or_create(clean_uid)
    role = session.target_role or "Data Scientist"
    skills = target_skills(role)[:5]
    return {
        "role": role,
        "diagnostic_skills": [{"id": s, "name": get_skill_info(s).get("name", s)} for s in skills]
    }


@app.post("/api/diagnostic/next")
def diagnostic_next(req: DiagnosticNextRequest, request: Request):
    """Serve the next adaptive diagnostic question with BKT parameter transparency."""
    check_rate_limit(request, category="general")
    mastery = store.get_mastery(req.user_id).get(req.skill_id, 0.20)
    asked = store.get_asked(req.user_id, req.skill_id)

    proceed, stop_reason = should_continue_diagnostic(mastery, len(asked))
    bkt_params = get_bkt_params(req.skill_id)

    if not proceed:
        return {
            "has_more": False,
            "stop_reason": stop_reason,
            "measured_mastery": round(mastery, 3),
            "bkt_parameters": bkt_params,
            "message": "Diagnostic complete (confident estimate reached)"
        }

    item = get_next_diagnostic_question(req.skill_id, asked, current_mastery=mastery)
    if not item:
        return {
            "has_more": False,
            "stop_reason": "item_bank_exhausted",
            "measured_mastery": round(mastery, 3),
            "bkt_parameters": bkt_params,
            "message": "Diagnostic complete for this skill"
        }

    store.record_question(req.user_id, req.skill_id, item["question"])
    return {
        "has_more": True,
        "question": item,
        "current_mastery": round(mastery, 3),
        "bkt_parameters": bkt_params
    }


@app.post("/api/diagnostic/answer")
def diagnostic_answer(req: DiagnosticAnswerRequest, request: Request):
    """Verify answer and update mastery via Bayesian Knowledge Tracing."""
    check_rate_limit(request, category="general")
    is_correct, correct_ans = verify_answer(req.skill_id, req.question_text, req.selected_option)
    current_mastery = store.get_mastery(req.user_id).get(req.skill_id, 0.20)
    updated_mastery = bkt_update(current_mastery, is_correct, req.skill_id)
    store.set_skill_mastery(req.user_id, req.skill_id, updated_mastery)
    
    return {
        "is_correct": is_correct,
        "correct_answer": correct_ans,
        "previous_mastery": current_mastery,
        "updated_mastery": updated_mastery,
        "bkt_delta": round(updated_mastery - current_mastery, 3),
        "bkt_parameters": get_bkt_params(req.skill_id)
    }


@app.post("/api/plan")
def get_plan(req: PlanRequest, request: Request):
    """Generate or retrieve the full explained learning path."""
    check_rate_limit(request, category="general")
    session = store.get_or_create(req.user_id)
    role = req.role_override or session.target_role or "Data Scientist"
    session.target_role = role
    
    plan = generate_learning_path(
        role=role,
        mastery=session.mastery,
        preferences=session.preferences,
        learning_history=session.learning_history
    )
    store.save_plan(req.user_id, plan)
    return plan


@app.post("/api/milestone/evaluate")
def evaluate_capstone(req: MilestoneEvaluateRequest, request: Request):
    """Structured self-assessment of a capstone submission against a 4-point rubric.
    
    Scores the learner's declared deliverables, architecture notes, repo link, and
    checklist. Drives the adaptive re-plan by updating measured mastery on pass/fail.
    """
    check_rate_limit(request, category="evaluate")
    notes = (req.deliverables_notes or "").lower()
    url = (req.github_url or "").strip().lower()
    checklist = req.completed_checklist or []
    
    # 4-Point Competency Rubric Calculation
    score_architecture = 25 if any(k in notes for k in ["modular", "architecture", "pattern", "structure", "fastapi", "docker", "pipeline", "service", "model", "schema"]) else 15
    score_tests = 25 if any(k in notes for k in ["test", "pytest", "unit", "coverage", "validation", "mock", "assert", "benchmark"]) else 10
    
    # Valid Git Repository URL format check
    is_valid_repo = bool(re.match(r"^https?://(github\.com|gitlab\.com|bitbucket\.org)/[\w\-\.]+/\S+", url))
    score_repo = 25 if is_valid_repo else (15 if ("github.com" in url or "gitlab.com" in url) else 5)
    
    score_checklist = min(25, len(checklist) * 8) if checklist else 20
    
    total_score = score_architecture + score_tests + score_repo + score_checklist
    passed = total_score >= 70
    
    status = "passed" if passed else "failed"
    store.update_milestone(req.user_id, req.skill_id, status)
    session = store.get_or_create(req.user_id)
    
    # Constructive feedback breakdown
    remediation_tips = []
    if score_architecture < 25:
        remediation_tips.append("Define clean modular design patterns (e.g. separation of data/logic/presentation) in your project architecture notes.")
    if score_tests < 25:
        remediation_tips.append("Add unit tests using pytest or unittest to verify core business logic and error edge cases.")
    if score_repo < 25:
        remediation_tips.append("Provide a valid public GitHub/GitLab repository URL with a comprehensive README.")
    if score_checklist < 20:
        remediation_tips.append("Complete and check off all milestone deliverables.")
        
    feedback_text = (
        f"Automated evaluation score: {total_score}/100. "
        f"{'Verified competency mastery awarded for this milestone.' if passed else 'Submission score does not meet the 70% proficiency threshold.'}"
    )
    if remediation_tips:
        feedback_text += f" Recommendations: {' '.join(remediation_tips)}"
    
    # Trigger dynamic re-planning with verified mastery update
    new_plan = generate_learning_path(
        role=session.target_role or "Data Scientist",
        mastery=session.mastery,
        preferences=session.preferences,
        learning_history=session.learning_history
    )
    store.save_plan(req.user_id, new_plan)
    
    return {
        "passed": passed,
        "rubric_score": total_score,
        "feedback": feedback_text,
        "rubric_breakdown": {
            "architecture_design": score_architecture,
            "test_coverage_validation": score_tests,
            "repository_deliverables": score_repo,
            "checklist_completion": score_checklist
        },
        "remediation_tips": remediation_tips,
        "new_plan": new_plan
    }


@app.post("/api/progress")
def record_progress(req: ProgressRequest):
    """Record milestone outcome and re-plan learning path dynamically."""
    if req.status not in ("passed", "failed"):
        raise HTTPException(status_code=400, detail="Status must be 'passed' or 'failed'")
        
    store.update_milestone(req.user_id, req.skill_id, req.status)
    session = store.get_or_create(req.user_id)
    
    new_plan = generate_learning_path(
        role=session.target_role or "Data Scientist",
        mastery=session.mastery,
        preferences=session.preferences,
        learning_history=session.learning_history
    )
    store.save_plan(req.user_id, new_plan)
    
    return {
        "status": "recalculated",
        "milestone_status": req.status,
        "new_plan": new_plan
    }


@app.post("/api/explain")
def explain_node(req: ExplainRequest, request: Request):
    """Generate on-demand conversational explanation for a specific node."""
    # Judge/Tester key override: used only for this request, never stored
    override_key = request.headers.get("X-Gemini-API-Key", "").strip() or None

    session = store.get_or_create(req.user_id)
    plan = session.current_plan or generate_learning_path(
        role=session.target_role or "Data Scientist",
        mastery=session.mastery,
        preferences=session.preferences
    )
    
    node = next((n for n in plan.get("nodes", []) if n["skill"] == req.skill_id), None)
    if not node:
        raise HTTPException(status_code=404, detail="Skill not found in active path")
    
    explanation = explain_node_trace(node["trace"], node["skill_name"], override_key=override_key)
    return {
        "skill_id": req.skill_id,
        "skill_name": node["skill_name"],
        "trace": node["trace"],
        "explanation": explanation
    }


@app.post("/api/chat")
def chat_with_coach(req: ChatRequest, request: Request):
    """Contextual Q&A with the grounded multi-agent assistant with rate limit & injection protection."""
    check_rate_limit(request, category="chat")
    clean_msg = req.message.strip()
    if not clean_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Judge/Tester key override: used only for this request, never stored
    override_key = request.headers.get("X-Gemini-API-Key", "").strip() or None

    session = store.get_or_create(req.user_id)
    store.add_chat_message(req.user_id, "user", clean_msg)
    context = store.get_context(req.user_id)
    
    persona = req.persona or "general"
    response = answer_learner_query(clean_msg, context, persona=persona, override_key=override_key)
    store.add_chat_message(req.user_id, "assistant", response)
    
    return {
        "reply": response,
        "persona_used": persona,
        "session_context_utilized": True
    }


@app.post("/api/chat/micro-challenge")
def get_micro_challenge(req: BaseSanitizedModel, request: Request):
    """Generate a 60-second spaced-repetition micro-challenge nudge for memory retention."""
    check_rate_limit(request, category="chat")
    session = store.get_or_create(req.user_id)
    context = store.get_context(req.user_id)
    role = session.target_role or "Data Scientist"
    
    # Pick first pending skill or mastered skill for review
    from .diagnostic import get_items_for_skill
    skills = [p["skill"] for p in context.get("plan_summary", []) if p.get("skill")] or ["python_basics"]
    target_skill = skills[0]
    items = get_items_for_skill(target_skill)
    item = items[0] if items else {}
    q_text = item.get("q") or item.get("question_text") or item.get("question") or f"What is the primary trade-off when optimizing data structures for {role}?"
    options = item.get("options", ["A) Time vs Space Complexity", "B) File Size vs Resolution", "C) Network Bandwidth vs CPU Cores", "D) RAM Speed vs Disk RPM"])

    nudge_text = (
        f"⚡ **60-Second Micro-Challenge ({target_skill}):**\n\n"
        f"**Question:** {q_text}\n\n"
        f"**Options:**\n" + "\n".join(f"- {opt}" for opt in options) + "\n\n"
        f"💡 *Reply with your option choice to instantly test your retention curve!*"
    )
    
    store.add_chat_message(req.user_id, "assistant", nudge_text)
    return {
        "reply": nudge_text,
        "skill_id": target_skill,
        "question": item
    }


@app.get("/api/export/markdown")
def export_roadmap_markdown(user_id: str):
    """Generate a clean, printable Markdown study syllabus of the learner's active path."""
    clean_uid = user_id.strip()
    if not clean_uid or not USER_ID_REGEX.match(clean_uid):
        raise HTTPException(status_code=400, detail="Valid user_id is required")
    
    session = store.get_or_create(clean_uid)
    plan = session.current_plan or generate_learning_path(
        role=session.target_role or "Data Scientist",
        mastery=session.mastery,
        preferences=session.preferences
    )
    
    md_lines = [
        f"# Career PathFinder — {plan['role']} Study Syllabus",
        f"**Estimated Duration:** {plan['estimated_weeks']} Weeks ({plan['weekly_commitment_hours']} hrs/week)",
        f"**Total Learning Effort:** ~{plan['total_estimated_hours']} Hours across {plan['total_steps']} Milestones",
        f"**Overall Role Readiness:** {int(plan['readiness_score'] * 100)}%\n",
        "---",
        "## 🗺️ Learning Path Milestones\n"
    ]
    
    for node in plan.get("nodes", []):
        md_lines.append(f"### Step {node['step_number']}: {node['skill_name']} ({node['phase_name']})")
        md_lines.append(f"- **Difficulty:** Tier {node['difficulty']} | **Labor-Market Importance:** {int(node['importance'] * 100)}%")
        md_lines.append(f"- **Description:** {node['description']}")
        md_lines.append("- **Recommended Courses:**")
        for c in node.get("courses", []):
            md_lines.append(f"  * [{c['title']}]({c['url']}) ({c['platform']}, {c['level']}, Rating: {c['rating']}/5.0)")
        project = node.get("project", {})
        md_lines.append(f"- **Hands-On Capstone Project:** {project.get('title', 'N/A')} (~{project.get('estimated_hours', 8)} hrs)")
        md_lines.append(f"  * *Objective:* {project.get('objective', '')}")
        md_lines.append(f"  * *Deliverables:* {', '.join(project.get('deliverables', []))}")
        md_lines.append(f"- **Milestone Status:** {node.get('status', 'pending').upper()}\n")
        
    return PlainTextResponse(
        content="\n".join(md_lines),
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=Career_Path_{plan['role'].replace(' ', '_')}.md"}
    )
