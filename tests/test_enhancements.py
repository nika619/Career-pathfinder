"""
Career PathFinder — Enhancement Tests
=====================================
Validates the enhancements added for the final submission:
1. Adaptive diagnostic termination (early-stop + difficulty targeting)
2. Open-world roles now produce a real weighted skill gap
3. Semantic embedding retrieval is active with a TF-IDF fallback
"""
import numpy as np
from backend.app.diagnostic import should_continue_diagnostic, get_next_diagnostic_question
from backend.app.gap import weighted_gap, get_role_profile
from backend.app import retrieval


# --- 1. Adaptive diagnostic termination ---
def test_diagnostic_stops_when_confidently_mastered():
    cont, reason = should_continue_diagnostic(current_mastery=0.95, num_asked=2)
    assert cont is False and reason == "confident_mastered"


def test_diagnostic_stops_when_confidently_gap():
    cont, reason = should_continue_diagnostic(current_mastery=0.05, num_asked=2)
    assert cont is False and reason == "confident_gap"


def test_diagnostic_respects_min_and_max_questions():
    # Below the minimum -> always continue, regardless of confidence.
    assert should_continue_diagnostic(0.99, 1)[0] is True
    # At the max -> always stop.
    assert should_continue_diagnostic(0.5, 4) == (False, "max_questions")


def test_diagnostic_continues_when_uncertain():
    cont, reason = should_continue_diagnostic(current_mastery=0.5, num_asked=2)
    assert cont is True and reason == ""


def test_adaptive_item_selection_targets_difficulty():
    # Higher mastery should skew the served item toward higher difficulty.
    q_low = get_next_diagnostic_question("python_basics", asked_questions=[], current_mastery=0.05)
    q_high = get_next_diagnostic_question("python_basics", asked_questions=[], current_mastery=0.95)
    assert q_low is not None and q_high is not None
    assert q_high["difficulty"] >= q_low["difficulty"]


# --- 2. Open-world gap ---
def test_open_world_role_produces_nonempty_gap():
    gap = weighted_gap("Cybersecurity Penetration Tester", mastery={})
    assert len(gap) >= 1
    assert all(0.0 <= g["importance"] <= 1.0 for g in gap)


def test_open_world_profile_labeled_estimated_not_onet():
    profile = get_role_profile("Cloud DevOps Engineer")
    assert len(profile) >= 1
    assert all(p["onet_source"] == "Estimated (heuristic)" for p in profile)


# --- 3. Semantic retrieval ---
def test_semantic_embeddings_loaded_and_active():
    assert retrieval._EMB_READY is True, "Precomputed embeddings should load at runtime"
    recs = retrieval.recommend_courses_for_skill("statistics", "Statistics", k=3)
    assert len(recs) == 3
    assert recs[0]["retrieval_mode"] == "semantic_embedding"
    assert 0.0 <= recs[0]["match_score"] <= 1.0


def test_semantic_scores_bounded_and_fallback_signal():
    s = retrieval._semantic_scores("statistics")
    assert s is not None
    assert float(np.min(s)) >= 0.0 and float(np.max(s)) <= 1.0
    # Unknown skill id -> None (signals TF-IDF fallback).
    assert retrieval._semantic_scores("__nonexistent_skill__") is None


# --- 4. Multi-Agent Guild, Mermaid Diagrams & Micro-Challenges ---
def test_multi_agent_persona_switching_in_chat():
    """Verify that different mentor personas adapt tone and return persona_used in response."""
    from backend.app.assistant import answer_learner_query
    ctx = {"target_role": "Software Developer", "readiness_percentage": 45}
    
    resp_algo = answer_learner_query("How to optimize loop?", ctx, persona="algo_specialist")
    assert "Complexity Target" in resp_algo or "Algorithm" in resp_algo

    resp_arch = answer_learner_query("How to scale API?", ctx, persona="system_architect")
    assert "System Architecture" in resp_arch or "API Contracts" in resp_arch

    resp_career = answer_learner_query("How to prepare for interview?", ctx, persona="career_strategist")
    assert "STAR" in resp_career or "Interview" in resp_career


def test_mermaid_diagram_generation_for_architecture_queries():
    """Verify that architecture or flowchart queries return valid Mermaid.js diagrams."""
    from backend.app.assistant import answer_learner_query
    ctx = {"target_role": "ML Engineer", "readiness_percentage": 30, "plan_summary": [{"skill": "python_basics"}, {"skill": "numpy"}]}
    
    resp_diag = answer_learner_query("Draw system architecture diagram", ctx, persona="general")
    assert "```mermaid" in resp_diag
    assert "graph TD" in resp_diag


def test_micro_challenge_endpoint():
    """Verify that /api/chat/micro-challenge returns spaced repetition questions."""
    from fastapi.testclient import TestClient
    from backend.app.main import app
    client = TestClient(app)
    
    res = client.post("/api/chat/micro-challenge", json={"user_id": "test_micro_user"})
    assert res.status_code == 200
    data = res.json()
    assert "60-Second Micro-Challenge" in data["reply"]
    assert "skill_id" in data
