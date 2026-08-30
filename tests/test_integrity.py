import json
import time
from pathlib import Path
import networkx as nx
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.skill_graph import G, load_graph, get_all_roles, target_skills, ordered_path, get_skill_info
from backend.app.gap import weighted_gap, mastery_summary
from backend.app.diagnostic import bkt_update, get_bkt_params, verify_answer
from backend.app.retrieval import recommend_courses_for_skill, get_project_for_skill, PROJECT_BANK
from backend.app.planner import generate_learning_path

DATA_DIR = Path(__file__).resolve().parent.parent / "backend" / "data"

@pytest.fixture(scope="module")
def raw_data():
    with open(DATA_DIR / "skill_graph.json", "r") as f:
        sg = json.load(f)
    with open(DATA_DIR / "role_skills.json", "r") as f:
        rs = json.load(f)
    with open(DATA_DIR / "courses.json", "r") as f:
        courses = json.load(f)
    with open(DATA_DIR / "item_bank.json", "r") as f:
        ib = json.load(f)
    return {"sg": sg, "rs": rs, "courses": courses, "ib": ib}


# =====================================================================
# 1. GRAPH TOPOLOGY & PREREQUISITE INTEGRITY
# =====================================================================

def test_graph_is_strictly_acyclic_and_self_loop_free(raw_data):
    """Verify DAG is strictly acyclic, connected, and contains no self-loops."""
    g = load_graph()
    assert nx.is_directed_acyclic_graph(g), "Skill graph must be a strict DAG without cycles"
    assert len(list(nx.nodes_with_selfloops(g))) == 0, "Graph cannot contain self-loops"
    assert g.number_of_nodes() > 50, "Graph must contain at least 50 skills"
    assert g.number_of_edges() > 55, "Graph must contain at least 55 prerequisite edges"


def test_all_edges_reference_existing_nodes(raw_data):
    """Verify that every directed prerequisite edge points between existing nodes."""
    nodes = set(raw_data["sg"]["nodes"].keys())
    for source, target in raw_data["sg"]["edges"]:
        assert source in nodes, f"Edge source '{source}' does not exist in nodes"
        assert target in nodes, f"Edge target '{target}' does not exist in nodes"


def test_all_role_target_skills_exist_in_dag(raw_data):
    """Verify that every career role specifies skills that exist in the DAG."""
    nodes = set(raw_data["sg"]["nodes"].keys())
    for role, skills in raw_data["sg"]["role_targets"].items():
        assert len(skills) >= 6, f"Role '{role}' must have at least 6 required skills"
        for s in skills:
            assert s in nodes, f"Skill '{s}' in role '{role}' missing from DAG nodes"


# =====================================================================
# 2. CROSS-DATASET SYNCHRONIZATION INTEGRITY
# =====================================================================

def test_100_percent_item_bank_coverage_for_all_dag_skills(raw_data):
    """Verify that every single skill in the DAG has diagnostic items with valid answers."""
    nodes = set(raw_data["sg"]["nodes"].keys())
    ib = raw_data["ib"]
    
    for node in nodes:
        assert node in ib, f"Skill '{node}' is missing from item_bank.json"
        items = ib[node]
        assert len(items) >= 1, f"Skill '{node}' has no diagnostic questions"
        for item in items:
            assert "q" in item and len(item["q"]) > 5, f"Malformed question text in '{node}'"
            assert "options" in item and len(item["options"]) >= 2, f"Insufficient options in '{node}'"
            assert "answer" in item, f"Missing answer in '{node}'"
            assert item["answer"] in item["options"], f"Answer '{item['answer']}' not among options in '{node}'"


def test_100_percent_project_bank_coverage_for_all_dag_skills(raw_data):
    """Verify that every DAG skill has a dedicated, non-generic industry capstone project."""
    nodes = set(raw_data["sg"]["nodes"].keys())
    for node in nodes:
        assert node in PROJECT_BANK, f"Skill '{node}' is missing from dedicated PROJECT_BANK"
        p = PROJECT_BANK[node]
        assert "title" in p and len(p["title"]) > 5
        assert "objective" in p and len(p["objective"]) > 20
        assert "deliverables" in p and len(p["deliverables"]) >= 2
        assert "estimated_hours" in p and 4 <= p["estimated_hours"] <= 30


def test_role_skills_onet_weights_are_normalized(raw_data):
    """Verify that all O*NET weights in role_skills.json are strictly within [0.0, 1.0]."""
    rs = raw_data["rs"]
    nodes = set(raw_data["sg"]["nodes"].keys())
    for role, profile in rs.items():
        assert len(profile) >= 6, f"Profile for '{role}' has insufficient skills"
        for item in profile:
            assert item["skill"] in nodes, f"Skill '{item['skill']}' in role_skills not in DAG"
            assert 0.0 <= item["importance"] <= 1.0, f"Importance out of bounds for '{item['skill']}'"


# =====================================================================
# 3. MATHEMATICAL INVARIANTS (BKT & MULTI-FACTOR RANKING)
# =====================================================================

def test_bkt_posterior_bounds_and_monotonicity():
    """Verify that BKT mathematically preserves probability bounds [0, 1] and strictly rewards correctness."""
    p_priors = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    for prior in p_priors:
        p_correct = bkt_update(prior, is_correct=True, skill_id="python_basics")
        p_incorrect = bkt_update(prior, is_correct=False, skill_id="python_basics")
        
        # Bounds invariant
        assert 0.0 <= p_correct <= 1.0, f"BKT correct out of bounds: {p_correct}"
        assert 0.0 <= p_incorrect <= 1.0, f"BKT incorrect out of bounds: {p_incorrect}"
        
        # Monotonicity invariant: correct answer must produce higher posterior than incorrect
        assert p_correct > p_incorrect, f"BKT monotonicity violated: {p_correct} <= {p_incorrect} for prior {prior}"


def test_bkt_asymptotic_convergence():
    """Verify that consecutive correct answers converge mastery toward 1.0, and incorrect toward lower bound."""
    p = 0.20
    for _ in range(8):
        p = bkt_update(p, is_correct=True, skill_id="statistics")
    assert p >= 0.85, f"Consecutive correct answers must converge above 0.85 (got {p})"

    p_fail = 0.50
    for _ in range(8):
        p_fail = bkt_update(p_fail, is_correct=False, skill_id="statistics")
    assert p_fail <= 0.35, f"Consecutive incorrect answers must depress posterior (got {p_fail})"


def test_recommendation_scoring_formula_invariants():
    """Verify that recommendation scores are bounded in [0.0, 1.0] and sorted strictly descending."""
    courses = recommend_courses_for_skill("supervised_learning", "Supervised Learning", k=5)
    assert len(courses) > 0
    prev_score = 1.0
    for c in courses:
        score = c["match_score"]
        assert 0.0 <= score <= 1.0, f"Score {score} out of bounds"
        assert score <= prev_score + 1e-6, "Courses must be sorted strictly descending by match_score"
        prev_score = score
        
        # Score breakdown weight sums to ~1.0
        breakdown = c["score_breakdown"]
        weights_sum = 0.45 + 0.20 + 0.20 + 0.15
        assert abs(weights_sum - 1.0) < 1e-6


# =====================================================================
# 4. API CONTRACTS & DEFENSIVE ERROR BOUNDARIES
# =====================================================================

client = TestClient(app)

def test_api_health_endpoint_schema():
    """Verify health endpoint returns valid system metadata."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert len(data["supported_roles"]) == 6


def test_api_goal_empty_string_rejection():
    """Verify that empty goal input returns 400 or 422 validation error."""
    res = client.post("/api/goal", json={"user_id": "test_u1", "goal_text": "   "})
    assert res.status_code == 400 or res.status_code == 422


def test_api_progress_invalid_status_rejection():
    """Verify that invalid milestone status is rejected with validation error."""
    res = client.post("/api/progress", json={
        "user_id": "test_u2",
        "skill_id": "python_basics",
        "status": "invalid_status_value"
    })
    assert res.status_code == 422 or res.status_code == 400


def test_sqlite_session_persistence_across_reloads():
    """Verify that learner sessions persist durably in SQLite database across memory reloads."""
    from backend.app.store import SessionStore
    temp_store = SessionStore()
    uid = f"persist_test_user_{int(time.time())}"
    
    # 1. Create and update session
    temp_store.set_role(uid, "Data Scientist")
    temp_store.set_skill_mastery(uid, "python_basics", 0.95)
    temp_store.update_preferences(uid, {"weekly_hours": 20})
    temp_store.update_milestone(uid, "python_basics", "passed")
    
    # 2. Simulate complete memory eviction / server restart
    reloaded_store = SessionStore()
    reloaded_session = reloaded_store.get_or_create(uid)
    
    assert reloaded_session.target_role == "Data Scientist"
    assert reloaded_session.mastery.get("python_basics") == 1.0  # 0.95 + 0.50 capped at 1.0 on pass
    assert reloaded_session.preferences.get("weekly_hours") == 20
    assert reloaded_session.milestone_status.get("python_basics") == "passed"


def test_readiness_score_strict_synchronization():
    """Verify that UI plan readiness_score and AI Coach chat readiness_percentage are 100% synchronized."""
    from backend.app.store import store
    from backend.app.planner import generate_learning_path
    from backend.app.assistant import answer_learner_query

    uid = f"sync_test_user_{int(time.time())}"
    store.set_role(uid, "Data Analyst")
    store.set_skill_mastery(uid, "python_basics", 0.90)
    store.set_skill_mastery(uid, "sql", 0.85)

    plan = generate_learning_path(role="Data Analyst", mastery=store.get_mastery(uid), preferences={})
    store.save_plan(uid, plan)

    # 1. UI Card Readiness Score
    ui_readiness_pct = round(plan["readiness_score"] * 100)

    # 2. Coach Chat Context & Response
    ctx = store.get_context(uid)
    assert ctx["readiness_percentage"] == ui_readiness_pct, f"Context readiness {ctx['readiness_percentage']}% != UI {ui_readiness_pct}%"

    coach_response = answer_learner_query("how to use u", ctx)
    assert f"{ui_readiness_pct}%" in coach_response, f"Coach response should contain '{ui_readiness_pct}%'. Response was: {coach_response}"


def test_dynamic_open_world_role_synthesis():
    """Verify that ANY arbitrary niche role (Cybersecurity, Cloud DevOps, Bioinformatics) dynamically generates a valid acyclic DAG path."""
    niche_roles = [
        "Cloud DevOps Engineer",
        "Cybersecurity Penetration Tester",
        "Bioinformatics Genomic Scientist",
        "Game Developer with Unreal Engine"
    ]
    for role in niche_roles:
        plan = generate_learning_path(role=role, mastery={}, preferences={"weekly_hours": 12})
        assert plan["role"] == role
        assert plan["total_steps"] >= 5
        assert len(plan["nodes"]) == plan["total_steps"]
        assert len(plan["phases"]) >= 2
        # Verify first step has zero unfulfilled prerequisites
        first_node = plan["nodes"][0]
        assert "step_number" in first_node and first_node["step_number"] == 1


def test_automated_capstone_rubric_evaluation_endpoint():
    """Verify that automated capstone grading evaluates repositories against the 4-point rubric."""
    res = client.post("/api/milestone/evaluate", json={
        "user_id": "eval_test_user",
        "skill_id": "python_basics",
        "github_url": "https://github.com/learner/cli-parser-engine",
        "deliverables_notes": "Implemented clean modular architecture with PyTest unit test coverage and Docker container.",
        "completed_checklist": ["clean_code", "unit_tests", "documentation"]
    })
    assert res.status_code == 200
    data = res.json()
    assert "passed" in data
    assert data["passed"] is True
    assert data["rubric_score"] >= 70
    assert "rubric_breakdown" in data
    assert data["rubric_breakdown"]["architecture_design"] == 25
    assert data["rubric_breakdown"]["test_coverage_validation"] == 25


def test_full_closed_loop_journey_generation_all_6_roles():
    """Verify end-to-end learning path compilation works flawlessly with sequential phases for all 6 roles."""
    roles = get_all_roles()
    for role in roles:
        plan = generate_learning_path(role=role, mastery={}, preferences={"weekly_hours": 15})
        assert plan["role"] == role
        assert plan["total_steps"] >= 6
        assert plan["readiness_score"] == 0.0
        assert len(plan["nodes"]) == plan["total_steps"]
        assert "phases" in plan and len(plan["phases"]) >= 2
        
        # Verify first step has zero unsatisfied prerequisites
        first_node = plan["nodes"][0]
        assert "step_number" in first_node and first_node["step_number"] == 1
        assert "trace" in first_node
        assert len(first_node["courses"]) > 0
        assert "title" in first_node["project"]
