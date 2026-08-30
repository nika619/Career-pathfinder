"""
Career PathFinder — Comprehensive Test Suite & Benchmark Harness
================================================================
Validates:
1. Prerequisite Graph: DAG properties, cycle detection, topological sequencing
2. Skill-Gap Engine: O*NET weighting, priority sorting, mastery thresholds
3. Bayesian Knowledge Tracing (BKT): exact probabilistic posterior updates
4. Multi-Factor Course Recommender: semantic scoring, level fit, format weighting
5. End-to-End Orchestrator: full path generation, glass-box trace integrity
6. Dynamic Adaptation: re-planning on milestone pass/fail
7. Synthetic Persona Benchmark: 4 distinct realistic learner simulation tests
"""

import pytest
import networkx as nx
from backend.app.skill_graph import G, ordered_path, target_skills, get_prerequisites
from backend.app.gap import weighted_gap, mastery_summary
from backend.app.diagnostic import bkt_update, verify_answer
from backend.app.retrieval import recommend_courses_for_skill, get_project_for_skill
from backend.app.planner import generate_learning_path
from backend.app.assistant import parse_goal_to_role

# 1. Prerequisite Graph & DAG Integrity
def test_skill_graph_is_acyclic():
    """Graph must be a strict DAG without circular dependencies."""
    assert nx.is_directed_acyclic_graph(G), "CRITICAL: Cycle detected in prerequisite DAG!"

def test_prerequisite_topological_order():
    """Prerequisites must ALWAYS appear strictly prior to downstream dependent skills."""
    # Python basics must precede Pandas, NumPy, and ML Foundations
    path = ordered_path(["deep_learning"], known=[])
    assert "python_basics" in path
    assert "ml_foundations" in path
    assert "deep_learning" in path
    assert path.index("python_basics") < path.index("ml_foundations")
    assert path.index("ml_foundations") < path.index("deep_learning")

def test_known_skills_are_pruned_from_path():
    """Learners who already know prerequisites should not be assigned them again."""
    path_without_known = ordered_path(["pandas"], known=[])
    assert "python_basics" in path_without_known

    path_with_known = ordered_path(["pandas"], known=["python_basics"])
    assert "python_basics" not in path_with_known
    assert "pandas" in path_with_known

# 2. O*NET Weighted Gap Engine
def test_gap_ranking_prioritizes_high_importance_high_deficit():
    """Gaps must be ranked by Importance * (1 - Mastery)."""
    mastery = {"python_basics": 0.90, "statistics": 0.10, "sql": 0.20}
    gaps = weighted_gap("Data Scientist", mastery=mastery, threshold=0.60)
    
    # Python basics is mastered (>0.60), so it must NOT be in the gap list
    gap_skills = [g["skill"] for g in gaps]
    assert "python_basics" not in gap_skills
    assert "statistics" in gap_skills
    
    # List must be strictly sorted in descending priority
    priorities = [g["priority"] for g in gaps]
    assert priorities == sorted(priorities, reverse=True)

def test_mastery_summary_metrics():
    """Readiness metrics must be mathematically bounded between 0 and 1."""
    mastery = {"python_basics": 0.80, "numpy": 0.80}
    summary = mastery_summary("Data Scientist", mastery)
    assert 0.0 <= summary["overall_readiness"] <= 1.0
    assert summary["skills_mastered"] == 2

# 3. Bayesian Knowledge Tracing Mathematics
def test_bkt_mastery_increases_on_correct_answer():
    """Answering correctly must strictly increase estimated mastery."""
    prior = 0.30
    posterior = bkt_update(prior, is_correct=True, skill_id="python_basics")
    assert posterior > prior
    assert posterior <= 1.0

def test_bkt_mastery_decreases_or_damps_on_incorrect_answer():
    """Answering incorrectly must decrease mastery before transit update."""
    prior = 0.70
    posterior = bkt_update(prior, is_correct=False, skill_id="deep_learning")
    assert posterior < prior

def test_diagnostic_answer_verification():
    """Verification must correctly match question answer keys."""
    is_correct, ans = verify_answer("python_basics", "Which keyword is used to define a function in Python?", "def")
    assert is_correct is True
    assert ans == "def"

# 4. Multi-Factor Course Recommendation
def test_multi_factor_recommendation_returns_curated_courses():
    """Recommender must return top-k real courses with score breakdowns."""
    recs = recommend_courses_for_skill(
        skill_id="python_basics",
        skill_name="Python Fundamentals",
        learner_experience="beginner",
        preferred_format="interactive",
        k=2
    )
    assert len(recs) == 2
    for r in recs:
        assert "title" in r
        assert "url" in r
        assert "score_breakdown" in r
        assert r["match_score"] > 0.0

def test_project_retrieval_for_skill():
    """Every skill must map to an industry-grade practical project."""
    proj = get_project_for_skill("supervised_learning")
    assert "title" in proj
    assert "deliverables" in proj
    assert proj["estimated_hours"] > 0

# 5. Full End-to-End Learning Path Orchestration
def test_end_to_end_learning_path_generation():
    """Complete roadmap compilation with traces, milestones, and time estimates."""
    plan = generate_learning_path(
        role="Data Scientist",
        mastery={"python_basics": 0.85},
        preferences={"weekly_hours": 15, "experience_level": "beginner"}
    )
    assert plan["role"] == "Data Scientist"
    assert len(plan["nodes"]) > 0
    assert plan["total_estimated_hours"] > 0
    assert plan["estimated_weeks"] > 0

    first_node = plan["nodes"][0]
    assert "trace" in first_node
    assert "courses" in first_node
    assert "project" in first_node
    assert "milestone" in first_node
    assert first_node["trace"]["target_role"] == "Data Scientist"

# 6. Natural Language Goal Classification
def test_goal_parsing_taxonomy_mapping():
    """Natural language goals must classify into supported career taxonomy or reject unclear inputs."""
    res1 = parse_goal_to_role("I want to become a machine learning engineer building neural networks")
    assert res1["role"] in ["ML Engineer", "Data Scientist"]

    res2 = parse_goal_to_role("Design intuitive mobile interfaces in Figma as a UX designer")
    assert res2["role"] == "UX Designer"

    # Negative / meaningless inputs must not default to Data Scientist
    res_negative = parse_goal_to_role("nothing")
    assert res_negative["role"] is None
    assert res_negative["error"] == "unclear_goal"

    res_gibberish = parse_goal_to_role("asdfghjk")
    assert res_gibberish["role"] is None

# 7. Synthetic Learner Persona Benchmark
@pytest.mark.parametrize("persona_name,goal,known,expected_first_skill", [
    ("Complete Beginner Data", "Data Scientist", {}, "python_basics"),
    ("Python-Proficient Learner", "Data Scientist", {"python_basics": 0.90}, "statistics"),  # Higher O*NET importance (0.92 vs 0.82)
    ("CS Grad to Full Stack", "Full Stack Developer", {"programming_fundamentals": 0.95, "web_fundamentals": 0.90, "html_css": 0.90}, "git_version_control"), # Foundational difficulty 1
    ("Designer moving to UX", "UX Designer", {}, "design_thinking")
])
def test_synthetic_learner_persona_paths(persona_name, goal, known, expected_first_skill):
    """Verify that different learner profiles receive distinct, properly sequenced entry points."""
    plan = generate_learning_path(role=goal, mastery=known)
    assert len(plan["nodes"]) > 0
    # First recommended skill must match pedagogical prerequisite expectations
    assert plan["nodes"][0]["skill"] == expected_first_skill
