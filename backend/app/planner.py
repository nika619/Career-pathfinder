"""
Career PathFinder — Learning Path Journey Orchestrator & Glass-Box Tracing
==========================================================================
Coordinates the full closed-loop engine:
1. Gathers learner profile, history, preferences, and measured mastery
2. Identifies ranked skill gaps via O*NET taxonomy
3. Traverses the prerequisite DAG to compute topological sequencing
4. Recommends multi-factor scored courses + hands-on industry projects
5. Attaches verifiable Glass-Box Reasoning Traces to every milestone node
6. Computes estimated time to completion and milestone checkpoints
"""

from typing import Any, Optional
from .skill_graph import target_skills, ordered_path, get_skill_info, get_prerequisites
from .gap import weighted_gap, mastery_summary
from .retrieval import recommend_courses_for_skill, get_project_for_skill

def generate_learning_path(
    role: str,
    mastery: dict[str, float],
    preferences: Optional[dict[str, Any]] = None,
    learning_history: Optional[list[str]] = None
) -> dict[str, Any]:
    """Compile the complete personalized learning roadmap.

    Args:
        role: Target career role (e.g., "Data Scientist")
        mastery: Measured mastery per skill (0.0 to 1.0)
        preferences: Learner constraints (weekly_hours, format, experience)
        learning_history: Claimed or completed skill IDs

    Returns:
        Structured roadmap payload with summary metrics, ordered milestone nodes,
        courses, projects, assessments, and glass-box traces.
    """
    prefs = preferences or {"weekly_hours": 10, "preferred_format": "interactive", "experience_level": "beginner"}
    known_skills = list(learning_history or []) + [s for s, score in mastery.items() if score >= 0.60]

    # 1. Calculate O*NET-grounded skill gap
    gaps = weighted_gap(role, mastery, threshold=0.60)
    gap_lookup = {g["skill"]: g for g in gaps}

    # 2. Extract target skills and sequence via Topological Sort on DAG
    targets = target_skills(role)
    importance_map = {g["skill"]: g["importance"] for g in gaps}
    sequence = ordered_path(targets, known=known_skills, role_skills_importance=importance_map)

    # 3. Assemble structured milestone nodes
    nodes = []
    total_estimated_hours = 0
    weekly_hours = max(1, prefs.get("weekly_hours", 10))

    for idx, skill_id in enumerate(sequence):
        meta = get_skill_info(skill_id)
        gap_info = gap_lookup.get(skill_id)
        importance = gap_info["importance"] if gap_info else 0.70
        curr_mastery = round(mastery.get(skill_id, 0.0), 2)
        prereqs = get_prerequisites(skill_id)

        # Multi-factor recommended courses
        courses = recommend_courses_for_skill(
            skill_id=skill_id,
            skill_name=meta.get("name", skill_id),
            learner_experience=prefs.get("experience_level", "beginner"),
            preferred_format=prefs.get("preferred_format", "interactive"),
            role_importance=importance,
            k=2
        )

        # Hands-on project
        project = get_project_for_skill(skill_id)
        step_hours = 10 + project.get("estimated_hours", 8)
        total_estimated_hours += step_hours

        # Determine learning phase and concurrent track capabilities
        diff = meta.get("difficulty", 1)
        if diff == 1:
            phase_num = 1
            phase_title = "Phase 1: Foundational Core"
        elif diff == 2:
            phase_num = 2
            phase_title = "Phase 2: Applied Tooling & Frameworks"
        else:
            phase_num = 3
            phase_title = "Phase 3: Production & Specialization"

        # Glass-Box Reasoning Trace (The core explainability differentiator)
        trace = {
            "target_role": role,
            "skill_importance_score": importance,
            "onet_classification": gap_info.get("onet_source", "Core Role Competency") if gap_info else "Prerequisite Competency",
            "measured_mastery_level": f"{int(curr_mastery * 100)}%",
            "prerequisites_required": [get_skill_info(p).get("name", p) for p in prereqs],
            "prerequisites_status": "All prerequisites satisfied or scheduled prior",
            "recommendation_rationale": (
                f"Selected as step #{idx + 1} ({phase_title}) because it closes a critical {int((1.0 - curr_mastery) * 100)}% "
                f"mastery deficit for '{role}'. Prerequisite constraints strictly require this foundational "
                f"competency before downstream advanced topics."
            )
        }

        # Outcome-based milestone assessment
        milestone = {
            "milestone_id": f"MS-{idx + 1}",
            "title": f"Milestone {idx + 1}: {meta.get('name', skill_id)} Mastery",
            "type": "capstone_and_quiz",
            "phase": phase_title,
            "pass_criteria": "Submit working project repository & achieve >=70% on adaptive diagnostic",
            "estimated_effort_hours": step_hours
        }

        nodes.append({
            "step_number": idx + 1,
            "phase": phase_num,
            "phase_name": phase_title,
            "skill": skill_id,
            "skill_name": meta.get("name", skill_id),
            "difficulty": diff,
            "cluster": meta.get("cluster", "general"),
            "description": meta.get("description", ""),
            "importance": importance,
            "current_mastery": curr_mastery,
            "courses": courses,
            "project": project,
            "milestone": milestone,
            "trace": trace,
            "status": "pending"
        })

    # Summary analytics & Phase breakdown
    summary = mastery_summary(role, mastery)
    estimated_weeks = max(1, round(total_estimated_hours / weekly_hours))

    # Compile phase summaries
    phases_map = {}
    for n in nodes:
        p_name = n["phase_name"]
        if p_name not in phases_map:
            phases_map[p_name] = {"name": p_name, "skills": [], "total_hours": 0}
        phases_map[p_name]["skills"].append(n["skill_name"])
        phases_map[p_name]["total_hours"] += (10 + n["project"].get("estimated_hours", 8))

    return {
        "role": role,
        "readiness_score": summary["overall_readiness"],
        "skills_mastered_count": summary["skills_mastered"],
        "total_skills_count": summary["skills_total"],
        "total_steps": len(nodes),
        "total_estimated_hours": total_estimated_hours,
        "estimated_weeks": estimated_weeks,
        "weekly_commitment_hours": weekly_hours,
        "phases": list(phases_map.values()),
        "gaps_identified": gaps,
        "nodes": nodes
    }
