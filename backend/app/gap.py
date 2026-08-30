"""
Career PathFinder — Weighted Skill-Gap Engine
==============================================
Computes the gap between a learner's current mastery and what a target
career role requires, ranked by importance × deficit.

Grounded in O*NET-derived role-skill profiles — the weights come from
real labour-market data, not from an LLM's opinion.

This is a pure function: deterministic, stateless, fully testable.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load O*NET-derived weighted role profiles at module level
with open(DATA_DIR / "role_skills.json", "r") as f:
    _role_skills = json.load(f)


def get_role_profile(role: str) -> list[dict]:
    """Return the weighted skill profile for a role.
    
    Supports both standard O*NET mapped roles and on-the-fly custom synthesized roles.
    """
    if role in _role_skills:
        return _role_skills[role]
        
    # Dynamically synthesize weighted profile for open-world role
    from .skill_graph import target_skills
    skills = target_skills(role)
    return [
        {
            "skill": s,
            "importance": round(0.75 + (0.02 * (idx % 10)), 2),
            "onet_source": "Estimated (heuristic)"
        }
        for idx, s in enumerate(skills)
    ]


def weighted_gap(
    role: str,
    mastery: dict[str, float],
    threshold: float = 0.6
) -> list[dict]:
    """Compute the ranked skill gap for a learner targeting a specific role.

    For each skill in the role's profile, if the learner's mastery is below
    the threshold, it's a gap. Gaps are ranked by priority = importance × (1 - mastery),
    so the most important and least-mastered skills surface first.

    Args:
        role: Target career role name.
        mastery: Dict mapping skill_id → mastery level (0.0 to 1.0).
        threshold: Mastery level above which a skill is considered "known" (default 0.6).

    Returns:
        List of gap dicts sorted by priority (descending), each containing:
        - skill: skill ID
        - importance: O*NET-derived importance weight (0-1)
        - onet_source: The O*NET category this maps to
        - mastery: learner's current mastery
        - priority: importance × (1 - mastery) — the urgency score
    """
    profile = get_role_profile(role)  # falls back to a synthesized profile for open-world roles
    gaps = []

    for item in profile:
        m = mastery.get(item["skill"], 0.0)
        if m < threshold:
            priority = round(item["importance"] * (1 - m), 3)
            gaps.append({
                "skill": item["skill"],
                "importance": item["importance"],
                "onet_source": item.get("onet_source", ""),
                "mastery": round(m, 3),
                "priority": priority,
            })

    return sorted(gaps, key=lambda x: x["priority"], reverse=True)


def mastery_summary(role: str, mastery: dict[str, float]) -> dict:
    """Generate a summary of the learner's overall readiness for a role.

    Args:
        role: Target career role name.
        mastery: Dict mapping skill_id → mastery level.

    Returns:
        Dict with overall_readiness, skills_mastered, skills_total,
        gap_count, and top_gaps.
    """
    profile = _role_skills.get(role, [])
    if not profile:
        return {"overall_readiness": 0, "skills_mastered": 0, "skills_total": 0}

    mastered = sum(1 for item in profile if mastery.get(item["skill"], 0.0) >= 0.6)
    total = len(profile)

    # Weighted readiness: sum(mastery × importance) / sum(importance)
    weighted_score = sum(
        mastery.get(item["skill"], 0.0) * item["importance"]
        for item in profile
    )
    total_importance = sum(item["importance"] for item in profile)
    readiness = round(weighted_score / total_importance, 3) if total_importance > 0 else 0

    gaps = weighted_gap(role, mastery)

    return {
        "overall_readiness": readiness,
        "skills_mastered": mastered,
        "skills_total": total,
        "gap_count": len(gaps),
        "top_gaps": gaps[:5],
    }
