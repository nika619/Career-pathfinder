"""
Career PathFinder — Adaptive Diagnostic & Bayesian Knowledge Tracing (BKT)
========================================================================
Implements Bayesian Knowledge Tracing (Corbett & Anderson standard model),
the foundational AI model used in Intelligent Tutoring Systems (ITS).

Instead of trusting subjective learner self-reports, BKT estimates the hidden
probability p(L_t) that a learner has mastered a specific skill given their
sequence of observed correct/incorrect responses.

Mathematical Model:
- Prior mastery: P(L_0)
- Transit probability: P(T) - chance of learning skill during step
- Slip probability: P(S) - knows skill but makes mistake
- Guess probability: P(G) - doesn't know skill but guesses right

Bayesian Posterior Update on observation:
If Correct:
  P(L_t | Correct) = [P(L_{t-1}) * (1 - P(S))] / [P(L_{t-1}) * (1 - P(S)) + (1 - P(L_{t-1})) * P(G)]
If Incorrect:
  P(L_t | Incorrect) = [P(L_{t-1}) * P(S)] / [P(L_{t-1}) * P(S) + (1 - P(L_{t-1})) * (1 - P(G))]

Learning Step Update:
  P(L_{t+1}) = P(L_t | Obs) + (1 - P(L_t | Obs)) * P(T)
"""

import json
from pathlib import Path
from typing import Optional, Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / "item_bank.json", "r") as f:
    _item_bank: dict[str, list[dict[str, Any]]] = json.load(f)

# Calibrated BKT parameters per difficulty tier
BKT_PARAMS = {
    "default": {"p_init": 0.20, "p_transit": 0.15, "p_slip": 0.10, "p_guess": 0.20},
    "foundational": {"p_init": 0.30, "p_transit": 0.20, "p_slip": 0.08, "p_guess": 0.25},
    "advanced": {"p_init": 0.15, "p_transit": 0.10, "p_slip": 0.12, "p_guess": 0.15}
}

def get_items_for_skill(skill_id: str) -> list[dict[str, Any]]:
    """Return all item bank diagnostic questions for a specific skill."""
    return _item_bank.get(skill_id, [])


def get_bkt_params(skill_id: str) -> dict[str, float]:
    """Return tuned BKT parameters according to skill complexity."""
    if "basics" in skill_id or "fundamentals" in skill_id:
        return BKT_PARAMS["foundational"]
    elif "deep" in skill_id or "system_design" in skill_id or "nlp" in skill_id:
        return BKT_PARAMS["advanced"]
    return BKT_PARAMS["default"]


def bkt_update(p_known: float, is_correct: bool, skill_id: str = "default") -> float:
    """Compute exact Bayesian Knowledge Tracing posterior mastery probability.

    Args:
        p_known: Prior probability of mastery (0.0 to 1.0)
        is_correct: Whether the learner answered correctly
        skill_id: Skill identifier to select tuned parameters

    Returns:
        Updated mastery probability p(L_{t+1}) in range [0.0, 1.0]
    """
    params = get_bkt_params(skill_id)
    p_slip = params["p_slip"]
    p_guess = params["p_guess"]
    p_transit = params["p_transit"]

    # Boundary safeguards
    p_k = max(0.01, min(0.99, p_known))

    if is_correct:
        numerator = p_k * (1.0 - p_slip)
        denominator = numerator + (1.0 - p_k) * p_guess
    else:
        numerator = p_k * p_slip
        denominator = numerator + (1.0 - p_k) * (1.0 - p_guess)

    posterior = numerator / denominator if denominator > 0 else p_k

    # Incorporate transition probability (learning opportunity)
    p_next = posterior + (1.0 - posterior) * p_transit
    return round(float(max(0.0, min(1.0, p_next))), 3)


# Adaptive stopping thresholds — the diagnostic ends early once mastery is
# confidently high or low, so we don't waste the learner's time.
DIAG_MIN_Q = 2       # always ask at least this many before an early stop
DIAG_MAX_Q = 4       # never ask more than this per skill
CONF_KNOWN = 0.85    # confidently mastered
CONF_UNKNOWN = 0.15  # confidently not mastered


def should_continue_diagnostic(current_mastery: float, num_asked: int) -> tuple[bool, str]:
    """Decide whether to keep quizzing this skill (adaptive termination).

    Returns (continue?, stop_reason). stop_reason is '' while continuing.
    """
    if num_asked >= DIAG_MAX_Q:
        return False, "max_questions"
    if num_asked < DIAG_MIN_Q:
        return True, ""
    if current_mastery >= CONF_KNOWN:
        return False, "confident_mastered"
    if current_mastery <= CONF_UNKNOWN:
        return False, "confident_gap"
    return True, ""


def get_next_diagnostic_question(
    skill_id: str,
    asked_questions: list[str],
    current_mastery: float = 0.3
) -> Optional[dict[str, Any]]:
    """Serve the next *adaptively selected* question for a given skill.

    Item selection targets a difficulty near the learner's current estimated
    standing (harder items as mastery rises), rather than a fixed order.

    Args:
        skill_id: Skill identifier
        asked_questions: Question texts already presented to this learner
        current_mastery: Current BKT mastery estimate, used to target difficulty

    Returns:
        Question dict or None if the item bank is exhausted for this skill
    """
    remaining = [it for it in _item_bank.get(skill_id, []) if it["q"] not in asked_questions]
    if not remaining:
        return None

    # Target difficulty ~ current standing (1..3); pick the closest unasked item.
    target_difficulty = max(1, min(3, round(current_mastery * 3) + 1))
    remaining.sort(key=lambda it: abs(it.get("difficulty", 1) - target_difficulty))
    item = remaining[0]

    return {
        "skill_id": skill_id,
        "question": item["q"],
        "options": item["options"],
        "difficulty": item.get("difficulty", 1)
    }


import html

def _normalize_ans(text: str) -> str:
    """Normalize answer string by unescaping HTML entities, trimming, stripping quotes/backticks/punctuation, and lowering."""
    if not text:
        return ""
    cleaned = html.unescape(str(text)).strip().lower()
    cleaned = cleaned.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
    changed = True
    while changed and len(cleaned) >= 2:
        changed = False
        for q in ('"', "'", '`'):
            if cleaned.startswith(q) and cleaned.endswith(q):
                cleaned = cleaned[1:-1].strip()
                changed = True
    return cleaned.strip()


def _normalize_q(text: str) -> str:
    """Normalize question text for robust matching."""
    if not text:
        return ""
    q = html.unescape(str(text)).strip().lower()
    q = q.replace("’", "'").replace("‘", "'").replace('“', '"').replace('”', '"')
    q = q.replace("`", "")
    return " ".join(q.split())


def verify_answer(skill_id: str, question_text: str, selected_option: str) -> tuple[bool, str]:
    """Verify if the chosen option is correct with ultra-robust normalization.

    Checks:
    1. Direct match in target skill_id
    2. Fallback search across all skills in the item bank if not found
    3. HTML entity unescaping, normalized and raw answer comparisons

    Returns:
        (is_correct, correct_answer)
    """
    if not question_text or not selected_option:
        return False, "Unknown"

    clean_q = html.unescape(str(question_text)).strip()
    clean_sel = html.unescape(str(selected_option)).strip()

    norm_selected = _normalize_ans(clean_sel)
    norm_q = _normalize_q(clean_q)

    # 1. First check in target skill_id
    items = list(_item_bank.get(skill_id, []))
    
    # 2. Fallback search across all skills if not found in skill_id
    if not items or not any(_normalize_q(it.get("q", "")) == norm_q for it in items):
        all_items = []
        for s_id, it_list in _item_bank.items():
            all_items.extend(it_list)
        items = all_items

    for item in items:
        raw_q = item.get("q", "")
        if (raw_q.strip() == clean_q or 
            _normalize_q(raw_q) == norm_q or 
            norm_q in _normalize_q(raw_q) or 
            _normalize_q(raw_q) in norm_q):
            
            raw_answer = item.get("answer", "")
            norm_answer = _normalize_ans(raw_answer)
            
            correct = (
                raw_answer.strip().lower() == clean_sel.lower() or
                norm_answer == norm_selected or
                clean_sel == raw_answer.strip()
            )
            return correct, raw_answer
            
    return False, "Unknown"

