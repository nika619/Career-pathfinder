"""
Test Diagnostic Coverage Across All 6 Supported Career Roles
"""
import json
from pathlib import Path
from backend.app.skill_graph import target_skills

DATA_DIR = Path(__file__).resolve().parent.parent / "backend" / "data"

def test_diagnostic_item_bank_covers_all_roles():
    with open(DATA_DIR / "item_bank.json", "r") as f:
        item_bank = json.load(f)
    
    roles = [
        "Data Scientist",
        "Data Analyst",
        "ML Engineer",
        "Software Developer",
        "Full Stack Developer",
        "UX Designer"
    ]
    
    for role in roles:
        skills = target_skills(role)[:5]
        covered_skills = [s for s in skills if s in item_bank and len(item_bank[s]) >= 2]
        assert len(covered_skills) >= 4, f"Role '{role}' needs at least 4 diagnostic skills in item bank, found {len(covered_skills)}"
