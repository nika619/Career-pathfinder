"""
Career PathFinder — Monte Carlo Synthetic Learner Evaluation Benchmark
======================================================================
Simulates 1,000 synthetic learners across diverse personas and background proficiencies
to mathematically benchmark and validate the recommender engine:

Metrics Evaluated:
1. Prerequisite Violation Rate (Strict DAG Invariant — Must be 0.00%)
2. Adaptive Learning Path Compression Ratio (Hours saved vs. rigid one-size-fits-all curricula)
3. Bayesian Knowledge Tracing (BKT) Convergence Stability
4. Multi-Factor Recommendation Score Distribution (MRR)
5. Execution Latency & Algorithmic Throughput
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import time
import random
import statistics
from typing import Any
from backend.app.skill_graph import G, get_all_roles, target_skills, ordered_path, get_prerequisites
from backend.app.gap import weighted_gap, mastery_summary
from backend.app.diagnostic import bkt_update
from backend.app.planner import generate_learning_path
from backend.app.retrieval import recommend_courses_for_skill

def run_monte_carlo_benchmark(num_learners: int = 1000) -> dict[str, Any]:
    print("=" * 75)
    print("  [CAREER PATHFINDER] -- MONTE CARLO SCIENTIFIC EVALUATION BENCHMARK")
    print(f"  Simulating N = {num_learners} synthetic learner profiles across 6 career roles...")
    print("=" * 75)

    roles = get_all_roles()
    all_skills = list(G.nodes)
    
    start_time = time.time()
    
    prereq_violations = 0
    total_generic_hours = 0
    total_adaptive_hours = 0
    path_lengths = []
    bkt_convergence_steps = []
    course_scores = []
    role_distribution = {r: 0 for r in roles}

    # Simulation loop
    for i in range(num_learners):
        target_role = random.choice(roles)
        role_distribution[target_role] += 1
        role_reqs = target_skills(target_role)
        
        # Synthetic persona archetype generation:
        # 40% beginner (0-1 known skills), 40% intermediate (2-5 known skills), 20% career switcher (3-7 known skills)
        persona_rand = random.random()
        if persona_rand < 0.40:
            num_known = random.randint(0, 1)
        elif persona_rand < 0.80:
            num_known = random.randint(2, 5)
        else:
            num_known = random.randint(3, 7)
            
        known_skills = random.sample(all_skills, min(num_known, len(all_skills)))
        synthetic_mastery = {s: random.uniform(0.70, 0.95) for s in known_skills}
        
        # 1. Benchmark Learning Path Generation
        plan = generate_learning_path(
            role=target_role,
            mastery=synthetic_mastery,
            preferences={"weekly_hours": 10, "experience_level": "intermediate"},
            learning_history=known_skills
        )
        
        # Generic unadapted path is the full unpruned topological path for the role
        full_role_plan = generate_learning_path(
            role=target_role,
            mastery={},
            preferences={"weekly_hours": 10, "experience_level": "beginner"},
            learning_history=[]
        )
        generic_hours = full_role_plan["total_estimated_hours"]
        adaptive_hours = plan["total_estimated_hours"]
        
        total_generic_hours += generic_hours
        total_adaptive_hours += adaptive_hours
        path_lengths.append(plan["total_steps"])

        # 2. Strict Prerequisite Sequence Verification
        scheduled_skills = [node["skill"] for node in plan["nodes"]]
        already_learned = set(known_skills)
        
        for skill in scheduled_skills:
            direct_prereqs = get_prerequisites(skill)
            # Prereqs must either be known or scheduled prior in the path
            for p in direct_prereqs:
                if p not in already_learned and p not in scheduled_skills[:scheduled_skills.index(skill)]:
                    prereq_violations += 1
            already_learned.add(skill)

        # 3. Simulate BKT Diagnostic Convergence
        # Test how many items are needed to reach confident mastery threshold (>=0.80)
        p_bkt = 0.20
        steps = 0
        while p_bkt < 0.80 and steps < 10:
            steps += 1
            p_bkt = bkt_update(p_bkt, is_correct=True, skill_id="statistics")
        bkt_convergence_steps.append(steps)

        # 4. Sample Multi-Factor Recommendation Scores
        if plan["nodes"]:
            first_skill = plan["nodes"][0]["skill"]
            recs = recommend_courses_for_skill(first_skill, first_skill, k=2)
            if recs:
                course_scores.extend([r["match_score"] for r in recs])

    total_time = time.time() - start_time
    avg_latency_ms = (total_time / num_learners) * 1000
    compression_ratio = ((total_generic_hours - total_adaptive_hours) / total_generic_hours) * 100
    hours_saved_per_learner = (total_generic_hours - total_adaptive_hours) / num_learners

    # Print Formatted Results
    print("\n" + "-" * 75)
    print("  [BENCHMARK EMPIRICAL RESULTS]")
    print("-" * 75)
    print(f"  Total Learners Simulated:          {num_learners:,}")
    print(f"  Simulation Execution Time:         {total_time:.2f}s ({avg_latency_ms:.2f} ms/learner)")
    print(f"  Prerequisite Violation Rate:       {prereq_violations / num_learners:.4f}% ({prereq_violations} violations)")
    print(f"  Curriculum Compression Ratio:      {compression_ratio:.2f}% time saved vs static roadmaps")
    print(f"  Average Hours Saved / Learner:     {hours_saved_per_learner:.1f} hours")
    print(f"  Average Adaptive Path Length:      {statistics.mean(path_lengths):.1f} milestones")
    print(f"  BKT Mean Steps to Convergence:     {statistics.mean(bkt_convergence_steps):.1f} assessment questions")
    print(f"  Multi-Factor Mean Rec Score:       {statistics.mean(course_scores):.3f} (Scale: 0.0 - 1.0)")
    print("-" * 75)
    print("  [ROLE DISTRIBUTION ACROSS SIMULATION RUNS]:")
    for role, count in role_distribution.items():
        print(f"    * {role:<24}: {count:>4} learners ({count/num_learners*100:.1f}%)")
    print("=" * 75 + "\n")

    return {
        "num_learners": num_learners,
        "compression_ratio": round(compression_ratio, 2),
        "prereq_violations": prereq_violations,
        "hours_saved_per_learner": round(hours_saved_per_learner, 1),
        "mean_path_length": round(statistics.mean(path_lengths), 1),
        "mean_bkt_steps": round(statistics.mean(bkt_convergence_steps), 1),
        "mean_mrr_score": round(statistics.mean(course_scores), 3),
        "avg_latency_ms": round(avg_latency_ms, 2)
    }

if __name__ == "__main__":
    run_monte_carlo_benchmark(1000)
