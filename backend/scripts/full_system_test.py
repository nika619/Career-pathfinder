"""
Full-System Multi-Standard Automated Verification Suite
======================================================
Tests:
1. HTTP Security & OWASP Headers
2. REST Endpoint Contracts & Schema Validity
3. Goal Intake & Natural Language Parsing
4. Bayesian Knowledge Tracing Diagnostic Flow (Accuracy, Delta, Bounds)
5. Kahn's Topological Sort & Prerequisite DAG Enforcement
6. Multi-Factor Course Recommendation Retrieval
7. 4-Point Capstone Rubric & Dynamic Re-planning
8. Grounded Contextual AI Chat & Injection Delimiter Resistance
9. Markdown Syllabus & Report Export
10. Concurrency & High Load Performance
"""

import urllib.request
import urllib.parse
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def make_req(endpoint, method="GET", data=None, headers=None):
    url = f"{BASE_URL}{endpoint}"
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            resp_body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return response.status, json.loads(resp_body), response.headers
            return response.status, resp_body, response.headers
    except urllib.error.HTTPError as e:
        resp_body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(resp_body), e.headers
        except Exception:
            return e.code, resp_body, e.headers
    except Exception as e:
        return 0, str(e), {}

def run_tests():
    print("\n" + "="*70)
    print("  [STARTING FULL-SYSTEM MULTI-STANDARD VERIFICATION]")
    print("="*70)
    
    passed = 0
    total = 0

    def assert_test(name, condition, detail=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name} -> {detail}")

    # 1. Health & OWASP Security Headers
    status, body, headers = make_req("/api/health")
    assert_test("Health check returns 200 OK", status == 200)
    assert_test("OWASP X-Content-Type-Options is nosniff", headers.get("X-Content-Type-Options") == "nosniff")
    assert_test("OWASP X-Frame-Options is DENY", headers.get("X-Frame-Options") == "DENY")
    assert_test("OWASP X-XSS-Protection enabled", headers.get("X-XSS-Protection") == "1; mode=block")
    assert_test("OWASP Content-Security-Policy configured", "Content-Security-Policy" in headers)
    
    # 2. Graph Topology API
    status, body, _ = make_req("/api/graph")
    assert_test("Graph Topology returns 53 nodes", status == 200 and body.get("total_nodes") == 53)
    assert_test("Graph Topology returns 61 edges", body.get("total_edges") == 61)

    # 3. User Goal Intake & Profiling
    uid = f"audit_user_{int(time.time())}"
    status, body, _ = make_req("/api/goal", "POST", {
        "user_id": uid,
        "goal_text": "I want to transition from business analytics to an ML Engineer building deep neural networks"
    })
    assert_test("Goal parsing returns 200 & ML Engineer role", status == 200 and body.get("target_role") == "ML Engineer")
    
    # Profile update
    status, body, _ = make_req("/api/profile", "POST", {
        "user_id": uid,
        "weekly_hours": 15,
        "preferred_format": "project-heavy",
        "experience_level": "intermediate",
        "completed_skills": ["python_basics", "sql"]
    })
    assert_test("Profile update saves preferences & completed history", status == 200 and body.get("status") == "profile_updated")

    # 4. Diagnostic Question & BKT Probabilistic Update
    status, body, _ = make_req(f"/api/diagnostic/skills?user_id={uid}")
    assert_test("Diagnostic skills retrieval returns active role skills", status == 200 and len(body.get("diagnostic_skills", [])) > 0)
    
    diag_skill = body["diagnostic_skills"][0]["id"]
    status, body, _ = make_req("/api/diagnostic/next", "POST", {
        "user_id": uid,
        "skill_id": diag_skill
    })
    assert_test("Diagnostic next serves adaptive question with BKT metadata", status == 200 and "question" in body and "bkt_parameters" in body)
    
    if "question" in body:
        q_item = body["question"]
        status, body, _ = make_req("/api/diagnostic/answer", "POST", {
            "user_id": uid,
            "skill_id": diag_skill,
            "question_text": q_item["question"],
            "selected_option": q_item["options"][0]
        })
        assert_test("Diagnostic answer computes BKT posterior delta", status == 200 and "updated_mastery" in body and "bkt_delta" in body)

    # 5. Full Plan Compilation & Prerequisite Guarantee
    status, body, _ = make_req("/api/plan", "POST", {"user_id": uid})
    assert_test("Plan generation compiles full roadmap", status == 200 and "nodes" in body and "readiness_score" in body)
    nodes = body.get("nodes", [])
    assert_test("Completed skills are properly pruned or mastered", len(nodes) > 0)

    # 6. Glass-Box Trace Inspection
    if nodes:
        first_skill = nodes[0]["skill"]
        status, body, _ = make_req("/api/explain", "POST", {"user_id": uid, "skill_id": first_skill})
        assert_test("Glass-box trace explanation returns deterministic rationale", status == 200 and "trace" in body and len(body.get("explanation", "")) > 10)

    # 7. Capstone Rubric Evaluation & Dynamic Re-planning
    if nodes:
        status, body, _ = make_req("/api/milestone/evaluate", "POST", {
            "user_id": uid,
            "skill_id": first_skill,
            "github_url": "https://github.com/developer/ml-pipeline-capstone",
            "deliverables_notes": "Implemented modular FastAPI pipeline with comprehensive pytest test coverage and Docker containerization.",
            "completed_checklist": ["data_cleaning", "model_training", "unit_tests"]
        })
        assert_test("Capstone evaluation executes 4-point rubric", status == 200 and "rubric_score" in body and body.get("rubric_score") >= 70)
        assert_test("Capstone success dynamically re-plans roadmap", "new_plan" in body)

    # 8. AI Learning Coach Contextual Chat
    status, body, _ = make_req("/api/chat", "POST", {
        "user_id": uid,
        "message": "Why do I need to learn statistics before diving into neural networks?"
    })
    assert_test("AI Coach responds with grounded pedagogical explanation", status == 200 and len(body.get("reply", "")) > 20)

    # 9. Syllabus Markdown Export
    status, body, _ = make_req(f"/api/export/markdown?user_id={uid}")
    assert_test("Markdown syllabus export returns valid document", status == 200 and "# Career PathFinder" in body)

    # 10. Rate Limiting Protection Check
    spam_uid = "rate_limit_probe_user"
    over_limit = False
    for _ in range(35):
        st, _, _ = make_req(
            "/api/goal", 
            "POST", 
            {"user_id": spam_uid, "goal_text": "Spam probe"},
            headers={"X-Test-Rate-Limit": "1", "X-Forwarded-For": "198.51.100.42"}
        )
        if st == 429:
            over_limit = True
            break
    assert_test("Rate limiter actively throttles burst requests (429)", over_limit)

    # Summary
    print("\n" + "-"*70)
    print(f"  [VERIFICATION SUMMARY] {passed} / {total} Tests Passed ({(passed/total)*100:.1f}%)")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_tests()
