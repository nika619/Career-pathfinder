# 🧭 Career PathFinder

> **AI-Powered Grounded, Adaptive & Explainable Learning Path Recommender**
> *Built for the HCL Amplified National Hackathon*

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-54%2F54%20Passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-gold.svg)](LICENSE)

---

## 🎯 The Problem

Online platforms offer thousands of disconnected courses. Standard recommenders suggest isolated items but never answer the real question: **"In what exact sequence must I learn these skills to reach my career goal without hitting prerequisite walls?"**

Naive LLM roadmappers make it worse — they **hallucinate** course titles that don't exist, recommend advanced topics before fundamentals, and can't justify why a step is scheduled.

**Career PathFinder** is built on one uncompromised principle:

> **The engine decides; the language model only phrases.**

Recommendations are grounded in real data (an authoritative skill taxonomy + a real course corpus + measured mastery). The LLM is fenced strictly to parsing goals, phrasing reasoning traces, and answering questions — it never invents the curriculum.

---

## 🏛️ System Architecture

```
                          [ LEARNER INTAKE ]
              Natural-language goal + time commitment
                                │
                                ▼
                     [ GOAL PARSER / CLASSIFIER ]        (Fenced Gemini + heuristic fallback)
                                │
                                ▼
                     [ O*NET-GROUNDED SKILL PROFILE ]    Role → weighted skills
                                │
                                ▼
                  [ ADAPTIVE DIAGNOSTIC (BKT) ]          Measured mastery, stops early once confident
                                │
                                ▼
                       [ SKILL-GAP ENGINE ]              Priority = Importance × (1 − Mastery)
                                │
                                ▼
                 [ PREREQUISITE DAG TRAVERSAL ]          Induced subgraph + Kahn's topological sort
                                │
                                ▼
             [ MULTI-FACTOR COURSE RECOMMENDER ]         Semantic embeddings + level + rating + format
                                │
                                ▼
                [ HANDS-ON CAPSTONE PROJECT BANK ]       Outcome-based project per milestone
                                │
                                ▼
                  [ GLASS-BOX REASONING TRACE ]          Verifiable rationale per node
                                │
                                ▼
                  [ DYNAMIC ADAPTIVE RE-PLANNER ]        Recomputes on milestone pass/fail
```

---

## 🧠 AI / ML Components

| Component | Foundation | File |
|---|---|---|
| **Prerequisite Skill Graph** | Directed Acyclic Graph + induced-subgraph **Kahn's topological sort** with pedagogical tie-breaking `(difficulty, −importance, id)` | `backend/app/skill_graph.py` |
| **Cognitive Mastery Estimation** | **Bayesian Knowledge Tracing** (Corbett–Anderson HMM) with calibrated slip/guess/transit priors; **adaptive** early-stop + difficulty-targeted item selection | `backend/app/diagnostic.py` |
| **Labour-Market Skill Gap** | O*NET-derived importance weights `[0,1]` × deficit: `Priority = Imp × (1 − Mastery)` | `backend/app/gap.py` |
| **Semantic Course Retrieval** | **Dense embeddings** (`all-MiniLM-L6-v2`, precomputed offline) + cosine, fused with level/rating/format into a multi-factor score. **TF-IDF lexical fallback** when embeddings are unavailable | `backend/app/retrieval.py` |
| **Fenced Conversational Layer** | Gemini 1.5 Flash restricted to parsing, phrasing traces, and grounded Q&A. **Deterministic fallback if no API key** | `backend/app/assistant.py` |

> **Retrieval note:** embeddings are precomputed offline (`backend/scripts/build_embeddings.py`) and shipped as small `.npy` files. The **runtime is numpy-only** — no torch, no API needed for retrieval — so it stays deploy-light while being genuinely semantic.

---

## 📊 By the numbers

| | |
|---|---|
| Skill graph | **53 nodes**, **61 prerequisite edges** (strict DAG) |
| Career roles | **6** (Data Scientist, Data Analyst, ML Engineer, Software Developer, Full Stack Developer, UX Designer) + open-world synthesis for arbitrary roles |
| Course corpus | **105** curated real courses (Coursera, Udemy, edX, and more) with URLs, levels, ratings, skill tags |
| Diagnostic bank | **139** questions across all 53 skills |
| REST API | **13** endpoints |
| Tests | **54 / 54 passing** |
| Live System Tests | **20 / 20 passing** |
| Benchmark | **0.00%** prerequisite-violation rate, **~0.8 ms/learner**, across **1,000** simulated learners |

---

## ♿ Accessibility & ⚡ Performance

**Accessibility.** Skip-to-content link, semantic landmarks, ARIA labels on icon controls, `aria-live` announcements on the coach and diagnostic feedback, dialog semantics (`role="dialog"`, `aria-modal`, Escape-to-close with focus restore) on the modals, always-visible keyboard focus rings, and full `prefers-reduced-motion` support — the ambient 3D scene renders a single static frame for reduced-motion users.

**Performance.** Responses are GZip-compressed; static assets carry cache headers for instant repeat loads; the 3D background caps its pixel ratio and idles when the tab is hidden; recommendation results are memoised so re-planning is instant. Deterministic endpoints respond in well under 100 ms.

**Resilience.** Every network call is guarded — failures surface as non-blocking toasts rather than a dead screen, and roadmap compilation shows a loading state.

## 🚀 Quickstart (local)

```bash
git clone https://github.com/<your-username>/career-pathfinder.git
cd career-pathfinder

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional — enables the Gemini-powered coach/explanations.
# The app runs fully without it (deterministic fallbacks).
cp .env.example .env         # then paste your GEMINI_API_KEY

python run.py                # serves the app at http://localhost:8000
```

Run the tests:

```bash
python -m pytest tests/ -v   # 54 passed
python backend/scripts/full_system_test.py # 20 / 20 passed
```

Run the Monte Carlo benchmark:

```bash
python benchmarks/run_benchmark.py
```

Regenerate embeddings (only if you change the courses/skills):

```bash
pip install -r backend/requirements-dev.txt
python backend/scripts/build_embeddings.py
```

---

## ☁️ Deploy (Render)

The app is a single service that serves both the API and the dashboard.

1. Push the repo to GitHub and create a **Render → Web Service** from it.
2. **Build command:** `pip install -r requirements.txt`
3. **Start command:** `python run.py`  *(binds `0.0.0.0` and honours `$PORT`)*
4. Add env var `GEMINI_API_KEY` (optional but recommended).
5. Open the service URL — the dashboard loads at `/`, health at `/api/health`.

*(Docker: `docker compose up --build` runs the same thing locally on port 8000.)*

---

## 📂 Repository structure

```
career-pathfinder/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI REST endpoints + static serving + security middleware
│   │   ├── skill_graph.py     # DAG prerequisite engine + topological pathing
│   │   ├── gap.py             # O*NET-weighted skill-gap engine
│   │   ├── diagnostic.py      # Adaptive Bayesian Knowledge Tracing
│   │   ├── retrieval.py       # Semantic + TF-IDF multi-factor recommender
│   │   ├── planner.py         # Full journey compiler + glass-box traces
│   │   ├── assistant.py       # Fenced Gemini layer (+ deterministic fallback)
│   │   └── store.py           # Dual-layer in-memory + SQLite (WAL) session store
│   ├── data/
│   │   ├── skill_graph.json   # Prerequisite DAG (53 nodes, 61 edges)
│   │   ├── role_skills.json   # O*NET-derived weights for 6 roles
│   │   ├── courses.json       # 105 curated real courses
│   │   ├── item_bank.json     # 139 diagnostic questions
│   │   ├── course_emb.npy     # Precomputed course embeddings (shipped)
│   │   ├── skill_emb.npy      # Precomputed skill embeddings (shipped)
│   │   └── emb_index.json     # Embedding model + skill-order index
│   ├── scripts/
│   │   ├── full_system_test.py    # Live integration test suite (20 tests)
│   │   ├── build_role_skills.py   # (dev) shape O*NET files
│   │   └── build_embeddings.py    # (dev) regenerate embeddings
│   ├── static/                # Dark editorial dashboard (HTML/CSS/JS + Three.js + SVG DAG)
├── tests/
│   ├── test_core.py           # Graph, gap, BKT, retrieval, e2e, persona benchmarks
│   ├── test_coverage.py       # Item-bank coverage across all roles
│   ├── test_integrity.py      # DAG/data integrity, math invariants, API contracts, persistence
│   ├── test_enhancements.py   # Adaptive diagnostic, open-world gap, semantic retrieval
│   └── test_security.py       # OWASP headers, rate limiting, sanitization, thread safety
├── benchmarks/
│   └── run_benchmark.py       # 1,000-learner Monte Carlo simulation
├── Dockerfile · docker-compose.yml · run.py
├── requirements.txt           # Pinned runtime deps (light: no torch, no pandas)
├── .env.example · .gitignore · .dockerignore · LICENSE
├── MASTER_PROJECT_HANDBOOK.md
├── SOLUTION_DOCUMENTATION.md
├── DEMO_VIDEO_SCRIPT.md
└── RESKIN.md
```

---

## 🏆 Scoring alignment

| Criterion | Weight | How Career PathFinder addresses it |
|---|---|---|
| **Functionality & Feature Completeness** | 25% | Complete closed loop across 6 roles + open-world synthesis: intake → adaptive BKT diagnostic → O*NET gap → topological roadmap → glass-box trace → dynamic re-plan → grounded coach. |
| **Problem Understanding & Design** | 20% | Targets the root cause — **sequence** and **grounding** — not "suggest a course." Engine-decides / LLM-phrases architecture. |
| **AI/ML Implementation** | 20% | Kahn's topological sort, Bayesian Knowledge Tracing (adaptive), **semantic embedding retrieval** with TF-IDF fallback, fenced LLM. |
| **Innovation & Creativity** | 15% | "We measure, we don't assume" (BKT) + "grounded, not generated" (O*NET + real corpus) + glass-box trace + outcome-based capstones + interactive DAG explorer. |
| **User Experience & Interface** | 10% | Restrained dark editorial UI, path-journey visualization, live coach; reduced-motion + keyboard focus + mobile support. |
| **Performance & Code Quality** | 10% | **54/54 tests**, **20/20 live integration tests**, clean modular design, pinned light dependencies, sub-100 ms deterministic endpoints, Monte Carlo–validated (0% violations). |

---

*Grounded, adaptive, explainable — running live, tested, and honestly documented.*
