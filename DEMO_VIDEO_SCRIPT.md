# 🎬 Career PathFinder — Demo Video Script

**HCL Amplified — AI-Powered Personalized Learning Path Recommender**

> **Target duration:** ~3 min 30 sec
> **Screen:** dual view (presenter webcam + browser recording of the deployed URL)
> **One-line thesis to land:** *We measure what you know and ground the path in real data — the engine decides, the model only phrases.*

---

## ⏱️ Scene-by-scene

### 0:00 – 0:35 · The problem & our architecture *(Problem Understanding — 20%)*
**Screen:** hero screen of Career PathFinder.
**Script:**
> *"Learning platforms have thousands of courses but no structure. Ask a generic AI for a roadmap and it hallucinates courses that don't exist and ignores prerequisites.*
> *Career PathFinder is a grounded, adaptive learning engine. It combines a prerequisite **skill graph**, an **O*NET** labour-market skill taxonomy, and **Bayesian Knowledge Tracing** into a closed loop. Our rule is simple: the engine decides what and in what order — the language model only phrases it."*

---

### 0:35 – 1:15 · Goal intake & adaptive diagnostic *(Functionality — 25%)*
**Screen:**
1. Click the *Data Scientist* quick-chip (or type: *"Commerce graduate wanting to become a Data Scientist"*).
2. Select `10 hrs/week`, `Interactive`.
3. Click **Analyze Role & Launch Diagnostic**.
4. Answer the diagnostic questions and watch the mastery gauge update — **note it stops early once confident.**

**Script:**
> *"A learner types their goal in plain English; the engine maps it to a role profile. Instead of trusting self-reported skill, we run an **adaptive cognitive diagnostic** powered by Bayesian Knowledge Tracing. Each answer updates the hidden mastery probability — and the diagnostic stops early the moment it's confident, so it doesn't waste the learner's time."*

---

### 1:15 – 2:05 · Prerequisite roadmap & glass-box trace *(AI/ML — 20%)*
**Screen:**
1. Click **Next Assessment Step** to load the dashboard roadmap.
2. Scroll the vertical journey.
3. Click **Inspect Trace** on Step 1 to expand the glass-box accordion.
4. Click **Explore Skill DAG (53 Nodes)** to open the interactive graph explorer.

**Script:**
> *"Here's the generated roadmap. Instead of keyword search, the engine runs **Kahn's topological sort** over a 53-skill directed acyclic graph — so no skill is ever scheduled before its prerequisites.*
> *Every milestone is **glass-box explainable**: Inspect Trace shows the exact rationale — O*NET importance, your measured mastery deficit, the semantic course match, and proof the prerequisites are satisfied. We can even open the full **DAG explorer** to see all 53 skills and 61 prerequisite edges."*

---

### 2:05 – 2:40 · Dynamic re-planning & capstones *(Innovation — 15%)*
**Screen:**
1. Close the graph modal.
2. Click **Need Remediation** on an active milestone — watch readiness recalculate and the chain re-sequence.
3. Highlight the **Capstone Project** box with concrete deliverables.
4. Click **Export Roadmap** to download the syllabus as markdown.

**Script:**
> *"Learning isn't static. If a learner struggles and flags remediation, the closed-loop engine drops the mastery estimate and instantly re-plans — scheduling reinforcement before advanced gateways. Every milestone ships a hands-on **capstone project** with testable deliverables, and the whole syllabus exports to markdown in one click."*

---

### 2:40 – 3:15 · Grounded coach & empirical rigor *(UX + Code Quality — 20%)*
**Screen:**
1. In the AI coach, ask: *"Why is statistics scheduled before machine learning?"*
2. Show the grounded answer.
3. Cut to the terminal: `python -m pytest tests/ -v` (**54 passed**) and the benchmark output.

**Script:**
> *"The AI coach is fenced to the active session — grounded mentorship, no hallucinating outside the curriculum. And on rigor: **54 out of 54 automated tests pass** and **20 out of 20 live integration tests pass**, and our Monte Carlo simulation of **1,000** synthetic learners shows a **0.00% prerequisite-violation rate** at sub-millisecond latency per learner."*

---

### 3:15 – 3:30 · Close
**Screen:** dashboard with the readiness ring; deployed URL visible.
**Script:**
> *"Grounded, adaptive, explainable — running live, fully tested, and honestly documented. Thank you."*

---

## 🏁 Pre-recording checklist
- [ ] App reachable at the **deployed URL** (warm it first — open it once so there's no cold-start stall).
- [ ] Browser at ~100% zoom, dark theme.
- [ ] Coach question ready to paste.
- [ ] Terminal tab open at `python -m pytest tests/ -v` (**54 passed**) and `python benchmarks/run_benchmark.py`.
- [ ] Foreground the **engine** (diagnostic, DAG, trace, re-plan) — the 3D scene is ambient, not the star.
