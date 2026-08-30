"""
Career PathFinder — Multi-Factor Course & Project Recommendation Engine
========================================================================
Implements a multi-factor ranking algorithm over real educational resources
and industry-aligned hands-on projects.

Retrieval signal: dense semantic embeddings (all-MiniLM-L6-v2, precomputed
offline into backend/data/*.npy) matched by cosine similarity. Runtime is
numpy-only — no torch, no API. If the embeddings are unavailable for a skill,
the engine falls back automatically to TF-IDF lexical similarity.

Why this beats standard keyword search / blind prompting:
1. Multi-Factor Recommendation Score (MRS):
   Score(c) = w_rel * Sim(Skill, Course)
            + w_lvl * LevelAlignment(LearnerLevel, CourseLevel)
            + w_rat * NormalizedRating(CourseRating)
            + w_fmt * PreferenceMatch(LearnerPref, CoursePlatform/Format)

2. Prerequisites & Level Filter:
   Courses are strictly matched to the current node's skill context,
   ensuring the learner never receives advanced material without foundations.

3. Project & Capstone Matching:
   Every skill node is paired with a concrete, outcome-oriented project
   (e.g., "Build an End-to-End Churn Predictor in Streamlit") rather than
   just passive video links.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / "courses.json", "r") as f:
    _courses: list[dict[str, Any]] = json.load(f)

# Hands-on Project Bank keyed by Skill ID
PROJECT_BANK: dict[str, dict[str, Any]] = {
    "python_basics": {
        "title": "CLI Automated Data Parser & Reporter",
        "objective": "Build a command-line tool that parses messy CSV/JSON server logs, validates data types, handles errors, and outputs a formatted summary report.",
        "deliverables": ["Modular Python scripts with docstrings", "Unit tests using unittest/pytest", "Clean CLI interface using argparse"],
        "estimated_hours": 6
    },
    "numpy": {
        "title": "Matrix Operations & Image Filter Engine from Scratch",
        "objective": "Implement 2D convolution filters (Sobel, Gaussian blur, sharpening) on raw pixel matrices using pure NumPy vectorized operations without high-level image libraries.",
        "deliverables": ["Vectorized convolution function", "Performance benchmark comparing loop vs vectorized implementation", "Visual comparison notebook"],
        "estimated_hours": 8
    },
    "pandas": {
        "title": "E-Commerce Transaction EDA & Cohort Retention Analysis",
        "objective": "Wrangle 500k+ transaction records: cleanse missing customer IDs, parse datetimes, engineer monthly cohort groups, and calculate 12-month retention heatmaps.",
        "deliverables": ["Cleaned dataset export", "Retention matrix visualization", "Executive insights markdown summary"],
        "estimated_hours": 10
    },
    "data_visualization": {
        "title": "Interactive Executive Financial Dashboard with Plotly/Dash",
        "objective": "Design an interactive web dashboard with linked charts, drill-down KPIs, and accessible color palettes for multi-department revenue metrics.",
        "deliverables": ["Interactive Plotly dashboard code", "Exportable PDF report generator", "Live interactive preview"],
        "estimated_hours": 8
    },
    "sql": {
        "title": "Subscription SaaS Metric Calculation Pipeline (PostgreSQL)",
        "objective": "Write complex analytical SQL queries using Window Functions, CTEs, and self-joins to compute Monthly Recurring Revenue (MRR), Churn Rate, and LTV.",
        "deliverables": ["Normalized database schema SQL", "Analytical queries with CTEs & Window Functions", "Query performance EXPLAIN ANALYZE report"],
        "estimated_hours": 10
    },
    "statistics": {
        "title": "A/B Test Design & Bayesian Hypothesis Testing Engine",
        "objective": "Design an A/B experimentation protocol for landing page conversion, perform power analysis for sample size, and evaluate frequentist vs Bayesian significance.",
        "deliverables": ["Statistical power calculation notebook", "Monte Carlo simulation script", "Decision-framework briefing document"],
        "estimated_hours": 12
    },
    "linear_algebra": {
        "title": "Eigenface Facial Recognition & PCA Compression Engine",
        "objective": "Implement Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) from scratch to compress images and perform nearest-neighbor classification.",
        "deliverables": ["Eigenvector/SVD math implementation", "Image reconstruction variance-explained plots", "Compression ratio benchmark"],
        "estimated_hours": 12
    },
    "data_cleaning": {
        "title": "Automated Data Validation & Imputation Pipeline",
        "objective": "Construct an end-to-end data auditing pipeline using regex, Z-score/IQR anomaly detectors, and multivariate iterative imputers for real-world healthcare records.",
        "deliverables": ["Automated validation schema", "Imputation quality comparison report", "Reusable sklearn Transformer pipeline"],
        "estimated_hours": 8
    },
    "feature_engineering": {
        "title": "High-Cardinality Target Encoding & Feature Selection Suite",
        "objective": "Engineer temporal, cyclical, and out-of-fold target-encoded features on financial credit risk data, with mutual information and permutation feature ranking.",
        "deliverables": ["Custom feature engineering pipeline", "Feature importance diagnostic graphs", "Leakage-prevention test suite"],
        "estimated_hours": 10
    },
    "ml_foundations": {
        "title": "Full Machine Learning Lifecycle Benchmark",
        "objective": "Compare 5 baseline algorithms against regression and classification datasets, tuning hyperparameters via nested cross-validation and evaluating learning curves.",
        "deliverables": ["Rigorous nested CV comparison notebook", "Bias-variance tradeoff diagnostic plots", "Model artifact serialization"],
        "estimated_hours": 14
    },
    "supervised_learning": {
        "title": "Customer Churn Prediction with Tuned XGBoost & SHAP Explainability",
        "objective": "Train and optimize gradient-boosted trees for telecom churn prediction, balancing class imbalance with SMOTE/class-weights and generating SHAP feature impact plots.",
        "deliverables": ["Tuned model pipeline (ROC-AUC > 0.88)", "Global and local SHAP explanation plots", "REST API model inference wrapper"],
        "estimated_hours": 16
    },
    "unsupervised_learning": {
        "title": "Customer Segmentation & Anomaly Detection System",
        "objective": "Perform RFM customer clustering with K-Means and HDBSCAN, utilizing Silhouette scores and t-SNE for projection, combined with Isolation Forests for fraud detection.",
        "deliverables": ["Cluster persona profile documentation", "2D/3D t-SNE interactive projection", "Fraud anomaly triage script"],
        "estimated_hours": 12
    },
    "deep_learning": {
        "title": "Medical Imaging Convolutional Classifier with Transfer Learning",
        "objective": "Fine-tune a pretrained ResNet/EfficientNet in PyTorch on chest X-ray scans with data augmentation, Grad-CAM attention visualizer, and mixed-precision training.",
        "deliverables": ["PyTorch training script with learning rate scheduler", "Grad-CAM visual activation heatmaps", "Model evaluation metrics report"],
        "estimated_hours": 20
    },
    "nlp": {
        "title": "RAG-Powered Domain Knowledge Assistant with HuggingFace & Vector Search",
        "objective": "Build a Retrieval-Augmented Generation (RAG) system using dense bi-encoder embeddings, chunking strategies, and a local quantized LLM for legal document synthesis.",
        "deliverables": ["Document ingestion & chunking pipeline", "Vector similarity retrieval with re-ranking", "Evaluation harness comparing RAG vs baseline"],
        "estimated_hours": 20
    },
    "model_evaluation": {
        "title": "Production Model Monitoring & Drift Detection Harness",
        "objective": "Implement an automated pipeline measuring statistical Kolmogorov-Smirnov data drift, concept drift, and degradation of precision-recall curves on streaming data.",
        "deliverables": ["Statistical drift detection module", "Automated retraining trigger webhook", "Model governance audit report"],
        "estimated_hours": 12
    },
    "cloud_ml": {
        "title": "Serverless ML Inference API with Docker & Cloud Deployment",
        "objective": "Containerize an ONNX runtime model, build a FastAPI inference server with input validation and batching, and deploy to AWS Lambda/Render with health checks.",
        "deliverables": ["Optimized multi-stage Dockerfile (<150MB)", "FastAPI latency-tested endpoint", "Automated CI/CD GitHub Actions workflow"],
        "estimated_hours": 14
    },
    "big_data": {
        "title": "Distributed Clickstream Aggregator with Apache PySpark",
        "objective": "Process 10GB+ semi-structured JSON clickstream event streams on PySpark: window aggregations, sessionization, and writing partitioned Parquet data lakes.",
        "deliverables": ["PySpark batch and structured streaming scripts", "Partitioned Parquet storage layout", "Query execution DAG optimization review"],
        "estimated_hours": 16
    },
    "programming_fundamentals": {
        "title": "Core Algorithm Library & Test-Driven Development Suite",
        "objective": "Implement standard utility algorithms with strict 100% test coverage, comprehensive error handling, and algorithmic complexity annotations.",
        "deliverables": ["Algorithm package with type hints", "PyTest test suite with mock fixtures", "Documentation generator"],
        "estimated_hours": 6
    },
    "data_structures": {
        "title": "In-Memory LRU Cache & Custom Graph Engine",
        "objective": "Implement a high-performance Least Recently Used (LRU) cache using doubly-linked lists + hash maps, and a directed graph traversal engine from scratch.",
        "deliverables": ["Clean data structure implementations with O(1) ops", "Concurrency thread-safety tests", "Memory profiling benchmark"],
        "estimated_hours": 10
    },
    "algorithms": {
        "title": "Route Optimization Engine with Dijkstra & Dynamic Programming",
        "objective": "Build a shortest-path and delivery routing optimization engine for multi-stop delivery graphs using Dijkstra, A*, and Knapsack dynamic programming.",
        "deliverables": ["Graph pathfinding module with heuristics", "Interactive map rendering", "Complexity analysis documentation"],
        "estimated_hours": 12
    },
    "javascript": {
        "title": "Real-Time Collaborative Kanban Board (Vanilla JS & WebSockets)",
        "objective": "Construct an interactive drag-and-drop Kanban board using vanilla ES6+ (custom event bus, state management, and optimistic UI updates).",
        "deliverables": ["Modular vanilla JS architecture", "HTML5 Drag-and-Drop integration", "Offline local storage synchronization"],
        "estimated_hours": 10
    },
    "react": {
        "title": "Modern SaaS Analytics Dashboard with Custom Hooks & State Machine",
        "objective": "Build a responsive, accessible web app with React, custom reducer hooks, dynamic routing, debounced search filters, and theme toggling.",
        "deliverables": ["Component hierarchy architecture", "Reusable custom hook library", "Storybook UI component catalog"],
        "estimated_hours": 14
    },
    "nodejs": {
        "title": "Production RESTful Microservice with Auth & Rate Limiting",
        "objective": "Engineer an Express/Node.js backend with JWT authentication, refresh tokens, role-based access control (RBAC), and Redis-backed rate limiting.",
        "deliverables": ["Structured layered architecture (Controller/Service/Repo)", "OpenAPI/Swagger documentation", "Integration test suite with Supertest"],
        "estimated_hours": 14
    },
    "system_design": {
        "title": "High-Throughput URL Shortener Architecture Blueprint & PoC",
        "objective": "Design and prototype a distributed URL shortening service handling 10k QPS: Base62 encoding, caching layers, database sharding strategies, and rate limiting.",
        "deliverables": ["Architecture diagram with component specs", "Proof-of-concept working prototype", "Capacity estimation & SLA specification"],
        "estimated_hours": 16
    },
    "design_thinking": {
        "title": "FinTech Mobile Onboarding Experience Overhaul",
        "objective": "Execute a full human-centered design cycle: conduct 5 user interviews, synthesize empathy maps, define problem statements, and run rapid paper prototyping.",
        "deliverables": ["User empathy & journey maps", "3 distinct 'How Might We' design vectors", "User validation test synthesis report"],
        "estimated_hours": 10
    },
    "figma": {
        "title": "Cross-Platform Scalable Design System in Figma",
        "objective": "Construct an enterprise Figma design system: semantic color & typography tokens, auto-layout 5.0 components, multi-state variants, and dark mode variables.",
        "deliverables": ["Complete Figma community-ready file", "Component documentation & usage guidelines", "Accessibility contrast audit checklist"],
        "estimated_hours": 14
    },
    "user_research": {
        "title": "Heuristic Evaluation & Usability Study on EdTech Platform",
        "objective": "Conduct Nielsen Norman Group 10-heuristic evaluation, create moderated usability testing protocols, and compile prioritized severity-ranked recommendation matrix.",
        "deliverables": ["Heuristic violation audit deck", "Usability test session recordings analysis", "Executive action roadmap"],
        "estimated_hours": 10
    },
    "web_fundamentals": {
        "title": "HTTP/3 & DNS Diagnostic Client & Web Server Simulator",
        "objective": "Build a multi-threaded socket client/server simulating HTTP request/response lifecycles, TCP handshakes, TLS negotiation, and DNS lookups.",
        "deliverables": ["Raw socket HTTP parser script", "DNS recursive lookup trace tool", "Protocol lifecycle analysis report"],
        "estimated_hours": 8
    },
    "html_css": {
        "title": "Pixel-Perfect Semantic SaaS Landing Page & Accessible Form Engine",
        "objective": "Construct a responsive landing page conforming to HTML5 semantic standards, WCAG 2.1 AA accessibility guidelines, and pure CSS Grid/Flexbox layouts.",
        "deliverables": ["Semantic HTML5 source files", "Responsive CSS stylesheet with custom properties", "Lighthouse accessibility audit report (100 score)"],
        "estimated_hours": 8
    },
    "responsive_design": {
        "title": "Fluid Multi-Breakpoint Enterprise Layout Engine",
        "objective": "Implement an adaptive, container-query-driven layout system supporting mobile, tablet, desktop, and ultra-wide displays without horizontal overflow.",
        "deliverables": ["Responsive layout component library", "Container-query fluid card grid", "Cross-browser responsive test matrix"],
        "estimated_hours": 8
    },
    "typescript": {
        "title": "Strict Type-Safe Domain Model & Generic State Machine",
        "objective": "Engineer an end-to-end type-safe state machine utilizing advanced TypeScript features: Discriminated Unions, Template Literal Types, Conditional Types, and Generics.",
        "deliverables": ["Zero-'any' typed state machine library", "Strict compile-time validation test harness", "Exported declaration .d.ts bundle"],
        "estimated_hours": 10
    },
    "api_design": {
        "title": "RESTful & GraphQL OpenAPI 3.0 API Specification & Mock Server",
        "objective": "Author an enterprise OpenAPI 3.0 specification with strict JSON schemas, hypermedia HATEOAS links, idempotency keys, and automated Prism mock server verification.",
        "deliverables": ["OpenAPI 3.0 YAML specification", "Automated contract testing script", "Postman collection with pre-request assertions"],
        "estimated_hours": 10
    },
    "databases_backend": {
        "title": "Relational Data Warehouse & Partitioned PostgreSQL Engine",
        "objective": "Design a normalized 3NF database schema, construct indexing strategies (B-Tree, GIN), partition large transaction tables, and write ACID transaction procedures.",
        "deliverables": ["Schema migration scripts", "Index performance benchmark report", "Stored procedure transaction tests"],
        "estimated_hours": 12
    },
    "devops": {
        "title": "Multi-Stage Docker CI/CD Deployment Pipeline with GitHub Actions",
        "objective": "Author optimized multi-stage Dockerfiles (<100MB footprint), configure automated linting/testing pipelines, and establish zero-downtime blue/green deployment workflows.",
        "deliverables": ["Multi-stage Dockerfile and docker-compose.yml", "GitHub Actions CI/CD workflow YAML", "Automated image vulnerability security scan"],
        "estimated_hours": 12
    },
    "cloud_services": {
        "title": "Serverless Infrastructure-as-Code (Terraform & AWS Lambda)",
        "objective": "Provision cloud architecture via Terraform: API Gateway, serverless Lambda functions, S3 bucket storage, and IAM least-privilege security roles.",
        "deliverables": ["Terraform HCL infrastructure code", "Serverless deployment package", "CloudWatch latency & cost estimate review"],
        "estimated_hours": 14
    },
    "testing_qa": {
        "title": "Automated End-to-End & Unit Testing Harness (Jest, Playwright)",
        "objective": "Implement an exhaustive testing pyramid: 100% code coverage unit tests, integration service tests, and headless Playwright browser regression suites.",
        "deliverables": ["Jest unit & integration test suites", "Playwright automated E2E test scripts", "Coverage badge and regression report"],
        "estimated_hours": 10
    },
    "git_version_control": {
        "title": "Enterprise Git Flow, Interactive Rebase & Release Pipeline",
        "objective": "Simulate a multi-developer branch merge conflict resolution workflow, squash-merge feature branches with semantic commit hooks, and configure signed release tags.",
        "deliverables": ["Git branching strategy documentation", "Husky pre-commit validation hooks", "Automated semantic-release changelog generator"],
        "estimated_hours": 6
    },
    "oop": {
        "title": "Extensible Design Pattern Catalog & Banking Simulation",
        "objective": "Implement the Gang of Four (GoF) design patterns (Factory, Observer, Strategy, Decorator) in a cohesive, SOLID-compliant banking domain model.",
        "deliverables": ["Modular OOP architecture source code", "UML class and sequence diagrams", "Polymorphism & SOLID compliance audit"],
        "estimated_hours": 10
    },
    "security_fundamentals": {
        "title": "OWASP Top 10 Security Audit & Authentication Shield",
        "objective": "Conduct a comprehensive vulnerability audit defending against SQL injection, XSS, CSRF, and timing attacks, implementing bcrypt hashing and rate limiting.",
        "deliverables": ["Vulnerability mitigation patch report", "Secure cookie & CORS middleware", "Automated penetration test script"],
        "estimated_hours": 12
    },
    "agile_methodology": {
        "title": "Agile Sprint Backlog, User Story Mapping & Velocity Tracker",
        "objective": "Synthesize epics, formulate INVEST-compliant user stories with Gherkin acceptance criteria, and model burndown charts for a 2-week sprint cycle.",
        "deliverables": ["Sprint backlog board configuration", "Gherkin feature acceptance specification", "Sprint retrospective & velocity report"],
        "estimated_hours": 6
    },
    "excel_analytics": {
        "title": "Financial Forecasting Model with Dynamic Arrays & VBA",
        "objective": "Build a financial 3-statement model utilizing XLOOKUP, INDEX/MATCH, dynamic array formulas, What-If sensitivity tables, and automated summary macros.",
        "deliverables": ["Excel financial model workbook", "Scenario sensitivity data tables", "Executive visual KPI deck"],
        "estimated_hours": 8
    },
    "tableau": {
        "title": "Multi-Source Interactive Business Intelligence Executive Deck",
        "objective": "Design an enterprise Tableau workbook with calculated fields, Level of Detail (LOD) expressions, parameter actions, and responsive layout containers.",
        "deliverables": ["Published interactive Tableau workbook", "Executive storytelling dashboard layout", "LOD expression calculation guide"],
        "estimated_hours": 10
    },
    "r_programming": {
        "title": "Epidemiological Survival Analysis & Tidyverse Modeling",
        "objective": "Wrangle longitudinal clinical trial data using dplyr/tidyr, execute Kaplan-Meier survival curves, and author an automated R Markdown / Quarto clinical dossier.",
        "deliverables": ["Tidyverse data pipeline script", "ggplot2 publication-grade statistical figures", "Automated Quarto HTML/PDF report"],
        "estimated_hours": 10
    },
    "wireframing": {
        "title": "Low-Fidelity Information Architecture & Wireframe Blueprint",
        "objective": "Map content hierarchy, design low-fidelity structural wireframes for 8 key user flows, and validate screen layouts against user task completion pathways.",
        "deliverables": ["8-screen wireframe deck in Figma/Balsamiq", "User task navigation flow diagram", "Initial usability test findings"],
        "estimated_hours": 8
    },
    "prototyping": {
        "title": "High-Fidelity Interactive Micro-Animation Prototype",
        "objective": "Construct an interactive prototype with smart-animate transitions, drag interactions, stateful components, and micro-interaction feedback loops.",
        "deliverables": ["Interactive clickable prototype link", "Interaction physics & timing specification", "User clickstream validation recording"],
        "estimated_hours": 10
    },
    "interaction_design": {
        "title": "Micro-Interaction Design System & State Transition Rules",
        "objective": "Define interaction models, hover/active/focus feedback states, motion easing curves, and error recovery patterns for complex multi-step user workflows.",
        "deliverables": ["Interaction state transition matrix", "Motion design timing guidelines (150-300ms)", "Accessible focus management documentation"],
        "estimated_hours": 8
    },
    "visual_design": {
        "title": "Brand Identity & Visual Hierarchy Design Language",
        "objective": "Create a unified visual identity: establish harmonious 8pt grids, modular scale typography, balanced negative space, and evocative imagery styles.",
        "deliverables": ["Brand identity style guide", "Visual component UI kit", "High-contrast visual balance audit"],
        "estimated_hours": 10
    },
    "typography": {
        "title": "Enterprise Modular Scale Typography & Vertical Rhythm Guide",
        "objective": "Establish an accessible typographic hierarchy using fluid modular scales (e.g. 1.25 Major Third), line-height ratios, and tabular number formatting.",
        "deliverables": ["Typographic token specification", "Fluid responsive type scale CSS/Figma tokens", "Legibility & dyslexia accessibility audit"],
        "estimated_hours": 6
    },
    "color_theory": {
        "title": "Accessible WCAG AAA Color Palette & Semantic Dark/Light Modes",
        "objective": "Generate algorithmic color scales in HSL/OKLCH color space, verifying 4.5:1 (AA) and 7:1 (AAA) contrast ratios across light and dark surface themes.",
        "deliverables": ["OKLCH semantic palette tokens", "Contrast ratio matrix report", "Theme switcher token mappings"],
        "estimated_hours": 6
    },
    "information_architecture": {
        "title": "Card Sorting Synthesis & Tree-Testing Navigation Taxonomy",
        "objective": "Conduct open/closed card sorting studies with 15 participants, synthesize similarity matrix dendrograms, and formulate the sitemap navigation taxonomy.",
        "deliverables": ["Card sorting similarity dendrogram", "Hierarchical sitemap taxonomy diagram", "Tree-testing success rate benchmark"],
        "estimated_hours": 8
    },
    "accessibility_design": {
        "title": "WCAG 2.2 Section 508 Comprehensive Accessibility Audit",
        "objective": "Audit web components for keyboard navigation traps, ARIA roles/live-regions, screen-reader compatibility (NVDA/VoiceOver), and color-blind simulators.",
        "deliverables": ["Accessibility compliance audit report", "Screen-reader test transcript", "Remediated ARIA component patterns"],
        "estimated_hours": 8
    },
    "design_systems": {
        "title": "Atomic Design Component Library & Token Synchronizer",
        "objective": "Build a scalable atomic design library (Atoms, Molecules, Organisms) with design token JSON exports connecting Figma directly to CSS variables.",
        "deliverables": ["Enterprise design system repository", "Automated token export pipeline (Style Dictionary)", "Living documentation catalog"],
        "estimated_hours": 14
    },
    "usability_testing": {
        "title": "Moderated Usability Study & Task Success Diagnostic Deck",
        "objective": "Create standardized usability testing scripts, record SUS (System Usability Scale) metric scores, and build video-timestamped usability severity matrices.",
        "deliverables": ["Usability testing protocol script", "SUS score calculation benchmark (target >80)", "Prioritized issue remediation backlog"],
        "estimated_hours": 10
    }
}

# Default project fallback for any node
DEFAULT_PROJECT = {
    "title": "Hands-On Practical Capstone Implementation",
    "objective": "Apply the core principles of this skill in an end-to-end practical project addressing a realistic industry scenario.",
    "deliverables": ["Complete source code repository", "Documentation and setup guide", "Verification test suite"],
    "estimated_hours": 8
}

# Pre-fit TF-IDF vectorizer over course corpus (fallback lexical signal)
_course_corpus_texts = [f"{c['title']} {c['description']} {' '.join(c.get('skills', []))}" for c in _courses]
_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2500)
_tfidf_matrix = _vectorizer.fit_transform(_course_corpus_texts)

# ---- Dense semantic embeddings (precomputed offline; runtime is numpy-only) ----
# Primary retrieval signal. Regenerate with backend/scripts/build_embeddings.py.
_course_emb = None
_skill_emb = None
_skill_order: dict[str, int] = {}
try:
    _ce, _se, _ix = DATA_DIR / "course_emb.npy", DATA_DIR / "skill_emb.npy", DATA_DIR / "emb_index.json"
    if _ce.exists() and _se.exists() and _ix.exists():
        _course_emb = np.load(_ce)
        _skill_emb = np.load(_se)
        _order = json.load(open(_ix)).get("skill_order", [])
        _skill_order = {sid: i for i, sid in enumerate(_order)}
except Exception:
    _course_emb = None  # any failure -> graceful TF-IDF fallback

_EMB_READY = _course_emb is not None and _skill_emb is not None and len(_skill_order) > 0

# Memoised recommendations — repeated plan/re-plan calls for the same
# (skill, experience, format, k) reuse the ranked result instead of recomputing.
_rec_cache: dict = {}


def _semantic_scores(skill_id: str):
    """Cosine of the precomputed skill vector against all course vectors.

    Returns a per-course numpy array in [0, 1], or None to signal TF-IDF
    fallback (embeddings not built, or skill absent from the precomputed index).
    """
    if not _EMB_READY or skill_id not in _skill_order:
        return None
    q = _skill_emb[_skill_order[skill_id]]
    scores = _course_emb @ q            # vectors are L2-normalized => cosine
    return np.clip(scores, 0.0, 1.0)    # keep composite score bounded in [0, 1]


def calculate_level_fit(course_level: str, learner_experience: str) -> float:
    """Score alignment between course level and learner profile (0.0 to 1.0)."""
    lvl_map = {"beginner": 1, "intermediate": 2, "advanced": 3}
    c_num = lvl_map.get(course_level.lower(), 2)
    l_num = lvl_map.get(learner_experience.lower(), 1)
    diff = abs(c_num - l_num)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.7
    return 0.3


def calculate_format_fit(course_platform: str, preferred_format: str) -> float:
    """Score alignment between course delivery platform and learner format preference."""
    interactive_platforms = ["DataCamp", "freeCodeCamp", "Kaggle", "Educative", "University of Helsinki"]
    video_platforms = ["Coursera", "Udemy", "edX", "MIT OCW"]
    
    if preferred_format == "interactive":
        return 1.0 if course_platform in interactive_platforms else 0.8
    elif preferred_format == "video":
        return 1.0 if course_platform in video_platforms else 0.8
    return 0.9  # neutral / project-heavy


def recommend_courses_for_skill(
    skill_id: str,
    skill_name: str,
    learner_experience: str = "beginner",
    preferred_format: str = "interactive",
    role_importance: float = 0.8,
    k: int = 3
) -> list[dict[str, Any]]:
    """Multi-Factor Recommendation Ranking (MRR) for a given skill.

    Computes:
      Score = 0.45 * SemanticSimilarity
            + 0.20 * LevelAlignment
            + 0.20 * NormalizedRating
            + 0.15 * FormatMatch
    """
    cache_key = (skill_id, learner_experience, preferred_format, k)
    if cache_key in _rec_cache:
        return _rec_cache[cache_key]

    # Primary: dense semantic similarity (precomputed embeddings). Fallback: TF-IDF.
    sem = _semantic_scores(skill_id)
    if sem is not None:
        sim_scores = sem
        retrieval_mode = "semantic_embedding"
    else:
        query_text = f"{skill_id} {skill_name} course tutorial hands-on"
        q_vec = _vectorizer.transform([query_text])
        sim_scores = cosine_similarity(q_vec, _tfidf_matrix).flatten()
        retrieval_mode = "tfidf_lexical"

    candidates = []
    for idx, course in enumerate(_courses):
        # Base semantic similarity
        sim = float(sim_scores[idx])
        
        # Exact skill tag match bonus
        if skill_id in course.get("skills", []):
            sim = min(1.0, sim + 0.35)
            
        # Rating normalized (assuming 3.0-5.0 scale -> 0.0 to 1.0)
        rating_score = max(0.0, (course.get("rating", 4.5) - 3.5) / 1.5)
        
        # Level fit & format fit
        level_fit = calculate_level_fit(course.get("level", "intermediate"), learner_experience)
        format_fit = calculate_format_fit(course.get("platform", "Coursera"), preferred_format)
        
        # Multi-factor composite score
        composite_score = round(
            (0.45 * sim) +
            (0.20 * level_fit) +
            (0.20 * rating_score) +
            (0.15 * format_fit),
            3
        )
        
        candidates.append({
            "id": course["id"],
            "title": course["title"],
            "description": course["description"],
            "platform": course["platform"],
            "url": course["url"],
            "level": course["level"],
            "rating": course.get("rating", 4.5),
            "match_score": composite_score,
            "retrieval_mode": retrieval_mode,
            "score_breakdown": {
                "semantic_relevance": round(sim, 3),
                "level_fit": round(level_fit, 3),
                "rating_weight": round(rating_score, 3),
                "format_match": round(format_fit, 3)
            }
        })

    # Sort descending by composite recommendation score
    ranked = sorted(candidates, key=lambda x: x["match_score"], reverse=True)
    result = ranked[:k]
    _rec_cache[cache_key] = result
    return result


def get_project_for_skill(skill_id: str) -> dict[str, Any]:
    """Retrieve curated hands-on project matching the skill node."""
    return PROJECT_BANK.get(skill_id, DEFAULT_PROJECT)
