"""
Career PathFinder — Offline Embedding Builder (DEV / BUILD-TIME ONLY)
=====================================================================
Precomputes dense semantic embeddings for the course corpus AND for every
skill node in the DAG, using a compact sentence-transformer. The resulting
vectors are saved to backend/data/ and shipped with the repo.

WHY OFFLINE: this keeps the *runtime* dependency-light. Production only needs
numpy to do the cosine lookup against these precomputed vectors — no torch,
no sentence-transformers, no API key. Retrieval is therefore genuinely
semantic yet fully deploy-safe.

REGENERATE (only if courses/skills change):
    pip install -r backend/requirements-dev.txt
    python backend/scripts/build_embeddings.py

Model is pinned; it must match retrieval.py's expectations (dim = 384).
"""

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"          # 384-dim, compact, CPU-friendly
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _norm(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a plain dot product equals cosine similarity."""
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    # ---- 1. Course corpus embeddings (row order == courses.json order) ----
    courses = json.load(open(DATA_DIR / "courses.json"))
    course_texts = [
        f"{c['title']}. {c['description']} Skills: {' '.join(c.get('skills', []))}"
        for c in courses
    ]
    course_emb = _norm(np.asarray(model.encode(course_texts), dtype="float32"))
    np.save(DATA_DIR / "course_emb.npy", course_emb)

    # ---- 2. Per-skill query embeddings (every DAG node) ----
    graph = json.load(open(DATA_DIR / "skill_graph.json"))
    skill_ids = list(graph["nodes"].keys())
    skill_texts = [
        f"{meta.get('name', sid)}. {meta.get('description', '')}"
        for sid, meta in graph["nodes"].items()
    ]
    skill_emb = _norm(np.asarray(model.encode(skill_texts), dtype="float32"))
    np.save(DATA_DIR / "skill_emb.npy", skill_emb)

    # ---- 3. Index so runtime can map skill_id -> row ----
    json.dump(
        {"model": MODEL_NAME, "dim": int(course_emb.shape[1]), "skill_order": skill_ids},
        open(DATA_DIR / "emb_index.json", "w"),
        indent=2,
    )

    print(f"Saved course_emb {course_emb.shape}, skill_emb {skill_emb.shape}, "
          f"index of {len(skill_ids)} skills (model={MODEL_NAME}).")


if __name__ == "__main__":
    main()
