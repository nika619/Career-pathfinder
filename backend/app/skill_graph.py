"""
Career PathFinder — Skill Graph Engine
=======================================
Loads the prerequisite DAG (directed acyclic graph) of skills and provides
topological pathing operations. This is a deterministic, testable core —
no LLM involved.

The graph encodes which skills must be learned before which others.
A learning path is the induced subgraph of everything the target role
needs minus what the learner already knows, returned in topological order
so prerequisites always come first.
"""

import json
from pathlib import Path
import networkx as nx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load skill graph data at module level (once, at import)
with open(DATA_DIR / "skill_graph.json", "r") as f:
    _data = json.load(f)


def load_graph() -> nx.DiGraph:
    """Build a NetworkX DiGraph from the skill graph JSON.

    Nodes carry metadata (name, difficulty, cluster).
    Edges are directed prerequisite relationships: A→B means A is required before B.

    Raises:
        AssertionError: If the graph contains cycles (would make topological sort impossible).
    """
    g = nx.DiGraph()
    for sid, meta in _data["nodes"].items():
        g.add_node(sid, **meta)
    g.add_edges_from(_data["edges"])
    assert nx.is_directed_acyclic_graph(g), "Skill graph has a cycle — cannot topologically sort!"
    return g


# Module-level graph instance — loaded once, shared across the app
G = load_graph()


def get_all_roles() -> list[str]:
    """Return all supported career roles."""
    return list(_data["role_targets"].keys())


def synthesize_custom_role_dag(role_query: str) -> list[str]:
    """Heuristically map an arbitrary/open-world career query to a set of DAG skills.

    Uses keyword-to-cluster affinity matching over the skill graph to assemble a
    plausible skill set for niche domains (e.g. Cybersecurity, DevOps, Bioinformatics,
    Game Dev). Prerequisite ordering is then handled by the DAG's topological sort.
    """
    lowered = role_query.lower()
    
    # 1. Domain keyword mappings for open-world career expansion
    cluster_scores = {}
    for sid, meta in _data["nodes"].items():
        name = meta.get("name", "").lower()
        desc = meta.get("description", "").lower()
        cluster = meta.get("cluster", "").lower()
        
        score = 0.0
        # Check domain affinities
        if any(w in lowered for w in ["devops", "cloud", "sre", "infrastructure"]) and cluster in ["cloud", "system", "programming"]:
            score += 0.8
        elif any(w in lowered for w in ["security", "cyber", "penetration", "crypto"]) and ("security" in sid or cluster in ["system", "programming"]):
            score += 0.9
        elif any(w in lowered for w in ["bio", "genom", "health", "pharma", "research"]) and cluster in ["data", "statistics", "ml"]:
            score += 0.85
        elif any(w in lowered for w in ["game", "graphics", "unreal", "unity", "3d"]) and cluster in ["programming", "design", "ui_ux"]:
            score += 0.85
        elif any(token in name or token in desc for token in lowered.split()):
            score += 0.6
            
        if score > 0:
            cluster_scores[sid] = score

    # Select top skills ranked by semantic relevance
    sorted_skills = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
    custom_targets = [s for s, _ in sorted_skills[:12]]
    
    # Fallback to general software developer foundation if query is completely novel
    if not custom_targets:
        custom_targets = ["programming_fundamentals", "data_structures", "algorithms", "git_version_control", "databases_backend", "api_design"]
        
    return custom_targets


def target_skills(role: str) -> list[str]:
    """Return the list of concrete skill IDs that a role requires.
    
    Supports both pre-baked role targets and on-the-fly dynamically synthesized custom roles.
    """
    if role in _data["role_targets"]:
        return _data["role_targets"][role]
    # Dynamically synthesize custom DAG skills for open-world role
    return synthesize_custom_role_dag(role)


def ordered_path(targets: list[str], known: list[str], role_skills_importance: dict[str, float] = None) -> list[str]:
    """Compute a prerequisite-respecting learning path with pedagogical tie-breaking.

    Given a set of target skills and the learner's known skills, computes
    the minimal set of skills to learn (targets + all ancestors, minus known)
    and returns them in pedagogical topological order — guaranteeing that no skill appears
    before its prerequisites, with concurrent skills sorted by difficulty ascending.

    Args:
        targets: List of skill IDs the learner needs to acquire.
        known: List of skill IDs the learner has already mastered.
        role_skills_importance: Optional dict of skill_id -> importance score for tie-breaking.

    Returns:
        Topologically sorted list of skill IDs to learn.
    """
    needed = set()
    for t in targets:
        if t in G.nodes:
            needed.add(t)
            needed |= nx.ancestors(G, t)  # all prerequisites, transitively

    # Remove skills the learner already knows
    needed -= set(known)

    if not needed:
        return []

    # Build the induced subgraph
    sub = G.subgraph(needed).copy()
    
    # In-degree based Kahn's algorithm with priority queue (difficulty, -importance)
    in_degree = {n: sub.in_degree(n) for n in sub.nodes}
    
    def sort_key(node):
        diff = _data["nodes"].get(node, {}).get("difficulty", 2)
        imp = (role_skills_importance or {}).get(node, 0.5)
        return (diff, -imp, node)
        
    ready = [n for n, deg in in_degree.items() if deg == 0]
    ready.sort(key=sort_key)
    
    ordered = []
    while ready:
        curr = ready.pop(0)
        ordered.append(curr)
        
        # Check successors
        for succ in sub.successors(curr):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                ready.append(succ)
                ready.sort(key=sort_key)
                
    return ordered


def get_skill_info(skill_id: str) -> dict:
    """Return metadata for a single skill node.

    Args:
        skill_id: The skill identifier.

    Returns:
        Dict with name, difficulty, cluster, description, and the skill_id.
    """
    if skill_id in _data["nodes"]:
        return {"skill_id": skill_id, **_data["nodes"][skill_id]}
    return {"skill_id": skill_id, "name": skill_id, "difficulty": 0, "cluster": "unknown"}


def get_prerequisites(skill_id: str) -> list[str]:
    """Return direct prerequisites of a skill.

    Args:
        skill_id: The skill identifier.

    Returns:
        List of skill IDs that are direct prerequisites.
    """
    if skill_id in G.nodes:
        return list(G.predecessors(skill_id))
    return []


def get_dependents(skill_id: str) -> list[str]:
    """Return skills that directly depend on this skill.

    Args:
        skill_id: The skill identifier.

    Returns:
        List of skill IDs that require this skill as a prerequisite.
    """
    if skill_id in G.nodes:
        return list(G.successors(skill_id))
    return []
