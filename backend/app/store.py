"""
Career PathFinder — Persistent SQLite & Memory-Cached Session Store
===================================================================
Manages state for active learner sessions with dual-layer architecture:
1. In-memory cache for fast API responses
2. SQLite persistent database (`backend/data/sessions.db`) for durable session retention across restarts

Tracks:
- Target career role & cluster
- Preferences (weekly hours, format, experience level, pace)
- Claimed background knowledge & learning history
- Real-time Bayesian Knowledge Tracing (BKT) mastery per skill
- Diagnostic questions served & response logs
- Generated learning path, parallel tracks & milestone completion status
- Interaction history for contextual AI coaching
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sessions.db"

def init_db():
    """Initialize SQLite tables for persistent session state with WAL high-concurrency mode."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
        # Enable Write-Ahead Logging (WAL) for unlimited concurrent readers + non-blocking writers
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                user_id TEXT PRIMARY KEY,
                target_role TEXT,
                preferences_json TEXT,
                learning_history_json TEXT,
                mastery_json TEXT,
                asked_questions_json TEXT,
                milestone_status_json TEXT,
                current_plan_json TEXT,
                chat_history_json TEXT,
                updated_at REAL
            )
        """)
        conn.commit()

init_db()


class LearnerSession:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.created_at = time.time()
        self.target_role: Optional[str] = None
        self.target_cluster: Optional[str] = None
        
        # Learner Profile & Preferences
        self.preferences: dict[str, Any] = {
            "weekly_hours": 10,
            "preferred_format": "interactive",  # 'video', 'interactive', 'project-heavy'
            "experience_level": "beginner",      # 'beginner', 'intermediate', 'advanced'
            "learning_pace": "moderate"          # 'accelerated', 'moderate', 'deep-dive'
        }
        
        # Initial learning history (courses/skills claimed)
        self.learning_history: list[str] = []
        
        # Measured mastery (Skill ID -> float 0.0 to 1.0)
        self.mastery: dict[str, float] = {}
        
        # Diagnostic item history: {skill_id: [question_text, ...]}
        self.asked_questions: dict[str, list[str]] = {}
        
        # Full compiled learning plan
        self.current_plan: Optional[dict[str, Any]] = None
        
        # Completed / Failed milestones
        self.milestone_status: dict[str, str] = {} # {skill_id: 'passed' | 'failed' | 'in_progress'}
        
        # Chat / Context history
        self.chat_history: list[dict[str, str]] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "target_role": self.target_role,
            "preferences": self.preferences,
            "learning_history": self.learning_history,
            "mastery": self.mastery,
            "milestone_status": self.milestone_status,
            "plan_generated": self.current_plan is not None
        }


from collections import OrderedDict

MAX_CACHE_SESSIONS = 5000


class SessionStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, LearnerSession] = OrderedDict()
        self._load_recent_from_db()

    def _load_recent_from_db(self, limit: int = 1000):
        """Pre-populate cache with most recent sessions from SQLite on startup."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,))
                for row in cursor.fetchall():
                    session = LearnerSession(row["user_id"])
                    session.target_role = row["target_role"]
                    session.preferences = json.loads(row["preferences_json"] or "{}")
                    session.learning_history = json.loads(row["learning_history_json"] or "[]")
                    session.mastery = json.loads(row["mastery_json"] or "{}")
                    session.asked_questions = json.loads(row["asked_questions_json"] or "{}")
                    session.milestone_status = json.loads(row["milestone_status_json"] or "{}")
                    session.current_plan = json.loads(row["current_plan_json"]) if row["current_plan_json"] else None
                    session.chat_history = json.loads(row["chat_history_json"] or "[]")
                    self._cache[session.user_id] = session
        except Exception as e:
            print(f"[WARN] Database load failed: {e}")

    def _load_single_from_db(self, user_id: str) -> Optional[LearnerSession]:
        """Lazy-rehydrate a session from SQLite if evicted from in-memory cache."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM sessions WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                if row:
                    session = LearnerSession(row["user_id"])
                    session.target_role = row["target_role"]
                    session.preferences = json.loads(row["preferences_json"] or "{}")
                    session.learning_history = json.loads(row["learning_history_json"] or "[]")
                    session.mastery = json.loads(row["mastery_json"] or "{}")
                    session.asked_questions = json.loads(row["asked_questions_json"] or "{}")
                    session.milestone_status = json.loads(row["milestone_status_json"] or "{}")
                    session.current_plan = json.loads(row["current_plan_json"]) if row["current_plan_json"] else None
                    session.chat_history = json.loads(row["chat_history_json"] or "[]")
                    return session
        except Exception as e:
            print(f"[WARN] SQLite single load failed for user {user_id}: {e}")
        return None

    def _sync_to_db(self, session: LearnerSession):
        """Persist session changes to SQLite (synchronous write-through with thread lock)."""
        with self._lock:
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO sessions (
                            user_id, target_role, preferences_json, learning_history_json,
                            mastery_json, asked_questions_json, milestone_status_json,
                            current_plan_json, chat_history_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session.user_id,
                        session.target_role,
                        json.dumps(session.preferences),
                        json.dumps(session.learning_history),
                        json.dumps(session.mastery),
                        json.dumps(session.asked_questions),
                        json.dumps(session.milestone_status),
                        json.dumps(session.current_plan) if session.current_plan else None,
                        json.dumps(session.chat_history),
                        time.time()
                    ))
                    conn.commit()
            except Exception as e:
                print(f"[WARN] SQLite sync failed for user {session.user_id}: {e}")

    def get_or_create(self, user_id: str) -> LearnerSession:
        with self._lock:
            if user_id in self._cache:
                self._cache.move_to_end(user_id)
                return self._cache[user_id]
            
            # Check SQLite
            session = self._load_single_from_db(user_id)
            if not session:
                session = LearnerSession(user_id)
            
            self._cache[user_id] = session
            if len(self._cache) > MAX_CACHE_SESSIONS:
                self._cache.popitem(last=False) # Evict oldest LRU session from RAM
            
            self._sync_to_db(session)
            return self._cache[user_id]

    def set_role(self, user_id: str, role: str) -> None:
        session = self.get_or_create(user_id)
        session.target_role = role
        self._sync_to_db(session)

    def update_preferences(self, user_id: str, prefs: dict[str, Any]) -> None:
        session = self.get_or_create(user_id)
        session.preferences.update(prefs)
        self._sync_to_db(session)

    def add_history(self, user_id: str, completed_skills: list[str]) -> None:
        session = self.get_or_create(user_id)
        session.learning_history.extend(completed_skills)
        for s in completed_skills:
            session.mastery[s] = max(session.mastery.get(s, 0.0), 0.85)
        self._sync_to_db(session)

    def get_mastery(self, user_id: str) -> dict[str, float]:
        return self.get_or_create(user_id).mastery

    def set_skill_mastery(self, user_id: str, skill_id: str, score: float) -> None:
        session = self.get_or_create(user_id)
        session.mastery[skill_id] = round(max(0.0, min(1.0, score)), 3)
        self._sync_to_db(session)

    def record_question(self, user_id: str, skill_id: str, q_text: str) -> None:
        session = self.get_or_create(user_id)
        if skill_id not in session.asked_questions:
            session.asked_questions[skill_id] = []
        session.asked_questions[skill_id].append(q_text)
        self._sync_to_db(session)

    def get_asked(self, user_id: str, skill_id: str) -> list[str]:
        return self.get_or_create(user_id).asked_questions.get(skill_id, [])

    def save_plan(self, user_id: str, plan: dict[str, Any]) -> None:
        session = self.get_or_create(user_id)
        session.current_plan = plan
        self._sync_to_db(session)

    def get_plan(self, user_id: str) -> Optional[dict[str, Any]]:
        return self.get_or_create(user_id).current_plan

    def update_milestone(self, user_id: str, skill_id: str, status: str) -> None:
        session = self.get_or_create(user_id)
        session.milestone_status[skill_id] = status
        current = session.mastery.get(skill_id, 0.20)
        if status == "passed":
            session.mastery[skill_id] = max(0.85, min(1.0, current + 0.50))
        elif status == "failed":
            session.mastery[skill_id] = round(max(0.05, current * 0.4), 3)
        self._sync_to_db(session)

    def add_chat_message(self, user_id: str, role: str, message: str) -> None:
        session = self.get_or_create(user_id)
        session.chat_history.append({"role": role, "message": message, "timestamp": time.time()})
        self._sync_to_db(session)

    def get_context(self, user_id: str) -> dict[str, Any]:
        session = self.get_or_create(user_id)
        from .gap import mastery_summary
        role = session.target_role or "Data Scientist"
        
        # Always prioritize active current_plan readiness_score to stay 100% synced with UI Hologram
        if session.current_plan and "readiness_score" in session.current_plan:
            readiness_score = session.current_plan["readiness_score"]
            skills_mastered = session.current_plan.get("skills_mastered_count", 0)
            skills_total = session.current_plan.get("total_skills_count", 0)
        else:
            summary = mastery_summary(role, session.mastery)
            readiness_score = summary["overall_readiness"]
            skills_mastered = summary["skills_mastered"]
            skills_total = summary["skills_total"]

        readiness_pct = round(readiness_score * 100)

        return {
            "target_role": role,
            "preferences": session.preferences,
            "mastery": session.mastery,
            "readiness_score": readiness_score,
            "readiness_percentage": readiness_pct,
            "skills_mastered": skills_mastered,
            "skills_total": skills_total,
            "milestone_status": session.milestone_status,
            "plan_summary": [
                {"skill": n.get("skill_name"), "status": session.milestone_status.get(n.get("skill"), "pending")}
                for n in (session.current_plan.get("nodes", []) if session.current_plan else [])
            ]
        }


# Singleton session store
store = SessionStore()
