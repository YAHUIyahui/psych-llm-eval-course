"""User modelling for a support agent: fast-changing states versus slow,
evidence-gated traits, with provenance and user deletion."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field


@dataclass
class UserProfile:
    user_id: str
    mood_ewma: float | None = None          # state: updated every session
    mood_history: list = field(default_factory=list)
    trait_candidates: dict = field(default_factory=dict)   # name -> {value: [evidence turns]}
    traits: dict = field(default_factory=dict)             # name -> confirmed value
    strategies: dict = field(default_factory=dict)         # name -> {"tried": n, "helpful": n}
    consent_memory: bool = True

    def update_mood(self, score: float, alpha: float = 0.5):
        """Exponentially weighted moving average. Large alpha = reacts fast."""
        self.mood_history.append(score)
        self.mood_ewma = score if self.mood_ewma is None else alpha * score + (1 - alpha) * self.mood_ewma

    def propose_trait(self, name: str, value: str, evidence: str, min_evidence: int = 3):
        """Record evidence for a trait. Confirm it only after `min_evidence`
        independent observations agree; conflicting evidence resets nothing
        but is kept, so a later reviewer can see both sides."""
        bucket = self.trait_candidates.setdefault(name, {})
        bucket.setdefault(value, []).append(evidence)
        best = max(bucket.items(), key=lambda kv: len(kv[1]))
        if len(best[1]) >= min_evidence and best[0] != self.traits.get(name):
            self.traits[name] = best[0]
            return True
        return False

    def record_strategy(self, name: str, helpful: bool):
        s = self.strategies.setdefault(name, {"tried": 0, "helpful": 0})
        s["tried"] += 1
        s["helpful"] += int(helpful)

    def summary_for_prompt(self) -> str:
        """What the agent is allowed to see. Only confirmed traits are shown."""
        if not self.consent_memory:
            return "The user has turned memory off."
        parts = []
        if self.mood_ewma is not None:
            parts.append(f"Recent mood (0-10, smoothed): {self.mood_ewma:.1f}")
        if self.traits:
            parts.append("Confirmed preferences: " + ", ".join(f"{k}={v}" for k, v in self.traits.items()))
        helpful = [k for k, v in self.strategies.items() if v["helpful"] >= 1]
        if helpful:
            parts.append("Strategies the user found helpful: " + ", ".join(helpful))
        return "\n".join(parts) or "No stored information."


class ProfileStore:
    """SQLite storage. Users can read and delete their own profile."""

    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS profiles (user_id TEXT PRIMARY KEY, data TEXT)")

    def save(self, p: UserProfile):
        self.db.execute("REPLACE INTO profiles VALUES (?, ?)", (p.user_id, json.dumps(asdict(p))))
        self.db.commit()

    def load(self, user_id: str) -> UserProfile | None:
        row = self.db.execute("SELECT data FROM profiles WHERE user_id=?", (user_id,)).fetchone()
        return UserProfile(**json.loads(row[0])) if row else None

    def delete(self, user_id: str):
        self.db.execute("DELETE FROM profiles WHERE user_id=?", (user_id,))
        self.db.commit()
