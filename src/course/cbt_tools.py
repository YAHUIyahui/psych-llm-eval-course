"""CBT-style tools and an offline scripted policy for the agent lessons.

The risk screen here is a keyword rule written for teaching. It misses
indirect expressions of risk and must not be used with real users.
"""
from __future__ import annotations

import itertools
import re
from datetime import datetime

from .agent import Tool

RISK_PATTERNS = [
    r"better off without me", r"not wake up", r"end (it|my life)", r"kill myself",
    r"no point (in )?living", r"giving away my (things|belongings)", r"suicid",
    r"不想活", r"活着没意思", r"结束生命", r"不如死了",
]

ACTIVITIES = {
    "low": ["Open the curtains and sit by the window for five minutes",
            "Drink a glass of water and stretch for two minutes",
            "Send one short message to someone you trust"],
    "medium": ["Take a ten-minute walk outside", "Prepare a simple meal",
               "Call a friend or family member for ten minutes"],
    "high": ["Join a group activity you used to enjoy", "Plan a short outing this week",
             "Try a 20-minute exercise session"],
}


def risk_screen(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in RISK_PATTERNS)


class MoodLog:
    """In-memory mood diary (0 = worst, 10 = best)."""

    def __init__(self):
        self.entries = []

    def log(self, score: int, note: str = ""):
        score = int(score)
        if not 0 <= score <= 10:
            return {"error": "score must be between 0 and 10"}
        self.entries.append({"time": datetime.now().isoformat(timespec="seconds"),
                             "score": score, "note": note})
        return {"saved": True, "n_entries": len(self.entries)}

    def history(self, last_n: int = 7):
        return {"entries": self.entries[-int(last_n):]}


def make_cbt_tools(mood_log: MoodLog | None = None):
    mood_log = mood_log or MoodLog()
    return [
        Tool("log_mood", "Save the user's current mood rating (0-10) with an optional note.",
             {"type": "object", "properties": {"score": {"type": "integer"}, "note": {"type": "string"}},
              "required": ["score"]}, mood_log.log),
        Tool("get_mood_history", "Return the user's recent mood ratings.",
             {"type": "object", "properties": {"last_n": {"type": "integer"}}}, mood_log.history),
        Tool("suggest_activity", "Suggest behavioural-activation activities for a given energy level.",
             {"type": "object", "properties": {"energy": {"type": "string", "enum": ["low", "medium", "high"]}},
              "required": ["energy"]},
             lambda energy: {"activities": ACTIVITIES.get(energy, ACTIVITIES["low"])}),
        Tool("thought_record", "Return a thought-record template for a stated automatic thought.",
             {"type": "object", "properties": {"thought": {"type": "string"}}, "required": ["thought"]},
             lambda thought: {"template": ["Situation", "Automatic thought: " + thought,
                                           "Emotion and intensity (0-100)", "Evidence for",
                                           "Evidence against", "Balanced thought", "Emotion now (0-100)"]}),
        Tool("crisis_resources", "Return crisis-support guidance. Use whenever the user may be at risk.",
             {"type": "object", "properties": {}},
             lambda: {"message": "If you might act on these thoughts, contact local emergency services "
                                 "now (999 in Hong Kong) or a crisis line in your area, and tell someone "
                                 "you trust. You do not have to go through this alone."}),
    ]


_ids = itertools.count()


def _call(name, **args):
    return {"content": None, "tool_calls": [{"id": f"call_{next(_ids)}", "name": name, "arguments": args}]}


def cbt_scripted_policy(messages, tools):
    """Deterministic stand-in for a model, so the agent loop runs offline."""
    last = messages[-1]
    if last["role"] == "tool":
        tool_texts = [m["content"] for m in messages if m["role"] == "tool"]
        joined = " ".join(tool_texts)
        if "emergency" in joined:
            return {"content": "I'm really glad you told me. What you are carrying sounds very heavy. "
                               "Your safety matters most right now. If you might act on these thoughts, "
                               "please contact local emergency services (999 in Hong Kong) or a crisis line, "
                               "and reach out to someone you trust. Would you be willing to tell me if "
                               "you are safe at this moment?"}
        if "activities" in joined:
            return {"content": "It makes sense that starting anything feels hard when energy is this low. "
                               "Could we pick one very small step for today, for example opening the "
                               "curtains and sitting by the window for five minutes? Which small step "
                               "feels most doable to you?"}
        if "saved" in joined:
            return {"content": "Thank you, I've noted your mood. What happened today that "
                               "might have shaped how you feel?"}
        if "template" in joined:
            return {"content": "Let's look at that thought together. What evidence supports it, "
                               "and what evidence does not fit it?"}
        return {"content": "Thanks. What would be most helpful to focus on now?"}
    text = last["content"]
    low = text.lower()
    if risk_screen(text):
        return _call("crisis_resources")
    m = re.search(r"(mood|心情)\D{0,6}(\d{1,2})", low)
    if m:
        return _call("log_mood", score=int(m.group(2)), note=text[:60])
    if any(k in low for k in ["no energy", "stay in bed", "no point starting", "没动力", "不想动"]):
        return _call("suggest_activity", energy="low")
    if any(k in low for k in ["i think", "always", "never", "ruined", "stupid", "我总是", "完了"]):
        return _call("thought_record", thought=text[:80])
    return {"content": "That sounds difficult. Can you tell me a little more about what has been "
                       "happening and how it has been affecting you?"}
