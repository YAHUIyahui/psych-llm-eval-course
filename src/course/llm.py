"""Minimal LLM clients.

Every client exposes `chat(messages, tools=None, temperature=0.0) -> dict` and
returns {"content": str | None, "tool_calls": [{"id", "name", "arguments"}],
"usage": dict}. The course runs offline with ScriptedLLM. To use a real model,
set environment variables and call get_client():

    LLM_API_KEY   your key
    LLM_BASE_URL  e.g. https://api.openai.com/v1, or any OpenAI-compatible
                  endpoint (DeepSeek, Qwen/DashScope compatible mode, etc.)
    LLM_MODEL     model name at that endpoint
"""
from __future__ import annotations

import json
import os
import time
import urllib.request


class ScriptedLLM:
    """Offline stand-in for a model. `policy(messages, tools)` returns the
    response dict. Every call is logged in `self.calls` for inspection."""

    def __init__(self, policy, name: str = "scripted"):
        self.policy, self.name, self.calls = policy, name, []

    def chat(self, messages, tools=None, temperature: float = 0.0, **kw):
        t0 = time.perf_counter()
        out = self.policy(messages, tools)
        out.setdefault("content", None)
        out.setdefault("tool_calls", [])
        out.setdefault("usage", {"prompt_tokens": sum(len(str(m.get("content", ""))) // 4 for m in messages),
                                 "completion_tokens": len(out["content"] or "") // 4})
        self.calls.append({"messages": len(messages), "latency_s": time.perf_counter() - t0,
                           "tool_calls": [c["name"] for c in out["tool_calls"]]})
        return out


class OpenAICompatible:
    """Client for any endpoint that speaks the OpenAI chat-completions format."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, timeout: int = 60, max_retries: int = 3):
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set LLM_API_KEY (never hard-code keys in notebooks).")
        self.timeout, self.max_retries = timeout, max_retries
        self.name = self.model

    def chat(self, messages, tools=None, temperature: float = 0.0, max_tokens: int = 800, seed=None):
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
                for t in tools]
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                break
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay *= 2  # exponential backoff for rate limits and timeouts
        msg = data["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": c["function"]["arguments"]}
            calls.append({"id": c["id"], "name": c["function"]["name"], "arguments": args})
        return {"content": msg.get("content"), "tool_calls": calls, "usage": data.get("usage", {})}


def get_client():
    """Real client if LLM_API_KEY is set, else None (notebooks then stay offline)."""
    return OpenAICompatible() if os.environ.get("LLM_API_KEY") else None
