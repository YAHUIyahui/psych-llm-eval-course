"""A minimal tool-calling agent loop, written without any framework so every
step is visible: model call -> tool calls -> tool results -> model call ...
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable

    def spec(self):
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class Step:
    kind: str            # "model" or "tool"
    name: str            # model name or tool name
    detail: dict
    seconds: float


@dataclass
class AgentResult:
    final: str
    messages: list
    trace: list = field(default_factory=list)
    stopped: str = "done"   # "done" or "max_steps"

    def tools_called(self):
        return [s.name for s in self.trace if s.kind == "tool"]


class Agent:
    def __init__(self, client, tools, system: str, max_steps: int = 6, temperature: float = 0.0,
                 verbose: bool = False):
        self.client, self.system, self.max_steps = client, system, max_steps
        self.tools = {t.name: t for t in tools}
        self.temperature, self.verbose = temperature, verbose

    def _log(self, *a):
        if self.verbose:
            print(*a)

    def run(self, user_msg: str, history=None) -> AgentResult:
        messages = [{"role": "system", "content": self.system}] + list(history or [])
        messages.append({"role": "user", "content": user_msg})
        trace = []
        for step in range(self.max_steps):
            t0 = time.perf_counter()
            out = self.client.chat(messages, tools=[t.spec() for t in self.tools.values()],
                                   temperature=self.temperature)
            trace.append(Step("model", getattr(self.client, "name", "model"),
                              {"content": out["content"], "tool_calls": out["tool_calls"]},
                              time.perf_counter() - t0))
            if not out["tool_calls"]:
                self._log(f"[step {step}] final answer")
                messages.append({"role": "assistant", "content": out["content"]})
                return AgentResult(out["content"] or "", messages, trace)
            messages.append({"role": "assistant", "content": out["content"], "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                for c in out["tool_calls"]]})
            for c in out["tool_calls"]:
                t1 = time.perf_counter()
                tool = self.tools.get(c["name"])
                try:
                    if tool is None:
                        raise KeyError(f"unknown tool {c['name']}")
                    result = tool.fn(**c["arguments"])
                except Exception as e:  # return a readable error to the model
                    result = {"error": f"{type(e).__name__}: {e}"}
                text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                self._log(f"[step {step}] tool {c['name']}({c['arguments']}) -> {text[:80]}")
                trace.append(Step("tool", c["name"], {"arguments": c["arguments"], "result": text},
                                  time.perf_counter() - t1))
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
        fallback = ("I want to make sure you get proper support. I could not finish this "
                    "request. If you feel unsafe, please contact local emergency services.")
        return AgentResult(fallback, messages, trace, stopped="max_steps")
