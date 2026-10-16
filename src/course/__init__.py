"""Course library: measurement and evaluation tools for psychology-focused
LLMs and agents. Each lesson imports what it needs from the submodules."""
from . import (agent, agent_eval, agreement, audit, bt, causal, cbt_tools, check,
               contamination, data, dawid_skene, irt, judge, llm, memory, plotting,
               repro, stats)

__version__ = "1.0.0"
