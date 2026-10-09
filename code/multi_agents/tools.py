"""Per-agent tool access.

Every agent calls its capabilities through a `ToolBox` of LangChain tools. A box only
contains the tools granted to its owner, so tool access is enforced rather than
conventional: the planning agent cannot reach the movement model, and only the
perception agent's box holds `run_movement_model` (the LoRA checkpoint). Calls and
their wall time are counted for the latency measures.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any, Callable, Sequence

from langchain_core.tools import BaseTool, StructuredTool


def make_tool(func: Callable[..., Any], name: str, description: str) -> BaseTool:
    return StructuredTool.from_function(func=func, name=name, description=description)


class ToolBox:
    def __init__(self, owner: str, tools: Sequence[BaseTool]) -> None:
        self.owner = owner
        self._tools = {tool.name: tool for tool in tools}
        self.calls: Counter[str] = Counter()
        self.milliseconds: defaultdict[str, float] = defaultdict(float)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def __call__(self, name: str, **arguments: Any) -> Any:
        if name not in self._tools:
            raise PermissionError(f"The {self.owner} agent has no access to tool {name!r}")
        started = time.perf_counter()
        result = self._tools[name].invoke(arguments)
        self.calls[name] += 1
        self.milliseconds[name] += 1000.0 * (time.perf_counter() - started)
        return result

    def describe(self) -> list[dict[str, str]]:
        return [{"name": tool.name, "description": tool.description} for tool in self._tools.values()]
