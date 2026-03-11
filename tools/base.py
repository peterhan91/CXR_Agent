"""
Base tool interface for CXR Agent tools.

Adapted from mimic_skills' LangChain BaseTool pattern, but designed
for Anthropic's native tool-use API. Each tool must:
1. Define an Anthropic-compatible JSON schema (to_anthropic_schema)
2. Implement a run() method that executes the tool

In mimic_skills, tools inherit from LangChain's BaseTool and implement
_run() with action_results dict. Here, tools wrap HuggingFace models
and return text observations that the agent can reason over.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Shared disk cache directory for all tools
_TOOL_CACHE_DIR = Path("cache/tools")


def _make_cache_key(tool_name: str, **kwargs) -> str:
    """Generate a deterministic cache key from tool name + params."""
    key_data = json.dumps({"tool": tool_name, **kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


# Shared caches (class-level, shared across all tool instances)
_memory_cache = {}  # (tool_name, cache_key) -> result


def cached_tool_call(tool_name: str, execute_fn, **kwargs) -> str:
    """Execute a tool call with in-memory + disk caching.

    Args:
        tool_name: Tool identifier for cache namespacing
        execute_fn: Callable that performs the actual tool execution
        **kwargs: Tool parameters

    Returns:
        Tool output string (from cache or fresh execution)
    """
    cache_key = _make_cache_key(tool_name, **kwargs)
    lookup = (tool_name, cache_key)

    # In-memory cache (fast path)
    if lookup in _memory_cache:
        logger.debug(f"Tool cache hit (memory): {tool_name}")
        return _memory_cache[lookup]

    # Disk cache (cross-session)
    disk_path = _TOOL_CACHE_DIR / tool_name / f"{cache_key}.json"
    if disk_path.exists():
        try:
            cached = json.loads(disk_path.read_text())
            result = cached["result"]
            _memory_cache[lookup] = result
            logger.debug(f"Tool cache hit (disk): {tool_name}")
            return result
        except (json.JSONDecodeError, KeyError):
            disk_path.unlink(missing_ok=True)

    # Cache miss — execute
    result = execute_fn(**kwargs)

    # Cache successful results only
    if result and not result.startswith("Error"):
        _memory_cache[lookup] = result
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_text(json.dumps({
            "tool": tool_name,
            "params": {k: v for k, v in kwargs.items()},
            "result": result,
        }))

    return result


class BaseCXRTool(ABC):
    """Base class for all CXR Agent tools.

    Analogous to mimic_skills' LangChain BaseTool subclasses
    (RunLaboratoryTests, RunImaging, DoPhysicalExamination, etc.)
    but adapted for Anthropic tool-use and HuggingFace model backends.

    Subclasses must implement:
    - name: Tool identifier used by the agent
    - description: What the tool does (shown to the agent)
    - input_schema: JSON schema for tool parameters
    - run(): Execute the tool and return text output
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in Anthropic API tool definitions."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description shown to the agent for tool selection."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict:
        """JSON Schema for the tool's input parameters."""
        ...

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool and return a text observation.

        Analogous to mimic_skills' BaseTool._run() which returns text
        observations that get appended to the agent's scratchpad.

        Returns:
            Text string describing the tool's output/findings.
        """
        ...

    def to_anthropic_schema(self) -> dict:
        """Convert tool definition to Anthropic API format.

        Returns the tool definition dict that gets passed to
        client.messages.create(tools=[...]).
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
