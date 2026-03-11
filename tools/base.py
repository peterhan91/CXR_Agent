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

from abc import ABC, abstractmethod
from typing import Any


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
        """JSON Schema for the tool's input parameters.

        Example:
            {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the CXR image"
                    }
                },
                "required": ["image_path"]
            }
        """
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
