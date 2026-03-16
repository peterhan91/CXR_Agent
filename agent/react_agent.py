"""
CXR Report Generation ReAct Agent using Anthropic native tool-use.

Adapted from mimic_skills' CustomZeroShotAgent (LangChain ZeroShotAgent),
redesigned to use Anthropic's structured tool-use API instead of text-based
ReAct parsing. This eliminates regex parsing failures and provides
reliable tool invocation.

Key differences from mimic_skills:
- No LangChain dependency; uses anthropic SDK directly
- Tools defined as Anthropic JSON schemas (not LangChain BaseTool)
- ReAct loop managed explicitly (not via AgentExecutor)
- CLEAR concept priors injected as structured context
- Goal is report generation, not diagnosis
"""

import base64
import mimetypes
import time
import logging
from typing import Any, Optional
from dataclasses import dataclass, field

import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential

from agent.prompts import (
    SYSTEM_PROMPT_WITH_SKILLS,
    SYSTEM_PROMPT_PLAIN,
    CONCEPT_PRIOR_TEMPLATE,
    SKILL_INJECTION_TEMPLATE,
    build_skills_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    tool_input: dict
    output: str
    duration_ms: float = 0.0


@dataclass
class AgentTrajectory:
    """Complete trajectory of an agent run, for logging and evolution."""
    image_id: str
    concept_prior: str
    steps: list = field(default_factory=list)
    final_report: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_duration_ms: float = 0.0


class CXRReActAgent:
    """
    Zero-shot ReAct agent for CXR report generation.

    Uses Claude Sonnet via Anthropic's native tool-use API to orchestrate
    multiple CXR analysis models. The agent receives CLEAR concept priors
    as structured context and iteratively calls tools to build a
    comprehensive radiology report.

    Analogous to mimic_skills' build_agent_executor_ZeroShot() but:
    - Uses Anthropic tool-use instead of LangChain AgentExecutor
    - ReAct loop is explicit (Thought in model text, Action via tool_use)
    - No text-based action parsing needed (DiagnosisWorkflowParser equivalent is native)

    Args:
        model: Anthropic model name (default: claude-sonnet-4-6)
        max_iterations: Maximum ReAct loop iterations (like mimic_skills' max_iterations=10)
        max_tokens: Max tokens per agent response
        temperature: Sampling temperature (0.0 for deterministic)
        tools: List of tool instances (must implement .to_anthropic_schema() and .run())
        skill_text: Optional evolved skill to inject into prompt (for future evotest)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_iterations: int = 10,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: list = None,
        skill_text: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        use_skills: bool = True,
    ):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tools = tools or []
        self.skill_text = skill_text
        # Adaptive thinking: "low", "medium", "high", or None to disable.
        # Same pattern as mimic_skills models.py.
        self.reasoning_effort = reasoning_effort
        self.use_skills = use_skills

        # Build tool schemas for Anthropic API
        self._tool_schemas = [t.to_anthropic_schema() for t in self.tools]
        # Map tool names to instances for execution
        self._tool_map = {t.name: t for t in self.tools}

    def _build_system_prompt(self, concept_prior_text: str) -> str:
        """Build the full system prompt with skills, concept priors, and optional evolved skill.

        Assembly order:
        1. Base system prompt (role + principles) — variant depends on whether skills are active
        2. Skill files (workflow, tool guidance, interpretation rules) — only if use_skills=True
        3. CLEAR concept prior for this specific image
        4. Optional evolved skill text (for evotest integration)
        """
        # Pick the right base prompt
        has_skills = self.use_skills or self.skill_text
        base_prompt = SYSTEM_PROMPT_WITH_SKILLS if has_skills else SYSTEM_PROMPT_PLAIN
        parts = [base_prompt]

        # Load skill files (clinical reasoning guidance)
        if self.use_skills:
            skills_text = build_skills_prompt()
            if skills_text:
                parts.append(skills_text)

        if concept_prior_text:
            parts.append(concept_prior_text)

        # Evolved skill overrides/supplements the default skills
        if self.skill_text:
            parts.append(SKILL_INJECTION_TEMPLATE.format(skill_text=self.skill_text))

        return "\n\n".join(parts)

    def _execute_tool(self, tool_name: str, tool_input: dict) -> ToolResult:
        """Execute a tool by name with given input.

        Uses the shared tool response cache for deterministic tools
        (same image + same params = same output). FactCheXcker verify
        is excluded from caching since its input (draft report) changes.

        Analogous to mimic_skills' AgentExecutor tool dispatch, but simpler
        since Anthropic API provides structured tool_name and tool_input
        (no regex parsing needed like DiagnosisWorkflowParser).
        """
        from tools.base import cached_tool_call

        start_time = time.time()

        tool = self._tool_map.get(tool_name)
        if tool is None:
            output = f"Error: Unknown tool '{tool_name}'. Available tools: {list(self._tool_map.keys())}"
            logger.warning(output)
        else:
            try:
                # Skip cache for stateful/variable-input tools
                # - factchexcker_verify: input changes per draft
                # - evidence_board: local stateful memory
                if tool_name in ("factchexcker_verify", "evidence_board"):
                    output = tool.run(**tool_input)
                else:
                    output = cached_tool_call(tool_name, tool.run, **tool_input)
                if output is None:
                    output = "(tool returned no output)"
            except Exception as e:
                output = f"Error executing {tool_name}: {e}"
                logger.error(output, exc_info=True)

        duration_ms = (time.time() - start_time) * 1000

        return ToolResult(
            tool_name=tool_name,
            tool_input=tool_input,
            output=output,
            duration_ms=duration_ms,
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def _api_call(self, **kwargs) -> anthropic.types.Message:
        """Make an Anthropic API call with retry and exponential backoff.

        Adapted from mimic_skills' models.py:anthropic_completion_with_backoff()
        which uses the same tenacity retry pattern for robustness against
        transient API errors (rate limits, 500s, network timeouts).
        """
        return self.client.messages.create(**kwargs)

    def run(
        self,
        image_path: str,
        concept_prior_text: str = "",
        image_id: str = "",
        prior_report: str = "",
        prior_image_path: str = "",
        clinical_context: str = "",
    ) -> AgentTrajectory:
        """
        Run the ReAct agent to generate a CXR report.

        This is the main entry point, analogous to mimic_skills'
        agent_executor.invoke({"input": patient_history}).

        The ReAct loop:
        1. Send context + message history to Claude Sonnet
        2. If response contains tool_use blocks: execute tools, append results
        3. If response is end_turn (text only): extract final report
        4. Repeat until final report or max_iterations

        Args:
            image_path: Path to the CXR image file
            concept_prior_text: Formatted CLEAR concept scores for this image
            image_id: Identifier for logging/trajectory tracking
            prior_report: Text of the most recent prior CXR report (if available)
            prior_image_path: Path to the prior CXR image (agent can pass to tools)
            clinical_context: Formatted clinical context (HPI + chief complaint)

        Returns:
            AgentTrajectory with full reasoning trace and final report
        """
        trajectory = AgentTrajectory(
            image_id=image_id,
            concept_prior=concept_prior_text,
        )

        # Reset evidence board for this study (fresh per run)
        from tools.evidence_board import EvidenceBoardTool
        for tool in self.tools:
            if isinstance(tool, EvidenceBoardTool):
                tool.board.reset()

        system_prompt = self._build_system_prompt(concept_prior_text)

        # Initial user message with the CXR image
        messages = [
            {
                "role": "user",
                "content": self._build_initial_message(
                    image_path,
                    prior_report=prior_report,
                    prior_image_path=prior_image_path,
                    clinical_context=clinical_context,
                ),
            }
        ]

        start_time = time.time()

        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Call Claude Sonnet with tool definitions
            # Adaptive thinking follows mimic_skills pattern: when enabled,
            # use higher max_tokens and no temperature/system constraints.
            if self.reasoning_effort:
                api_kwargs = {
                    "model": self.model,
                    "max_tokens": 16000,
                    "messages": messages,
                    "system": system_prompt,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": self.reasoning_effort},
                }
            else:
                api_kwargs = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": system_prompt,
                    "messages": messages,
                    "temperature": self.temperature,
                }
            if self._tool_schemas:
                api_kwargs["tools"] = self._tool_schemas

            response = self._api_call(**api_kwargs)

            # Track token usage
            trajectory.total_input_tokens += response.usage.input_tokens
            trajectory.total_output_tokens += response.usage.output_tokens

            # Process response content blocks
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Check stop_reason for truncation
            # Analogous to mimic_skills' stop word handling in models.py
            if response.stop_reason == "max_tokens":
                logger.warning(
                    f"Response truncated (max_tokens={self.max_tokens}). "
                    "Consider increasing max_tokens."
                )
                trajectory.steps.append({
                    "iteration": iteration + 1,
                    "type": "truncated_response",
                    "stop_reason": response.stop_reason,
                })

            # Check if the model wants to use tools
            tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]

            if not tool_use_blocks:
                # If truncated with no tool calls, the text is likely incomplete —
                # ask the model to continue rather than treating garbage as final report
                if response.stop_reason == "max_tokens":
                    logger.warning("Truncated response with no tool calls, requesting continuation")
                    messages.append({
                        "role": "user",
                        "content": "Your response was truncated. Please continue and complete your report.",
                    })
                    continue

                # No tool calls — model is done reasoning, extract final report
                text_blocks = [b for b in assistant_content if b.type == "text"]
                final_text = "\n".join(b.text for b in text_blocks)
                trajectory.final_report = final_text
                trajectory.steps.append({
                    "iteration": iteration + 1,
                    "type": "final_report",
                    "text": final_text,
                })
                logger.info("Agent produced final report")
                break

            # Execute each tool call and collect results
            n_parallel = len(tool_use_blocks)
            tool_results = []
            for tool_block in tool_use_blocks:
                result = self._execute_tool(tool_block.name, tool_block.input)
                trajectory.steps.append({
                    "iteration": iteration + 1,
                    "type": "tool_call",
                    "tool_name": result.tool_name,
                    "tool_input": result.tool_input,
                    "tool_output": result.output,
                    "duration_ms": result.duration_ms,
                    "parallel_count": n_parallel,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result.output,
                })

            # Append tool results as user message (Anthropic API convention)
            messages.append({"role": "user", "content": tool_results})

            # Log thinking blocks (adaptive thinking) and reasoning text
            thinking_blocks = [b for b in assistant_content if b.type == "thinking"]
            if thinking_blocks:
                thinking_text = "\n".join(b.thinking for b in thinking_blocks)
                logger.debug(f"Agent thinking: {thinking_text[:300]}...")
                trajectory.steps.append({
                    "iteration": iteration + 1,
                    "type": "thinking",
                    "text": thinking_text,
                })

            text_blocks = [b for b in assistant_content if b.type == "text"]
            if text_blocks:
                reasoning = "\n".join(b.text for b in text_blocks)
                logger.info(f"Agent reasoning: {reasoning[:200]}...")
                trajectory.steps.append({
                    "iteration": iteration + 1,
                    "type": "reasoning",
                    "text": reasoning,
                })

        else:
            # Max iterations reached without final report
            # Force extraction from last response (analogous to mimic_skills'
            # truncation + forced diagnosis in _construct_scratchpad Tier 2)
            logger.warning(f"Max iterations ({self.max_iterations}) reached, forcing report extraction")
            trajectory.steps.append({
                "iteration": self.max_iterations,
                "type": "max_iterations_reached",
            })
            if not trajectory.final_report:
                trajectory.final_report = self._force_final_report(
                    messages, system_prompt, trajectory
                )

        trajectory.total_duration_ms = (time.time() - start_time) * 1000
        return trajectory

    def _encode_image(self, image_path: str) -> dict | None:
        """Encode an image file as a base64 Anthropic image content block.

        16-bit PNGs (PadChest-GR, RexGradient) are always normalized to 8-bit
        first — the Anthropic API does not render mode-I PNGs correctly.
        If the base64 payload exceeds the Anthropic 5 MB limit, the image is
        re-encoded as JPEG with decreasing quality until it fits.
        """
        _MAX_B64_BYTES = 5_242_880  # 5 MB Anthropic limit
        _SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
        mime_type = mimetypes.guess_type(image_path)[0]
        if mime_type not in _SUPPORTED_MIME_TYPES:
            logger.warning(
                f"MIME type '{mime_type}' for {image_path} not supported by Anthropic API, "
                f"defaulting to image/png. Supported: {_SUPPORTED_MIME_TYPES}"
            )
            mime_type = "image/png"
        try:
            with open(image_path, "rb") as f:
                raw_bytes = f.read()
        except FileNotFoundError:
            logger.warning(f"Image file not found: {image_path}")
            return None

        # Always normalize 16-bit PNGs to 8-bit before encoding
        from PIL import Image
        import io
        import numpy as np
        img = Image.open(io.BytesIO(raw_bytes))
        if img.mode in ("I", "I;16"):
            arr = np.array(img, dtype=np.float64)
            arr = arr - arr.min()
            mx = arr.max()
            if mx > 0:
                arr = (arr / mx * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
            img = Image.fromarray(arr, mode="L").convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            raw_bytes = buf.getvalue()
            mime_type = "image/png"
            logger.info(f"Converted 16-bit PNG to 8-bit: {image_path}")

        image_data = base64.standard_b64encode(raw_bytes).decode("utf-8")

        # If over 5 MB, re-encode as JPEG with decreasing quality
        if len(image_data) > _MAX_B64_BYTES:
            if img.mode != "RGB":
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            for quality in (90, 80, 70, 50):
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                image_data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
                if len(image_data) <= _MAX_B64_BYTES:
                    logger.info(
                        f"Compressed {image_path} to JPEG q={quality} "
                        f"({len(image_data)/1024/1024:.1f} MB b64)"
                    )
                    break
            mime_type = "image/jpeg"

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": image_data,
            },
        }

    def _build_initial_message(
        self,
        image_path: str,
        prior_report: str = "",
        prior_image_path: str = "",
        clinical_context: str = "",
    ) -> list:
        """Build the initial user message with CXR image and instructions.

        Analogous to mimic_skills' "Patient History: {input}" in the prompt,
        but includes the actual image for multimodal models.
        """
        content = []

        # Add the CXR image if path provided
        if image_path:
            img_block = self._encode_image(image_path)
            if img_block:
                content.append(img_block)

        # Build the task instruction text
        text_parts = [
            f"Generate a radiology report for this chest X-ray. "
            f"The image file path for tool calls is: {image_path}",
        ]

        # Clinical context (HPI + chief complaint)
        if clinical_context:
            text_parts.append(f"\nCLINICAL CONTEXT:\n{clinical_context}")

        # Prior study (report text + image path for tool calls)
        if prior_report:
            prior_section = f"\nPRIOR STUDY REPORT:\n{prior_report}"
            if prior_image_path:
                prior_section += (
                    f"\n\nPrior study image path: {prior_image_path}"
                )
            text_parts.append(prior_section)

        content.append({
            "type": "text",
            "text": "\n".join(text_parts),
        })

        return content

    def _force_final_report(
        self,
        messages: list,
        system_prompt: str,
        trajectory: AgentTrajectory,
    ) -> str:
        """Force the agent to produce a final report when max iterations reached.

        Analogous to mimic_skills' truncation strategy in
        _construct_scratchpad() Tier 2 where the agent is forced to provide
        "Final Diagnosis and Treatment" when context overflows.
        """
        messages_copy = list(messages)
        messages_copy.append({
            "role": "user",
            "content": (
                "You have used all available tool calls. Based on all the information "
                "gathered so far, please synthesize your final radiology report now. "
                "Include FINDINGS and IMPRESSION sections."
            ),
        })

        force_kwargs = {
            "model": self.model,
            "system": system_prompt,
            "messages": messages_copy,
        }
        if self.reasoning_effort:
            force_kwargs["max_tokens"] = 16000
            force_kwargs["thinking"] = {"type": "adaptive"}
            force_kwargs["output_config"] = {"effort": self.reasoning_effort}
        else:
            force_kwargs["max_tokens"] = self.max_tokens
            force_kwargs["temperature"] = self.temperature

        response = self._api_call(**force_kwargs)

        # Track tokens for forced report (was missing before)
        trajectory.total_input_tokens += response.usage.input_tokens
        trajectory.total_output_tokens += response.usage.output_tokens

        text_blocks = [b for b in response.content if b.type == "text"]
        return "\n".join(b.text for b in text_blocks)
