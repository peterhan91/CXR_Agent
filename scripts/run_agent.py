#!/usr/bin/env python3
"""
Main entry point for the CXR Report Generation Agent.

Adapted from mimic_skills' codes_Hager/.../run.py (Hydra-based),
simplified for initial development. Uses YAML config instead of Hydra.

Usage:
    python scripts/run_agent.py --image path/to/cxr.png
    python scripts/run_agent.py --image path/to/cxr.png --config configs/config.yaml
    python scripts/run_agent.py --image_dir path/to/images/ --output results/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.react_agent import CXRReActAgent
from clear.concept_scorer import CLEARConceptScorer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_tools(config: dict) -> list:
    """Build tool instances based on config.

    Each tool is a thin HTTP client to a FastAPI model server.
    Enable/disable and set endpoints in configs/config.yaml.
    """
    from tools import (
        CheXagent2ReportTool,
        CheXagent2SRRGTool,
        CheXagent2GroundingTool,
        CheXagent2ClassifyTool,
        CheXagent2VQATool,
        CheXOneReportTool,
        CheXzeroClassifyTool,
        CXRFoundationClassifyTool,
        MedVersaReportTool,
        MedVersaClassifyTool,
        MedVersaDetectTool,
        MedVersaSegmentTool,
        MedVersaVQATool,
        BiomedParseSegmentTool,
        MedSAMSegmentTool,
        MedSAM3SegmentTool,
        FactCheXckerVerifyTool,
    )

    logger = logging.getLogger(__name__)
    tool_config = config.get("tools", {})

    # Map config keys to tool classes
    tool_registry = {
        "chexagent2": CheXagent2ReportTool,
        "chexagent2_srrg": CheXagent2SRRGTool,
        "chexagent2_classify": CheXagent2ClassifyTool,
        "chexagent2_grounding": CheXagent2GroundingTool,
        "chexagent2_vqa": CheXagent2VQATool,
        "chexone": CheXOneReportTool,
        "chexzero": CheXzeroClassifyTool,
        "cxr_foundation": CXRFoundationClassifyTool,
        "medversa": MedVersaReportTool,
        "medversa_classify": MedVersaClassifyTool,
        "medversa_detect": MedVersaDetectTool,
        "medversa_segment": MedVersaSegmentTool,
        "medversa_vqa": MedVersaVQATool,
        "biomedparse": BiomedParseSegmentTool,
        "medsam": MedSAMSegmentTool,
        "medsam3": MedSAM3SegmentTool,
        "factchexcker": FactCheXckerVerifyTool,
    }

    tools = []
    for key, tool_cls in tool_registry.items():
        entry = tool_config.get(key, {})
        if entry.get("enabled", False):
            endpoint = entry.get("endpoint", "http://localhost:8000")
            tools.append(tool_cls(endpoint=endpoint))
            logger.info(f"Enabled tool: {key} -> {endpoint}")
        else:
            logger.debug(f"Skipped tool: {key} (disabled)")

    logger.info(f"Built {len(tools)} tools: {[t.name for t in tools]}")
    return tools


def run_single_image(
    image_path: str,
    config: dict,
    scorer: CLEARConceptScorer = None,
    agent: CXRReActAgent = None,
) -> dict:
    """Run the agent on a single CXR image.

    Analogous to a single patient run in mimic_skills:
    agent_executor.invoke({"input": patient_history})
    """
    logger = logging.getLogger(__name__)
    image_id = Path(image_path).stem

    # Get CLEAR concept prior
    concept_prior_text = ""
    if scorer is not None:
        logger.info(f"Computing CLEAR concept scores for {image_id}...")
        top_k = config.get("clear", {}).get("top_k", 20)
        concept_prior_text = scorer.score_image(image_path, top_k=top_k)
        logger.info(f"Concept prior computed ({top_k} top concepts)")

    # Run the agent
    logger.info(f"Running agent on {image_id}...")
    trajectory = agent.run(
        image_path=image_path,
        concept_prior_text=concept_prior_text,
        image_id=image_id,
    )

    logger.info(
        f"Agent finished: {len(trajectory.steps)} steps, "
        f"{trajectory.total_input_tokens} input tokens, "
        f"{trajectory.total_output_tokens} output tokens, "
        f"{trajectory.total_duration_ms:.0f}ms"
    )

    return {
        "image_id": image_id,
        "image_path": image_path,
        "report": trajectory.final_report,
        "num_steps": len(trajectory.steps),
        "input_tokens": trajectory.total_input_tokens,
        "output_tokens": trajectory.total_output_tokens,
        "duration_ms": trajectory.total_duration_ms,
        "trajectory": trajectory.steps,
    }


def main():
    parser = argparse.ArgumentParser(description="CXR Report Generation Agent")
    parser.add_argument("--image", type=str, help="Path to single CXR image")
    parser.add_argument("--image_dir", type=str, help="Directory of CXR images")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--no_clear", action="store_true", help="Skip CLEAR concept scoring")
    parser.add_argument("--no_skills", action="store_true", help="Run without skill files")
    parser.add_argument("--skill_path", type=str, help="Path to evolved skill file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger = logging.getLogger(__name__)

    # Initialize CLEAR scorer
    scorer = None
    if not args.no_clear:
        clear_config = config.get("clear", {})
        scorer = CLEARConceptScorer(
            model_path=clear_config.get("model_path"),
            concepts_path=clear_config.get("concepts_path"),
            dinov2_model_name=clear_config.get("dinov2_model_name", "dinov2_vitb14"),
            image_resolution=clear_config.get("image_resolution", 448),
        )
        logger.info("Loading CLEAR model (one-time cost)...")
        scorer.load()
        logger.info("CLEAR model ready")

    # Build tools
    tools = build_tools(config)

    # Load skill if provided
    skill_text = None
    skill_path = args.skill_path or config.get("skill", {}).get("path")
    if skill_path and os.path.exists(skill_path):
        with open(skill_path, "r") as f:
            skill_text = f.read()
        # Strip YAML frontmatter if present (from mimic_skills convention)
        if skill_text.startswith("---"):
            parts = skill_text.split("---", 2)
            if len(parts) >= 3:
                skill_text = parts[2].strip()
        logger.info(f"Loaded skill from {skill_path} ({len(skill_text)} chars)")

    # Initialize agent
    agent_config = config.get("agent", {})
    # Skills can be disabled via --no_skills flag or config
    use_skills = not args.no_skills and config.get("skill", {}).get("enabled", True)

    agent = CXRReActAgent(
        model=agent_config.get("model", "claude-sonnet-4-6"),
        max_iterations=agent_config.get("max_iterations", 10),
        max_tokens=agent_config.get("max_tokens", 4096),
        temperature=agent_config.get("temperature", 0.0),
        tools=tools,
        skill_text=skill_text,
        reasoning_effort=agent_config.get("reasoning_effort"),
        use_skills=use_skills,
    )

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run on single image or directory
    if args.image:
        result = run_single_image(args.image, config, scorer, agent)

        # Print report
        print("\n" + "=" * 80)
        print(f"REPORT FOR: {result['image_id']}")
        print("=" * 80)
        print(result["report"])
        print("=" * 80)

        # Save result
        output_path = os.path.join(args.output, f"{result['image_id']}_result.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to {output_path}")

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_files = sorted(
            list(image_dir.glob("*.png"))
            + list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.jpeg"))
            + list(image_dir.glob("*.dcm"))
        )
        logger.info(f"Found {len(image_files)} images in {image_dir}")

        results = []
        for img_path in image_files:
            result = run_single_image(str(img_path), config, scorer, agent)
            results.append(result)

        # Save all results
        output_path = os.path.join(args.output, "all_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"All results saved to {output_path}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
