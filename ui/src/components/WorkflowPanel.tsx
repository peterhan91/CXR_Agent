"use client";

import { useStudyStore } from "@/stores/studyStore";
import type { Trajectory, TrajectoryStep } from "@/lib/types";

function cleanFeedback(raw: string): string {
  let text = raw;
  text = text.replace(/^Attending\s+feedback[^:]*:\s*/i, "");
  text = text.replace(/\s*Re-examine with your tools and revise your report\.?\s*$/i, "");
  text = text.replace(/\s*Continue your investigation with this in mind and produce your final report\.?\s*$/i, "");
  return text.trim();
}

interface WorkflowPanelProps {
  trajectories: Record<string, Trajectory>;
}

// Color by tool type
function toolColor(name: string): string {
  if (name.includes("report") || name.includes("srrg")) return "text-blue-400";
  if (name.includes("classify")) return "text-purple-400";
  if (name.includes("vqa")) return "text-cyan-400";
  if (name.includes("grounding")) return "text-yellow-400";
  if (name.includes("segment")) return "text-green-400";
  if (name.includes("verify") || name.includes("fact")) return "text-red-400";
  return "text-text-secondary";
}

function toolIcon(name: string): string {
  if (name.includes("report") || name.includes("srrg")) return "R";
  if (name.includes("classify")) return "C";
  if (name.includes("vqa")) return "Q";
  if (name.includes("grounding")) return "G";
  if (name.includes("segment")) return "S";
  if (name.includes("verify") || name.includes("fact")) return "V";
  return "T";
}

/**
 * Detect the feedback injection boundary in RITL trajectories.
 * Both rerun and checkpoint trajectories reset iteration to 1
 * after the feedback is injected, so we look for an iteration
 * that drops back down vs the previous step.
 */
function findFeedbackBoundary(steps: TrajectoryStep[]): number {
  for (let i = 1; i < steps.length; i++) {
    if (steps[i].iteration < steps[i - 1].iteration) {
      return i;
    }
  }
  return -1;
}

function FeedbackDivider({ feedback }: { feedback: string }) {
  return (
    <div className="my-3 mx-1">
      <div className="flex items-center gap-2 mb-2">
        <div className="flex-1 h-px bg-semantic-orange/40" />
        <span className="text-[10px] text-semantic-orange font-semibold uppercase tracking-wider px-2">
          Feedback Injected
        </span>
        <div className="flex-1 h-px bg-semantic-orange/40" />
      </div>
      <div className="bg-semantic-orange/10 border border-semantic-orange/20 rounded-lg px-3 py-2 text-xs text-semantic-orange leading-relaxed">
        {cleanFeedback(feedback)}
      </div>
    </div>
  );
}

function StepCard({
  step,
  isExpanded,
  onToggle,
  isPostFeedback,
}: {
  step: TrajectoryStep;
  isExpanded: boolean;
  onToggle: () => void;
  isPostFeedback: boolean;
}) {
  return (
    <button
      onClick={onToggle}
      className="w-full text-left group"
    >
      <div className={`flex items-start gap-2 py-1.5 px-2 rounded-lg hover:bg-bg-elevated transition-colors ${
        isPostFeedback ? "border-l-2 border-semantic-orange/30" : ""
      }`}>
        {/* Icon */}
        <span
          className={`w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold bg-bg-elevated flex-shrink-0 mt-0.5 ${toolColor(
            step.tool_name
          )}`}
        >
          {toolIcon(step.tool_name)}
        </span>

        {/* Name + duration */}
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2">
            <span className={`text-xs font-medium ${toolColor(step.tool_name)}`}>
              {step.tool_name}
            </span>
            {step.parallel_count && step.parallel_count > 1 && (
              <span className="text-[10px] text-text-tertiary">
                +{step.parallel_count - 1} parallel
              </span>
            )}
            <span className="text-[10px] text-text-tertiary ml-auto flex-shrink-0">
              {step.duration_ms > 0
                ? `${(step.duration_ms / 1000).toFixed(1)}s`
                : "cached"}
            </span>
          </div>

          {/* Expanded output */}
          {isExpanded && (
            <pre className="mt-2 text-[11px] text-text-secondary bg-bg-surface rounded p-2 overflow-x-auto whitespace-pre-wrap max-h-64 overflow-y-auto border border-separator">
              {step.tool_output}
            </pre>
          )}
        </div>
      </div>
    </button>
  );
}

export default function WorkflowPanel({ trajectories }: WorkflowPanelProps) {
  const { selectedModel, expandedSteps, toggleStep } = useStudyStore();

  // Pick trajectory matching selected model
  const trajectoryKey =
    selectedModel in trajectories
      ? selectedModel
      : Object.keys(trajectories)[0];
  const traj = trajectories[trajectoryKey];

  if (!traj) {
    return (
      <div className="flex items-center justify-center h-full text-text-tertiary text-sm">
        No trajectory available
      </div>
    );
  }

  const steps = traj.steps.filter((s) => s.type === "tool_call");
  const feedbackIdx = traj.feedback ? findFeedbackBoundary(steps) : -1;
  const totalTime = traj.wall_time_ms / 1000;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 pt-4 pb-2 border-b border-separator">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
          Agent Trajectory
        </h2>
        <div className="flex gap-3 mt-1 text-[10px] text-text-tertiary">
          <span>{steps.length} tool calls</span>
          <span>{totalTime.toFixed(1)}s total</span>
          <span>{traj.input_tokens?.toLocaleString()} tokens in</span>
        </div>
        {feedbackIdx >= 0 && (
          <div className="flex gap-2 mt-1.5 text-[10px]">
            <span className="text-text-tertiary">{feedbackIdx} pre-feedback</span>
            <span className="text-semantic-orange">{steps.length - feedbackIdx} post-feedback</span>
          </div>
        )}
      </div>

      {/* Steps */}
      <div className="flex-1 overflow-y-auto px-2 py-2">
        {steps.map((step, i) => (
          <div key={i}>
            {i === feedbackIdx && traj.feedback && (
              <FeedbackDivider feedback={traj.feedback} />
            )}
            <StepCard
              step={step}
              isExpanded={expandedSteps.has(i)}
              onToggle={() => toggleStep(i)}
              isPostFeedback={feedbackIdx >= 0 && i >= feedbackIdx}
            />
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="px-4 py-2 border-t border-separator flex flex-wrap gap-3 text-[10px] text-text-tertiary">
        <span className="text-blue-400">R Report</span>
        <span className="text-purple-400">C Classify</span>
        <span className="text-cyan-400">Q VQA</span>
        <span className="text-yellow-400">G Ground</span>
        <span className="text-green-400">S Segment</span>
        <span className="text-red-400">V Verify</span>
      </div>
    </div>
  );
}
