"use client";

import { useRef, useEffect } from "react";
import { useStudyStore } from "@/stores/studyStore";
import type { Trajectory, TrajectoryStep } from "@/lib/types";

function cleanFeedback(raw: string): string {
  let text = raw;
  text = text.replace(/^Attending\s+feedback[^:]*:\s*/i, "");
  text = text.replace(/\s*Re-examine with your tools and revise your report\.?\s*$/i, "");
  text = text.replace(/\s*Continue your investigation with this in mind and produce your final report\.?\s*$/i, "");
  return text.trim();
}

interface FeatureContext {
  hasMetadata: boolean;
  hasPrior: boolean;
  hasLateral: boolean;
}

interface WorkflowPanelProps {
  trajectories: Record<string, Trajectory>;
  featureContext?: FeatureContext;
  lateralImagePath?: string;
}

// Color by tool type
function toolColor(name: string): string {
  if (name.includes("temporal") || name.includes("longitudinal")) return "text-violet-400";
  if (name.includes("report") || name.includes("srrg")) return "text-blue-400";
  if (name.includes("classify")) return "text-purple-400";
  if (name.includes("vqa")) return "text-cyan-400";
  if (name.includes("grounding")) return "text-yellow-400";
  if (name.includes("segment")) return "text-green-400";
  if (name.includes("verify") || name.includes("fact")) return "text-red-400";
  return "text-text-secondary";
}

function toolIcon(name: string): string {
  if (name.includes("temporal") || name.includes("longitudinal")) return "T";
  if (name.includes("report") || name.includes("srrg")) return "R";
  if (name.includes("classify")) return "C";
  if (name.includes("vqa")) return "Q";
  if (name.includes("grounding")) return "G";
  if (name.includes("segment")) return "S";
  if (name.includes("verify") || name.includes("fact")) return "V";
  return "?";
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

function FeatureBanner({ color, label }: { color: string; label: string }) {
  return (
    <div className={`flex items-center gap-2 mx-1 my-1.5 px-2.5 py-1 rounded border-l-2 ${color}`}>
      <span className="text-[10px] font-medium">{label}</span>
    </div>
  );
}

/** Inline marker after specific tool calls */
function InlineMarker({ label, color }: { label: string; color: string }) {
  return (
    <div className={`flex items-center gap-1.5 mx-3 my-0.5 text-[10px] ${color}`}>
      <span className="w-1 h-1 rounded-full bg-current" />
      <span className="font-medium">{label}</span>
    </div>
  );
}

function StepList({
  steps,
  feedbackIdx,
  feedback,
  expandedSteps,
  toggleStep,
  isLive,
  featureContext,
  lateralImagePath,
}: {
  steps: TrajectoryStep[];
  feedbackIdx: number;
  feedback?: string;
  expandedSteps: Set<number>;
  toggleStep: (i: number) => void;
  isLive: boolean;
  featureContext?: FeatureContext;
  lateralImagePath?: string;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevCount = useRef(0);

  // Auto-scroll to bottom when new steps appear during live run
  useEffect(() => {
    if (isLive && steps.length > prevCount.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
    prevCount.current = steps.length;
  }, [steps.length, isLive]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto px-2 py-2">
      {/* Top-of-list feature banners */}
      {featureContext?.hasMetadata && (
        <FeatureBanner color="border-accent bg-accent/8 text-accent" label="Clinical context provided" />
      )}
      {featureContext?.hasPrior && (
        <FeatureBanner color="border-violet-400 bg-violet-400/8 text-violet-400" label="Prior study comparison active" />
      )}
      {featureContext?.hasLateral && (
        <FeatureBanner color="border-teal-400 bg-teal-400/8 text-teal-400" label="Lateral view provided" />
      )}

      {steps.map((step, i) => (
        <div key={i} className={isLive && i >= prevCount.current - 1 ? "animate-step-pop-in" : ""}>
          {i === feedbackIdx && (
            <>
              <FeatureBanner color="border-semantic-orange bg-semantic-orange/8 text-semantic-orange" label="Feedback received — agent re-reasoning" />
              {feedback && <FeedbackDivider feedback={feedback} />}
            </>
          )}
          <StepCard
            step={step}
            isExpanded={expandedSteps.has(i)}
            onToggle={() => toggleStep(i)}
            isPostFeedback={feedbackIdx >= 0 && i >= feedbackIdx}
          />
          {/* Inline markers after specific tool calls */}
          {(step.tool_name.includes("temporal") || step.tool_name.includes("longitudinal")) && (
            <InlineMarker label="Temporal comparison triggered" color="text-violet-400" />
          )}
          {step.tool_name.includes("grounding") && (
            <InlineMarker label="Grounding generated" color="text-yellow-400" />
          )}
          {lateralImagePath && (
            JSON.stringify(step.tool_input || {}).includes("lateral") ||
            JSON.stringify(step.tool_input || {}).includes(lateralImagePath) ||
            (step.tool_output || "").toLowerCase().includes("lateral")
          ) && (
            <InlineMarker label="Lateral view analyzed" color="text-teal-400" />
          )}
        </div>
      ))}
      {isLive && (
        <div className="flex items-center gap-2 py-2 px-2 text-[10px] text-semantic-orange animate-pulse">
          <span className="w-2 h-2 border border-semantic-orange border-t-transparent rounded-full animate-spin" />
          Waiting for next tool call...
        </div>
      )}
    </div>
  );
}

export default function WorkflowPanel({ trajectories, featureContext, lateralImagePath }: WorkflowPanelProps) {
  const { selectedModel, expandedSteps, toggleStep, liveRuns } = useStudyStore();

  // Pick trajectory: live run trajectory first, then pre-computed
  const liveRun = liveRuns[selectedModel];
  const liveTraj = liveRun?.trajectory as Trajectory | null;

  const trajectoryKey =
    selectedModel in trajectories
      ? selectedModel
      : Object.keys(trajectories)[0];
  const precomputedTraj = trajectories[trajectoryKey];

  const traj = liveTraj || precomputedTraj;
  const hasPendingFeedback = !!(liveRun as unknown as Record<string, unknown> | null)?._pendingFeedback;
  const isLiveInitial = !!liveTraj && (liveRun?.status === "running" || liveRun?.status === "queued");
  const isLiveRitl = !!(liveRun?.status === "running" && hasPendingFeedback);
  const isLive = isLiveInitial || isLiveRitl;

  if (!traj) {
    const isRunning = liveRun?.status === "queued" || liveRun?.status === "running";
    return (
      <div className="flex flex-col items-center justify-center h-full text-text-tertiary text-sm gap-2">
        {isRunning ? (
          <>
            <div className="w-6 h-6 border-2 border-semantic-orange border-t-transparent rounded-full animate-spin" />
            <span className="text-semantic-orange text-xs">{liveRun?.message || "Starting..."}</span>
          </>
        ) : (
          "No trajectory available"
        )}
      </div>
    );
  }

  const baseSteps = traj.steps.filter((s) => s.type === "tool_call" && s.tool_name);

  // Append RITL feedback steps if available
  const ritlSteps = (liveRun?.ritl_trajectory_steps as TrajectoryStep[] | null) || [];
  const filteredRitlSteps = ritlSteps.filter((s) => s.type === "tool_call" && s.tool_name);
  const ritlFeedback = liveRun?.ritl_result
    ? (liveRun.ritl_result as Record<string, string>).feedback
    : (liveRun as unknown as Record<string, unknown> | null)?._pendingFeedback as string | null;

  const steps = filteredRitlSteps.length > 0
    ? [...baseSteps, ...filteredRitlSteps]
    : baseSteps;

  // Feedback boundary: either from pre-computed traj or from live RITL
  let feedbackIdx = traj.feedback ? findFeedbackBoundary(baseSteps) : -1;
  const feedbackText = traj.feedback || ritlFeedback || undefined;
  if (feedbackIdx < 0 && (filteredRitlSteps.length > 0 || ritlFeedback)) {
    feedbackIdx = baseSteps.length; // boundary is where RITL steps begin
  }

  const totalTime = traj.wall_time_ms / 1000;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 pt-4 pb-2 border-b border-separator">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-text-secondary flex items-center gap-2">
          Agent Trajectory
          {isLive && (
            <span className="inline-flex items-center gap-1 text-semantic-orange font-normal normal-case tracking-normal">
              <span className="w-1.5 h-1.5 rounded-full bg-semantic-orange animate-pulse" />
              Live
            </span>
          )}
        </h2>
        <div className="flex gap-3 mt-1 text-[10px] text-text-tertiary">
          <span>{steps.length} tool calls</span>
          <span>{totalTime.toFixed(1)}s</span>
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
      <StepList
        steps={steps}
        feedbackIdx={feedbackIdx}
        feedback={feedbackText}
        expandedSteps={expandedSteps}
        toggleStep={toggleStep}
        isLive={isLive}
        featureContext={featureContext}
        lateralImagePath={lateralImagePath}
      />

      {/* Legend */}
      <div className="px-4 py-2 border-t border-separator flex flex-wrap gap-3 text-[10px] text-text-tertiary">
        <span className="text-blue-400">R Report</span>
        <span className="text-purple-400">C Classify</span>
        <span className="text-cyan-400">Q VQA</span>
        <span className="text-violet-400">T Temporal</span>
        <span className="text-yellow-400">G Ground</span>
        <span className="text-green-400">S Segment</span>
        <span className="text-red-400">V Verify</span>
      </div>
    </div>
  );
}
