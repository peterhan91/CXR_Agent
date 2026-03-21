"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useStudyStore } from "@/stores/studyStore";
import type { Prediction, TestStudy } from "@/lib/types";
import { startRun, pollRun, submitFeedback } from "@/lib/api";
import type { RunStatus } from "@/lib/api";
import ReportDiff from "./ReportDiff";

interface ReportPanelProps {
  study: TestStudy;
  predictions: Record<string, Prediction>;
  models: string[];
}

function ReportSection({
  label,
  text,
  color,
}: {
  label: string;
  text: string;
  color: string;
}) {
  if (!text) return null;
  return (
    <div>
      <h3 className={`text-xs font-semibold uppercase tracking-wider mb-2 ${color}`}>
        {label}
      </h3>
      <p className="text-sm text-text-primary leading-relaxed whitespace-pre-wrap">
        {text}
      </p>
    </div>
  );
}

function cleanFeedback(raw: string): string {
  let text = raw;
  // Strip prefixes like "Attending feedback:" or "Attending feedback before you write the report:"
  text = text.replace(/^Attending\s+feedback[^:]*:\s*/i, "");
  // Strip trailing instruction boilerplate
  text = text.replace(/\s*Re-examine with your tools and revise your report\.?\s*$/i, "");
  text = text.replace(/\s*Continue your investigation with this in mind and produce your final report\.?\s*$/i, "");
  return text.trim();
}

function FeedbackBanner({ feedback }: { feedback: string }) {
  return (
    <div className="bg-semantic-orange/10 border border-semantic-orange/30 rounded-panel px-4 py-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-semantic-orange mb-1">
        Attending Feedback
      </h3>
      <p className="text-sm text-text-primary leading-relaxed">{cleanFeedback(feedback)}</p>
    </div>
  );
}

/** Strip markdown to plain text */
function stripMarkdown(text: string): string {
  return text
    .replace(/\*\*([^*]*)\*\*/g, "$1")      // **bold** → bold
    .replace(/^\s*\*\s+/gm, "")             // * bullet lines → remove bullet
    .replace(/^\s*[-•]\s+/gm, "")           // - bullet → remove
    .replace(/\*([^*]+)\*/g, "$1")           // *italic* → italic
    .replace(/^\s*#+\s*/gm, "")             // ### heading → remove
    .replace(/\n{3,}/g, "\n\n")             // collapse blank lines
    .trim();
}

function parseReport(report: string) {
  // 1. Standard agent format: FINDINGS: ... IMPRESSION: ...
  // 1. Standard agent format: line starts with FINDINGS: (not **Key Findings:**)
  const findingsMatch = report.match(
    /^FINDINGS:\s*([\s\S]*?)(?=\n\s*IMPRESSION:|$)/im
  );
  const impressionMatch = report.match(/^IMPRESSION:\s*([\s\S]*?)$/im);

  if (findingsMatch) {
    let findings = findingsMatch[1]?.trim() || "";
    const impression = impressionMatch?.[1]?.trim() || "";
    // Strip CheXagent-2 [Category] tags and **bold** if present
    if (findings.includes("[") && findings.includes("]")) {
      findings = findings.replace(/\[.*?\]\s*/g, "").replace(/\*\*([^*]*)\*\*/g, "$1").trim();
    }
    return { findings, impression };
  }

  // 2. MedGemma / markdown VLM format
  //    Split by known section headers, then strip markdown
  const lines = report.split("\n");
  const sections: Record<string, string[]> = {};
  let currentSection = "preamble";

  for (const line of lines) {
    const stripped = line.replace(/\*/g, "").trim().toLowerCase();
    // Findings headers: "Key Findings:", "Key findings include:", "Chest X-ray findings:", "Report:", etc.
    if (/^(key\s+|chest\s+x-?ray\s+)?findings\s*(include)?:?$/.test(stripped) ||
        /^report:?$/.test(stripped) ||
        /shows\s+several\s+(key\s+|notable\s+)?findings:?$/.test(stripped)) {
      currentSection = "findings";
      continue;
    }
    // Impression headers: "Overall Impression:", "Impression:", or "Overall, ..." / "Overall: ..."
    if (/^(overall\s+)?impression:?$/.test(stripped)) {
      currentSection = "impression";
      continue;
    }
    if (/^overall[,:]/.test(stripped)) {
      currentSection = "impression";
      // This line has content after "Overall, " — keep it
      const afterOverall = line.replace(/\*/g, "").replace(/^[^,:]+[,:]\s*/i, "").trim();
      if (afterOverall) {
        if (!sections[currentSection]) sections[currentSection] = [];
        sections[currentSection].push(afterOverall);
      }
      continue;
    }
    if (/^disclaimer:?/i.test(stripped) || /^it is important to note/i.test(stripped)) {
      currentSection = "disclaimer";
      continue;
    }
    if (!sections[currentSection]) sections[currentSection] = [];
    sections[currentSection].push(line);
  }

  if (sections["findings"] || sections["impression"]) {
    // If no explicit findings header, treat preamble as findings
    const findingsLines = sections["findings"] || sections["preamble"] || [];
    return {
      findings: stripMarkdown(findingsLines.join("\n")),
      impression: stripMarkdown((sections["impression"] || []).join("\n")),
    };
  }

  // 3. MedGemma without explicit headers — bullet list is findings,
  //    "Overall Impression" embedded with ** markers
  if (report.includes("*   **")) {
    // Everything before "Overall Impression" or "Disclaimer" is findings
    const parts = report.split(/\*?\*?(?:Overall\s+)?Impression:?\*?\*?/i);
    const findingsRaw = parts[0] || "";
    const impressionRaw = (parts[1] || "").split(/\*?\*?Disclaimer/i)[0] || "";
    // Strip preamble (first line before bullets)
    const findingsClean = findingsRaw.replace(/^[^*]*\n/, "");
    return {
      findings: stripMarkdown(findingsClean),
      impression: stripMarkdown(impressionRaw),
    };
  }

  // 4. CheXagent-2 format: [Category] **text**
  if (report.includes("[") && report.includes("]")) {
    const cleaned = report
      .replace(/\[.*?\]\s*/g, "")
      .replace(/\*\*([^*]*)\*\*/g, "$1");
    return { findings: cleaned.trim(), impression: "" };
  }

  // 5. Plain text (CheXOne etc.)
  return { findings: report.trim(), impression: "" };
}

// Map model selector IDs to run modes
const MODEL_TO_RUN_MODE: Record<string, string> = {
  agent_initial: "agent",
  agent_initial_guided: "agent_guided",
  chexagent2: "chexagent2",
  chexone: "chexone",
  medgemma: "medgemma",
};

export default function ReportPanel({
  study,
  predictions,
  models,
}: ReportPanelProps) {
  const { selectedModel, setSelectedModel, showGroundTruth, toggleGroundTruth } =
    useStudyStore();

  // Run state — from store (shared with WorkflowPanel)
  const liveRuns = useStudyStore((s) => s.liveRuns);
  const setLiveRun = useStudyStore((s) => s.setLiveRun);
  const updateLiveRun = useStudyStore((s) => s.updateLiveRun);
  const [feedback, setFeedback] = useState("");
  const [showFeedbackInput, setShowFeedbackInput] = useState(false);
  const [ritlTab, setRitlTab] = useState<"revised" | "original" | "diff">("revised");
  const pollRefs = useRef<Record<string, NodeJS.Timeout>>({});

  useEffect(() => {
    return () => { Object.values(pollRefs.current).forEach(clearInterval); };
  }, []);

  const startPolling = useCallback((runId: string, modelKey: string) => {
    if (pollRefs.current[modelKey]) clearInterval(pollRefs.current[modelKey]);
    pollRefs.current[modelKey] = setInterval(async () => {
      try {
        const status = await pollRun(runId);
        setLiveRun(modelKey, status);
        if (status.status === "complete" || status.status === "error") {
          clearInterval(pollRefs.current[modelKey]);
          delete pollRefs.current[modelKey];
        }
      } catch {
        clearInterval(pollRefs.current[modelKey]);
        delete pollRefs.current[modelKey];
      }
    }, 2000);
  }, [setLiveRun]);

  const handleRunMode = useCallback(async (modelKey: string) => {
    const mode = MODEL_TO_RUN_MODE[modelKey];
    if (!mode) return;
    setShowFeedbackInput(false);
    setFeedback("");
    const empty: RunStatus = { run_id: "", study_id: study.study_id, mode, status: "queued", message: "Starting...", result: null, trajectory: null, ritl_result: null, ritl_trajectory_steps: null, elapsed_ms: 0 };
    setLiveRun(modelKey, empty);
    try {
      const { run_id } = await startRun(study.study_id, mode);
      startPolling(run_id, modelKey);
    } catch (e) {
      setLiveRun(modelKey, { ...empty, status: "error", message: String(e) });
    }
  }, [study.study_id, startPolling, setLiveRun]);

  const BASELINE_MODELS = ["chexagent2", "chexone", "medgemma"];

  const handleRunBaselines = useCallback(async () => {
    for (const key of BASELINE_MODELS) {
      if (!predictions[key]) {
        handleRunMode(key);
      }
    }
  }, [predictions, handleRunMode]);

  const handleFeedback = useCallback(async () => {
    const agentRun = liveRuns["agent_initial"] || liveRuns["agent_initial_guided"];
    if (!agentRun || !feedback.trim()) return;
    if (agentRun.status !== "complete" && agentRun.status !== "awaiting_feedback") {
      alert("Agent run must complete or reach checkpoint before giving feedback.");
      return;
    }
    try {
      const modelKey = liveRuns["agent_initial"] ? "agent_initial" : "agent_initial_guided";
      const feedbackText = feedback.trim();
      await submitFeedback(agentRun.run_id, feedbackText);
      // Store feedback text on the run so trajectory panel can show it during streaming
      updateLiveRun(modelKey, { _pendingFeedback: feedbackText } as Partial<RunStatus>);
      setShowFeedbackInput(false);
      setFeedback("");
      setRitlTab("revised");
      startPolling(agentRun.run_id, modelKey);
    } catch (e) {
      const modelKey = liveRuns["agent_initial"] ? "agent_initial" : "agent_initial_guided";
      setLiveRun(modelKey, { ...agentRun, status: "error", message: String(e) });
    }
  }, [liveRuns, feedback, startPolling, setLiveRun]);

  // Use live result if available, otherwise pre-computed
  const liveRun = liveRuns[selectedModel] || null;
  const hasRitl = !!(liveRun?.ritl_result);
  const origPred = liveRun?.status === "complete" && liveRun.result
    ? (liveRun.result as unknown as Prediction)
    : predictions[selectedModel];
  const ritlPred = hasRitl ? (liveRun!.ritl_result as unknown as Prediction) : null;
  const pred = hasRitl && ritlTab === "revised" ? ritlPred! : origPred;
  const parsedPred = pred ? parseReport(pred.report_pred) : null;
  const parsedOrig = origPred ? parseReport(origPred.report_pred) : null;
  const parsedRitl = ritlPred ? parseReport(ritlPred.report_pred) : null;
  const parsedGT = {
    findings: study.findings,
    impression: study.impression,
  };

  // For RITL models, get the base prediction to show diff
  const isRITL = selectedModel.includes("ritl");
  const basePred = isRITL ? predictions[pred?.base_model || "agent_initial"] : null;
  const parsedBase = basePred ? parseReport(basePred.report_pred) : null;

  // Always show runnable models + any that have predictions
  const RUNNABLE = ["agent_initial", "chexagent2", "chexone", "medgemma"];
  const allBaseModels = Array.from(new Set([
    ...RUNNABLE,
    ...models.filter((m) => !m.includes("ritl")),
  ]));
  const ritlModels = models.filter((m) => m.includes("ritl"));

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Model selector */}
      <div className="px-4 pt-4 pb-2 border-b border-separator space-y-2">
        <div className="flex items-center gap-2 flex-wrap">
          {allBaseModels.map((m) => {
            const hasPred = !!predictions[m];
            const lr = liveRuns[m];
            const isRunning = lr?.status === "queued" || lr?.status === "running";
            const isDone = lr?.status === "complete";
            const label = m.replace("agent_initial", "Agent").replace("chexagent2", "CheXagent-2").replace("chexone", "CheXOne").replace("medgemma", "MedGemma");
            return (
              <button
                key={m}
                onClick={() => setSelectedModel(m)}
                className={`px-3 py-1 text-xs rounded-full transition-colors inline-flex items-center gap-1 ${
                  selectedModel === m
                    ? "bg-accent text-white"
                    : hasPred || isDone
                      ? "bg-bg-elevated text-white hover:text-white/80"
                      : "bg-bg-elevated/50 text-text-tertiary hover:text-white border border-dashed border-separator"
                }`}
              >
                {label}
                {isRunning && <span className="inline-block w-2 h-2 border border-current border-t-transparent rounded-full animate-spin" />}
                {isDone && !hasPred && <span className="text-semantic-green">&#10003;</span>}
              </button>
            );
          })}
        </div>
        {ritlModels.length > 0 && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[10px] text-semantic-orange font-semibold uppercase tracking-wider">RITL</span>
              {ritlModels.map((m) => (
                <button
                  key={m}
                  onClick={() => setSelectedModel(m)}
                  className={`px-3 py-1 text-xs rounded-full transition-colors ${
                    selectedModel === m
                      ? "bg-semantic-orange text-black"
                      : "bg-semantic-orange/10 text-semantic-orange hover:bg-semantic-orange/20"
                  }`}
                >
                  {m.includes("checkpoint") ? "Guided" : "Revised"}
                </button>
              ))}
            </div>
            {selectedModel.includes("ritl") && (
              <p className="text-xs text-white leading-snug">
                {selectedModel.includes("checkpoint")
                  ? "Attending reviewed findings mid-loop before the agent wrote its report."
                  : "Attending reviewed the draft report, then the agent re-investigated and revised."}
              </p>
            )}
          </div>
        )}
        <div className="flex gap-3 text-[10px]">
          <button
            onClick={toggleGroundTruth}
            className={`uppercase tracking-wider transition-colors ${
              showGroundTruth ? "text-semantic-green" : "text-text-tertiary hover:text-text-secondary"
            }`}
          >
            {showGroundTruth ? "Hide" : "Show"} Ground Truth
          </button>
          {MODEL_TO_RUN_MODE[selectedModel] && (pred || liveRun?.status === "complete") && (
            <button
              onClick={() => handleRunMode(selectedModel)}
              className="px-3 py-1 text-[10px] rounded-full bg-bg-elevated text-text-primary hover:text-white transition-colors"
            >
              Re-run
            </button>
          )}
        </div>
      </div>

      {/* Report content */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {/* RITL feedback banner */}
        {pred?.feedback && <FeedbackBanner feedback={pred.feedback} />}

        {/* RITL tab switcher — Original / Revised / Diff */}
        {hasRitl && (
          <div className="flex items-center gap-1 bg-bg-elevated rounded-full p-0.5 w-fit">
            {(["revised", "original", "diff"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setRitlTab(tab)}
                className={`px-3 py-1 text-xs rounded-full transition-colors capitalize ${
                  ritlTab === tab
                    ? "bg-semantic-orange text-black font-semibold"
                    : "text-white/70 hover:text-white"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        )}


        {/* Side-by-side: Generated vs GT */}
        {showGroundTruth ? (
          <div className="space-y-4">
            {/* Column headers */}
            <div className="grid grid-cols-2 gap-3">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-text-secondary">
                Generated
              </div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-semantic-green">
                Ground Truth
              </div>
            </div>

            {/* Findings side-by-side */}
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-text-tertiary mb-2">
                Findings
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <p className="text-sm text-text-primary leading-relaxed whitespace-pre-wrap">
                  {parsedPred?.findings || <span className="text-text-tertiary italic">—</span>}
                </p>
                <p className="text-sm text-semantic-green/70 leading-relaxed whitespace-pre-wrap bg-semantic-green/5 rounded p-2">
                  {parsedGT.findings || <span className="text-text-tertiary italic">—</span>}
                </p>
              </div>
            </div>

            {/* Impression side-by-side */}
            {(parsedPred?.impression || parsedGT.impression) && (
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-text-tertiary mb-2">
                  Impression
                </h3>
                <div className="grid grid-cols-2 gap-3">
                  <p className="text-sm text-text-primary leading-relaxed whitespace-pre-wrap">
                    {parsedPred?.impression || <span className="text-text-tertiary italic">—</span>}
                  </p>
                  <p className="text-sm text-semantic-green/70 leading-relaxed whitespace-pre-wrap bg-semantic-green/5 rounded p-2">
                    {parsedGT.impression || <span className="text-text-tertiary italic">—</span>}
                  </p>
                </div>
              </div>
            )}

            {/* Run buttons inside GT view when no prediction */}
            {!pred && !liveRun && (
              <div className="flex gap-2 pt-2">
                {selectedModel === "agent_initial" && (
                  <button onClick={() => handleRunMode("agent_initial")} className="px-4 py-1.5 text-xs font-semibold rounded-full bg-semantic-green text-black hover:bg-semantic-green/80">Run Agent</button>
                )}
                {selectedModel !== "agent_initial" && MODEL_TO_RUN_MODE[selectedModel] && (
                  <button onClick={() => handleRunMode(selectedModel)} className="px-4 py-1.5 text-xs font-semibold rounded-full bg-semantic-green text-black hover:bg-semantic-green/80">
                    Run {selectedModel.replace("chexagent2", "CheXagent-2").replace("chexone", "CheXOne").replace("medgemma", "MedGemma")}
                  </button>
                )}
                <button onClick={handleRunBaselines} className="px-4 py-1.5 text-xs rounded-full bg-accent text-white hover:bg-accent-hover">Run All Baselines</button>
              </div>
            )}
          </div>
        ) : isRITL && parsedBase && parsedPred ? (
          /* Diff view: original → revised for RITL */
          <div className="space-y-4">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-text-tertiary">
              Changes from original
              <span className="ml-2 font-normal">
                (<span className="text-semantic-red">removed</span>
                {" / "}
                <span className="text-semantic-green">added</span>)
              </span>
            </div>
            <ReportDiff
              original={parsedBase.findings}
              revised={parsedPred.findings}
              label="Findings"
            />
            <ReportDiff
              original={parsedBase.impression}
              revised={parsedPred.impression}
              label="Impression"
            />
          </div>
        ) : (
          <>
            {/* Report display — plain or diff depending on RITL tab */}
            {hasRitl && ritlTab === "diff" && parsedOrig && parsedRitl ? (
              <div className="space-y-4">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-text-tertiary">
                  Changes from original
                  <span className="ml-2 font-normal">
                    (<span className="text-semantic-red">removed</span>
                    {" / "}
                    <span className="text-semantic-green">added</span>)
                  </span>
                </div>
                <ReportDiff
                  original={parsedOrig.findings}
                  revised={parsedRitl.findings}
                  label="Findings"
                />
                <ReportDiff
                  original={parsedOrig.impression}
                  revised={parsedRitl.impression}
                  label="Impression"
                />
              </div>
            ) : parsedPred ? (
              <>
                <ReportSection
                  label="Findings"
                  text={parsedPred.findings}
                  color="text-text-secondary"
                />
                <ReportSection
                  label="Impression"
                  text={parsedPred.impression}
                  color="text-text-secondary"
                />
              </>
            ) : null}
            {!pred && !liveRun && (
              <div className="text-center py-8 space-y-4">
                <p className="text-sm text-text-tertiary">
                  No result yet.
                </p>
                <div className="flex flex-col items-center gap-3">
                  {selectedModel === "agent_initial" && (
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleRunMode("agent_initial")}
                        className="px-5 py-2 text-sm font-semibold rounded-full bg-semantic-green text-black hover:bg-semantic-green/80 transition-colors"
                      >
                        Run Agent (Revised)
                      </button>
                      <button
                        onClick={() => {
                          // Store guided run under "agent_initial" key so it shows in the same tab
                          const mode = "agent_guided";
                          const empty: RunStatus = { run_id: "", study_id: study.study_id, mode, status: "queued", message: "Starting guided agent...", result: null, trajectory: null, ritl_result: null, ritl_trajectory_steps: null, elapsed_ms: 0 };
                          setLiveRun("agent_initial", empty);
                          setShowFeedbackInput(false);
                          setFeedback("");
                          startRun(study.study_id, mode).then(({ run_id }) => {
                            startPolling(run_id, "agent_initial");
                          }).catch((e) => {
                            setLiveRun("agent_initial", { ...empty, status: "error", message: String(e) });
                          });
                        }}
                        className="px-5 py-2 text-sm font-semibold rounded-full bg-semantic-orange text-black hover:bg-semantic-orange/80 transition-colors"
                      >
                        Run Agent (Guided)
                      </button>
                    </div>
                  )}
                  {selectedModel !== "agent_initial" && MODEL_TO_RUN_MODE[selectedModel] && (
                    <button
                      onClick={() => handleRunMode(selectedModel)}
                      className="px-5 py-2 text-sm font-semibold rounded-full bg-semantic-green text-black hover:bg-semantic-green/80 transition-colors"
                    >
                      Run {selectedModel.replace("chexagent2", "CheXagent-2").replace("chexone", "CheXOne").replace("medgemma", "MedGemma")}
                    </button>
                  )}
                  <button
                    onClick={handleRunBaselines}
                    className="px-4 py-1.5 text-xs rounded-full bg-accent text-white hover:bg-accent-hover transition-colors"
                  >
                    Run All Baselines
                  </button>
                </div>
              </div>
            )}
            {/* Live run status */}
            {liveRun && (liveRun.status === "queued" || liveRun.status === "running") && (
              <div className="text-center py-8 space-y-2">
                <div className="inline-block w-6 h-6 border-2 border-semantic-orange border-t-transparent rounded-full animate-spin" />
                <p className="text-sm text-semantic-orange">
                  {liveRun.message}
                </p>
                <p className="text-[10px] text-text-tertiary">
                  {(liveRun.elapsed_ms / 1000).toFixed(0)}s elapsed
                </p>
              </div>
            )}
            {liveRun?.status === "awaiting_feedback" && (
              <div className="py-3 space-y-3">
                <div className="bg-semantic-orange/10 border border-semantic-orange/30 rounded-panel px-4 py-3">
                  <p className="text-xs font-semibold uppercase tracking-wider text-semantic-orange mb-2">
                    Agent Summary — Awaiting Review
                  </p>
                  <div className="text-sm text-text-primary leading-relaxed whitespace-pre-wrap">
                    {liveRun.message}
                  </div>
                </div>
              </div>
            )}
            {liveRun?.status === "error" && (
              <div className="text-center py-4 space-y-2">
                <p className="text-sm text-semantic-red">{liveRun.message}</p>
                <button
                  onClick={() => handleRunMode(selectedModel)}
                  className="px-4 py-1 text-xs rounded-full bg-bg-elevated text-text-primary hover:text-white"
                >
                  Retry
                </button>
              </div>
            )}
          </>
        )}

        {/* Stats */}
        {pred && (
          <div className="border-t border-separator pt-3 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-text-tertiary">
            {pred.num_steps != null && <span>{pred.num_steps} steps</span>}
            {pred.wall_time_ms != null && (
              <span>{(pred.wall_time_ms / 1000).toFixed(1)}s</span>
            )}
            {pred.input_tokens != null && (
              <span>{pred.input_tokens.toLocaleString()} in / {pred.output_tokens?.toLocaleString()} out tokens</span>
            )}
            {pred.unused_tools && pred.unused_tools.length > 0 && (
              <span>unused: {pred.unused_tools.join(", ")}</span>
            )}
          </div>
        )}

        {/* RITL feedback — available for live agent runs (Revised or Guided), supports multiple rounds */}
        {(liveRun?.status === "complete" || liveRun?.status === "awaiting_feedback") &&
         (liveRun.mode === "agent" || liveRun.mode === "agent_guided") && (
          <div className="border-t border-separator pt-3 space-y-2">
            {!showFeedbackInput ? (
              <button
                onClick={() => setShowFeedbackInput(true)}
                className="px-4 py-1.5 text-xs rounded-full bg-semantic-orange/10 text-semantic-orange hover:bg-semantic-orange/20 transition-colors"
              >
                Give Attending Feedback
              </button>
            ) : (
              <div className="space-y-2">
                <textarea
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  placeholder="e.g. I don't see effusion — re-examine the costophrenic angles"
                  className="w-full bg-bg-surface border border-separator rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-semantic-orange resize-none"
                  rows={3}
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleFeedback}
                    disabled={!feedback.trim()}
                    className="px-4 py-1 text-xs font-semibold rounded-full bg-semantic-orange text-black hover:bg-semantic-orange/80 disabled:opacity-50"
                  >
                    Submit & Re-run
                  </button>
                  <button
                    onClick={() => { setShowFeedbackInput(false); setFeedback(""); }}
                    className="px-3 py-1 text-xs rounded-full bg-bg-elevated text-text-secondary"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
