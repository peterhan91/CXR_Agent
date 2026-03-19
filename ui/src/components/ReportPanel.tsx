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
  const findingsMatch = report.match(
    /FINDINGS:\s*([\s\S]*?)(?=\n\s*IMPRESSION:|$)/i
  );
  const impressionMatch = report.match(/IMPRESSION:\s*([\s\S]*?)$/i);

  if (findingsMatch) {
    return {
      findings: findingsMatch[1]?.trim() || "",
      impression: impressionMatch?.[1]?.trim() || "",
    };
  }

  // 2. MedGemma / markdown VLM format
  //    Split by known section headers, then strip markdown
  const lines = report.split("\n");
  const sections: Record<string, string[]> = {};
  let currentSection = "preamble";

  for (const line of lines) {
    const stripped = line.replace(/\*/g, "").trim().toLowerCase();
    if (/^key\s+findings:?$/i.test(stripped) || /^findings:?$/i.test(stripped)) {
      currentSection = "findings";
      continue;
    }
    if (/^(overall\s+)?impression:?$/i.test(stripped)) {
      currentSection = "impression";
      continue;
    }
    if (/^disclaimer:?/i.test(stripped)) {
      currentSection = "disclaimer";
      continue;
    }
    if (!sections[currentSection]) sections[currentSection] = [];
    sections[currentSection].push(line);
  }

  if (sections["findings"] || sections["impression"]) {
    return {
      findings: stripMarkdown((sections["findings"] || []).join("\n")),
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

  // Run state
  const [liveRun, setLiveRun] = useState<RunStatus | null>(null);
  const [feedback, setFeedback] = useState("");
  const [showFeedbackInput, setShowFeedbackInput] = useState(false);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  // Clear live run when switching models
  useEffect(() => {
    setLiveRun(null);
    setShowFeedbackInput(false);
    setFeedback("");
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  }, [selectedModel]);

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const startPolling = useCallback((runId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const status = await pollRun(runId);
        setLiveRun(status);
        if (status.status === "complete" || status.status === "error") {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }, 2000);
  }, []);

  const handleRun = useCallback(async () => {
    const mode = MODEL_TO_RUN_MODE[selectedModel];
    if (!mode) return;
    setLiveRun({ run_id: "", study_id: study.study_id, mode, status: "queued", message: "Starting...", result: null, trajectory: null, ritl_result: null, ritl_trajectory_steps: null, elapsed_ms: 0 });
    try {
      const { run_id } = await startRun(study.study_id, mode);
      startPolling(run_id);
    } catch (e) {
      setLiveRun((prev) => prev ? { ...prev, status: "error", message: String(e) } : null);
    }
  }, [selectedModel, study.study_id, startPolling]);

  const handleFeedback = useCallback(async () => {
    if (!liveRun || !feedback.trim()) return;
    try {
      await submitFeedback(liveRun.run_id, feedback.trim());
      setShowFeedbackInput(false);
      setFeedback("");
      startPolling(liveRun.run_id);
    } catch (e) {
      setLiveRun((prev) => prev ? { ...prev, status: "error", message: String(e) } : null);
    }
  }, [liveRun, feedback, startPolling]);

  // Use live result if available, otherwise pre-computed
  const pred = liveRun?.status === "complete" && liveRun.result
    ? (liveRun.result as unknown as Prediction)
    : predictions[selectedModel];
  const parsedPred = pred ? parseReport(pred.report_pred) : null;
  const parsedGT = {
    findings: study.findings,
    impression: study.impression,
  };

  // For RITL models, get the base prediction to show diff
  const isRITL = selectedModel.includes("ritl");
  const basePred = isRITL ? predictions[pred?.base_model || "agent_initial"] : null;
  const parsedBase = basePred ? parseReport(basePred.report_pred) : null;

  // Organize models: base models first, then RITL variants
  const baseModels = models.filter(
    (m) => !m.includes("ritl")
  );
  const ritlModels = models.filter((m) => m.includes("ritl"));

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Model selector */}
      <div className="px-4 pt-4 pb-2 border-b border-separator space-y-2">
        <div className="flex items-center gap-2 flex-wrap">
          {baseModels.map((m) => (
            <button
              key={m}
              onClick={() => setSelectedModel(m)}
              className={`px-3 py-1 text-xs rounded-full transition-colors ${
                selectedModel === m
                  ? "bg-accent text-white"
                  : "bg-bg-elevated text-text-secondary hover:text-text-primary"
              }`}
            >
              {m.replace("agent_initial", "Agent").replace("chexagent2", "CheXagent-2").replace("chexone", "CheXOne").replace("medgemma", "MedGemma").replace("medversa", "MedVersa")}
            </button>
          ))}
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
        </div>
      </div>

      {/* Report content */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {/* RITL feedback banner */}
        {pred?.feedback && <FeedbackBanner feedback={pred.feedback} />}


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
            {/* Single-column generated report */}
            {parsedPred && (
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
            )}
            {!pred && !liveRun && MODEL_TO_RUN_MODE[selectedModel] && (
              <div className="text-center py-8 space-y-3">
                <p className="text-sm text-text-tertiary">
                  No pre-computed result for this model.
                </p>
                <button
                  onClick={handleRun}
                  className="px-5 py-2 text-sm font-semibold rounded-full bg-semantic-green text-black hover:bg-semantic-green/80 transition-colors"
                >
                  Run {selectedModel === "agent_initial" ? "Agent" : selectedModel.replace("chexagent2", "CheXagent-2").replace("chexone", "CheXOne").replace("medgemma", "MedGemma")}
                </button>
              </div>
            )}
            {!pred && !liveRun && !MODEL_TO_RUN_MODE[selectedModel] && (
              <p className="text-sm text-text-tertiary italic">
                No prediction from this model for this study.
              </p>
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
            {liveRun?.status === "error" && (
              <div className="text-center py-4 space-y-2">
                <p className="text-sm text-semantic-red">{liveRun.message}</p>
                <button
                  onClick={handleRun}
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

        {/* RITL feedback — available for live agent runs */}
        {liveRun?.status === "complete" && liveRun.mode === "agent" && !liveRun.ritl_result && (
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

        {/* RITL revised result */}
        {liveRun?.ritl_result && (
          <div className="border-t border-separator pt-3 space-y-2">
            <div className="bg-semantic-orange/10 border border-semantic-orange/30 rounded-panel px-3 py-2">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-semantic-orange">
                Revised after feedback
              </span>
            </div>
            <div className="text-sm text-text-primary leading-relaxed whitespace-pre-wrap">
              {(liveRun.ritl_result as Record<string, string>).report_pred}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
