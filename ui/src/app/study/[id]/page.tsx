"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { getStudyDetail } from "@/lib/api";
import type { StudyDetailResponse } from "@/lib/types";
import { useStudyStore } from "@/stores/studyStore";
import CXRViewer from "@/components/CXRViewer";
import ReportPanel from "@/components/ReportPanel";
import WorkflowPanel from "@/components/WorkflowPanel";

export default function StudyViewerPage() {
  const params = useParams();
  const studyId = params.id as string;
  const [data, setData] = useState<StudyDetailResponse | null>(null);
  const [error, setError] = useState<string>("");
  const resetStudy = useStudyStore((s) => s.resetStudy);

  useEffect(() => {
    if (!studyId) return;
    resetStudy();
    setData(null);
    setError("");
    getStudyDetail(decodeURIComponent(studyId))
      .then(setData)
      .catch((e) => setError(e.message));
  }, [studyId, resetStudy]);

  if (error) {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <div className="text-semantic-red text-sm">{error}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <div className="text-text-tertiary text-sm">Loading study...</div>
      </div>
    );
  }

  const { study, predictions, trajectories } = data;
  const models = Object.keys(predictions);

  // Metadata badges
  const dataset = study.dataset;
  const viewPos = study.metadata?.view_position;
  const chexpertLabels = study.metadata?.chexpert_labels || {};
  const positiveLabels = Object.entries(chexpertLabels)
    .filter(([, v]) => v === "1.0")
    .map(([k]) => k);

  return (
    <div className="h-screen bg-bg flex flex-col overflow-hidden">
      {/* Top bar */}
      <header className="border-b border-separator px-4 py-2 flex items-center gap-4 flex-shrink-0">
        <Link
          href="/"
          className="text-xs text-accent hover:text-accent-hover transition-colors"
        >
          &larr; Back
        </Link>
        <h1 className="text-sm font-semibold text-text-primary">
          {study.study_id}
        </h1>
        <span className="text-[10px] text-text-tertiary">{dataset}</span>
        {viewPos && (
          <span className="text-[10px] text-text-tertiary">{viewPos}</span>
        )}
        {study.is_followup && (
          <span className="text-[10px] text-semantic-orange">Follow-up</span>
        )}
        {positiveLabels.length > 0 && (
          <div className="flex gap-1 ml-auto">
            {positiveLabels.map((label) => (
              <span
                key={label}
                className="px-2 py-0.5 text-[10px] rounded bg-bg-elevated text-text-secondary"
              >
                {label}
              </span>
            ))}
          </div>
        )}
      </header>

      {/* 3-panel layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: CXR Image (~5/12) */}
        <div className="w-5/12 border-r border-separator">
          <CXRViewer
            imagePath={study.image_path}
            priorImagePath={study.prior_study?.image_path}
          />
        </div>

        {/* Middle: Report (~1/3) */}
        <div className="w-1/3 border-r border-separator">
          <ReportPanel
            study={study}
            predictions={predictions}
            models={models}
          />
        </div>

        {/* Right: Agent Workflow */}
        <div className="flex-1">
          <WorkflowPanel trajectories={trajectories} />
        </div>
      </div>
    </div>
  );
}
