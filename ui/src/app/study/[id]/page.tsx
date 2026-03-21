"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { getStudyDetail } from "@/lib/api";
import type { StudyDetailResponse } from "@/lib/types";
import { useStudyStore } from "@/stores/studyStore";
import CXRViewer from "@/components/CXRViewer";
import MetadataBar from "@/components/MetadataBar";
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
  const chexpertLabels = study.metadata?.chexpert_labels || {};
  const positiveLabels = Object.entries(chexpertLabels)
    .filter(([, v]) => v === "1.0")
    .map(([k]) => k);

  // Feature context for trajectory highlights
  const meta = study.metadata;
  const hasAge = !!(meta?.age ?? meta?.admission_info?.demographics?.age);
  const hasIndication = !!(meta?.indication || "").trim();
  const featureContext = {
    hasMetadata: hasAge || hasIndication,
    hasPrior: !!(study.prior_study?.image_path),
    hasLateral: !!(study.lateral_image_path),
  };

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
        <span className="text-xs text-text-tertiary">{dataset}</span>
        {positiveLabels.length > 0 && (
          <div className="flex gap-1 ml-auto">
            {positiveLabels.map((label) => (
              <span
                key={label}
                className="px-2 py-0.5 text-xs rounded bg-bg-elevated text-text-secondary"
              >
                {label}
              </span>
            ))}
          </div>
        )}
      </header>

      {/* 3-panel layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: CXR Image + Metadata (~5/12) */}
        <div className="w-5/12 border-r border-separator flex flex-col">
          <div className="flex-1 min-h-0">
            <CXRViewer
              imagePath={study.image_path}
              priorImagePath={study.prior_study?.image_path}
              priorStudyDate={study.prior_study?.study_date}
              lateralImagePath={study.lateral_image_path}
            />
          </div>
          <MetadataBar study={study} />
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
          <WorkflowPanel
            trajectories={trajectories}
            featureContext={featureContext}
            lateralImagePath={study.lateral_image_path}
          />
        </div>
      </div>
    </div>
  );
}
