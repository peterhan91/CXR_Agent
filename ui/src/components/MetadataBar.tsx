"use client";

import { useState } from "react";
import type { TestStudy } from "@/lib/types";

interface MetadataBarProps {
  study: TestStudy;
}

export default function MetadataBar({ study }: MetadataBarProps) {
  const [hidden, setHidden] = useState(false);
  const meta = study.metadata;
  if (!meta) return null;

  // Resolve age/sex from top-level or admission_info
  const age =
    meta.age ?? meta.admission_info?.demographics?.age;
  const sex =
    meta.sex ?? meta.admission_info?.demographics?.gender;
  const indication = (meta.indication || "").trim();
  const rawComparison = (meta.comparison || "").trim();
  const viewPos = meta.view_position;

  // When a linked prior exists, show its actual study date instead of the
  // raw Comparison text (which may have inconsistent de-identified dates).
  const priorDate = study.prior_study?.study_date;
  const comparison = priorDate
    ? `${priorDate.slice(0, 4)}-${priorDate.slice(4, 6)}-${priorDate.slice(6, 8)}`
    : rawComparison;

  // Don't render if nothing to show
  const hasDemographics = age != null || sex;
  const hasClinical = !!indication || !!comparison;
  if (!hasDemographics && !hasClinical && !viewPos) return null;

  // Format age+sex as compact badge
  const sexLabel = sex
    ? String(sex).toUpperCase() === "M" || String(sex).toUpperCase() === "MALE"
      ? "M"
      : "F"
    : "";
  const demoLabel = age != null ? `${age}${sexLabel}` : sexLabel;

  if (hidden) {
    return (
      <div className="bg-bg-surface border-t border-separator px-3 py-1.5 flex items-center">
        <button
          onClick={() => setHidden(false)}
          className="px-3.5 py-1 rounded bg-bg-elevated text-[13px] text-white hover:bg-bg-elevated/80 transition-colors"
        >
          Show Clinical Info
        </button>
      </div>
    );
  }

  return (
    <div className="bg-bg-surface border-t border-separator">
      <div className="px-3 py-2 flex items-center gap-2 flex-wrap">
        {demoLabel && (
          <span className="px-2.5 py-1 rounded bg-bg-elevated text-xs text-white font-bold">
            {demoLabel}
          </span>
        )}
        {viewPos && (
          <span className="px-2.5 py-1 rounded bg-bg-elevated text-xs text-white">
            {viewPos}
          </span>
        )}
        {study.is_followup && (
          <span className="px-2.5 py-1 rounded bg-bg-elevated text-xs text-semantic-orange">
            Follow-up
          </span>
        )}
        {indication && (
          <span className="px-2.5 py-1 rounded bg-bg-elevated text-xs text-white w-full break-words">
            <span className="font-bold">Indication: </span>
            {indication}
          </span>
        )}
        {comparison && comparison !== "___" && comparison !== "___." && (
          <span className="px-2.5 py-1 rounded bg-bg-elevated text-xs text-white max-w-full">
            <span className="font-bold">Comparison: </span>
            {comparison}
          </span>
        )}
        <button
          onClick={() => setHidden(true)}
          className="ml-auto px-3 py-1 rounded bg-bg-elevated text-xs text-white hover:bg-bg-elevated/80 transition-colors"
        >
          Hide
        </button>
      </div>
    </div>
  );
}
