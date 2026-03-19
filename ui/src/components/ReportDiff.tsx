"use client";

import { diffWords } from "diff";

interface ReportDiffProps {
  original: string;
  revised: string;
  label: string;
}

export default function ReportDiff({ original, revised, label }: ReportDiffProps) {
  if (!original && !revised) return null;

  const changes = diffWords(original || "", revised || "");

  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wider text-text-tertiary mb-2">
        {label}
      </h3>
      <div className="text-sm leading-relaxed whitespace-pre-wrap">
        {changes.map((part, i) => {
          if (part.added) {
            return (
              <span
                key={i}
                className="bg-semantic-green/20 text-semantic-green rounded-sm px-0.5"
              >
                {part.value}
              </span>
            );
          }
          if (part.removed) {
            return (
              <span
                key={i}
                className="bg-semantic-red/20 text-semantic-red line-through rounded-sm px-0.5"
              >
                {part.value}
              </span>
            );
          }
          return (
            <span key={i} className="text-text-primary">
              {part.value}
            </span>
          );
        })}
      </div>
    </div>
  );
}
